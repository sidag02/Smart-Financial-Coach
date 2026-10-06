#!/usr/bin/env bash
# Check chat on the deployed app the way a visitor uses it (FR-13 to FR-15 design, §6): sign in
# as a demo user, ask one question, and pass only on a sourced answer the grounding check let
# through. A redirect, "isn't available", "couldn't be reached" or the safe message fails it.
# The demo password comes from SFC_DEMO_PASSWORD, or from the deployed app's own secret; costs one
# coach answer.
#
#   deploy/azure/check-chat.sh                      # the deployed app
#   deploy/azure/check-chat.sh http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ $# -ge 1 ]; then
  BASE="${1%/}"
else
  source deploy/azure/config.sh
  BASE="https://$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv)"
fi
# The demo password: from the environment (a local server), else the deployed app's own secret
PASSWORD="${SFC_DEMO_PASSWORD:-}"
if [ -z "$PASSWORD" ]; then
  source deploy/azure/config.sh
  PASSWORD="$(az containerapp secret show --name "$CONTAINER_APP" --resource-group \
    "$RESOURCE_GROUP" --secret-name demo-password --query value -o tsv)"
fi
[ -n "$PASSWORD" ] || { echo "set SFC_DEMO_PASSWORD" >&2; exit 1; }
EMAIL="${CHECK_EMAIL:-maya@example.com}"
QUESTION="${CHECK_QUESTION:-How much did I spend last month?}"

jar="$(mktemp)"
trap 'rm -f "$jar"' EXIT
health="$(curl -fsS "$BASE/healthz")"
grep -q '"coach":{' <<<"$health" || { echo "chat is off: $health" >&2; exit 1; }
status="$(curl -s -o /dev/null -w '%{http_code}' -c "$jar" -b "$jar" \
  -H "Origin: $BASE" --data-urlencode "email=$EMAIL" --data-urlencode "password=$PASSWORD" \
  "$BASE/signin")"
[ "$status" = 303 ] || { echo "sign-in failed: HTTP $status" >&2; exit 1; }
answer="$(curl -fsS -c "$jar" -b "$jar" -H "Origin: $BASE" \
  --data-urlencode "question=$QUESTION" "$BASE/chat")"
if grep -Eq "isn&#39;t available|couldn&#39;t be reached|couldn&#39;t check every number" <<<"$answer"; then
  echo "chat answered without a grounded answer:" >&2
  sed -e 's/<[^>]*>//g' <<<"$answer" | tr -s ' \n' | head -c 600 >&2
  exit 1
fi
grep -q 'class="src-chip"' <<<"$answer" || { echo "the answer cites no source" >&2; exit 1; }
echo "Chat is live: $(grep -o '"model":"[^"]*"' <<<"$health"), a sourced answer for $EMAIL"
