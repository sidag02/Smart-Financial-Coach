#!/usr/bin/env bash
# Make chat live on the deployed app (FR-13 to FR-15 design, §6): give it the Anthropic key, read
# from .env (SFC_LLM_API_KEY or ANTHROPIC_API_KEY) so it never appears on the command line or in
# shell history, pin the coach model, and size the total chat limit to the key's spending cap.
# Then wait for the new revision and check chat answers a demo user with a sourced answer.
#
#   CHAT_TOTAL_PER_HOUR=150 deploy/azure/set-llm-key.sh
#
# CHAT_TOTAL_PER_HOUR is the most answers an hour across all visitors (owner decision 7: a $20 cap
# that must last the demo): at most (cap left) / (dollars per answer x demo hours), from the coach
# suite's API run. SFC_LLM_MODEL defaults to claude-sonnet-5-5 (decision 1). Turn chat off
# afterwards with deploy/azure/remove-llm-key.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
: "${CHAT_TOTAL_PER_HOUR:?set CHAT_TOTAL_PER_HOUR, sized to the spending cap on the key (see above)}"
MODEL="${SFC_LLM_MODEL:-claude-sonnet-5-5}"
KEY="$(grep -E '^(SFC_LLM_API_KEY|ANTHROPIC_API_KEY)=' .env | head -1 | cut -d= -f2-)"
[ -n "$KEY" ] || { echo "no SFC_LLM_API_KEY or ANTHROPIC_API_KEY in .env" >&2; exit 1; }
az containerapp secret set --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --secrets "llm-api-key=$KEY" -o none
az containerapp update --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --set-env-vars SFC_LLM_API_KEY=secretref:llm-api-key "SFC_LLM_MODEL=$MODEL" \
    "SFC_CHAT_MESSAGES_PER_HOUR_TOTAL=$CHAT_TOTAL_PER_HOUR" -o none
FQDN="$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)"
echo "Key set ($MODEL, at most $CHAT_TOTAL_PER_HOUR answers an hour); waiting for the restart"
for attempt in $(seq 1 30); do
  if curl -fsS "https://$FQDN/healthz" | grep -q "\"model\":\"$MODEL\""; then break; fi
  sleep 10
done
deploy/azure/check-chat.sh "https://$FQDN"
