#!/usr/bin/env bash
# Make chat live on the deployed app (FR-13 to FR-15 design, §6), once you've added the Anthropic
# key to the container app by hand: Azure Portal > Container Apps > sfc-web > Settings > Secrets >
# Add, named llm-api-key. The key never passes through this script, a file or the repo.
#
# This points SFC_LLM_API_KEY at that secret, pins the coach model, sizes the total chat limit to
# the key's spending cap, waits for the new revision and checks chat answers a demo user.
#
#   CHAT_TOTAL_PER_HOUR=150 deploy/azure/enable-chat.sh
#
# CHAT_TOTAL_PER_HOUR is the most answers an hour across all visitors (owner decision 7: a $20 cap
# that must last the demo): at most (cap left) / (dollars per answer x demo hours). SFC_LLM_MODEL
# defaults to claude-sonnet-5-5 (decision 1). Turn chat off with deploy/azure/remove-llm-key.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
: "${CHAT_TOTAL_PER_HOUR:?set CHAT_TOTAL_PER_HOUR, sized to the spending cap on the key (see above)}"
MODEL="${SFC_LLM_MODEL:-claude-sonnet-5-5}"
SECRETS="$(az containerapp secret list --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query "[].name" -o tsv)"
grep -qx "llm-api-key" <<<"$SECRETS" || {
  echo "add the key first: Portal > Container Apps > $CONTAINER_APP > Settings > Secrets >" \
    "Add, named llm-api-key" >&2
  exit 1
}
az containerapp update --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --set-env-vars SFC_LLM_API_KEY=secretref:llm-api-key "SFC_LLM_MODEL=$MODEL" \
    "SFC_CHAT_MESSAGES_PER_HOUR_TOTAL=$CHAT_TOTAL_PER_HOUR" -o none
FQDN="$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)"
echo "Chat set up ($MODEL, at most $CHAT_TOTAL_PER_HOUR answers an hour); waiting for the new revision"
# The update makes a new revision. Wait until it's ready, takes all the traffic and reports the
# model, as deploy.yml waits for a deploy: until then the old revision can still answer, and a
# rerun with the same model would otherwise check before the new limit is live (review on #76)
ready=""
for attempt in $(seq 1 30); do
  latest=$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
    --query properties.latestRevisionName -o tsv) || true
  live=$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
    --query properties.latestReadyRevisionName -o tsv) || true
  traffic=$(az containerapp revision show --name "$CONTAINER_APP" \
    --resource-group "$RESOURCE_GROUP" --revision "$latest" \
    --query properties.trafficWeight -o tsv) || true
  if [ -n "$latest" ] && [ "$latest" = "$live" ] && [ "$traffic" = "100" ] &&
    curl -fsS "https://$FQDN/healthz" | grep -q "\"model\":\"$MODEL\""; then
    ready=yes
    break
  fi
  sleep 10
done
[ -n "$ready" ] || { echo "timed out waiting for the new revision ($latest)" >&2; exit 1; }
deploy/azure/check-chat.sh "https://$FQDN"
