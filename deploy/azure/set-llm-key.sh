#!/usr/bin/env bash
# Make chat live on the deployed app (FR-13 to FR-15 design, §6). The Anthropic key stays in Key
# Vault (deploy/azure/key-vault.sh, then the Portal or put-llm-key.sh): the app's secret is a
# reference to it, read through the app's managed identity, so the key is never in .env, the
# repo, an image or a command line. Also pins the coach model and sizes the total chat limit to
# the key's spending cap, then waits for the restart and checks chat answers a demo user.
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
SECRET_URI="$(az keyvault secret show --vault-name "$KEY_VAULT" --name "$LLM_KEY_SECRET" \
  --query id -o tsv)" || { echo "no $LLM_KEY_SECRET in $KEY_VAULT: run key-vault.sh, add the key" >&2; exit 1; }
SECRET_URI="${SECRET_URI%/*}"  # the secret without its version: the app follows rotations
# A new role assignment can take a few minutes to reach Key Vault: retry until the app can read it
for attempt in $(seq 1 12); do
  if az containerapp secret set --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
    --secrets "llm-api-key=keyvaultref:$SECRET_URI,identityref:system" -o none 2>/dev/null; then
    break
  fi
  [ "$attempt" = 12 ] && { echo "the app can't read $KEY_VAULT yet: rerun key-vault.sh" >&2; exit 1; }
  sleep 15
done
az containerapp update --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --set-env-vars SFC_LLM_API_KEY=secretref:llm-api-key "SFC_LLM_MODEL=$MODEL" \
    "SFC_CHAT_MESSAGES_PER_HOUR_TOTAL=$CHAT_TOTAL_PER_HOUR" -o none
FQDN="$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)"
echo "Key referenced from $KEY_VAULT ($MODEL, at most $CHAT_TOTAL_PER_HOUR answers an hour)"
for attempt in $(seq 1 30); do
  if curl -fsS "https://$FQDN/healthz" | grep -q "\"model\":\"$MODEL\""; then break; fi
  sleep 10
done
deploy/azure/check-chat.sh "https://$FQDN"
