#!/usr/bin/env bash
# Turn chat off on the deployed app once the demo is over (owner decision 7: the key is live for
# the demo only). Removes the app's reference to the key; chat then says the coach isn't
# available and the rest of the app works as before (NFR-6). The key stays in Key Vault until
# you delete it there; revoke it in the Anthropic console as well.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
az containerapp update --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --remove-env-vars SFC_LLM_API_KEY -o none
az containerapp secret remove --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --secret-names llm-api-key -o none
FQDN="$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)"
for attempt in $(seq 1 30); do
  if curl -fsS "https://$FQDN/healthz" | grep -q '"coach":null'; then
    echo "Chat is off. Delete $LLM_KEY_SECRET from $KEY_VAULT and revoke the key in the Anthropic console."
    exit 0
  fi
  sleep 10
done
echo "the app still reports a coach after 5 minutes" >&2
exit 1
