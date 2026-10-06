#!/usr/bin/env bash
# Create the demo's Key Vault for the Anthropic key, with Azure RBAC: the signed-in owner may
# write its secrets, and the container app's managed identity may only read them. Idempotent.
# Then put the key in it: in the Azure Portal (Key vaults > the vault > Secrets > Generate/Import,
# named anthropic-api-key) or with deploy/azure/put-llm-key.sh, which asks for it without echo.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
echo "== Key Vault $KEY_VAULT"
az keyvault show --name "$KEY_VAULT" -o none 2>/dev/null ||
  az keyvault create --name "$KEY_VAULT" --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" --enable-rbac-authorization true -o none
VAULT_ID="$(az keyvault show --name "$KEY_VAULT" --query id -o tsv)"
OWNER="$(az ad signed-in-user show --query id -o tsv)"
APP="$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query identity.principalId -o tsv)"
[ -n "$APP" ] || { echo "the container app has no system-assigned identity" >&2; exit 1; }
grant() {  # role, principal, principal type
  if [ -z "$(az role assignment list --scope "$VAULT_ID" --assignee "$2" --role "$1" \
    --query "[0].id" -o tsv)" ]; then
    az role assignment create --scope "$VAULT_ID" --role "$1" --assignee-object-id "$2" \
      --assignee-principal-type "$3" -o none
  fi
}
grant "Key Vault Secrets Officer" "$OWNER" User
grant "Key Vault Secrets User" "$APP" ServicePrincipal
echo "Ready. Put the key in $KEY_VAULT as $LLM_KEY_SECRET (Portal, or deploy/azure/put-llm-key.sh)."
echo "Role assignments can take a few minutes to apply."
