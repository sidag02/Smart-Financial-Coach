#!/usr/bin/env bash
# Store the Anthropic key in Key Vault, typed at a hidden prompt: it never reaches .env, the
# repo, the command line or shell history. The same as adding it in the Azure Portal.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
read -rsp "Anthropic API key (not shown): " KEY; echo
[ -n "$KEY" ] || { echo "no key entered" >&2; exit 1; }
printf '%s' "$KEY" | az keyvault secret set --vault-name "$KEY_VAULT" --name "$LLM_KEY_SECRET" \
  --file /dev/stdin --query "attributes.updated" -o tsv
unset KEY
echo "Stored as $LLM_KEY_SECRET in $KEY_VAULT."
