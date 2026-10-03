#!/usr/bin/env bash
# Give the deployed app the Anthropic key, read from .env (SFC_LLM_API_KEY or ANTHROPIC_API_KEY)
# so it never appears on the command line or in shell history. Restarts the app.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
KEY="$(grep -E '^(SFC_LLM_API_KEY|ANTHROPIC_API_KEY)=' .env | head -1 | cut -d= -f2-)"
[ -n "$KEY" ] || { echo "no SFC_LLM_API_KEY or ANTHROPIC_API_KEY in .env" >&2; exit 1; }
az containerapp secret set --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --secrets "llm-api-key=$KEY" -o none
az containerapp update --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --set-env-vars SFC_LLM_API_KEY=secretref:llm-api-key -o none
echo "Key set; the app restarts with chat enabled."
