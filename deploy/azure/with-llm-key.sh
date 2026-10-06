#!/usr/bin/env bash
# Run one local command with the Anthropic key from Key Vault in its environment only, e.g. the
# coach suite's API run (FR-13 to FR-15 design, §5):
#
#   deploy/azure/with-llm-key.sh uv run sfc-coach eval --backend api --latency-cost
#
# The key is read into this process and its child, never written to a file.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
[ $# -ge 1 ] || { echo "usage: $0 command [args...]" >&2; exit 1; }
ANTHROPIC_API_KEY="$(az keyvault secret show --vault-name "$KEY_VAULT" --name "$LLM_KEY_SECRET" \
  --query value -o tsv)"
export ANTHROPIC_API_KEY
exec "$@"
