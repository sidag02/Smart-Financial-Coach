# Names for the Oct 6, 2026 demo deployment (Web App UI, "Demo build"). Sourced by the scripts.
LOCATION="${LOCATION:-eastus}"
RESOURCE_GROUP="rg-sfc-demo"
CONTAINER_ENV="sfc-demo-env"
CONTAINER_APP="sfc-web"
DEPLOY_IDENTITY="id-sfc-github-deploy"
GITHUB_REPO="sidag02/Smart-Financial-Coach"
GITHUB_ENVIRONMENT="demo"
SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
TENANT_ID="$(az account show --query tenantId -o tsv)"
# Registry names are global: derive a stable one from the subscription
ACR_NAME="sfcdemo$(printf '%s' "$SUBSCRIPTION_ID" | shasum | cut -c1-8)"
# The Anthropic key lives in Key Vault, never in .env or the repo; the app reads it through its
# managed identity (FR-13 to FR-15 design, §6). Vault names are global too
KEY_VAULT="kv-sfc-$(printf '%s' "$SUBSCRIPTION_ID" | shasum | cut -c1-8)"
LLM_KEY_SECRET="anthropic-api-key"
