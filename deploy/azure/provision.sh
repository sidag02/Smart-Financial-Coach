#!/usr/bin/env bash
# Create the demo's Azure resources and first deployment, and wire up the GitHub pipeline.
# Safe to re-run: existing resources are kept.
#
#   SFC_DEMO_PASSWORD=... deploy/azure/provision.sh
#
# Needs: az (signed in), gh (signed in), and build/demo (uv run sfc-web build-demo ...).
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
: "${SFC_DEMO_PASSWORD:?set SFC_DEMO_PASSWORD, the shared demo password (8+ characters)}"
[ -f build/demo/dataset.sqlite ] || { echo "build/demo is missing: run sfc-web build-demo" >&2; exit 1; }
TAG="$(git rev-parse --short HEAD)"
IMAGE="$ACR_NAME.azurecr.io/sfc-web:$TAG"

echo "== Providers and CLI extension"
az extension add --name containerapp --upgrade --yes --only-show-errors
for provider in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry \
  Microsoft.ManagedIdentity; do
  az provider register --namespace "$provider" --wait
done

echo "== Resource group $RESOURCE_GROUP ($LOCATION)"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" \
  --tags project=smart-financial-coach purpose=demo delete-after=2026-10-06 -o none

echo "== Registry $ACR_NAME"
az acr show --name "$ACR_NAME" -o none 2>/dev/null ||
  az acr create --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --sku Basic -o none

echo "== Image $IMAGE (built in Azure)"
az acr build --registry "$ACR_NAME" --image "sfc-web:$TAG" --image sfc-web:latest \
  --platform linux/amd64 . -o none

echo "== Container Apps environment $CONTAINER_ENV"
az containerapp env show --name "$CONTAINER_ENV" --resource-group "$RESOURCE_GROUP" -o none 2>/dev/null ||
  az containerapp env create --name "$CONTAINER_ENV" --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" -o none

echo "== Container app $CONTAINER_APP"
if az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" -o none 2>/dev/null; then
  az containerapp update --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
    --image "$IMAGE" -o none
else
  SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  # One replica: chat history and rate limits live in memory (Web App UI, "Demo build")
  az containerapp create --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINER_ENV" --image "$IMAGE" \
    --registry-server "$ACR_NAME.azurecr.io" --registry-identity system \
    --ingress external --target-port 8000 --min-replicas 1 --max-replicas 1 \
    --cpu 1 --memory 2Gi \
    --secrets "demo-password=$SFC_DEMO_PASSWORD" "session-secret=$SESSION_SECRET" \
    --env-vars SFC_DEMO_PASSWORD=secretref:demo-password \
      SFC_SESSION_SECRET=secretref:session-secret SFC_TRUSTED_PROXY_HOPS=1 -o none
fi

echo "== Pipeline identity $DEPLOY_IDENTITY (GitHub OIDC, no stored Azure secret)"
az identity show --name "$DEPLOY_IDENTITY" --resource-group "$RESOURCE_GROUP" -o none 2>/dev/null ||
  az identity create --name "$DEPLOY_IDENTITY" --resource-group "$RESOURCE_GROUP" -o none
CLIENT_ID="$(az identity show --name "$DEPLOY_IDENTITY" --resource-group "$RESOURCE_GROUP" --query clientId -o tsv)"
PRINCIPAL_ID="$(az identity show --name "$DEPLOY_IDENTITY" --resource-group "$RESOURCE_GROUP" --query principalId -o tsv)"
# The subject GitHub's OIDC token carries. Repos with immutable subjects send owner and repo IDs
# (repo:owner@123/name@456:...), so ask GitHub for the prefix instead of assuming repo:owner/name
SUB_PREFIX="$(gh api "repos/$GITHUB_REPO/actions/oidc/customization/sub" \
  --jq '.sub_claim_prefix // empty' 2>/dev/null || true)"
SUB_PREFIX="${SUB_PREFIX:-repo:$GITHUB_REPO}"
CREDENTIAL="github-demo"
[ "$SUB_PREFIX" = "repo:$GITHUB_REPO" ] || CREDENTIAL="github-demo-ids"
az identity federated-credential show --name "$CREDENTIAL" --identity-name "$DEPLOY_IDENTITY" \
  --resource-group "$RESOURCE_GROUP" -o none 2>/dev/null ||
  az identity federated-credential create --name "$CREDENTIAL" --identity-name "$DEPLOY_IDENTITY" \
    --resource-group "$RESOURCE_GROUP" --issuer https://token.actions.githubusercontent.com \
    --subject "$SUB_PREFIX:environment:$GITHUB_ENVIRONMENT" \
    --audiences api://AzureADTokenExchange -o none
# Contributor on this resource group only: build images and update the app, nothing else
RG_ID="$(az group show --name "$RESOURCE_GROUP" --query id -o tsv)"
az role assignment list --assignee "$PRINCIPAL_ID" --scope "$RG_ID" --role Contributor \
  --query "[0].id" -o tsv | grep -q . ||
  az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal --role Contributor --scope "$RG_ID" -o none

echo "== GitHub environment $GITHUB_ENVIRONMENT (deploys from main only)"
gh api -X PUT "repos/$GITHUB_REPO/environments/$GITHUB_ENVIRONMENT" --input - >/dev/null <<JSON
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
gh api "repos/$GITHUB_REPO/environments/$GITHUB_ENVIRONMENT/deployment-branch-policies" \
  --jq '.branch_policies[].name' | grep -qx main ||
  gh api -X POST "repos/$GITHUB_REPO/environments/$GITHUB_ENVIRONMENT/deployment-branch-policies" \
    -f name=main -f type=branch >/dev/null
# Repository variable that switches the deploy workflow on; teardown.sh removes it
gh variable set DEMO_DEPLOY --repo "$GITHUB_REPO" --body on
for pair in "AZURE_CLIENT_ID=$CLIENT_ID" "AZURE_TENANT_ID=$TENANT_ID" \
  "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID" "ACR_NAME=$ACR_NAME" \
  "RESOURCE_GROUP=$RESOURCE_GROUP" "CONTAINER_APP=$CONTAINER_APP"; do
  gh variable set "${pair%%=*}" --env "$GITHUB_ENVIRONMENT" --repo "$GITHUB_REPO" --body "${pair#*=}"
done

FQDN="$(az containerapp show --name "$CONTAINER_APP" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)"
echo "== Up: https://$FQDN"
