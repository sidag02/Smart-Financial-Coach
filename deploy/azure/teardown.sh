#!/usr/bin/env bash
# Delete everything the demo created: the resource group (app, registry, logs, identity) and the
# GitHub environment with its variables, and the Key Vault with the Anthropic key in it. Revoke the
# key in the Anthropic console separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
az group delete --name "$RESOURCE_GROUP" --yes
# A deleted vault is kept 90 days and holds its name: purge it so provisioning again works
az keyvault purge --name "$KEY_VAULT" -o none 2>/dev/null || true
gh variable delete DEMO_DEPLOY --repo "$GITHUB_REPO" || true  # switches the deploy workflow off
gh api -X DELETE "repos/$GITHUB_REPO/environments/$GITHUB_ENVIRONMENT"
echo "Deleted $RESOURCE_GROUP and the $GITHUB_ENVIRONMENT environment."
