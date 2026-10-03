#!/usr/bin/env bash
# Delete everything the demo created: the resource group (app, registry, logs, identity) and the
# GitHub environment with its variables. Revoke the Anthropic key in the Console separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/azure/config.sh
az group delete --name "$RESOURCE_GROUP" --yes
gh variable delete DEMO_DEPLOY --repo "$GITHUB_REPO" || true  # switches the deploy workflow off
gh api -X DELETE "repos/$GITHUB_REPO/environments/$GITHUB_ENVIRONMENT"
echo "Deleted $RESOURCE_GROUP and the $GITHUB_ENVIRONMENT environment."
