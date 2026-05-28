#!/usr/bin/env bash
# cluster/sync-results.sh — rsync cluster outputs + logs back to local.
set -euo pipefail
source "$(dirname "$0")/config.sh"

mkdir -p "${LOCAL_PROJECT}/cluster/outputs"
mkdir -p "${LOCAL_PROJECT}/cluster/logs"

echo "==> Pulling outputs from ${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/outputs/"
rsync -avz \
    "${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/outputs/" \
    "${LOCAL_PROJECT}/cluster/outputs/"

echo ""
echo "==> Pulling logs from ${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/logs/"
rsync -avz \
    "${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/logs/" \
    "${LOCAL_PROJECT}/cluster/logs/"

echo ""
echo "==> Results sync complete."
