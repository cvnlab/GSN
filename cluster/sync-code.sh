#!/usr/bin/env bash
# cluster/sync-code.sh — rsync the GSN repo to the Kempner cluster.
#
# Syncs:
#   1. cluster/  — job scripts, config, benchmark driver
#   2. gsn/      — the package under test
#   3. tests/    — so we can run pytest on the cluster too
#   4. setup.py + requirements.txt — install metadata
set -euo pipefail
source "$(dirname "$0")/config.sh"

# ── 1. Cluster scripts ─────────────────────────────────────────────────────
echo "==> Syncing cluster/ → ${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/"
ssh "$CLUSTER_HOST" "mkdir -p '${CLUSTER_PROJECT}/cluster'"
rsync -avz --delete \
    --exclude='logs/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    "${LOCAL_PROJECT}/cluster/" \
    "${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/"

# ── 2. GSN package ─────────────────────────────────────────────────────────
echo ""
echo "==> Syncing gsn/ → ${CLUSTER_HOST}:${CLUSTER_PROJECT}/gsn/"
rsync -avz --delete \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='*.egg-info/' \
    "${LOCAL_PROJECT}/gsn/" \
    "${CLUSTER_HOST}:${CLUSTER_PROJECT}/gsn/"

# ── 3. Tests ───────────────────────────────────────────────────────────────
echo ""
echo "==> Syncing tests/ → ${CLUSTER_HOST}:${CLUSTER_PROJECT}/tests/"
rsync -avz --delete \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='speedup_bench_data/' \
    --exclude='gsn_equivalence_test_data/' \
    --exclude='speedup_magnitude.png' \
    --exclude='speedup_magnitude.json' \
    "${LOCAL_PROJECT}/tests/" \
    "${CLUSTER_HOST}:${CLUSTER_PROJECT}/tests/"

# ── 4. Install metadata ────────────────────────────────────────────────────
echo ""
echo "==> Syncing setup.py + requirements.txt"
rsync -avz \
    "${LOCAL_PROJECT}/setup.py" \
    "${LOCAL_PROJECT}/requirements.txt" \
    "${CLUSTER_HOST}:${CLUSTER_PROJECT}/" 2>/dev/null || true

echo ""
echo "==> Code sync complete."
