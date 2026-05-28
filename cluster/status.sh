#!/usr/bin/env bash
# cluster/status.sh — show squeue + recent sacct for this user.
set -euo pipefail
source "$(dirname "$0")/config.sh"

ssh "$CLUSTER_HOST" bash -s <<REMOTE_STATUS
echo "==> squeue (running/queued)"
squeue -u $CLUSTER_USER --format='%.10i %.20j %.10T %.10M %.10L %.5C %.10R'
echo ""
echo "==> sacct (last 24h)"
sacct -u $CLUSTER_USER --starttime now-24hours \
    --format=JobID,JobName%25,State,Elapsed,ExitCode,NodeList%20
REMOTE_STATUS
