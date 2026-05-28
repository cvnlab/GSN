#!/usr/bin/env bash
# cluster/run.sh — Idempotent benchmark runner.
#
# For each requested nunits value, computes the expected shard filename
# and only submits SLURM tasks for sizes whose shard is MISSING locally.
# Reruns are cheap: re-invoking with the same BENCH_* vars after a
# partial failure only re-submits the failed sizes.
#
# During polling we rsync the shard directory on every tick so completed
# shards appear locally as soon as written, not only at the end of the
# array. Once all expected shards are present locally we merge them into
# a consolidated JSON.
#
# Usage:
#   bash cluster/run.sh benchmark_gsn              # full pipeline
#   bash cluster/run.sh benchmark_gsn --no-poll    # fire-and-forget submit
#   bash cluster/run.sh benchmark_gsn --merge-only # skip submit, just merge
set -euo pipefail
source "$(dirname "$0")/config.sh"

SCRIPT="${1:?Usage: $0 <script_id> [--no-poll|--merge-only]}"
MODE="full"
for arg in "${@:2}"; do
    case "$arg" in
        --no-poll)    MODE="no_poll" ;;
        --merge-only) MODE="merge_only" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

JOB_SCRIPT="cluster/jobs/${SCRIPT}.sh"
if [ ! -f "${LOCAL_PROJECT}/${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script not found: ${LOCAL_PROJECT}/${JOB_SCRIPT}"
    exit 1
fi

# ── Config ─────────────────────────────────────────────────────────────────
BENCH_NUNITS="${BENCH_NUNITS:-100 200 500 1000 2000}"
BENCH_NCONDS="${BENCH_NCONDS:-80}"
BENCH_NTRIALS="${BENCH_NTRIALS:-4}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"

SHARD_DIR_NAME="${SCRIPT}_shards"
LOCAL_SHARD_DIR="${LOCAL_PROJECT}/cluster/outputs/${SHARD_DIR_NAME}"
REMOTE_SHARD_DIR="${CLUSTER_PROJECT}/cluster/outputs/${SHARD_DIR_NAME}"
mkdir -p "$LOCAL_SHARD_DIR"

shard_name_for() {
    echo "shard_N${1}_C${BENCH_NCONDS}_T${BENCH_NTRIALS}_rep${BENCH_REPEATS}.json"
}

# ── Figure out which sizes are missing ─────────────────────────────────────
read -r -a NUNITS_ARR <<< "$BENCH_NUNITS"
MISSING=()
PRESENT=()
for N in "${NUNITS_ARR[@]}"; do
    if [ -f "${LOCAL_SHARD_DIR}/$(shard_name_for "$N")" ]; then
        PRESENT+=("$N")
    else
        MISSING+=("$N")
    fi
done

echo "=========================================="
echo "  Config"
echo "=========================================="
echo "  sizes requested:  ${NUNITS_ARR[*]}"
echo "  already present:  ${PRESENT[*]:-(none)}"
echo "  missing (to run): ${MISSING[*]:-(none)}"
echo "  nconds=${BENCH_NCONDS}  ntrials=${BENCH_NTRIALS}  repeats=${BENCH_REPEATS}"
echo "  shard dir (local):  ${LOCAL_SHARD_DIR}"
echo "  shard dir (remote): ${REMOTE_SHARD_DIR}"
echo ""

# ── Submit if there's work and not merge-only ──────────────────────────────
JOBID=""
if [ "$MODE" != "merge_only" ] && [ "${#MISSING[@]}" -gt 0 ]; then
    echo "=========================================="
    echo "  Sync code"
    echo "=========================================="
    bash "$(dirname "$0")/sync-code.sh"
    echo ""

    echo "=========================================="
    echo "  Submit SLURM array job"
    echo "=========================================="
    NEW_NUNITS="${MISSING[*]}"
    ARRAY_RANGE="0-$(( ${#MISSING[@]} - 1 ))"
    echo "  nunits for this submit: $NEW_NUNITS"
    echo "  array range:            $ARRAY_RANGE"

    # Submit on the cluster. Environment is passed via --export.
    SUBMIT_OUT=$(ssh "$CLUSTER_HOST" "cd '${CLUSTER_PROJECT}' && \
        sbatch --array=${ARRAY_RANGE} \
               --export=ALL,BENCH_NUNITS='${NEW_NUNITS}',\
BENCH_NCONDS=${BENCH_NCONDS},BENCH_NTRIALS=${BENCH_NTRIALS},BENCH_REPEATS=${BENCH_REPEATS} \
               '${JOB_SCRIPT}'")
    echo "  $SUBMIT_OUT"
    JOBID=$(echo "$SUBMIT_OUT" | grep -oE '[0-9]+' | head -1)
    echo "  job id: $JOBID"
    echo ""
fi

# ── Poll until all expected shards are present locally ─────────────────────
if [ "$MODE" == "full" ]; then
    echo "=========================================="
    echo "  Poll for shards"
    echo "=========================================="
    DELAY="$POLL_INITIAL"
    while true; do
        ssh "$CLUSTER_HOST" "mkdir -p '${REMOTE_SHARD_DIR}'" >/dev/null 2>&1 || true
        rsync -azq \
            "${CLUSTER_HOST}:${REMOTE_SHARD_DIR}/" \
            "${LOCAL_SHARD_DIR}/" 2>/dev/null || true

        # Count present
        PRESENT_COUNT=0
        for N in "${NUNITS_ARR[@]}"; do
            if [ -f "${LOCAL_SHARD_DIR}/$(shard_name_for "$N")" ]; then
                PRESENT_COUNT=$(( PRESENT_COUNT + 1 ))
            fi
        done
        TS=$(date '+%H:%M:%S')
        echo "  [$TS] shards: $PRESENT_COUNT / ${#NUNITS_ARR[@]}"
        if [ "$PRESENT_COUNT" -eq "${#NUNITS_ARR[@]}" ]; then
            break
        fi
        sleep "$DELAY"
        # Exponential backoff up to POLL_MAX.
        DELAY=$(awk -v d="$DELAY" -v b="$POLL_BACKOFF" -v m="$POLL_MAX" \
                'BEGIN{nd=d*b; if(nd>m) nd=m; printf "%d", nd}')
    done
    echo ""

    # Also pull logs for inspection.
    rsync -azq \
        "${CLUSTER_HOST}:${CLUSTER_PROJECT}/cluster/logs/" \
        "${LOCAL_PROJECT}/cluster/logs/" 2>/dev/null || true
fi

# ── Merge shards into consolidated JSON ────────────────────────────────────
if [ "$MODE" != "no_poll" ]; then
    echo "=========================================="
    echo "  Merge shards"
    echo "=========================================="
    MERGED_JSON="${LOCAL_PROJECT}/cluster/outputs/${SCRIPT}_merged.json"
    python3 "${LOCAL_PROJECT}/cluster/scripts/merge_shards.py" \
        --shard-dir "$LOCAL_SHARD_DIR" \
        --output    "$MERGED_JSON"
    echo "  merged: $MERGED_JSON"
fi

echo ""
echo "==> Done."
