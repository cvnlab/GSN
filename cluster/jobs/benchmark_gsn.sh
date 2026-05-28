#!/usr/bin/env bash
#SBATCH --job-name=gsn_bench
#SBATCH --account=kempner_konkle_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=cluster/logs/benchmark_gsn_%A_%a.out
#SBATCH --error=cluster/logs/benchmark_gsn_%A_%a.err

# cluster/jobs/benchmark_gsn.sh
#
# SLURM **array** job: one task per nunits size in BENCH_NUNITS, so each
# H100 handles a single nunits value in parallel. Wall-clock = slowest
# single size, not the sum.
#
# Each task writes its own shard to
# cluster/outputs/benchmark_gsn_shards/shard_N<N>_C<nconds>_T<ntrials>_rep<repeats>.json
# and cluster/run.sh merges them into a consolidated JSON afterward.
#
# Backends timed per task:
#   python-torch-cuda     opt['device']='cuda'
#   python-torch-cpu      opt['device']='cpu' (still uses torch.linalg)
#   python-numpy          batched_nll._HAS_TORCH monkey-patched to False
#
# Sweep with:
#   BENCH_NUNITS="500 1000 2000" bash cluster/run.sh benchmark_gsn
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source "${PROJECT_ROOT}/cluster/config.sh"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON — did you run cluster/setup.sh?"
    exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Expandable CUDA allocator helps when the (S, N, N) shrunken-cov stack
# at large N would otherwise fragment reserved memory blocks.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────
BENCH_NUNITS="${BENCH_NUNITS:-100 200 500 1000 2000}"
BENCH_NCONDS="${BENCH_NCONDS:-80}"
BENCH_NTRIALS="${BENCH_NTRIALS:-4}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"

# Pick this task's nunits from BENCH_NUNITS via SLURM_ARRAY_TASK_ID.
read -r -a NUNITS_ARR <<< "$BENCH_NUNITS"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#NUNITS_ARR[@]}" ]; then
    echo "ERROR: task index $IDX out of range for BENCH_NUNITS=($BENCH_NUNITS)"
    exit 1
fi
N="${NUNITS_ARR[$IDX]}"

SHARD_DIR="cluster/outputs/benchmark_gsn_shards"
mkdir -p "$SHARD_DIR"
SHARD_NAME="shard_N${N}_C${BENCH_NCONDS}_T${BENCH_NTRIALS}_rep${BENCH_REPEATS}.json"
OUTPUT_JSON="${SHARD_DIR}/${SHARD_NAME}"

echo "========================================"
echo "Array job:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "This task:  N=${N}"
echo "All sizes:  ${BENCH_NUNITS}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "torch:      $($PYTHON -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())' 2>&1)"
echo "Started:    $(date)"
echo "========================================"

$PYTHON cluster/scripts/benchmark_gsn.py \
    --nunits "$N" \
    --nconds "$BENCH_NCONDS" \
    --ntrials "$BENCH_NTRIALS" \
    --repeats "$BENCH_REPEATS" \
    --output "$OUTPUT_JSON"

echo "========================================"
echo "Shard:      ${OUTPUT_JSON}"
echo "Finished:   $(date)"
echo "========================================"
