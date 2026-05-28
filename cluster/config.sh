#!/usr/bin/env bash
# cluster/config.sh — Shared configuration for the Kempner cluster pipeline.
# Source from other scripts: source "$(dirname "$0")/config.sh"
set -euo pipefail

# ── SSH / cluster ────────────────────────────────────────────────────────────
# Assumes ~/.ssh/config has a `cannon` alias (or equivalent) that resolves to
# the Kempner login node and handles auth.
CLUSTER_HOST="cannon"
CLUSTER_USER="jacobprince"
CLUSTER_PROJECT="/n/home11/jacobprince/projects/GSN-speedup"

# ── Local paths ──────────────────────────────────────────────────────────────
LOCAL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── SLURM defaults ───────────────────────────────────────────────────────────
# kempner_h100 = 1× H100 80 GB. GSN's batched-Cholesky fits comfortably under
# 40 GB at N=2000; the 64 GB request includes margin for the biconvex iter.
SLURM_ACCOUNT="kempner_konkle_lab"
SLURM_PARTITION="kempner_h100"
SLURM_TIME="0-01:00:00"
SLURM_GPUS=1
SLURM_CPUS=16
SLURM_MEM="64G"

# ── Conda env ─────────────────────────────────────────────────────────────────
# Lives in lab storage so it doesn't count against home-dir quota. Keep
# separate from any PSN-updated env so the two stacks don't share a torch.
CONDA_PREFIX="/n/holylabs/LABS/konkle_lab/Users/jacobprince/conda_envs/gsn-speedup"

# ── Poll settings (seconds) ──────────────────────────────────────────────────
POLL_INITIAL=15
POLL_MAX=120
POLL_BACKOFF=1.5

# ── rsync excludes for code sync ─────────────────────────────────────────────
RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='cluster/logs/'
    --exclude='cluster/outputs/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.DS_Store'
    --exclude='*.egg-info/'
    --exclude='*.npy'
    --exclude='*.npz'
    --exclude='*.h5'
    --exclude='*.mat'
    --exclude='*.png'
    --exclude='*.jpg'
    --exclude='.ipynb_checkpoints/'
    --exclude='tests/speedup_bench_data/'
    --exclude='tests/gsn_equivalence_test_data/'
    --exclude='tests/speedup_magnitude.json'
)
