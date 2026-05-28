#!/usr/bin/env bash
# cluster/setup.sh — One-time setup on the Kempner cluster.
#
# Syncs code, makes output directories, builds the conda env, installs torch
# with CUDA 12.1, installs the GSN package in editable mode.
# Safe to re-run: the conda-env block short-circuits if the env exists.
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo "==> One-time cluster setup"

echo "--- Step 1: Syncing code ---"
bash "$(dirname "$0")/sync-code.sh"

echo "--- Step 2: Creating directories ---"
ssh "$CLUSTER_HOST" bash -s <<REMOTE_SETUP
set -euo pipefail
mkdir -p "${CLUSTER_PROJECT}/cluster/logs"
mkdir -p "${CLUSTER_PROJECT}/cluster/outputs"
ls -la "${CLUSTER_PROJECT}/cluster/"
REMOTE_SETUP

echo "--- Step 3: Conda env + CUDA-enabled torch ---"
ssh "$CLUSTER_HOST" bash -s <<CONDA_SETUP
set -euo pipefail

module load python/3.12.5-fasrc01 2>/dev/null || module load python 2>/dev/null || true

# Prefer mamba/micromamba for environment creation — conda's classic
# solver takes ~10 min for a fresh python3.11 env; mamba does it in ~1.
if command -v mamba >/dev/null 2>&1; then
    SOLVER_CMD="mamba"
elif command -v micromamba >/dev/null 2>&1; then
    SOLVER_CMD="micromamba"
else
    # Fall back to conda but force the libmamba solver (~3x faster
    # than the classic solver). Available in conda >= 23.10.
    SOLVER_CMD="conda"
    CONDA_EXTRA="--solver=libmamba"
fi
CONDA_EXTRA="\${CONDA_EXTRA:-}"

if [ -d "${CONDA_PREFIX}" ]; then
    echo "Env already exists at ${CONDA_PREFIX}"
    echo "To recreate: rm -rf ${CONDA_PREFIX}"
else
    echo "Creating env at ${CONDA_PREFIX} using \$SOLVER_CMD"
    mkdir -p "\$(dirname "${CONDA_PREFIX}")"
    \$SOLVER_CMD create \$CONDA_EXTRA --prefix "${CONDA_PREFIX}" python=3.11 -y
fi

# Use env's own pip/python directly. `conda activate` is unreliable in
# non-interactive SSH shells.
ENV_PIP="${CONDA_PREFIX}/bin/pip"
ENV_PYTHON="${CONDA_PREFIX}/bin/python"

echo "Installing requirements..."
\$ENV_PIP install --upgrade pip

# torch with CUDA 12.1 — matches the Kempner H100 driver as of 2025-06.
# Adjust the --index-url if the cluster CUDA changes.
\$ENV_PIP install --index-url https://download.pytorch.org/whl/cu121 torch

\$ENV_PIP install -r "${CLUSTER_PROJECT}/cluster/requirements.txt"

# Install GSN in editable mode so imports resolve from any cwd.
\$ENV_PIP install -e "${CLUSTER_PROJECT}"

echo ""
echo "== Environment ready =="
echo "  Python:  \$(\$ENV_PYTHON --version)"
echo "  torch:   \$(\$ENV_PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  CUDA:    \$(\$ENV_PYTHON -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)' 2>/dev/null || echo 'N/A')"
CONDA_SETUP

echo "==> Cluster setup complete."
echo ""
echo "Next: run a benchmark with"
echo "  bash cluster/run.sh benchmark_gsn"
