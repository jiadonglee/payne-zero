#!/bin/bash
# Cloud Agent bootstrap for Payne Zero.
#
# Installs the package into the system Python interpreter so that both
# `python3 -c "import payne_zero_synthesis"` and the `payne-zero-*` console
# scripts work in every shell without activating a virtual environment (the
# environment schema has no way to persist PATH/venv activation).
#
# Idempotent: safe to re-run to refresh dependencies and caches.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# 1. Runtime data via Git LFS: physics tables, source catalogs, and the three
#    atmosphere-initializer checkpoints (~7 GB). A plain checkout leaves these
#    as pointer files, so pull them before the package validates them.
echo "[payne-zero cloud-setup] pulling Git LFS runtime data"
git lfs pull

# 2. Python dependencies into the SYSTEM interpreter. CPU-only torch keeps the
#    environment lean; Cloud Agent VMs have no GPU and the synthesis kernels
#    fall back to CPU. Installing torch first satisfies the `torch>=2.2`
#    constraint so the editable install does not pull the large CUDA wheels.
echo "[payne-zero cloud-setup] installing Python dependencies (system interpreter, CPU torch)"
sudo python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
sudo python3 -m pip install --break-system-packages -e ".[tutorial]"

# 3. Stage/verify runtime data and build the synthesis cache with the project
#    installer, reusing the work above:
#      * SKIP_LFS_PULL   - already pulled in step 1
#      * SKIP_PIP_INSTALL - already installed in step 2
#      * SKIP_ATMOSPHERE_PREWARM - the prewarm's cool-giant branch peaks above
#        the 16 GB Cloud Agent VM and is OOM-killed. Atmosphere Numba kernels
#        compile lazily on the first solve instead. Hot/solar converged solves
#        and all spectral synthesis run within the available memory.
echo "[payne-zero cloud-setup] staging runtime data and building the synthesis cache"
PYTHON=python3 \
PAYNE_ZERO_SKIP_LFS_PULL=1 \
PAYNE_ZERO_SKIP_PIP_INSTALL=1 \
PAYNE_ZERO_SKIP_ATMOSPHERE_PREWARM=1 \
  ./install.sh

echo "[payne-zero cloud-setup] done"
