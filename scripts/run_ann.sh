#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate ann
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$ROOT_DIR/.cache/torch}"
"$@"
