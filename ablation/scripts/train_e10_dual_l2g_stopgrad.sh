#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29610}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${USE_H20_CUDA_COMPAT:-1}" == "1" ]]; then
  source ablation/scripts/setup_h20_cuda_compat.sh
fi

"$PYTHON_BIN" run/train_uem.py CONFIG ablation/configs/e10_dual_l2g_stopgrad_w8_u84k.yaml
