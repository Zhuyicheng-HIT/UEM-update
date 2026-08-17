#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" run/train_uem.py CONFIG ablation/configs/e14_explicit_task_token_w8_u84k.yaml
