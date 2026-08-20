#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {e17|e18|e19|e20} [YACS overrides ...]" >&2
  exit 2
fi

EXPERIMENT="${1,,}"
shift

case "$EXPERIMENT" in
  e17) CONFIG_PATH="ablation/configs/e17_task_global_weight_w8_u84k.yaml" ;;
  e18) CONFIG_PATH="ablation/configs/e18_task_film_w8_u84k.yaml" ;;
  e19) CONFIG_PATH="ablation/configs/e19_smplx_geometry_w8_u84k.yaml" ;;
  e20) CONFIG_PATH="ablation/configs/e20_mixed_batch_w8_u84k.yaml" ;;
  *)
    echo "Unknown experiment '$EXPERIMENT'; expected e17, e18, e19, or e20." >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# This is the layout on the current training server.  An explicitly exported
# UEM_DATA_DIR always wins, so the launcher remains portable to other hosts.
SERVER_DATA_DIR="$PROJECT_ROOT/../datasets/ee4d_motion_uniegomotion"
if [[ -z "${UEM_DATA_DIR:-}" && -d "$SERVER_DATA_DIR" ]]; then
  export UEM_DATA_DIR="$SERVER_DATA_DIR"
fi

DATA_DIR="${UEM_DATA_DIR:-$PROJECT_ROOT/data/ee4d_motion_uniegomotion}"
STATS_PATH="$DATA_DIR/uniegomotion/v4_beta_ee_train_stats.pt"
if [[ ! -f "$STATS_PATH" ]]; then
  echo "Missing training data/statistics: $STATS_PATH" >&2
  echo "Set UEM_DATA_DIR to the EE4D UniEgoMotion dataset root." >&2
  exit 1
fi

if [[ "$EXPERIMENT" == "e19" ]]; then
  SMPLX_DIR="${SMPLX_MODEL_PATH:-$PROJECT_ROOT/body_models/smplx}"
  if [[ ! -f "$SMPLX_DIR/SMPLX_NEUTRAL.npz" ]]; then
    echo "E19 requires SMPLX_NEUTRAL.npz under $SMPLX_DIR" >&2
    echo "Set SMPLX_MODEL_PATH to the SMPL-X model directory." >&2
    exit 1
  fi
fi

TRAIN_COMMAND=("$PYTHON_BIN" run/train_uem.py CONFIG "$CONFIG_PATH" "$@")
echo "Launching ${EXPERIMENT^^}: GPUs=$CUDA_VISIBLE_DEVICES, data=$DATA_DIR"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${TRAIN_COMMAND[@]}"
