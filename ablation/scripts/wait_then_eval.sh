#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "Usage: $0 ID TRAIN_PID MAX_EPOCHS TRAIN_LOG CONFIG_PATH CUDA_DEVICES EVAL_SUFFIX EVAL_LOG" >&2
  exit 2
fi

ID="$1"
TRAIN_PID="$2"
MAX_EPOCHS="$3"
TRAIN_LOG="$4"
CONFIG_PATH="$5"
CUDA_DEVICES="$6"
EVAL_SUFFIX="$7"
EVAL_LOG="$8"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

timestamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

echo "[$(timestamp)] $ID: monitoring training PID $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

echo "[$(timestamp)] $ID: training process exited; verifying normal completion"
if ! grep -Eq "Trainer\.fit.*stopped.*max_epochs.*=.*${MAX_EPOCHS}.*reached" "$TRAIN_LOG"; then
  echo "[$(timestamp)] $ID: training did not report max_epochs=${MAX_EPOCHS}; evaluation will not start" >&2
  exit 1
fi

EXP_PATH=$(
  awk '
    /^TRAIN:/ { in_train=1; next }
    /^[^[:space:]]/ { in_train=0 }
    in_train && /EXP_PATH:/ { sub(/^[[:space:]]*EXP_PATH:[[:space:]]*/, ""); print; exit }
  ' "$CONFIG_PATH"
)
if [[ -z "$EXP_PATH" || ! -f "$EXP_PATH/last.ckpt" ]]; then
  echo "[$(timestamp)] $ID: final checkpoint not found at $EXP_PATH/last.ckpt" >&2
  exit 1
fi

for task in recon gen fore; do
  metric_path="$EXP_PATH/metrics_3d_ee4d_${task}${EVAL_SUFFIX}_keyjoints.pkl"
  if [[ -e "$metric_path" ]]; then
    echo "[$(timestamp)] $ID: refusing to overwrite existing metric $metric_path" >&2
    exit 1
  fi
done

echo "[$(timestamp)] $ID: starting Euler-10 evaluation on CUDA devices $CUDA_DEVICES"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-1}" \
bash ablation/scripts/eval_checkpoint.sh \
  "$CONFIG_PATH" \
  last_ckpt \
  10 \
  "$EVAL_SUFFIX" \
  > "$EVAL_LOG" 2>&1
status=$?

if [[ "$status" -eq 0 ]]; then
  echo "[$(timestamp)] $ID: evaluation completed successfully; log=$EVAL_LOG"
else
  echo "[$(timestamp)] $ID: evaluation failed with status $status; log=$EVAL_LOG" >&2
fi

exit "$status"
