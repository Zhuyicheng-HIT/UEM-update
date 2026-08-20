#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/exp/ablation/launch_logs/e17_e20_$RUN_TAG}"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(e17 e18 e19 e20)
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
PIDS=()

terminate_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for index in "${!EXPERIMENTS[@]}"; do
  experiment="${EXPERIMENTS[$index]}"
  gpu_pair="${GPU_PAIRS[$index]}"
  log_file="$LOG_DIR/${experiment}.log"
  echo "Launching ${experiment^^} on GPU $gpu_pair -> $log_file"
  CUDA_VISIBLE_DEVICES="$gpu_pair" \
    nohup bash "ablation/scripts/train_e17_e20.sh" "$experiment" "$@" \
    >"$log_file" 2>&1 &
  PIDS+=("$!")
  echo "  PID ${PIDS[-1]}"
done

printf '%s\n' "${PIDS[@]}" >"$LOG_DIR/pids.txt"
echo "All four jobs launched. Logs: $LOG_DIR"

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
