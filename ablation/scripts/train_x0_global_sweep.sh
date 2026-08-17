#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

scripts=(
  ablation/scripts/train_e5_x0_global_w2.sh
  ablation/scripts/train_e6_x0_global_w4.sh
  ablation/scripts/train_e7_x0_global_w8.sh
  ablation/scripts/train_e8_x0_rot6_trans12.sh
)
gpu_sets=("0,1" "2,3" "4,5" "6,7")
exp_dirs=(
  exp/ablation/e5_x0_global_w2_u84k
  exp/ablation/e6_x0_global_w4_u84k
  exp/ablation/e7_x0_global_w8_u84k
  exp/ablation/e8_x0_rot6_trans12_u84k
)

# Preflight every run before starting any process.
for index in "${!scripts[@]}"; do
  script="${scripts[$index]}"
  exp_dir="${exp_dirs[$index]}"
  if pgrep -f "$script" >/dev/null; then
    echo "Refusing to launch: $script is already running." >&2
    exit 1
  fi
  if [[ "${ALLOW_EXISTING:-0}" != "1" ]] && {
    [[ -e "$exp_dir/last.ckpt" ]] || [[ -s "$exp_dir/train.log" ]]
  }; then
    echo "Refusing to reuse non-empty experiment directory: $exp_dir" >&2
    echo "Set ALLOW_EXISTING=1 only when an intentional resume is configured." >&2
    exit 1
  fi
done

for index in "${!scripts[@]}"; do
  script="${scripts[$index]}"
  gpu_set="${gpu_sets[$index]}"
  exp_dir="${exp_dirs[$index]}"
  mkdir -p "$exp_dir"
  CUDA_VISIBLE_DEVICES="$gpu_set" PYTHONUNBUFFERED=1 nohup bash "$script" \
    >"$exp_dir/train.log" 2>&1 &
  echo "Started $(basename "$script") on GPUs $gpu_set: PID=$!, log=$exp_dir/train.log"
done
