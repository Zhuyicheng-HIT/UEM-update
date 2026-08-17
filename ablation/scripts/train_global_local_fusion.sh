#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

scripts=(
  ablation/scripts/train_e9_dual_no_fusion.sh
  ablation/scripts/train_e10_dual_l2g_stopgrad.sh
  ablation/scripts/train_e11_dual_g2l_stopgrad.sh
  ablation/scripts/train_e12_dual_bidir_gated.sh
)
gpu_sets=("0,1" "2,3" "4,5" "6,7")
ports=("29609" "29610" "29611" "29612")
exp_dirs=(
  exp/ablation/e9_dual_no_fusion_w8_u84k
  exp/ablation/e10_dual_l2g_stopgrad_w8_u84k
  exp/ablation/e11_dual_g2l_stopgrad_w8_u84k
  exp/ablation/e12_dual_bidir_gated_w8_u84k
)

if [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)" -ne 0 ]]; then
  echo "Refusing to launch because at least one GPU already has a compute process." >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >&2
  exit 1
fi

# Complete all checks before launching any process, so failures remain atomic.
for index in "${!scripts[@]}"; do
  script="${scripts[$index]}"
  port="${ports[$index]}"
  exp_dir="${exp_dirs[$index]}"
  if pgrep -f "[b]ash $script" >/dev/null; then
    echo "Refusing to launch: $script is already running." >&2
    exit 1
  fi
  if ss -H -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "Refusing to launch: TCP port $port is already in use." >&2
    exit 1
  fi
  if [[ "${ALLOW_EXISTING:-0}" != "1" ]] && {
    [[ -e "$exp_dir/last.ckpt" ]] || [[ -s "$exp_dir/train.log" ]]
  }; then
    echo "Refusing to reuse non-empty experiment directory: $exp_dir" >&2
    echo "Set ALLOW_EXISTING=1 only for an intentional resume." >&2
    exit 1
  fi
done

for index in "${!scripts[@]}"; do
  script="${scripts[$index]}"
  gpu_set="${gpu_sets[$index]}"
  port="${ports[$index]}"
  exp_dir="${exp_dirs[$index]}"
  mkdir -p "$exp_dir"
  CUDA_VISIBLE_DEVICES="$gpu_set" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$port" \
    PYTHONUNBUFFERED=1 \
    nohup bash "$script" >"$exp_dir/train.log" 2>&1 &
  launcher_pid=$!
  echo "$launcher_pid" >"$exp_dir/launcher.pid"
  echo "Started $(basename "$script") on GPUs $gpu_set (port $port): PID=$launcher_pid, log=$exp_dir/train.log"
done
