#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPEED_GPU="${SPEED_GPU:-0}"
SPEED_NUM_SAMPLES="${SPEED_NUM_SAMPLES:-256}"
SPEED_BATCH_SIZE="${SPEED_BATCH_SIZE:-64}"
SPEED_REPEATS="${SPEED_REPEATS:-3}"
SPEED_INCLUDE_POSTPROCESS="${SPEED_INCLUDE_POSTPROCESS:-1}"
SPEED_ALLOW_BUSY_GPU="${SPEED_ALLOW_BUSY_GPU:-0}"
SPEED_OVERWRITE="${SPEED_OVERWRITE:-0}"
SPEED_METHOD_ORDER="${SPEED_METHOD_ORDER:-diffusion_first}"
OUTPUT_DIR="${SPEED_OUTPUT_DIR:-exp/speed_diffusion_vs_e7_n256_b64}"

DIFFUSION_CONFIG="config/uem.yaml"
DIFFUSION_EXP="exp/uem_v4b_dinov2"
E7_CONFIG="ablation/configs/e7_x0_global_w8_u84k.yaml"
E7_EXP="exp/ablation/e7_x0_global_w8_u84k"
DIFFUSION_JSON="$OUTPUT_DIR/diffusion_speed.json"
E7_JSON="$OUTPUT_DIR/e7_speed.json"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
for checkpoint in "$DIFFUSION_EXP/last.ckpt" "$E7_EXP/last.ckpt"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "Checkpoint not found or empty: $checkpoint" >&2
    exit 1
  fi
done
if ! [[ "$SPEED_NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$SPEED_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$SPEED_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Sample count, batch size, and repeats must be positive integers." >&2
  exit 2
fi
if [[ "$SPEED_INCLUDE_POSTPROCESS" != "0" && "$SPEED_INCLUDE_POSTPROCESS" != "1" ]]; then
  echo "SPEED_INCLUDE_POSTPROCESS must be 0 or 1." >&2
  exit 2
fi
if [[ "$SPEED_METHOD_ORDER" != "diffusion_first" && "$SPEED_METHOD_ORDER" != "e7_first" ]]; then
  echo "SPEED_METHOD_ORDER must be diffusion_first or e7_first." >&2
  exit 2
fi

check_gpu_idle() {
  if [[ "$SPEED_ALLOW_BUSY_GPU" == "1" ]]; then
    return
  fi
  active_pids="$(
    nvidia-smi -i "$SPEED_GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true
  )"
  if [[ -n "$active_pids" ]]; then
    echo "Refusing to benchmark on busy GPU $SPEED_GPU (active PIDs: $active_pids)." >&2
    echo "Wait for an idle GPU; SPEED_ALLOW_BUSY_GPU=1 is available only for intentional diagnostics." >&2
    exit 1
  fi
}
check_gpu_idle

if [[ "$SPEED_OVERWRITE" != "1" ]] && { [[ -e "$DIFFUSION_JSON" ]] || [[ -e "$E7_JSON" ]]; }; then
  echo "Benchmark output already exists under $OUTPUT_DIR." >&2
  echo "Choose SPEED_OUTPUT_DIR or set SPEED_OVERWRITE=1 for an intentional replacement." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
nvidia-smi -i "$SPEED_GPU" -q >"$OUTPUT_DIR/gpu_before.txt"
"$PYTHON_BIN" - <<'PY' >"$OUTPUT_DIR/software_versions.txt"
import platform
import pytorch_lightning
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("pytorch_lightning", pytorch_lightning.__version__)
PY

common_args=(
  --num-samples "$SPEED_NUM_SAMPLES"
  --batch-size "$SPEED_BATCH_SIZE"
  --repeats "$SPEED_REPEATS"
  --tasks recon gen fore
)
if [[ "$SPEED_INCLUDE_POSTPROCESS" == "1" ]]; then
  common_args+=(--include-postprocess)
fi

run_diffusion() {
  check_gpu_idle
  echo "Running official Diffusion benchmark on physical GPU $SPEED_GPU..."
  CUDA_VISIBLE_DEVICES="$SPEED_GPU" "$PYTHON_BIN" ablation/scripts/benchmark_inference_speed.py \
    --config "$DIFFUSION_CONFIG" \
    --exp-path "$DIFFUSION_EXP" \
    --label "Official Diffusion" \
    --output "$DIFFUSION_JSON" \
    "${common_args[@]}" \
    >"$OUTPUT_DIR/diffusion_stdout.log" 2>"$OUTPUT_DIR/diffusion_stderr.log"
}

run_e7() {
  check_gpu_idle
  echo "Running E7 Flow Matching benchmark on physical GPU $SPEED_GPU..."
  CUDA_VISIBLE_DEVICES="$SPEED_GPU" "$PYTHON_BIN" ablation/scripts/benchmark_inference_speed.py \
    --config "$E7_CONFIG" \
    --exp-path "$E7_EXP" \
    --label "E7 x0 global-w8" \
    --output "$E7_JSON" \
    "${common_args[@]}" \
    >"$OUTPUT_DIR/e7_stdout.log" 2>"$OUTPUT_DIR/e7_stderr.log"
}

if [[ "$SPEED_METHOD_ORDER" == "diffusion_first" ]]; then
  run_diffusion
  run_e7
else
  run_e7
  run_diffusion
fi

"$PYTHON_BIN" ablation/scripts/summarize_speed_comparison.py \
  --diffusion "$DIFFUSION_JSON" \
  --e7 "$E7_JSON" \
  --output-dir "$OUTPUT_DIR" \
  | tee "$OUTPUT_DIR/summary_stdout.log"
nvidia-smi -i "$SPEED_GPU" -q >"$OUTPUT_DIR/gpu_after.txt"

echo "Benchmark completed: $OUTPUT_DIR/speed_comparison.md"
