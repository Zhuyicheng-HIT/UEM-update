#!/usr/bin/env bash
# Source this file before multi-GPU training on the current H20/driver-535 host.
# CUDA forward compatibility supplies libcuda, while NVML must match the host
# kernel driver.  Putting the NVML-only shim first prevents a 570-vs-535 NVML
# mismatch without replacing the forward-compatible libcuda.

CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/usr/local/cuda-12.8/compat}"
CUDA_RUNTIME_DIR="${CUDA_RUNTIME_DIR:-/usr/local/cuda-12.8/lib64}"
HOST_NVML="$(readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1)"
NVML_SHIM_DIR="${NVML_SHIM_DIR:-/tmp/uniegomotion-nvml-host}"

if [[ ! -r "$CUDA_COMPAT_DIR/libcuda.so.1" ]]; then
  echo "Missing CUDA forward-compat library: $CUDA_COMPAT_DIR/libcuda.so.1" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -z "$HOST_NVML" || ! -r "$HOST_NVML" ]]; then
  echo "Unable to resolve the host NVML library." >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$NVML_SHIM_DIR"
ln -sfn "$HOST_NVML" "$NVML_SHIM_DIR/libnvidia-ml.so.1"
ln -sfn "$HOST_NVML" "$NVML_SHIM_DIR/libnvidia-ml.so"

export LD_LIBRARY_PATH="$NVML_SHIM_DIR:$CUDA_COMPAT_DIR:$CUDA_RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# NCCL kernels are JIT-compiled for sm90 with this forward-compat stack.
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
