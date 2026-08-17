#!/usr/bin/env python3
"""Benchmark UniEgoMotion inference without training or changing checkpoints.

The benchmark fixes the validation subset and measures model sampling with CUDA
events. Dataset preparation, host/device transfer, checkpoint loading, warmup,
and optional SMPL-X post-processing are reported separately so they cannot be
silently mixed into the core sampler latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

# Allow direct execution from any working directory without requiring callers
# to preconfigure PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytorch_lightning as pl
import torch

from config.defaults import get_cfg_defaults
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from module.ema import apply_ema_weights_from_checkpoint
from module.uem_module import UEM_Module
from utils.torch_utils import careful_collate_fn, to_device


SUPPORTED_TASKS = ("recon", "gen", "fore")


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _batch_size(mapping: dict) -> int:
    sizes = {len(value) for value in mapping.values()}
    if len(sizes) != 1:
        raise ValueError(f"Condition batch has inconsistent sizes: {sorted(sizes)}")
    return sizes.pop()


def _load_cfg(config_path: str, exp_path: str):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.defrost()
    cfg.TRAIN.EXP_PATH = str(Path(exp_path).resolve())
    cfg.MODEL.CKPT_PATH = "last_ckpt"
    cfg.EVAL.NUM_GPUS = 1
    cfg.freeze()
    return cfg


def _build_dataset(cfg):
    return EE4D_Motion_Dataset(
        data_dir=cfg.DATA.DATA_DIR,
        split="val",
        repre_type=cfg.DATA.REPRE_TYPE,
        cond_img_feat=cfg.DATA.COND_IMG_FEAT,
        cond_traj=cfg.DATA.COND_TRAJ,
        window=cfg.DATA.WINDOW,
        img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
        cond_betas=cfg.DATA.COND_BETAS,
    )


def _prepare_batches(ds, tasks: list[str], num_samples: int, batch_size: int):
    eval_indices = list(range(0, len(ds), 10))[:num_samples]
    if len(eval_indices) != num_samples:
        raise RuntimeError(f"Requested {num_samples} samples, found {len(eval_indices)}")

    prepared = {}
    reference_keys = None
    for task in tasks:
        task_batches = []
        task_keys = []
        for start in range(0, num_samples, batch_size):
            indices = eval_indices[start : start + batch_size]
            samples = [ds.process_sample_for_task(ds[index], task) for index in indices]
            for sample in samples:
                task_keys.append(
                    f"{sample['misc']['seq_name']}_start_{sample['misc']['start_idx'] // 3}"
                )
            task_batches.append(careful_collate_fn(samples))
        if reference_keys is None:
            reference_keys = task_keys
        elif task_keys != reference_keys:
            raise RuntimeError(f"The sample order changed for task {task}")
        prepared[task] = task_batches

    assert reference_keys is not None
    key_digest = hashlib.sha256("\n".join(reference_keys).encode()).hexdigest()
    return prepared, reference_keys, key_digest


def _load_model(cfg, checkpoint_path: Path, device: torch.device):
    model = UEM_Module.load_from_checkpoint(
        str(checkpoint_path), cfg=cfg, map_location="cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema_applied = apply_ema_weights_from_checkpoint(model.model, checkpoint)
    del checkpoint
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    return model, ema_applied


def _postprocess_motion(ds, batch: dict, output: torch.Tensor):
    if "motion" not in batch["misc"]:
        raise ValueError("Speed comparison expects full-body motion checkpoints")
    batch["pred"]["motion"] = output.to("cpu")
    decoded = ds.ret_to_full_sequence(batch)
    del decoded


def _benchmark_task(
    *,
    model,
    ds,
    batches: list[dict],
    device: torch.device,
    repeats: int,
    window_seconds: float,
    include_postprocess: bool,
):
    batch_gpu_seconds = []
    batch_wall_seconds = []
    batch_transfer_seconds = []
    postprocess_seconds = []
    repeat_summaries = []
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0

    for repeat in range(repeats):
        torch.manual_seed(62 + repeat)
        torch.cuda.manual_seed_all(62 + repeat)
        repeat_gpu_seconds = 0.0
        repeat_wall_seconds = 0.0
        repeat_transfer_seconds = 0.0
        repeat_postprocess_seconds = 0.0
        repeat_samples = 0

        for batch in batches:
            transfer_start = time.perf_counter()
            y = to_device(batch["y"], device)
            torch.cuda.synchronize(device)
            transfer_seconds = time.perf_counter() - transfer_start
            current_batch_size = _batch_size(y)

            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            start_event.record()
            with torch.inference_mode():
                output = model.sample(y, current_batch_size)
            end_event.record()
            torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - wall_start
            gpu_seconds = start_event.elapsed_time(end_event) / 1000.0

            peak_allocated_bytes = max(
                peak_allocated_bytes, torch.cuda.max_memory_allocated(device)
            )
            peak_reserved_bytes = max(
                peak_reserved_bytes, torch.cuda.max_memory_reserved(device)
            )

            postprocess_time = 0.0
            # Post-processing is deterministic pipeline overhead. Measuring it
            # once avoids multiplying an expensive CPU/SMPL-X stage by repeats.
            if include_postprocess and repeat == 0:
                postprocess_start = time.perf_counter()
                _postprocess_motion(ds, batch, output)
                postprocess_time = time.perf_counter() - postprocess_start
                postprocess_seconds.append(postprocess_time)

            batch_gpu_seconds.append(gpu_seconds)
            batch_wall_seconds.append(wall_seconds)
            batch_transfer_seconds.append(transfer_seconds)
            repeat_gpu_seconds += gpu_seconds
            repeat_wall_seconds += wall_seconds
            repeat_transfer_seconds += transfer_seconds
            repeat_postprocess_seconds += postprocess_time
            repeat_samples += current_batch_size
            del output, y, start_event, end_event

        repeat_summaries.append(
            {
                "repeat": repeat,
                "samples": repeat_samples,
                "gpu_seconds": repeat_gpu_seconds,
                "wall_seconds": repeat_wall_seconds,
                "host_to_device_seconds": repeat_transfer_seconds,
                "postprocess_seconds": repeat_postprocess_seconds,
            }
        )

    samples_per_repeat = repeat_summaries[0]["samples"]
    if any(item["samples"] != samples_per_repeat for item in repeat_summaries):
        raise RuntimeError("The number of samples changed between repeats")
    total_samples = samples_per_repeat * repeats
    total_gpu_seconds = sum(item["gpu_seconds"] for item in repeat_summaries)
    total_wall_seconds = sum(item["wall_seconds"] for item in repeat_summaries)
    repeat_gpu_values = [item["gpu_seconds"] for item in repeat_summaries]
    repeat_wall_values = [item["wall_seconds"] for item in repeat_summaries]
    mean_transfer_seconds = (
        sum(item["host_to_device_seconds"] for item in repeat_summaries) / repeats
    )
    measured_postprocess_seconds = sum(postprocess_seconds)
    prepared_pipeline_seconds = (
        total_wall_seconds / repeats
        + mean_transfer_seconds
        + measured_postprocess_seconds
    )

    return {
        "repeat_summaries": repeat_summaries,
        "batch_gpu_seconds": batch_gpu_seconds,
        "batch_wall_seconds": batch_wall_seconds,
        "batch_host_to_device_seconds": batch_transfer_seconds,
        "batch_postprocess_seconds": postprocess_seconds,
        "gpu_seconds_mean_per_run": total_gpu_seconds / repeats,
        "gpu_seconds_std_per_run": (
            statistics.stdev(repeat_gpu_values) if repeats > 1 else 0.0
        ),
        "wall_seconds_mean_per_run": total_wall_seconds / repeats,
        "wall_seconds_std_per_run": (
            statistics.stdev(repeat_wall_values) if repeats > 1 else 0.0
        ),
        "host_to_device_seconds_mean_per_run": mean_transfer_seconds,
        "postprocess_seconds_one_run": measured_postprocess_seconds,
        "prepared_pipeline_seconds_one_run": prepared_pipeline_seconds,
        "gpu_ms_per_sample": total_gpu_seconds * 1000.0 / total_samples,
        "throughput_samples_per_second": total_samples / total_gpu_seconds,
        "realtime_multiplier": total_samples * window_seconds / total_gpu_seconds,
        "batch_gpu_p50_seconds": _percentile(batch_gpu_seconds, 50),
        "batch_gpu_p90_seconds": _percentile(batch_gpu_seconds, 90),
        "batch_gpu_p95_seconds": _percentile(batch_gpu_seconds, 95),
        "batch_gpu_mean_seconds": statistics.fmean(batch_gpu_seconds),
        "peak_allocated_mib": peak_allocated_bytes / (1024.0**2),
        "peak_reserved_mib": peak_reserved_bytes / (1024.0**2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--include-postprocess", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("num-samples, batch-size, and repeats must be positive")
    invalid_tasks = set(args.tasks) - set(SUPPORTED_TASKS)
    if invalid_tasks:
        raise ValueError(f"Unsupported tasks: {sorted(invalid_tasks)}")

    cfg = _load_cfg(args.config, args.exp_path)
    checkpoint_path = Path(cfg.TRAIN.EXP_PATH) / "last.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    pl.seed_everything(62, workers=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", 0)

    dataset_start = time.perf_counter()
    ds = _build_dataset(cfg)
    dataset_load_seconds = time.perf_counter() - dataset_start

    preparation_start = time.perf_counter()
    prepared, sample_keys, sample_key_digest = _prepare_batches(
        ds, args.tasks, args.num_samples, args.batch_size
    )
    batch_preparation_seconds = time.perf_counter() - preparation_start

    model_load_start = time.perf_counter()
    model, ema_applied = _load_model(cfg, checkpoint_path, device)
    model_load_seconds = time.perf_counter() - model_load_start
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    inference_dtype = str(next(model.model.parameters()).dtype)

    first_batch = prepared[args.tasks[0]][0]
    first_y = to_device(first_batch["y"], device)
    with torch.inference_mode():
        warmup_start = time.perf_counter()
        warmup_output = model.sample(first_y, _batch_size(first_y))
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
    del warmup_output, first_y

    task_results = {}
    for task in args.tasks:
        task_results[task] = _benchmark_task(
            model=model,
            ds=ds,
            batches=prepared[task],
            device=device,
            repeats=args.repeats,
            window_seconds=cfg.DATA.WINDOW / 10.0,
            include_postprocess=args.include_postprocess,
        )

    generative_type = str(cfg.MODEL.GENERATIVE_TYPE).lower()
    if generative_type in {"flow", "flow_matching", "flow-matching"}:
        nfe = cfg.FLOW.NUM_STEPS
        solver = cfg.FLOW.SOLVER
    else:
        nfe = cfg.MODEL.DIFFUSION_STEPS
        solver = "ancestral_ddpm"

    payload = {
        "schema_version": 1,
        "label": args.label,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "generative_type": generative_type,
        "solver": solver,
        "nfe": int(nfe),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "tasks": args.tasks,
        "include_postprocess": args.include_postprocess,
        "sample_key_sha256": sample_key_digest,
        "first_sample_key": sample_keys[0],
        "last_sample_key": sample_keys[-1],
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "inference_dtype": inference_dtype,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "parameter_count": parameter_count,
        "ema_applied": ema_applied,
        "dataset_load_seconds": dataset_load_seconds,
        "batch_preparation_seconds": batch_preparation_seconds,
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "results": task_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
