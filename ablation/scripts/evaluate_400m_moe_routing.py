#!/usr/bin/env python3
"""Evaluate 400M Motion Expert routing modes and inspect its router.

The script keeps the checkpoint frozen and evaluates four execution modes on
identical validation windows and identical Flow noise: trained Top-2, Top-1,
Shared-only, and deterministic random Top-2.  Router statistics are collected
from the trained Top-2 pass at the native four-frame chunk granularity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from config.defaults import finalize_sparse_joint_config, get_cfg_defaults
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from eval.compute_sparse_joint_metrics import (
    K12_INDICES,
    _accumulate,
    _as_joint_blocks,
    _new_stats,
    _stats_to_json,
)
from module.ema import apply_ema_weights_from_checkpoint
from module.uem_module import UEM_Module
from utils.task_conditioning import TASKS, apply_task_conditioning
from utils.torch_utils import careful_collate_fn, to_device


ROUTING_MODES = ("top2", "top1", "shared_only", "random_top2")


def build_cfg(config: str, checkpoint: str, data_dir: str | None = None):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.defrost()
    cfg.TRAIN.EXP_PATH = str(Path(checkpoint).resolve().parent)
    cfg.MODEL.CKPT_PATH = str(Path(checkpoint).resolve())
    if data_dir is not None:
        cfg.DATA.DATA_DIR = str(Path(data_dir).resolve())
    cfg.TRAIN.ONLY_VALIDATE = True
    cfg.freeze()
    return finalize_sparse_joint_config(cfg)


def build_dataset(cfg):
    return EE4D_Motion_Dataset(
        data_dir=cfg.DATA.DATA_DIR,
        split="val",
        repre_type=cfg.DATA.REPRE_TYPE,
        cond_img_feat=cfg.DATA.COND_IMG_FEAT,
        cond_traj=cfg.DATA.COND_TRAJ,
        window=cfg.DATA.WINDOW,
        img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
        cond_betas=cfg.DATA.COND_BETAS,
        sparse_joint_indices=list(cfg.SPARSE_JOINTS.INDICES),
    )


def load_model(cfg, checkpoint: str, device: torch.device):
    model = UEM_Module.load_from_checkpoint(checkpoint, cfg=cfg, map_location="cpu")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    using_ema = apply_ema_weights_from_checkpoint(model.model, checkpoint_data)
    del checkpoint_data
    logger.info(f"EMA weights: {using_ema}")
    return model.to(device).eval()


def make_batches(dataset, num_samples: int, batch_size: int):
    indices = list(range(0, len(dataset), 10))[:num_samples]
    batches = []
    pending = []
    for index in tqdm(indices, desc="Loading validation windows"):
        sample = dataset[index]
        sample["misc"]["eval_valid_frames"] = sample["y"]["valid_frames"].clone()
        pending.append(sample)
        if len(pending) == batch_size:
            batches.append(careful_collate_fn(pending))
            pending = []
    if pending:
        batches.append(careful_collate_fn(pending))
    return indices, batches


class RouterCollector:
    """Accumulate chunk-level routing statistics through forward hooks."""

    def __init__(self, model, tasks, num_layers, num_experts, chunk_size):
        shape = (len(tasks), num_layers, num_experts)
        self.tasks = tuple(tasks)
        self.chunk_size = int(chunk_size)
        self.prob_sum = np.zeros(shape, dtype=np.float64)
        self.top1_count = np.zeros(shape, dtype=np.float64)
        self.top2_count = np.zeros(shape, dtype=np.float64)
        self.token_count = np.zeros(shape[:2], dtype=np.float64)
        self.entropy_sum = np.zeros(shape[:2], dtype=np.float64)
        self.switch_count = np.zeros(shape[:2], dtype=np.float64)
        self.switch_pairs = np.zeros(shape[:2], dtype=np.float64)
        self.current_task = None
        self.current_valid = None
        self.handles = []
        for layer_index, block in enumerate(model.model.tsfm):
            self.handles.append(block.ff.register_forward_hook(self._hook(layer_index)))

    def _hook(self, layer_index):
        def callback(module, _inputs, _output):
            if self.current_task is None or module.last_router_probs is None:
                return
            probs = module.last_router_probs[:, 1:].float().cpu()
            top = module.last_top_indices[:, 1:].cpu()
            valid = self.current_valid.bool().cpu()
            batch, frames, experts = probs.shape
            chunk = self.chunk_size
            n_chunks = (frames + chunk - 1) // chunk
            pad = n_chunks * chunk - frames
            if pad:
                valid = torch.nn.functional.pad(valid, (0, pad), value=False)
            valid_chunks = valid.view(batch, n_chunks, chunk).any(dim=-1)
            chunk_probs = probs[:, ::chunk]
            chunk_top = top[:, ::chunk]
            selected_probs = chunk_probs[valid_chunks]
            selected_top = chunk_top[valid_chunks]
            if selected_probs.numel() == 0:
                return

            task_index = self.tasks.index(self.current_task)
            self.prob_sum[task_index, layer_index] += selected_probs.sum(0).numpy()
            self.token_count[task_index, layer_index] += selected_probs.shape[0]
            entropy = -(selected_probs.clamp_min(1e-12) * selected_probs.clamp_min(1e-12).log()).sum(-1)
            self.entropy_sum[task_index, layer_index] += float(entropy.sum())
            for slot in range(selected_top.shape[-1]):
                counts = torch.bincount(selected_top[:, slot], minlength=experts).double().numpy()
                self.top2_count[task_index, layer_index] += counts
                if slot == 0:
                    self.top1_count[task_index, layer_index] += counts

            top1 = chunk_top[..., 0]
            if n_chunks > 1:
                adjacent = valid_chunks[:, 1:] & valid_chunks[:, :-1]
                switched = (top1[:, 1:] != top1[:, :-1]) & adjacent
                self.switch_count[task_index, layer_index] += float(switched.sum())
                self.switch_pairs[task_index, layer_index] += float(adjacent.sum())

        return callback

    def close(self):
        for handle in self.handles:
            handle.remove()

    def result(self):
        denom = np.maximum(self.token_count[..., None], 1.0)
        top2_denom = np.maximum(2.0 * self.token_count[..., None], 1.0)
        prob = self.prob_sum / denom
        top1 = self.top1_count / denom
        top2 = self.top2_count / top2_denom
        entropy = self.entropy_sum / np.maximum(self.token_count, 1.0)
        entropy_norm = entropy / np.log(self.prob_sum.shape[-1])
        switches = self.switch_count / np.maximum(self.switch_pairs, 1.0)
        task_top1 = self.top1_count.sum(axis=1)
        expert_total = np.maximum(task_top1.sum(axis=0, keepdims=True), 1.0)
        task_preference = task_top1 / expert_total
        return {
            "mean_router_probability": prob.tolist(),
            "top1_load_fraction": top1.tolist(),
            "top2_slot_fraction": top2.tolist(),
            "normalized_entropy": entropy_norm.tolist(),
            "chunk_switch_rate": switches.tolist(),
            "task_preference_given_expert": task_preference.tolist(),
            "token_count": self.token_count.tolist(),
        }


def plot_router_stats(stats, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    loads = np.asarray(stats["top1_load_fraction"])
    entropy = np.asarray(stats["normalized_entropy"])
    switches = np.asarray(stats["chunk_switch_rate"])
    preference = np.asarray(stats["task_preference_given_expert"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        image = axes[task_index].imshow(loads[task_index], aspect="auto", vmin=0.0, vmax=max(0.2, loads.max()))
        axes[task_index].set_title(f"{task}: Top-1 expert load")
        axes[task_index].set_xlabel("Routed expert")
        axes[task_index].set_ylabel("Transformer layer")
        axes[task_index].set_xticks(range(loads.shape[-1]))
        axes[task_index].set_xticklabels(range(1, loads.shape[-1] + 1))
    fig.colorbar(image, ax=axes, shrink=0.85)
    fig.savefig(output_dir / "router_load_by_task_layer.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        axes[0].plot(np.arange(1, entropy.shape[1] + 1), entropy[task_index], marker="o", label=task)
        axes[1].plot(np.arange(1, switches.shape[1] + 1), switches[task_index], marker="o", label=task)
    axes[0].set(title="Normalized router entropy", xlabel="Layer", ylabel="Entropy / log(11)", ylim=(0, 1.02))
    axes[1].set(title="Top-1 switch rate across chunks", xlabel="Layer", ylabel="Switch rate", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output_dir / "router_entropy_and_switching.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    bottom = np.zeros(preference.shape[1])
    for task_index, task in enumerate(TASKS):
        axis.bar(np.arange(1, preference.shape[1] + 1), preference[task_index], bottom=bottom, label=task)
        bottom += preference[task_index]
    axis.set(title="Task composition of each expert's Top-1 assignments", xlabel="Routed expert", ylabel="Fraction", ylim=(0, 1))
    axis.set_xticks(range(1, preference.shape[1] + 1))
    axis.legend(ncol=3)
    fig.savefig(output_dir / "router_task_preference.png", dpi=180)
    plt.close(fig)


def evaluate(args):
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    cfg = build_cfg(args.config, args.checkpoint, args.data_dir)
    dataset = build_dataset(cfg)
    indices, batches = make_batches(dataset, args.num_samples, args.batch_size)
    model = load_model(cfg, args.checkpoint, device)
    collector = RouterCollector(
        model,
        TASKS,
        len(model.model.tsfm),
        model.model.tsfm[0].ff.num_experts,
        model.model.tsfm[0].ff.chunk_size,
    )

    results = {}
    for mode_index, mode in enumerate(ROUTING_MODES):
        logger.info(f"Evaluating routing mode: {mode}")
        model.model.set_moe_inference_mode(mode, seed=args.seed)
        mode_result = {"metrics": {}, "timing": {}}
        for task_index, task in enumerate(TASKS):
            stats = _new_stats(len(K12_INDICES))
            # Warm-up is deliberately excluded from timing and router counts.
            warm_y = apply_task_conditioning(
                batches[0]["y"], task, forecast_prefix=cfg.DATA.WINDOW // 4
            )
            collector.current_task = None
            with torch.inference_mode():
                _ = model.sample(to_device(warm_y, device), len(batches[0]["misc"]["motion"]), cond_scale=cfg.TRAIN.COND_SCALE)
            torch.cuda.synchronize(device)

            # Restore both Flow noise and private random-routing sequence so
            # every mode sees exactly the same initial Gaussian samples.
            model.model.set_moe_inference_mode(mode, seed=args.seed)
            torch.manual_seed(args.seed + 1000 * task_index)
            torch.cuda.manual_seed_all(args.seed + 1000 * task_index)
            elapsed_ms = 0.0
            samples = 0
            for batch in tqdm(batches, desc=f"{mode}/{task}"):
                y = apply_task_conditioning(
                    batch["y"], task, forecast_prefix=cfg.DATA.WINDOW // 4
                )
                y_device = to_device(y, device)
                # Router collection performs deliberate GPU-to-CPU copies and
                # must not contaminate the CUDA-event latency measurement.
                collector.current_task = None
                collector.current_valid = batch["misc"]["eval_valid_frames"]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode():
                    pred = model.sample(y_device, len(batch["misc"]["motion"]), cond_scale=cfg.TRAIN.COND_SCALE)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms += start.elapsed_time(end)
                samples += pred.shape[0]

                pred_raw = dataset.denormalize(pred.detach().cpu(), "motion")
                gt_raw = dataset.denormalize(batch["misc"]["motion"], "motion")
                pred_blocks = _as_joint_blocks(pred_raw, K12_INDICES, is_sparse=True)
                gt_blocks = _as_joint_blocks(gt_raw, K12_INDICES, is_sparse=True)
                _accumulate(stats, pred_blocks, gt_blocks, batch["misc"]["eval_valid_frames"].bool(), task)
            collector.current_task = None
            mode_result["metrics"][task] = _stats_to_json(stats, K12_INDICES)
            mode_result["timing"][task] = {
                "gpu_time_ms": elapsed_ms,
                "ms_per_window": elapsed_ms / samples,
                "windows_per_second": 1000.0 * samples / elapsed_ms,
                "batch_size": args.batch_size,
                "num_windows": samples,
            }
        results[mode] = mode_result

    # Router diagnostics use a separate untimed traversal.  This preserves a
    # clean speed comparison while still collecting every layer/Flow call.
    logger.info("Collecting trained Top-2 router statistics in a separate untimed pass.")
    model.model.set_moe_inference_mode("top2", seed=args.seed)
    for task_index, task in enumerate(TASKS):
        torch.manual_seed(args.seed + 1000 * task_index)
        torch.cuda.manual_seed_all(args.seed + 1000 * task_index)
        for batch in tqdm(batches, desc=f"router-stats/{task}"):
            y = apply_task_conditioning(
                batch["y"], task, forecast_prefix=cfg.DATA.WINDOW // 4
            )
            collector.current_task = task
            collector.current_valid = batch["misc"]["eval_valid_frames"]
            with torch.inference_mode():
                _ = model.sample(
                    to_device(y, device),
                    len(batch["misc"]["motion"]),
                    cond_scale=cfg.TRAIN.COND_SCALE,
                )
    collector.current_task = None

    collector.close()
    router_stats = collector.result()
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "joint_indices": list(K12_INDICES),
        "routing_modes": list(ROUTING_MODES),
        "num_windows": len(indices),
        "seed": args.seed,
        "noise_protocol": "same task-specific Flow RNG seed for every routing mode",
        "timing_protocol": "CUDA events, 10-step Euler Flow, one untimed warm-up batch per mode/task",
        "results": results,
        "router": {
            "protocol": "trained Top-2, all 10 Flow evaluations, valid four-frame chunks only",
            "tasks": list(TASKS),
            "num_layers": len(model.model.tsfm),
            "num_routed_experts": model.model.tsfm[0].ff.num_experts,
            "chunk_size": model.model.tsfm[0].ff.chunk_size,
            **router_stats,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    plot_router_stats(router_stats, output_path.parent / "router_figures")
    logger.info(f"Saved routing ablation to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=62)
    evaluate(parser.parse_args())
