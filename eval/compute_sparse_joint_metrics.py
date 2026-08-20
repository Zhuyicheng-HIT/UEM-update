#!/usr/bin/env python3
"""Evaluate a common sparse SMPL22 joint set without a recovery model.

This evaluator deliberately stays in the v4_beta representation space.  A
sparse checkpoint produces K*9+45 features, while dense checkpoints produce
243 features; both are converted to the same ordered K*9 joint blocks before
metrics are accumulated.  This makes K12, E7, and the original diffusion
checkpoint directly comparable without decoding or modifying unobserved
joints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from tqdm.auto import tqdm

from config.defaults import finalize_sparse_joint_config, get_cfg_defaults
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from dataset.representation_utils import full_to_sparse_motion
from module.ema import apply_ema_weights_from_checkpoint
from module.uem_module import UEM_Module
from utils.rotation_conversions import rotation_6d_to_matrix
from utils.task_conditioning import TASKS
from utils.torch_utils import careful_collate_fn, to_device


K12_INDICES = (0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21)
PAPER_WINDOWS = {
    "recon": (0, None),
    "gen": (0, 20),
    "fore": (20, 40),
}
FPS = 10.0


def _init_distributed() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed sparse evaluation requires CUDA.")
        torch.cuda.set_device(local_rank)
        # The model forward still runs on one CUDA device per rank.  Gloo is
        # used only for the tiny CPU metric reductions because this host's
        # CUDA/NCCL combination reports ``invalid device function`` even for a
        # scalar all-reduce.
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world_size, distributed


def _build_cfg(args: argparse.Namespace):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.defrost()
    cfg.TRAIN.EXP_PATH = str(Path(args.checkpoint).resolve().parent)
    cfg.MODEL.CKPT_PATH = str(Path(args.checkpoint).resolve())
    cfg.TRAIN.ONLY_VALIDATE = True
    cfg.EVAL.BATCH_SIZE = int(args.batch_size)
    cfg.EVAL.NUM_SAMPLES = int(args.num_samples)
    cfg.freeze()
    return finalize_sparse_joint_config(cfg)


def _joint_indices(cfg):
    sparse_cfg = getattr(cfg, "SPARSE_JOINTS", None)
    if sparse_cfg is not None and sparse_cfg.ENABLED:
        indices = tuple(int(x) for x in sparse_cfg.INDICES)
        if indices != K12_INDICES:
            raise ValueError(
                "This comparison is fixed to the trained K12 set "
                f"{list(K12_INDICES)}, but config contains {list(indices)}."
            )
        return indices, True
    return K12_INDICES, False


def _rotation_geodesic_deg(pred_rot6d: torch.Tensor, gt_rot6d: torch.Tensor) -> torch.Tensor:
    pred_rot = rotation_6d_to_matrix(pred_rot6d)
    gt_rot = rotation_6d_to_matrix(gt_rot6d)
    relative = pred_rot.transpose(-1, -2) @ gt_rot
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine) * (180.0 / torch.pi)


def _new_stats(k: int) -> dict[str, torch.Tensor]:
    return {
        "frames": torch.zeros((), dtype=torch.float64),
        "vel_frames": torch.zeros((), dtype=torch.float64),
        "pos_sum_mm": torch.zeros((), dtype=torch.float64),
        "rot_sum_deg": torch.zeros((), dtype=torch.float64),
        "pos_vel_sum_mm_s": torch.zeros((), dtype=torch.float64),
        "pos_joint_sum_mm": torch.zeros(k, dtype=torch.float64),
        "rot_joint_sum_deg": torch.zeros(k, dtype=torch.float64),
        "pos_vel_joint_sum_mm_s": torch.zeros(k, dtype=torch.float64),
    }


def _accumulate(stats, pred, gt, valid_frames, task):
    """Accumulate metrics for a batch of [B,T,K,9] raw features."""
    k = pred.shape[2]
    start, fixed_end = PAPER_WINDOWS[task]
    for i in range(pred.shape[0]):
        valid_len = int(valid_frames[i].to(dtype=torch.long).sum().item())
        end = valid_len if fixed_end is None else min(valid_len, fixed_end)
        if end <= start:
            continue
        pred_seg = pred[i, start:end]
        gt_seg = gt[i, start:end]
        pos_err = torch.linalg.vector_norm(pred_seg[..., 6:9] - gt_seg[..., 6:9], dim=-1) * 1000.0
        rot_err = _rotation_geodesic_deg(pred_seg[..., :6], gt_seg[..., :6])
        stats["frames"] += float(pos_err.numel())
        stats["pos_sum_mm"] += pos_err.double().sum().cpu()
        stats["rot_sum_deg"] += rot_err.double().sum().cpu()
        stats["pos_joint_sum_mm"] += pos_err.double().sum(dim=0).cpu()
        stats["rot_joint_sum_deg"] += rot_err.double().sum(dim=0).cpu()

        if end - start >= 2:
            pred_vel = (pred_seg[1:, ..., 6:9] - pred_seg[:-1, ..., 6:9]) * FPS * 1000.0
            gt_vel = (gt_seg[1:, ..., 6:9] - gt_seg[:-1, ..., 6:9]) * FPS * 1000.0
            vel_err = torch.linalg.vector_norm(pred_vel - gt_vel, dim=-1)
            stats["vel_frames"] += float(vel_err.numel())
            stats["pos_vel_sum_mm_s"] += vel_err.double().sum().cpu()
            stats["pos_vel_joint_sum_mm_s"] += vel_err.double().sum(dim=0).cpu()


def _as_joint_blocks(motion: torch.Tensor, joint_indices, *, is_sparse: bool) -> torch.Tensor:
    """Return only the ordered K*9 body blocks, dropping the 45D auxiliary tail."""
    if not is_sparse:
        motion = full_to_sparse_motion(motion, joint_indices)
    body_dim = len(joint_indices) * 9
    if motion.shape[-1] < body_dim:
        raise ValueError(f"Motion has {motion.shape[-1]} features, expected at least {body_dim}.")
    return motion[..., :body_dim].contiguous().view(motion.shape[0], motion.shape[1], len(joint_indices), 9)


def _reduce_stats(stats, distributed, device):
    if not distributed:
        return stats
    for value in stats.values():
        reduced = value.to(device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        value.copy_(reduced.cpu())
    return stats


def _stats_to_json(stats, indices):
    frames = max(float(stats["frames"].item()), 1.0)
    vel_frames = max(float(stats["vel_frames"].item()), 1.0)
    pos_joint = stats["pos_joint_sum_mm"] / (frames / len(indices))
    rot_joint = stats["rot_joint_sum_deg"] / (frames / len(indices))
    vel_joint = stats["pos_vel_joint_sum_mm_s"] / (vel_frames / len(indices))
    return {
        "num_joint_frames": int(round(frames)),
        "num_velocity_frames": int(round(vel_frames)),
        "position_error_mm": float(stats["pos_sum_mm"].item() / frames),
        "rotation_error_deg": float(stats["rot_sum_deg"].item() / frames),
        "position_velocity_error_mm_s": float(stats["pos_vel_sum_mm_s"].item() / vel_frames),
        "per_joint_position_error_mm": {
            str(joint): float(value) for joint, value in zip(indices, pos_joint.tolist())
        },
        "per_joint_rotation_error_deg": {
            str(joint): float(value) for joint, value in zip(indices, rot_joint.tolist())
        },
        "per_joint_position_velocity_error_mm_s": {
            str(joint): float(value) for joint, value in zip(indices, vel_joint.tolist())
        },
    }


def _load_model(cfg, checkpoint, device):
    logger.info(f"Loading checkpoint: {checkpoint}")
    model = UEM_Module.load_from_checkpoint(checkpoint, cfg=cfg, map_location="cpu")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if apply_ema_weights_from_checkpoint(model.model, checkpoint_data):
        logger.info("Using EMA weights stored in checkpoint.")
    del checkpoint_data
    return model.to(device).eval()


def main(args):
    rank, local_rank, world_size, distributed = _init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(62 + rank)
    cfg = _build_cfg(args)
    joint_indices, is_sparse = _joint_indices(cfg)

    ds = EE4D_Motion_Dataset(
        data_dir=cfg.DATA.DATA_DIR,
        split="val",
        repre_type=cfg.DATA.REPRE_TYPE,
        cond_img_feat=cfg.DATA.COND_IMG_FEAT,
        cond_traj=cfg.DATA.COND_TRAJ,
        window=cfg.DATA.WINDOW,
        img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
        cond_betas=cfg.DATA.COND_BETAS,
        sparse_joint_indices=list(joint_indices) if is_sparse else None,
    )
    model = _load_model(cfg, args.checkpoint, device)

    eval_indices = list(range(0, len(ds), 10))
    if args.num_samples > 0:
        eval_indices = eval_indices[: args.num_samples]
    local_indices = eval_indices[rank::world_size]
    if rank == 0:
        logger.info(
            f"{args.name}: {len(eval_indices)} validation windows, K12={list(joint_indices)}, "
            f"world_size={world_size}, batch_per_rank={args.batch_size}"
        )

    task_stats = {task: _new_stats(len(joint_indices)) for task in TASKS}
    batch_size = int(args.batch_size)
    for task in TASKS:
        batch = []
        for idx in tqdm(local_indices, disable=rank != 0, desc=f"{args.name}/{task}"):
            sample = ds[idx]
            sample["misc"]["eval_valid_frames"] = sample["y"]["valid_frames"].clone()
            batch.append(ds.process_sample_for_task(sample, task))
            if len(batch) < batch_size:
                continue

            batch_data = careful_collate_fn(batch)
            valid_frames = batch_data["misc"]["eval_valid_frames"].bool()
            y = to_device(batch_data["y"], device)
            with torch.inference_mode():
                pred = model.sample(y, len(batch), cond_scale=cfg.TRAIN.COND_SCALE)
            if isinstance(pred, tuple):
                pred = pred[-1]
            pred_raw = ds.denormalize(pred.detach().cpu(), "motion")
            gt_raw = ds.denormalize(batch_data["misc"]["motion"].detach().cpu(), "motion")
            pred_raw = _as_joint_blocks(pred_raw, joint_indices, is_sparse=is_sparse)
            gt_raw = _as_joint_blocks(gt_raw, joint_indices, is_sparse=is_sparse)
            _accumulate(task_stats[task], pred_raw, gt_raw, valid_frames, task)
            batch = []

        if batch:
            batch_data = careful_collate_fn(batch)
            valid_frames = batch_data["misc"]["eval_valid_frames"].bool()
            y = to_device(batch_data["y"], device)
            with torch.inference_mode():
                pred = model.sample(y, len(batch), cond_scale=cfg.TRAIN.COND_SCALE)
            if isinstance(pred, tuple):
                pred = pred[-1]
            pred_raw = ds.denormalize(pred.detach().cpu(), "motion")
            gt_raw = ds.denormalize(batch_data["misc"]["motion"].detach().cpu(), "motion")
            pred_raw = _as_joint_blocks(pred_raw, joint_indices, is_sparse=is_sparse)
            gt_raw = _as_joint_blocks(gt_raw, joint_indices, is_sparse=is_sparse)
            _accumulate(task_stats[task], pred_raw, gt_raw, valid_frames, task)

        _reduce_stats(task_stats[task], distributed, device)
        if distributed:
            dist.barrier()

    if rank == 0:
        result = {
            "name": args.name,
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "joint_indices": list(joint_indices),
            "sparse_model_output": bool(is_sparse),
            "representation": "v4_beta canonical joint blocks",
            "units": {"position": "mm", "rotation": "degree", "velocity": "mm/s"},
            "protocol": PAPER_WINDOWS,
            "num_windows": len(eval_indices),
            "metrics": {task: _stats_to_json(task_stats[task], joint_indices) for task in TASKS},
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        logger.info(f"Saved sparse-joint metrics to {output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=256)
    main(parser.parse_args())
