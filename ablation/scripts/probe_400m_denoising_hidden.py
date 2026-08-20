#!/usr/bin/env python3
"""Frozen linear probes for step-1, step-5 and step-10 Motion Expert tokens.

The 400M model and its EMA parameters remain frozen.  A separate, identical
LayerNorm+Linear probe is fitted for each denoising step to predict the K12
future trajectory (frames 20--39) from the corresponding frame hidden tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
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

from ablation.scripts.evaluate_400m_moe_routing import build_cfg, build_dataset, load_model
from eval.compute_sparse_joint_metrics import K12_INDICES, _accumulate, _new_stats, _stats_to_json
from utils.task_conditioning import apply_task_conditioning
from utils.torch_utils import careful_collate_fn, to_device


STEP_INDICES = (0, 4, 9)
STEP_NAMES = ("step1", "step5", "step10")
TARGET_START = 20
TARGET_END = 40
BODY_DIM = len(K12_INDICES) * 9


class LinearMotionProbe(torch.nn.Module):
    def __init__(self, hidden_dim=768, output_dim=BODY_DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden):
        return self.net(hidden)


def choose_indices(dataset, count, seed, stride=None):
    if stride is not None:
        candidates = np.arange(0, len(dataset), stride, dtype=np.int64)
    else:
        candidates = np.arange(len(dataset), dtype=np.int64)
    if count > len(candidates):
        raise ValueError(f"Requested {count} samples from only {len(candidates)} candidates.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=count, replace=False)
    return chosen.tolist()


def _run_hidden_steps(model, y, batch_size, device):
    hiddens = {}
    step_times = {}
    iterator = model.flow.sample_loop_progressive(
        model.model,
        (batch_size, model.window, model.model.input_feats),
        model_kwargs={"y": y, "cond_scale": model.cfg.TRAIN.COND_SCALE},
        noise=None,
        progress=False,
        return_one_step_hidden=True,
    )
    for step_index, output in enumerate(iterator):
        if step_index in STEP_INDICES:
            name = STEP_NAMES[STEP_INDICES.index(step_index)]
            hiddens[name] = output["one_step_hidden"][:, TARGET_START:TARGET_END].detach().cpu().half()
            step_times[name] = float(output["t"][0].item())
    if set(hiddens) != set(STEP_NAMES):
        raise RuntimeError(f"Expected hidden steps {STEP_NAMES}, collected {tuple(hiddens)}.")
    return hiddens, step_times


def extract_features(model, dataset, indices, batch_size, device, seed, description):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.model.set_moe_inference_mode("top2", seed=seed)
    collected = {name: [] for name in STEP_NAMES}
    targets = []
    masks = []
    all_times = None
    pending = []
    for position, index in enumerate(tqdm(indices, desc=description)):
        sample = dataset[index]
        sample["misc"]["eval_valid_frames"] = sample["y"]["valid_frames"].clone()
        pending.append(sample)
        if len(pending) < batch_size and position + 1 < len(indices):
            continue
        batch = careful_collate_fn(pending)
        pending = []
        y = apply_task_conditioning(batch["y"], "fore", forecast_prefix=TARGET_START)
        y = to_device(y, device)
        with torch.inference_mode():
            hidden, step_times = _run_hidden_steps(model, y, len(batch["misc"]["motion"]), device)
        all_times = step_times
        for name in STEP_NAMES:
            collected[name].append(hidden[name])
        targets.append(batch["misc"]["motion"][:, TARGET_START:TARGET_END, :BODY_DIM].half())
        masks.append(batch["misc"]["eval_valid_frames"][:, TARGET_START:TARGET_END].bool())
    return (
        {name: torch.cat(parts, dim=0) for name, parts in collected.items()},
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        all_times,
    )


def train_probe(hidden, target, mask, dev_hidden, dev_target, dev_mask, args, seed):
    device = torch.device(args.probe_device)
    torch.manual_seed(seed)
    probe = LinearMotionProbe(hidden.shape[-1], target.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    train_x = hidden[mask].float()
    train_y = target[mask].float()
    dev_x = dev_hidden[dev_mask].float()
    dev_y = dev_target[dev_mask].float()
    generator = torch.Generator().manual_seed(seed)
    best = {"mse": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(args.probe_epochs):
        probe.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        loss_sum = 0.0
        count = 0
        for start in range(0, len(permutation), args.probe_batch_size):
            ids = permutation[start : start + args.probe_batch_size]
            x = train_x.index_select(0, ids).to(device)
            y = train_y.index_select(0, ids).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(probe(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            count += len(ids)
        probe.eval()
        dev_loss_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(dev_x), args.probe_batch_size):
                x = dev_x[start : start + args.probe_batch_size].to(device)
                y = dev_y[start : start + args.probe_batch_size].to(device)
                dev_loss_sum += float(torch.nn.functional.mse_loss(probe(x), y, reduction="sum"))
        train_mse = loss_sum / max(count, 1)
        dev_mse = dev_loss_sum / max(dev_y.numel(), 1)
        history.append({"epoch": epoch + 1, "train_mse": train_mse, "dev_mse": dev_mse})
        if dev_mse < best["mse"]:
            best = {
                "mse": dev_mse,
                "epoch": epoch + 1,
                "state": {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()},
            }
    probe.load_state_dict(best["state"])
    return probe, best, history


def evaluate_probe(probe, hidden, target, mask, dataset, args):
    device = torch.device(args.probe_device)
    predictions = torch.zeros_like(target, dtype=torch.float32)
    probe.eval()
    with torch.inference_mode():
        for start in range(0, len(hidden), args.probe_eval_batch_size):
            predictions[start : start + args.probe_eval_batch_size] = probe(
                hidden[start : start + args.probe_eval_batch_size].float().to(device)
            ).cpu()
    normalized_mse = torch.nn.functional.mse_loss(predictions[mask], target.float()[mask]).item()
    mean = dataset.stats["motion_mean"][:BODY_DIM].view(1, 1, BODY_DIM)
    std = dataset.stats["motion_std"][:BODY_DIM].view(1, 1, BODY_DIM) + 1e-6
    pred_raw = (predictions * std + mean).view(len(predictions), TARGET_END - TARGET_START, len(K12_INDICES), 9)
    gt_raw = (target.float() * std + mean).view(len(target), TARGET_END - TARGET_START, len(K12_INDICES), 9)
    stats = _new_stats(len(K12_INDICES))
    _accumulate(stats, pred_raw, gt_raw, mask, "recon")
    return normalized_mse, _stats_to_json(stats, K12_INDICES), predictions


def hidden_noise_stability(reference, alternatives, mask):
    result = {}
    for name in STEP_NAMES:
        pairs = []
        ref = torch.nn.functional.normalize(reference[name].float(), dim=-1)
        for alt_set in alternatives:
            alt = torch.nn.functional.normalize(alt_set[name].float(), dim=-1)
            pairs.append(float((ref * alt).sum(-1)[mask].mean()))
        result[name] = {
            "mean_cosine_to_seed_reference": float(np.mean(pairs)),
            "per_alternative_seed_cosine": pairs,
        }
    return result


def plot_probe_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(STEP_NAMES)
    position = [results[name]["geometry"]["position_error_mm"] for name in labels]
    rotation = [results[name]["geometry"]["rotation_error_deg"] for name in labels]
    velocity = [results[name]["geometry"]["position_velocity_error_mm_s"] for name in labels]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, values, title, ylabel in zip(
        axes,
        (position, rotation, velocity),
        ("Future position probe", "Future rotation probe", "Future velocity probe"),
        ("mm", "degree", "mm/s"),
    ):
        bars = axis.bar(labels, values, color=("#4C78A8", "#F58518", "#54A24B"))
        axis.bar_label(bars, fmt="%.2f", fontsize=8)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "denoising_hidden_probe.png", dpi=180)
    plt.close(fig)


def main(args):
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    cfg = build_cfg(args.config, args.checkpoint, args.data_dir)
    model = load_model(cfg, args.checkpoint, device)

    logger.info("Loading official train split for probe fitting/dev selection.")
    # build_dataset is validation-specific, so construct the train twin
    # explicitly with identical representation settings.
    from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset

    train_dataset = EE4D_Motion_Dataset(
        data_dir=cfg.DATA.DATA_DIR,
        split="train",
        repre_type=cfg.DATA.REPRE_TYPE,
        cond_img_feat=cfg.DATA.COND_IMG_FEAT,
        cond_traj=cfg.DATA.COND_TRAJ,
        window=cfg.DATA.WINDOW,
        img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
        cond_betas=cfg.DATA.COND_BETAS,
        sparse_joint_indices=list(cfg.SPARSE_JOINTS.INDICES),
    )
    train_indices = choose_indices(train_dataset, args.train_samples + args.dev_samples, args.seed)
    train_hidden_all, train_target_all, train_mask_all, step_times = extract_features(
        model,
        train_dataset,
        train_indices,
        args.extract_batch_size,
        device,
        args.seed,
        "Extract train/dev hidden",
    )
    split = args.train_samples
    train_hidden = {name: value[:split] for name, value in train_hidden_all.items()}
    dev_hidden = {name: value[split:] for name, value in train_hidden_all.items()}
    train_target, dev_target = train_target_all[:split], train_target_all[split:]
    train_mask, dev_mask = train_mask_all[:split], train_mask_all[split:]
    del train_hidden_all, train_target_all, train_mask_all, train_dataset
    gc.collect()

    logger.info("Loading official validation split for the held-out probe test.")
    val_dataset = build_dataset(cfg)
    val_indices = choose_indices(val_dataset, args.val_samples, args.seed + 1, stride=10)
    val_hidden, val_target, val_mask, _ = extract_features(
        model,
        val_dataset,
        val_indices,
        args.extract_batch_size,
        device,
        args.seed,
        "Extract held-out validation hidden",
    )

    results = {}
    output_dir = Path(args.output).parent
    checkpoint_dir = output_dir / "probe_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for step_offset, name in enumerate(STEP_NAMES):
        logger.info(f"Training frozen-token probe: {name}")
        probe, best, history = train_probe(
            train_hidden[name],
            train_target,
            train_mask,
            dev_hidden[name],
            dev_target,
            dev_mask,
            args,
            args.seed + step_offset,
        )
        normalized_mse, geometry, _ = evaluate_probe(
            probe, val_hidden[name], val_target, val_mask, val_dataset, args
        )
        torch.save({"state_dict": probe.state_dict(), "step": name}, checkpoint_dir / f"{name}.pt")
        results[name] = {
            "flow_t": step_times[name],
            "probe_parameters": sum(parameter.numel() for parameter in probe.parameters()),
            "best_dev_epoch": best["epoch"],
            "best_dev_normalized_mse": best["mse"],
            "heldout_normalized_mse": normalized_mse,
            "geometry": geometry,
            "training_history": history,
        }

    # Re-run a small fixed subset with alternative noise seeds.  This is a
    # representation stability diagnostic only; probes are not retrained.
    stability_count = min(args.stability_samples, len(val_indices))
    stability_indices = val_indices[:stability_count]
    reference = {name: value[:stability_count] for name, value in val_hidden.items()}
    stability_mask = val_mask[:stability_count]
    alternatives = []
    alternative_seeds = []
    for offset in range(1, args.stability_seeds):
        alt_seed = args.seed + offset
        alt_hidden, _, _, _ = extract_features(
            model,
            val_dataset,
            stability_indices,
            args.extract_batch_size,
            device,
            alt_seed,
            f"Noise stability seed {alt_seed}",
        )
        alternatives.append(alt_hidden)
        alternative_seeds.append(alt_seed)
    stability = hidden_noise_stability(reference, alternatives, stability_mask) if alternatives else {}

    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "protocol": {
            "frozen_backbone": True,
            "routing": "trained Top-2",
            "task": "forecasting",
            "observed_frames": [0, TARGET_START - 1],
            "probe_target_frames": [TARGET_START, TARGET_END - 1],
            "target": "normalized K12 6D rotation + 3D position blocks",
            "probe": "LayerNorm(768) + Linear(768,108), independently fitted per step",
            "train_split": "official train",
            "test_split": "official val, stride-10 candidates",
            "train_samples": args.train_samples,
            "dev_samples": args.dev_samples,
            "val_samples": args.val_samples,
            "noise_reference_seed": args.seed,
            "noise_alternative_seeds": alternative_seeds,
        },
        "step_times": step_times,
        "results": results,
        "noise_stability": stability,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    plot_probe_results(results, output_path.parent / "probe_figures")
    logger.info(f"Saved denoising-token probe results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-device", default="cuda:0")
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--dev-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--probe-eval-batch-size", type=int, default=8192)
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--stability-samples", type=int, default=64)
    parser.add_argument("--stability-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=62)
    main(parser.parse_args())
