"""Training-only differentiable SMPL-X kinematic losses for E19.

The implementation deliberately stops at the 55-joint SMPL-X kinematic
tree. It computes shape-dependent rest joints and forward kinematics, but
never constructs the 10k-vertex mesh, keeping memory practical for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.rotation_conversions as rc
from utils.pca_conversions import pca_to_matrix


def _transform_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    bottom = torch.zeros(*rotation.shape[:-2], 1, 4, device=rotation.device, dtype=rotation.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat((torch.cat((rotation, translation[..., None]), dim=-1), bottom), dim=-2)


def _inverse_kinematics(global_rotation: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    local = [global_rotation[..., 0, :, :]]
    for joint in range(1, global_rotation.shape[-3]):
        parent_rotation = global_rotation[..., int(parents[joint]), :, :]
        local.append(parent_rotation.transpose(-1, -2) @ global_rotation[..., joint, :, :])
    return torch.stack(local, dim=-3)


def _forward_kinematics(
    local_rotation: torch.Tensor,
    rest_joints: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """Return posed joint locations without evaluating SMPL-X vertices."""

    relative = rest_joints.clone()
    relative[:, 1:] -= rest_joints[:, parents[1:]]
    local_transform = _transform_matrix(local_rotation, relative)
    transforms = [local_transform[:, 0]]
    for joint in range(1, local_rotation.shape[1]):
        transforms.append(transforms[int(parents[joint])] @ local_transform[:, joint])
    return torch.stack(transforms, dim=1)[..., :3, 3]


def _masked_huber(error: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    """Reduce a BxTx... error to one robust loss per sample."""

    if error.shape[:2] != mask.shape:
        raise ValueError(
            f"Geometry mask {tuple(mask.shape)} does not match error prefix {tuple(error.shape[:2])}."
        )
    element_loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=delta, reduction="none")
    expanded_mask = mask.to(dtype=element_loss.dtype)
    for _ in range(error.ndim - 2):
        expanded_mask = expanded_mask.unsqueeze(-1)
    feature_count = error[0, 0].numel() if error.shape[1] > 0 else 1
    numerator = (element_loss * expanded_mask).flatten(1).sum(dim=1)
    denominator = mask.to(dtype=element_loss.dtype).sum(dim=1) * feature_count
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )


def geometry_terms_from_joints(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    target_contacts: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    delta: float,
) -> dict[str, torch.Tensor]:
    """Compute metric-aligned geometry terms from SMPL-X kinematic joints."""

    if pred_joints.shape != target_joints.shape or pred_joints.ndim != 4:
        raise ValueError("Predicted and target joints must share shape BxTxJx3.")
    if pred_joints.shape[2] < 55:
        raise ValueError("SMPL-X geometry loss requires the first 55 kinematic joints.")
    loss_mask = torch.as_tensor(loss_mask, device=pred_joints.device, dtype=torch.bool)
    contacts = torch.as_tensor(target_contacts, device=pred_joints.device).clamp(0.0, 1.0)

    body_error = pred_joints[:, :, :22] - target_joints[:, :, :22]
    root_error = body_error[:, :, 0]
    pred_relative = pred_joints[:, :, 1:22] - pred_joints[:, :, :1]
    target_relative = target_joints[:, :, 1:22] - target_joints[:, :, :1]

    left_hand_error = (
        pred_joints[:, :, 25:40] - pred_joints[:, :, 20:21]
        - (target_joints[:, :, 25:40] - target_joints[:, :, 20:21])
    )
    right_hand_error = (
        pred_joints[:, :, 40:55] - pred_joints[:, :, 21:22]
        - (target_joints[:, :, 40:55] - target_joints[:, :, 21:22])
    )
    hand_error = torch.cat((left_hand_error, right_hand_error), dim=2)

    foot = pred_joints[:, :, [10, 11]]
    target_foot = target_joints[:, :, [10, 11]]
    contact_mask = loss_mask & (contacts.max(dim=-1).values > 0.5)
    foot_height = (foot[..., 2] - target_foot[..., 2]) * contacts

    if pred_joints.shape[1] > 1:
        foot_velocity = foot[:, 1:] - foot[:, :-1]
        velocity_mask = loss_mask[:, 1:] & loss_mask[:, :-1]
        velocity_contact = contacts[:, 1:] * contacts[:, :-1]
        foot_velocity = foot_velocity * velocity_contact[..., None]
        velocity_mask &= velocity_contact.max(dim=-1).values > 0.5
    else:
        foot_velocity = foot[:, :0]
        velocity_mask = loss_mask[:, :0]

    return {
        "joint": _masked_huber(body_error, loss_mask, delta),
        "root": _masked_huber(root_error, loss_mask, delta),
        "relative": _masked_huber(pred_relative - target_relative, loss_mask, delta),
        "hand": _masked_huber(hand_error, loss_mask, delta),
        "foot_velocity": _masked_huber(foot_velocity, velocity_mask, delta),
        "foot_height": _masked_huber(foot_height, contact_mask, delta),
    }


class SMPLXGeometryLoss(nn.Module):
    """Decode normalized v4_beta motion and supervise SMPL-X joint geometry."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        if cfg.DATA.REPRE_TYPE != "v4_beta":
            raise ValueError("E19 SMPL-X geometry loss currently requires DATA.REPRE_TYPE=v4_beta.")

        geometry_cfg = cfg.TRAIN.GEOMETRY_LOSS
        self.weight = float(geometry_cfg.WEIGHT)
        self.warmup_steps = int(geometry_cfg.WARMUP_STEPS)
        self.task_weights = tuple(float(value) for value in geometry_cfg.TASK_WEIGHTS)
        if len(self.task_weights) != 3 or any(value < 0.0 for value in self.task_weights):
            raise ValueError("GEOMETRY_LOSS.TASK_WEIGHTS must contain three non-negative values.")
        self.component_weights = {
            "joint": float(geometry_cfg.JOINT_WEIGHT),
            "root": float(geometry_cfg.ROOT_WEIGHT),
            "relative": float(geometry_cfg.RELATIVE_WEIGHT),
            "hand": float(geometry_cfg.HAND_WEIGHT),
            "foot_velocity": float(geometry_cfg.FOOT_VELOCITY_WEIGHT),
            "foot_height": float(geometry_cfg.FOOT_HEIGHT_WEIGHT),
        }
        self.huber_delta = float(geometry_cfg.HUBER_DELTA)
        if self.weight < 0.0 or self.warmup_steps < 0 or self.huber_delta <= 0.0:
            raise ValueError("Geometry weight/warmup/delta configuration is invalid.")

        stats_path = Path(cfg.DATA.DATA_DIR) / "uniegomotion" / "v4_beta_ee_train_stats.pt"
        if not stats_path.is_file():
            raise FileNotFoundError(f"E19 motion normalization statistics not found at {stats_path}.")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        self.register_buffer("motion_mean", stats["motion_mean"].float(), persistent=False)
        self.register_buffer("motion_std", stats["motion_std"].float(), persistent=False)

        # Import lazily so all non-E19 experiments remain usable without the
        # optional SMPL-X package/model files in the current environment.
        from dataset.smpl_utils import get_smpl

        smpl = get_smpl("smplx")
        self.register_buffer("v_template", smpl.v_template.detach().float().clone(), persistent=False)
        self.register_buffer("shapedirs", smpl.shapedirs[..., :10].detach().float().clone(), persistent=False)
        self.register_buffer("joint_regressor", smpl.J_regressor.detach().float().clone(), persistent=False)
        self.register_buffer("parents", smpl.parents[:55].detach().long().clone(), persistent=False)
        self.register_buffer(
            "left_hand_components", smpl.left_hand_components.detach().float().clone(), persistent=False
        )
        self.register_buffer(
            "right_hand_components", smpl.right_hand_components.detach().float().clone(), persistent=False
        )

    def _rest_joints(self, betas: torch.Tensor) -> torch.Tensor:
        shaped_vertices = self.v_template[None] + torch.einsum(
            "bl,vcl->bvc", betas, self.shapedirs
        )
        return torch.einsum("jv,bvc->bjc", self.joint_regressor[:55], shaped_vertices)

    def _decode(self, normalized_motion: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        motion = normalized_motion.float() * (self.motion_std + 1.0e-6) + self.motion_mean
        batch_size, window, feature_dim = motion.shape
        if feature_dim != 243:
            raise ValueError(f"v4_beta geometry expects 243 features, got {feature_dim}.")

        local_rottrans = motion[..., :198].reshape(batch_size, window, 22, 9)
        local_transform = _transform_matrix(
            rc.rotation_6d_to_matrix(local_rottrans[..., :6]),
            local_rottrans[..., 6:9],
        )
        delta = motion[..., 198:207]
        delta_transform = _transform_matrix(
            rc.rotation_6d_to_matrix(delta[..., :6]), delta[..., 6:9]
        )
        inverse_canonical = [delta_transform[:, 0]]
        for frame in range(1, window):
            inverse_canonical.append(inverse_canonical[-1] @ delta_transform[:, frame])
        inverse_canonical = torch.stack(inverse_canonical, dim=1)
        global_transform = inverse_canonical[:, :, None] @ local_transform
        global_rotation = global_transform[..., :3, :3]
        global_translation = global_transform[..., :3, 3]
        body_rotation = _inverse_kinematics(global_rotation, self.parents[:22])

        left_hand = pca_to_matrix(motion[..., 207:219], self.left_hand_components)
        right_hand = pca_to_matrix(motion[..., 219:231], self.right_hand_components)
        identity = torch.eye(3, device=motion.device, dtype=motion.dtype).expand(
            batch_size, window, 3, 3, 3
        )
        local_rotation = torch.cat((body_rotation, identity, left_hand, right_hand), dim=2)

        valid = padding_mask.to(dtype=motion.dtype)
        betas = (motion[..., 233:243] * valid[..., None]).sum(dim=1)
        betas = betas / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        rest_joints = self._rest_joints(betas)
        repeated_rest = rest_joints[:, None].expand(-1, window, -1, -1).reshape(-1, 55, 3)
        posed = _forward_kinematics(
            local_rotation.reshape(-1, 55, 3, 3), repeated_rest, self.parents
        ).reshape(batch_size, window, 55, 3)
        translation = global_translation[:, :, 0] - rest_joints[:, None, 0]
        joints = posed + translation[:, :, None]
        return joints, motion[..., 231:233]

    def forward(
        self,
        pred_xstart: torch.Tensor,
        target_xstart: torch.Tensor,
        conditioning: Mapping[str, torch.Tensor],
        *,
        step: int,
    ) -> dict[str, torch.Tensor]:
        loss_mask = torch.as_tensor(
            conditioning.get("loss_mask", conditioning["valid_frames"]),
            device=pred_xstart.device,
            dtype=torch.bool,
        )
        padding_mask = torch.as_tensor(
            conditioning.get("padding_mask", conditioning["valid_frames"]),
            device=pred_xstart.device,
            dtype=torch.bool,
        )
        task_id = torch.as_tensor(
            conditioning.get("task_id", 0), device=pred_xstart.device, dtype=torch.long
        )
        if task_id.ndim == 0:
            task_id = task_id.expand(pred_xstart.shape[0])
        if task_id.shape != (pred_xstart.shape[0],):
            raise ValueError("Geometry loss requires one task_id per sample.")

        device_type = pred_xstart.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pred_joints, _ = self._decode(pred_xstart, padding_mask)
            with torch.no_grad():
                target_joints, target_contacts = self._decode(target_xstart.detach(), padding_mask)
            terms = geometry_terms_from_joints(
                pred_joints,
                target_joints,
                target_contacts,
                loss_mask,
                delta=self.huber_delta,
            )
            raw = sum(self.component_weights[name] * terms[name] for name in self.component_weights)
            task_scale = torch.tensor(
                self.task_weights, device=raw.device, dtype=raw.dtype
            )[task_id]
            warmup = 1.0 if self.warmup_steps == 0 else min(1.0, (int(step) + 1) / self.warmup_steps)
            terms["raw"] = raw
            terms["loss"] = raw * task_scale * (self.weight * warmup)
            terms["warmup"] = raw.new_full(raw.shape, warmup)
        return terms


__all__ = ["SMPLXGeometryLoss", "geometry_terms_from_joints"]
