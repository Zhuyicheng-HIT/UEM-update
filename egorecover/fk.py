"""Optional SMPL22 FK with fixed bootstrap shape and no GT fallbacks.

The caller provides a licensed SMPL-X layer (dataset.smpl_utils.get_smpl()).
Importing this module alone does not require assets. Actual asset-backed FK
must be validated before these outputs become formal utility labels.
"""

import torch
from torch import nn

from utils.pca_conversions import pca_to_matrix


class FixedShapeFK(nn.Module):
    def __init__(self, smpl, beta_boot):
        super().__init__()
        if beta_boot.shape != (10,) or not bool(torch.isfinite(beta_boot).all()):
            raise ValueError("Provide ten finite beta coefficients from the common model bootstrap.")
        self.smpl = smpl.eval().requires_grad_(False)
        self.register_buffer("beta_boot", beta_boot.detach().clone())
        self.parents = [int(value) for value in smpl.parents[:22]]
        if (
            len(self.parents) != 22
            or self.parents[0] != -1
            or any(not 0 <= parent < joint for joint, parent in enumerate(self.parents[1:], start=1))
        ):
            raise ValueError("Expected topologically ordered SMPL22 parents.")
        with torch.no_grad():
            root = self.smpl(betas=self.beta_boot[None]).joints[0, 0]
        self.register_buffer("root_offset", root.detach().clone())

    def train(self, mode=True):
        super().train(mode)
        self.smpl.eval()
        return self

    def forward(self, state):
        if state.joints.shape[-3:] != (22, 4, 4) or state.auxiliary.shape[-1] != 36:
            raise ValueError("Expected dense BodyState with 22 transforms and 36 auxiliary channels.")
        prefix = state.joints.shape[:-3]
        if state.auxiliary.shape[:-1] != prefix:
            raise ValueError("Body transforms and auxiliary channels must have identical batch dimensions.")
        joints = state.joints.reshape(-1, 22, 4, 4)
        auxiliary = state.auxiliary.reshape(-1, 36)
        global_rotation = joints[..., :3, :3]
        local = torch.stack(
            [
                (
                    global_rotation[:, joint]
                    if parent < 0
                    else global_rotation[:, parent].transpose(-1, -2) @ global_rotation[:, joint]
                )
                for joint, parent in enumerate(self.parents)
            ],
            dim=1,
        )
        count = len(joints)
        output = self.smpl(
            global_orient=local[:, 0],
            body_pose=local[:, 1:22],
            betas=self.beta_boot.expand(count, -1),
            transl=joints[:, 0, :3, 3] - self.root_offset,
            left_hand_pose=pca_to_matrix(auxiliary[:, :12], self.smpl.left_hand_components),
            right_hand_pose=pca_to_matrix(auxiliary[:, 12:24], self.smpl.right_hand_components),
            return_verts=False,
        )
        result = output.joints[:, :22].reshape(*prefix, 22, 3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("SMPL FK produced nonfinite joints.")
        return result
