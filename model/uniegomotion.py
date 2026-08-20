import copy
import math
import numpy as np
import torch
import torch.nn as nn
from model.core import DecoderBlock
from model.core import EncoderBlock
from model.core import MoEDecoderBlock
from model.core import MoEEncoderBlock
from loguru import logger


_MOTION_FEATURE_DIMS = {"v1_beta": 234, "v4_beta": 243, "v5_beta": 243}
_SPARSE_SUPPORTED_REPRESENTATIONS = {"v4_beta", "v5_beta"}
_SMPL_BODY_JOINTS = 22
_JOINT_TRANSFORM_DIM = 9
_V4_BETA_AUX_DIM = 45


def get_sparse_joint_indices(cfg):
    """Return validated SMPL22 indices, or an empty tuple when disabled."""
    sparse_cfg = getattr(cfg, "SPARSE_JOINTS", None)
    if sparse_cfg is None or not bool(getattr(sparse_cfg, "ENABLED", False)):
        return ()

    indices = tuple(getattr(sparse_cfg, "INDICES", ()))
    if not indices:
        raise ValueError("SPARSE_JOINTS.INDICES must contain at least one SMPL22 joint index.")
    invalid_types = [index for index in indices if isinstance(index, bool) or not isinstance(index, int)]
    if invalid_types:
        raise ValueError(f"SPARSE_JOINTS.INDICES must contain integers, got {invalid_types}.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"SPARSE_JOINTS.INDICES must be unique, got {list(indices)}.")
    invalid_indices = [index for index in indices if index < 0 or index >= _SMPL_BODY_JOINTS]
    if invalid_indices:
        raise ValueError(
            "SPARSE_JOINTS.INDICES contains indices outside the SMPL22 range "
            f"[0, 21]: {invalid_indices}."
        )
    return indices


def get_motion_feature_dim(cfg):
    """Resolve the model's motion feature dimension from representation/config."""
    repre_type = cfg.DATA.REPRE_TYPE
    if repre_type not in _MOTION_FEATURE_DIMS:
        raise ValueError(
            f"Unsupported DATA.REPRE_TYPE {repre_type!r}; expected one of {sorted(_MOTION_FEATURE_DIMS)}."
        )

    sparse_joint_indices = get_sparse_joint_indices(cfg)
    if not sparse_joint_indices:
        return _MOTION_FEATURE_DIMS[repre_type]
    if repre_type not in _SPARSE_SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            "SPARSE_JOINTS is supported only for v4_beta or v5_beta motion "
            f"representations, got {repre_type!r}."
        )
    return _JOINT_TRANSFORM_DIM * len(sparse_joint_indices) + _V4_BETA_AUX_DIM


def get_global_feature_slice(cfg):
    """Return the contiguous 9D global delta slice for dense or sparse motion."""

    sparse_joint_indices = get_sparse_joint_indices(cfg)
    if sparse_joint_indices:
        start = _JOINT_TRANSFORM_DIM * len(sparse_joint_indices)
        return start, start + _JOINT_TRANSFORM_DIM
    return int(cfg.FLOW.GLOBAL_FEATURE_START), int(cfg.FLOW.GLOBAL_FEATURE_END)


def mask_it(mask, cond, replace_token):
    # mask: B bool
    # cond: B x ... x D
    # replace_token: D
    expand_dims = [1] * (len(cond.shape) - len(mask.shape))
    mask = mask.view(*mask.shape, *expand_dims).float()  # B x ... x 1

    expand_dims = [1] * (len(cond.shape) - 1)
    mask_token = replace_token.view(*expand_dims, -1)  # 1 x ... x D
    cond = cond * (1.0 - mask) + mask * mask_token
    return cond


class GlobalLocalOutputHead(nn.Module):
    """Decode a shared feature sequence with separate local/global branches.

    The global representation occupies a contiguous slice in the original
    feature vector, while the local representation is the concatenation of
    everything before and after that slice.  Both directional gates are
    always evaluated.  A constant activity mask selects the experiment's
    fusion topology, which keeps the parameter set and DDP graph identical
    for all four topology ablations.
    """

    MODES = {
        "no_fusion",
        "local_to_global",
        "global_to_local",
        "bidirectional",
    }

    def __init__(
        self,
        latent_dim,
        output_dim,
        global_start,
        global_end,
        mode,
        dropout=0.1,
        gate_init=-4.0,
        stop_gradient=True,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(
                f"Unsupported global/local branch mode {mode!r}; "
                f"expected one of {sorted(self.MODES)}."
            )
        if not 0 <= global_start < global_end <= output_dim:
            raise ValueError(
                "The global output slice must satisfy "
                f"0 <= start < end <= {output_dim}, got [{global_start}, {global_end})."
            )

        self.mode = mode
        self.output_dim = int(output_dim)
        self.global_start = int(global_start)
        self.global_end = int(global_end)
        self.global_dim = self.global_end - self.global_start
        self.local_dim = self.output_dim - self.global_dim
        self.stop_gradient = bool(stop_gradient)

        def make_adapter():
            return nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(latent_dim),
            )

        self.local_adapter = make_adapter()
        self.global_adapter = make_adapter()
        self.local_output = nn.Linear(latent_dim, self.local_dim)
        self.global_output = nn.Linear(latent_dim, self.global_dim)

        # Per-channel gates can select which branch features to exchange.  A
        # sigmoid(-4) starts at ~0.018, so the initial model is close to the
        # no-fusion variant without blocking gradients to an active gate.
        self.local_to_global_gate = nn.Parameter(torch.full((latent_dim,), float(gate_init)))
        self.global_to_local_gate = nn.Parameter(torch.full((latent_dim,), float(gate_init)))

        local_to_global_active = mode in {"local_to_global", "bidirectional"}
        global_to_local_active = mode in {"global_to_local", "bidirectional"}
        self.register_buffer(
            "local_to_global_active",
            torch.tensor(float(local_to_global_active)),
            persistent=False,
        )
        self.register_buffer(
            "global_to_local_active",
            torch.tensor(float(global_to_local_active)),
            persistent=False,
        )

    def _source_feature(self, feature):
        return feature.detach() if self.stop_gradient else feature

    def fusion_gate_values(self):
        """Return effective mean gate strengths for lightweight monitoring."""
        return {
            "local_to_global": self.local_to_global_active
            * torch.sigmoid(self.local_to_global_gate).mean(),
            "global_to_local": self.global_to_local_active
            * torch.sigmoid(self.global_to_local_gate).mean(),
        }

    def forward(self, shared_feature):
        local_base = self.local_adapter(shared_feature)
        global_base = self.global_adapter(shared_feature)

        local_to_global = (
            self.local_to_global_active
            * torch.sigmoid(self.local_to_global_gate)
            * self._source_feature(local_base)
        )
        global_to_local = (
            self.global_to_local_active
            * torch.sigmoid(self.global_to_local_gate)
            * self._source_feature(global_base)
        )

        global_feature = global_base + local_to_global
        local_feature = local_base + global_to_local
        local_output = self.local_output(local_feature)
        global_output = self.global_output(global_feature)

        # Local is non-contiguous in v4_beta: [0:start] + [end:output_dim].
        return torch.cat(
            (
                local_output[..., : self.global_start],
                global_output,
                local_output[..., self.global_start :],
            ),
            dim=-1,
        )


class UniEgoMotion(nn.Module):
    def __init__(
        self,
        cfg,
        dropout=0.1,
    ):
        super().__init__()

        self.cfg = cfg

        latent_dim = 768
        ff_size = 768 * 2
        num_layers = 12
        num_heads = 12

        self.img_feat_type = cfg.DATA.IMG_FEAT_TYPE
        self.cond_img_feat = cfg.DATA.COND_IMG_FEAT
        self.cond_betas = cfg.DATA.COND_BETAS
        self.encoder_tsfm = cfg.MODEL.ENCODER_TSFM
        self.finetune_type = cfg.MODEL.FINETUNE_TYPE
        self.output_branch_mode = str(getattr(cfg.MODEL, "OUTPUT_BRANCH_MODE", "single")).lower()
        self.cond_task = bool(getattr(cfg.MODEL, "COND_TASK", False))
        motion_expert_cfg = getattr(cfg.MODEL, "MOTION_EXPERT", None)
        self.motion_expert_enabled = bool(
            motion_expert_cfg is not None and getattr(motion_expert_cfg, "ENABLED", False)
        )
        self.motion_expert_cfg = motion_expert_cfg
        if self.motion_expert_enabled and cfg.MODEL.LEARN_TRAJ:
            raise ValueError("MOTION_EXPERT is a K12 motion model and cannot be combined with LEARN_TRAJ.")

        self.generative_type = str(getattr(cfg.MODEL, "GENERATIVE_TYPE", "diffusion")).lower()
        flow_types = {"flow", "flow_matching", "flow-matching"}
        diffusion_types = {"diffusion", "gaussian_diffusion", "ddpm"}
        if self.generative_type not in flow_types | diffusion_types:
            raise ValueError(f"Unsupported MODEL.GENERATIVE_TYPE: {self.generative_type}")
        self.is_flow_matching = self.generative_type in flow_types

        if self.finetune_type is not None:
            assert self.finetune_type in ["gen", "fore", "recon"]
            logger.warning(f"Using finetune type {self.finetune_type}.")

        self.repre_type = cfg.DATA.REPRE_TYPE
        if self.repre_type not in _MOTION_FEATURE_DIMS:
            raise ValueError(
                f"Unsupported DATA.REPRE_TYPE {self.repre_type!r}; "
                f"expected one of {sorted(_MOTION_FEATURE_DIMS)}."
            )
        traj_dim = {"v1_beta": 9, "v4_beta": 18, "v5_beta": 18}[self.repre_type]
        self.latent_dim = latent_dim

        if cfg.MODEL.LEARN_TRAJ:
            self.input_feats = traj_dim
            self.sparse_joint_indices = ()
            self.global_feature_start = int(cfg.FLOW.GLOBAL_FEATURE_START)
            self.global_feature_end = int(cfg.FLOW.GLOBAL_FEATURE_END)
            logger.warning("LEARNING TRAJ... USING SMALLER MODEL.")
            latent_dim = 512
            ff_size = 512 * 2
            num_layers = 8
            num_heads = 8
        else:
            self.input_feats = get_motion_feature_dim(cfg)
            self.sparse_joint_indices = get_sparse_joint_indices(cfg)
            self.global_feature_start, self.global_feature_end = get_global_feature_slice(cfg)
            if self.sparse_joint_indices:
                logger.warning(
                    "Using sparse SMPL22 body features: "
                    f"indices={list(self.sparse_joint_indices)}, input_feats={self.input_feats}, "
                    f"global=[{self.global_feature_start}, {self.global_feature_end})."
                )

        if self.motion_expert_enabled:
            # The 400M expert is deliberately scoped to the K12 experiment.
            # Keeping this restriction explicit prevents accidentally comparing
            # differently parameterized dense/K3/K6/K10 models under one name.
            if len(self.sparse_joint_indices) != 12 or self.input_feats != 153:
                raise ValueError(
                    "MOTION_EXPERT currently requires the K12 sparse v4_beta layout "
                    "(12 joints and 153 motion features)."
                )
            logger.warning(
                "Using the 400M K12 Motion Expert: "
                f"{getattr(motion_expert_cfg, 'NUM_ROUTED_EXPERTS', 11)} routed experts, "
                f"top-{motion_expert_cfg.TOP_K} routing, "
                f"chunk={getattr(motion_expert_cfg, 'CHUNK_SIZE', 4)}."
            )

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.img_feat_dim = 768
        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            self.img_feat_dim = 1024
        if self.img_feat_type in ["egovideo"]:
            self.img_feat_dim = 512

        # self.cond_mask_prob = cond_mask_prob
        self.cond_mask_prob = {
            "traj": 0.5,
            "clip": 0.1,
            "subseq_frames": 0.5,
        }

        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.embed_traj_cond = nn.Linear(traj_dim, self.latent_dim)
        self.embed_clip_cond = nn.Linear(self.img_feat_dim, self.latent_dim)
        self.embed_text_cond = nn.Linear(768, self.latent_dim)  # Not used
        self.embed_text_cond.requires_grad_(False)
        if self.cond_task:
            num_tasks = int(getattr(cfg.MODEL, "NUM_TASKS", 3))
            if num_tasks != 3:
                raise ValueError(
                    "Explicit task conditioning uses the fixed recon/fore/gen mapping and requires NUM_TASKS=3."
                )
            self.embed_task_cond = nn.Embedding(num_tasks, self.latent_dim)
            logger.warning("Using explicit recon/fore/gen task embeddings.")

        if self.cond_betas:
            self.embed_betas = nn.Linear(10, self.latent_dim)
            logger.warning("Using betas.")

        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "clip": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "text": nn.Parameter(torch.randn(self.latent_dim) * 0.05),  # Not used
            }
        )

        self.zero_mask_token = cfg.MODEL.ZERO_MASK_TOKEN
        if self.zero_mask_token:
            logger.warning("Using zero mask token.")
            del self.mask_tokens
            self.mask_tokens = nn.ParameterDict(
                {
                    "traj": nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False),
                    "clip": nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False),
                    "text": nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False),  # Not used
                }
            )

        # Keep these keys for old checkpoint compatibility, but exclude the
        # unused text condition parameters from DDP gradient synchronization.
        self.mask_tokens["text"].requires_grad_(False)

        self._last_moe_aux_loss = None
        if self.encoder_tsfm is not None:
            assert self.encoder_tsfm in ["add"]
            logger.warning("Using encoder.")
            block_cls = MoEEncoderBlock if self.motion_expert_enabled else EncoderBlock
            block_kwargs = {}
            if self.motion_expert_enabled:
                block_kwargs = {
                    "num_experts": int(getattr(motion_expert_cfg, "NUM_ROUTED_EXPERTS", 11)),
                    "top_k": int(motion_expert_cfg.TOP_K),
                    "router_jitter": float(getattr(motion_expert_cfg, "ROUTER_JITTER", 0.0)),
                    "shared_expert": bool(getattr(motion_expert_cfg, "SHARED_EXPERT", True)),
                    "chunk_size": int(getattr(motion_expert_cfg, "CHUNK_SIZE", 4)),
                    "router_conditioned": bool(getattr(motion_expert_cfg, "CONDITIONED_ROUTER", True)),
                    "routed_gate_init": float(getattr(motion_expert_cfg, "ROUTED_GATE_INIT", 0.05)),
                }
            self.tsfm = nn.ModuleList(
                [block_cls(self.latent_dim, self.num_heads, self.dropout, 2, **block_kwargs) for _ in range(self.num_layers)]
            )
        else:
            block_cls = MoEDecoderBlock if self.motion_expert_enabled else DecoderBlock
            block_kwargs = {}
            if self.motion_expert_enabled:
                block_kwargs = {
                    "num_experts": int(getattr(motion_expert_cfg, "NUM_ROUTED_EXPERTS", 11)),
                    "top_k": int(motion_expert_cfg.TOP_K),
                    "router_jitter": float(getattr(motion_expert_cfg, "ROUTER_JITTER", 0.0)),
                    "shared_expert": bool(getattr(motion_expert_cfg, "SHARED_EXPERT", True)),
                    "chunk_size": int(getattr(motion_expert_cfg, "CHUNK_SIZE", 4)),
                    "router_conditioned": bool(getattr(motion_expert_cfg, "CONDITIONED_ROUTER", True)),
                    "routed_gate_init": float(getattr(motion_expert_cfg, "ROUTED_GATE_INIT", 0.05)),
                }
            self.tsfm = nn.ModuleList(
                [block_cls(self.latent_dim, self.num_heads, self.dropout, 2, **block_kwargs) for _ in range(self.num_layers)]
            )

        if self.output_branch_mode == "single":
            self.output_process = nn.Linear(self.latent_dim, self.input_feats)
            self.global_local_output = None
        else:
            if cfg.MODEL.LEARN_TRAJ:
                raise ValueError("Global/local output branches are not supported with MODEL.LEARN_TRAJ.")
            if self.repre_type not in _SPARSE_SUPPORTED_REPRESENTATIONS:
                raise ValueError(
                    "Global/local output branches require the v4_beta or v5_beta representation, "
                    f"got {self.repre_type!r} with {self.input_feats} features."
                )
            self.output_process = None
            self.global_local_output = GlobalLocalOutputHead(
                latent_dim=self.latent_dim,
                output_dim=self.input_feats,
                global_start=self.global_feature_start,
                global_end=self.global_feature_end,
                mode=self.output_branch_mode,
                dropout=self.dropout,
                gate_init=getattr(cfg.MODEL, "FUSION_GATE_INIT", -4.0),
                stop_gradient=getattr(cfg.MODEL, "FUSION_STOP_GRAD", True),
            )
            logger.warning(
                "Using Global/Local output branches: "
                f"mode={self.output_branch_mode}, "
                f"global=[{self.global_feature_start}, {self.global_feature_end}), "
                f"stop_gradient={getattr(cfg.MODEL, 'FUSION_STOP_GRAD', True)}."
            )

    def fusion_gate_values(self):
        if self.global_local_output is None:
            return {}
        return self.global_local_output.fusion_gate_values()

    def get_moe_auxiliary_loss(self):
        """Return the mean router load-balancing loss from the latest forward."""
        if not self.motion_expert_enabled or self._last_moe_aux_loss is None:
            return None
        return self._last_moe_aux_loss

    def initialize_motion_expert_from_dense(self, dense_state, noise_std=0.01):
        """Initialize the new MoE from a dense K12 model state.

        Attention/condition/output weights keep their exact names.  Dense FFN
        parameters are copied into each layer's independent Shared Expert;
        routed experts are then cloned from that shared branch with a small
        perturbation.  The method returns the number of copied tensors.
        """
        if not self.motion_expert_enabled:
            return 0
        own_state = self.state_dict()
        copied = 0
        with torch.no_grad():
            for key, target in own_state.items():
                if key.startswith("tsfm.") and ".ff.shared." in key:
                    dense_key = key.replace(".ff.shared.", ".ff.", 1)
                else:
                    dense_key = key
                source = dense_state.get(dense_key)
                if source is None or tuple(source.shape) != tuple(target.shape):
                    continue
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                copied += 1

            for block in self.tsfm:
                if hasattr(block.ff, "initialize_routed_from_shared"):
                    block.ff.initialize_routed_from_shared(noise_std=noise_std)
        logger.warning(f"Initialized Motion Expert from dense K12 state: copied {copied} tensors.")
        return copied

    def mask_cond_finetune(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D

        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        inp_shape = cond.shape
        assert cond_type in ["traj", "clip", "motion"]

        # For recon, do not mask anything
        if self.finetune_type == "recon":
            return cond

        # mask everything
        mask = torch.ones(*cond.shape[:2], device=cond.device)  # B x T

        if self.finetune_type == "gen":
            if cond_type == "traj":
                pass
            elif cond_type == "clip":
                mask[:, 0] = 0  # do not mask the first frame
            elif cond_type == "motion":
                pass

        # prediction based on past images and trajectories
        elif self.finetune_type in ["fore"]:

            avail = self.cfg.DATA.WINDOW // 4
            if cond_type != "motion":
                mask[:, :avail] = 0  # do not mask the first avail frames
            else:
                pass

        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond.shape == inp_shape
        return cond

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D
        if self.finetune_type is not None:
            return self.mask_cond_finetune(cond, cond_type, cond_mask)

        # for inference of recon, fore and gen, masks are already provided.
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        if not self.training:
            # technically, this should never happen because we always do inference with cond_mask.
            return cond
        inp_shape = cond.shape

        # During training, mask whole condition randomly based on probability.
        # To simulate generation, we additionally mask subsequent frames (after first frame) with some probability.
        # Overall, this will simulate recon and gen tasks with some probability.

        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob[cond_type]  # B
        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond_type in ["traj", "clip"]

        # Mask condition of subsequent frames using probability. For trajectory, we also mask the first frame.
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob["subseq_frames"]  # B
        if cond_type == "traj":
            cond = mask_it(mask, cond, self.mask_tokens[cond_type])
        else:
            subsequent_cond = mask_it(mask, cond[:, 1:], self.mask_tokens[cond_type])
            cond = torch.cat((cond[:, :1], subsequent_cond), axis=1)

        assert cond.shape == inp_shape
        return cond

    def pos_enc_and_process_img_feat(self, x, enc_imgs):
        if self.img_feat_type in ["dinov2_reg"]:
            # B x T x 5 x D
            for i in range(5):
                enc_imgs[:, :, i] = self.pos_enc(enc_imgs[:, :, i])
            return enc_imgs
        if self.img_feat_type in ["clip_all", "dinov2", "egovideo"]:
            enc_imgs = self.pos_enc(enc_imgs)
            return enc_imgs
        else:
            raise ValueError(f"img_feat_type is {self.img_feat_type}")

    def forward(self, x, timesteps, y, cond_scale=None, diffusion=None, return_hidden=False):
        """
        x_t: B x T x F
        timesteps: B
        y: dict
        """
        if cond_scale is not None:
            x_cond = self.forward(x, timesteps, y, return_hidden=return_hidden)  # conditional output
            uncond_y = {"valid_frames": y["valid_frames"]}
            if "task_id" in y:
                # Task identity is an instruction, not a modality to drop for CFG.
                uncond_y["task_id"] = y["task_id"]
            x_uncond = self.forward(x, timesteps, uncond_y, return_hidden=return_hidden)  # unconditional output

            # Both predicted x_start (diffusion) and velocity (flow matching)
            # support classifier-free guidance by linear output interpolation.
            if return_hidden:
                x_cond, h_cond = x_cond
                x_uncond, h_uncond = x_uncond
                x_scaled = x_uncond + (x_cond - x_uncond) * cond_scale
                h_scaled = h_uncond + (h_cond - h_uncond) * cond_scale
                return x_scaled, h_scaled
            return x_uncond + (x_cond - x_uncond) * cond_scale

        B, T, F = x.shape
        x = self.input_process(x)  # B x T x D
        x = self.pos_enc(x)  # B x T x D

        traj_mask = y["traj_mask"] if "traj_mask" in y else None
        img_mask = y["img_mask"] if "img_mask" in y else None

        enc_time = self.embed_timestep(timesteps)  # B x D
        if self.cond_task:
            task_id = y.get("task_id")
            if task_id is None:
                # Raw validation batches are reconstruction batches.
                task_id = torch.zeros(B, device=x.device, dtype=torch.long)
            else:
                task_id = torch.as_tensor(task_id, device=x.device, dtype=torch.long)
                if task_id.ndim == 0:
                    task_id = task_id.expand(B)
                if task_id.shape != (B,):
                    raise ValueError(f"task_id must have shape ({B},), got {tuple(task_id.shape)}.")
            enc_time = enc_time + self.embed_task_cond(task_id)

        pose_router_cond = x
        if "traj" in y:  # B x T x F
            enc_traj = self.embed_traj_cond(y["traj"])  # B x T x D
            enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)
            x = x + enc_traj
        else:
            x = x + self.mask_tokens["traj"]

        if "betas" in y:
            assert self.cond_betas
            enc_betas = self.embed_betas(y["betas"])  # B x D
            x = x + enc_betas[:, None, :]  # B x T x D

        all_cond = [enc_time[:, None]]  # B x 1 x D
        all_cond_mask = [torch.ones(B, 1, dtype=torch.long, device=x.device)]  # B x 1
        if "img_embs" in y:
            enc_imgs = self.embed_clip_cond(y["img_embs"])
            enc_imgs = self.mask_cond(enc_imgs, "clip", img_mask)
            enc_imgs = self.pos_enc_and_process_img_feat(x, enc_imgs)

            enc_img_mask = y["valid_img_embs"]  # B x C

            # Keep a frame-aligned egovideo condition for the chunk router
            # before flattening DINO register tokens into the attention context.
            router_ego_cond = enc_imgs.mean(dim=2) if enc_imgs.ndim == 4 else enc_imgs
            if router_ego_cond.shape[1] != T:
                router_ego_cond = router_ego_cond.mean(dim=1, keepdim=True).expand(-1, T, -1)

            if self.img_feat_type == "dinov2_reg":
                enc_imgs = enc_imgs.flatten(1, 2)  # B x T x 5 x D to B x T5 x D
                enc_img_mask = enc_img_mask[:, :, None].repeat(1, 1, 5).flatten(1, 2)  # B x T to B x T5

            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)
        else:  # if self.cond_img_feat:  # Model was trained with img feats. Use mask tokens here.
            enc_imgs = self.mask_tokens["clip"].view(1, 1, -1).repeat(B, T, 1)
            enc_imgs = self.pos_enc_and_process_img_feat(x, enc_imgs)
            enc_img_mask = y["valid_frames"]
            router_ego_cond = enc_imgs

            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)

        x = torch.cat((enc_time[:, None], x), axis=1)  # B x (T+1) x D
        mask = y["valid_frames"]  # B x T where valid are 1
        mask = torch.cat((torch.ones((B, 1), device=mask.device, dtype=mask.dtype), mask), dim=1)  # B x (T+1)
        mask = mask[:, None, None, :]  # B x 1 x 1 x (T+1)

        context_mask = torch.cat(all_cond_mask, dim=1)  # B x C
        context_mask = context_mask[:, None, None, :]  # B x 1 x 1 x C
        context = torch.cat(all_cond, dim=1)  # B x C x D

        router_conditions = None
        if self.motion_expert_enabled:
            router_conditions = {
                "pose": pose_router_cond,
                "ego": router_ego_cond,
                "timestep": enc_time,
                "valid_frames": y["valid_frames"].bool(),
            }

        if self.encoder_tsfm:
            if self.encoder_tsfm == "add":
                x = x + context
            else:
                raise ValueError(f"encoder_tsfm is {self.encoder_tsfm}")

            for enc in self.tsfm:
                if self.motion_expert_enabled:
                    x = enc(x=x, mask=mask, router_conditions=router_conditions)
                else:
                    x = enc(x=x, mask=mask)
        else:
            for dec in self.tsfm:
                decoder_kwargs = {
                    "x": x,
                    "context": context,
                    "mask": mask,
                    "context_mask": context_mask,
                }
                if self.motion_expert_enabled:
                    decoder_kwargs["router_conditions"] = router_conditions
                x = dec(**decoder_kwargs)

        if self.motion_expert_enabled:
            aux_losses = [
                block.ff.last_aux_loss
                for block in self.tsfm
                if hasattr(block.ff, "last_aux_loss") and block.ff.last_aux_loss is not None
            ]
            self._last_moe_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        hidden = x[:, 1 : 1 + T].contiguous()
        x = hidden
        if self.global_local_output is None:
            x = self.output_process(x)  # B x T x (J x F)
        else:
            x = self.global_local_output(x)
        x = x.view(B, T, F)

        if "repaint_mask" in y:
            if self.is_flow_matching:
                raise ValueError(
                    "Flow Matching cannot apply repaint inside the velocity model; "
                    "known values must be constrained along the interpolation path in the sampler."
                )
            assert self.cfg.MODEL.PREDICT_XSTART
            repaint_mask = y["repaint_mask"]
            repaint_value = y["repaint_value"]
            x = x * (1 - repaint_mask) + repaint_mask * repaint_value

        if return_hidden:
            return x, hidden
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe[None])  # 1 x Tmax x D

    def forward(self, x):
        # x is B x T x D
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pos_enc, min_period=4e-3, max_period=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = pos_enc  # .pe is 1 x Tmax x D

        if not 0 < min_period < max_period:
            raise ValueError(f"Expected 0 < min_period < max_period, got {min_period} and {max_period}.")
        half_dim = self.latent_dim // 2
        periods = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), half_dim))
        angular_frequencies = (2.0 * math.pi) / periods
        # Non-persistent keeps state_dict keys compatible with existing
        # diffusion checkpoints while still following module device moves.
        self.register_buffer("continuous_angular_frequencies", angular_frequencies, persistent=False)

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # timesteps is (B,)
        if timesteps.ndim != 1:
            raise ValueError(f"Expected one timestep per batch item, got shape {tuple(timesteps.shape)}.")

        if timesteps.dtype.is_floating_point:
            # OpenPI-style continuous sin/cos features. Flow time is expected
            # in [0, 1], with t=0 at data and t=1 at Gaussian noise.
            continuous_time = timesteps.to(dtype=torch.float32)
            frequencies = self.continuous_angular_frequencies.to(dtype=torch.float32)
            angles = continuous_time[:, None] * frequencies[None, :]
            t = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
            if t.shape[-1] < self.latent_dim:
                t = torch.cat((t, torch.zeros_like(t[:, :1])), dim=-1)
            t = t.to(dtype=self.time_embed[0].weight.dtype)
        else:
            # Preserve the original discrete diffusion embedding exactly.
            t = self.pos_enc.pe[0, timesteps.long()]  # B x D
        return self.time_embed(t)  # B x D


if __name__ == "__main__":
    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.MODEL = lambda: None
    cfg.DATA.COND_IMG_FEAT = True
    cfg.DATA.COND_BETAS = True
    cfg.DATA.IMG_FEAT_TYPE = "clip_all"
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.MODEL.LEARN_TRAJ = False

    model = UniEgoMotion(cfg)
    x = torch.randn(2, 10, 224)
    timesteps = torch.tensor([342, 21])

    y = {
        "traj": torch.randn(2, 10, 9),
        "img_embs": torch.randn(2, 10, 768),
        "valid_frames": torch.ones(2, 10).long(),
        "valid_img_embs": torch.ones(2, 10).long(),
        "betas": torch.randn(2, 10),
    }
    out = model(x, timesteps, y)

    import IPython

    IPython.embed()
