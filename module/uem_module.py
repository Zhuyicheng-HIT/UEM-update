import math
import os
import pytorch_lightning as pl
import torch
import torch.distributed
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR, StepLR

from model.uniegomotion import UniEgoMotion
from model.motion_lstm import Motion_LSTM
from model.motion_unet import Motion_Unet

from module.utils import cfg_to_dict, create_gaussian_diffusion
from module.task_sampler import ExplicitTaskSchedule
from mydiffusion.flow_matching import FlowMatching
from mydiffusion.resample import create_named_schedule_sampler
from mydiffusion.gaussian_diffusion import sum_flat
from utils.task_conditioning import TASKS, apply_task_conditioning


class UEM_Module(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        if not cfg.TRAIN.ONLY_VALIDATE:
            self.save_hyperparameters(cfg_to_dict(cfg))
        self.cfg = cfg
        self.window = self.cfg.DATA.WINDOW
        self.model_name = cfg.MODEL.MODEL_NAME
        self.learn_traj = cfg.MODEL.LEARN_TRAJ
        self.task_sampler_enabled = bool(getattr(cfg.TRAIN.TASK_SAMPLER, "ENABLED", False))
        self.task_schedule = None
        self.forecast_prefix = int(getattr(cfg.TRAIN.TASK_SAMPLER, "FORECAST_PREFIX", self.window // 4))
        self._adaptive_val_sums = None
        self._adaptive_val_counts = None
        if self.learn_traj:
            assert not self.cfg.DATA.COND_TRAJ

        self.is_lstm = False
        if self.model_name == "uem":
            self.model = UniEgoMotion(self.cfg)
        elif self.model_name == "lstm":
            self.model = Motion_LSTM(self.cfg)
            self.is_lstm = True
        elif self.model_name == "unet":
            self.model = Motion_Unet(self.cfg)
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        self._initialize_motion_expert()

        if self.is_lstm:
            if self.task_sampler_enabled:
                raise ValueError("Explicit E13--E16 task sampling is currently supported only by the flow UEM model.")
            return

        self.generative_type = getattr(cfg.MODEL, "GENERATIVE_TYPE", "diffusion").lower()
        if self.generative_type in {"flow", "flow_matching"}:
            flow_cfg = cfg.FLOW
            self.flow = FlowMatching(
                num_steps=flow_cfg.NUM_STEPS,
                solver=flow_cfg.SOLVER,
                beta_alpha=flow_cfg.BETA_ALPHA,
                beta_beta=flow_cfg.BETA_BETA,
                t_min=flow_cfg.T_MIN,
                prediction_type=flow_cfg.PREDICTION_TYPE,
                global_weight=flow_cfg.GLOBAL_WEIGHT,
                global_feature_start=flow_cfg.GLOBAL_FEATURE_START,
                global_feature_end=flow_cfg.GLOBAL_FEATURE_END,
                global_rotation_weight=getattr(flow_cfg, "GLOBAL_ROT_WEIGHT", None),
                global_translation_weight=getattr(flow_cfg, "GLOBAL_TRANS_WEIGHT", None),
            )
            self.diffusion = None
            self.schedule_sampler_type = None
            self.schedule_sampler = None
        elif self.generative_type == "diffusion":
            self.diffusion = create_gaussian_diffusion(cfg)
            self.flow = None
            self.schedule_sampler_type = "uniform"
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        else:
            raise ValueError(
                f"Unknown generative type {self.generative_type!r}; expected 'diffusion' or 'flow'."
            )
        if self.task_sampler_enabled:
            if self.generative_type not in {"flow", "flow_matching"}:
                raise ValueError("Explicit task target masks require MODEL.GENERATIVE_TYPE=flow.")
            self.task_schedule = ExplicitTaskSchedule.from_config(cfg)
            logger.warning(
                "Using explicit task training: "
                f"mode={self.task_schedule.mode}, total_steps={self.task_schedule.total_steps}, "
                f"forecast_prefix={self.forecast_prefix}."
            )
        self.last_iters = []

    def _initialize_motion_expert(self):
        """Load dense K12 weights into Shared Experts before DDP wrapping."""
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        if self.is_lstm or expert_cfg is None or not bool(getattr(expert_cfg, "ENABLED", False)):
            return
        init_path = getattr(expert_cfg, "INIT_DENSE_CKPT_PATH", None)
        if not init_path:
            logger.warning("Motion Expert has no INIT_DENSE_CKPT_PATH; using random shared/routed weights.")
            return
        if not os.path.isfile(init_path):
            raise FileNotFoundError(
                f"Motion Expert dense initialization state does not exist: {init_path}. "
                "Create it from the existing K12 checkpoint before training."
            )
        logger.warning(f"Loading dense K12 initialization state from {init_path}")
        state = torch.load(init_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        # Accept either a raw model state or a Lightning state with model.* keys.
        dense_state = {}
        for key, value in state.items():
            dense_state[key[6:] if key.startswith("model.") else key] = value
        copied = self.model.initialize_motion_expert_from_dense(
            dense_state,
            noise_std=float(getattr(expert_cfg, "ROUTED_INIT_NOISE", 0.01)),
        )
        if copied == 0:
            raise RuntimeError("Dense initialization state did not match any Motion Expert tensors.")

    def _set_motion_expert_train_phase(self, epoch):
        """Progressively expose routed, shared and backbone parameters."""
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        if self.is_lstm or expert_cfg is None or not bool(getattr(expert_cfg, "ENABLED", False)):
            return
        warmup_epochs = int(getattr(expert_cfg, "ROUTED_WARMUP_EPOCHS", 10))
        shared_epoch = int(getattr(expert_cfg, "SHARED_UNFREEZE_EPOCH", 30))
        for name, parameter in self.model.named_parameters():
            if ".ff." not in name:
                # Keep the original input/attention/condition backbone frozen
                # until the routed branch has learned a useful partition.
                parameter.requires_grad = epoch >= shared_epoch
                continue
            is_router_or_routed = ".ff.router" in name or ".ff.routed_experts." in name or ".ff.routed_gate" in name
            is_shared = ".ff.shared." in name
            if is_router_or_routed:
                parameter.requires_grad = True
            elif is_shared:
                parameter.requires_grad = epoch >= shared_epoch
            else:
                parameter.requires_grad = epoch >= shared_epoch
        if epoch == 0:
            logger.warning(
                "Motion Expert curriculum: routed experts/router train first; "
                f"shared/backbone unfreeze at epoch {shared_epoch}."
            )

    def on_train_epoch_start(self):
        self._set_motion_expert_train_phase(int(self.current_epoch))

    def _batch_for_task(self, batch, task):
        conditioned_batch = dict(batch)
        conditioned_batch["y"] = apply_task_conditioning(
            batch["y"],
            task,
            forecast_prefix=self.forecast_prefix,
        )
        return conditioned_batch

    def configure_optimizers(self):
        fused = getattr(self.cfg.TRAIN, "FUSED_ADAMW", False) and torch.cuda.is_available()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            fused=fused,
        )

        scheduler_name = getattr(self.cfg.TRAIN, "SCHEDULER", "step").lower()
        if scheduler_name == "cosine_warmup":
            warmup_epochs = max(0, getattr(self.cfg.TRAIN, "WARMUP_EPOCHS", 0))
            total_epochs = getattr(self.cfg.TRAIN, "SCHEDULER_TOTAL_EPOCHS", 0)
            if total_epochs <= 0:
                total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            min_lr_ratio = getattr(self.cfg.TRAIN, "MIN_LR_RATIO", 0.1)

            def lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = StepLR(optimizer, max(1, self.cfg.TRAIN.NUM_EPOCHS - 50), gamma=0.1)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        ]

    def _add_motion_expert_auxiliary_loss(self, loss, mode):
        """Add router load-balancing regularization for the optional MoE expert."""
        if not hasattr(self.model, "get_moe_auxiliary_loss"):
            return loss
        aux_loss = self.model.get_moe_auxiliary_loss()
        if aux_loss is None:
            return loss
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        aux_weight = float(getattr(expert_cfg, "LOAD_BALANCE_WEIGHT", 0.0))
        if aux_weight <= 0:
            return loss
        self.log(
            f"{mode}/moe_aux_loss",
            aux_loss,
            on_step=mode == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.DATA.BATCH_SIZE,
        )
        return loss + aux_weight * aux_loss

    def on_train_start(self):
        self._set_motion_expert_train_phase(int(self.current_epoch))
        if self.cfg.MODEL.CKPT_PATH is None:
            return
        if self.cfg.TRAIN.USE_CKPT_LR:
            logger.warning("Using LR from checkpoint.")
            return
        logger.warning("Discarding LR of optimizer dict and using config LR.")
        for g in self.optimizers().param_groups:
            g["lr"] = self.cfg.TRAIN.LR
        for g in self.optimizers().param_groups:
            g["weight_decay"] = self.cfg.TRAIN.WEIGHT_DECAY

    def training_step(self, batch, batch_idx, mode="train"):
        active_task = None
        if mode == "train" and self.task_schedule is not None:
            active_task = self.task_schedule.task_for_step(int(self.global_step))
            batch = self._batch_for_task(batch, active_task)
            self.log(
                "train/task_id",
                float(TASKS.index(active_task)),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                batch_size=batch["misc"]["motion"].shape[0],
            )

        if self.is_lstm:
            x = batch["misc"]["traj"] if self.learn_traj else batch["misc"]["motion"]
            y = batch["y"]
            pred_x = self.model(x.clone(), y)
            target = x

            mask = y["valid_frames"]
            mask = mask.view(list(mask.shape) + [1] * (x.ndim - mask.ndim))
            mask = mask.expand_as(target)
            loss_here = sum_flat(((target - pred_x) * mask) ** 2)
            denom = sum_flat(mask)
            loss = (loss_here / denom).mean()
            self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
            return loss

        x = batch["misc"]["traj"] if self.learn_traj else batch["misc"]["motion"]
        if self.generative_type in {"flow", "flow_matching"}:
            losses = self.flow.training_losses(self.model, x, model_kwargs={"y": batch["y"]})
            loss = losses["loss"].mean()
            loss = self._add_motion_expert_auxiliary_loss(loss, mode)
            self.log(
                f"{mode}/loss",
                loss,
                on_step=mode == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
            for group_name in ("local_mse", "global_mse"):
                if group_name in losses:
                    self.log(
                        f"{mode}/{group_name}",
                        losses[group_name].mean(),
                        on_step=mode == "train",
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=x.shape[0],
                    )
            gate_values = self.model.fusion_gate_values() if hasattr(self.model, "fusion_gate_values") else {}
            for direction, gate in gate_values.items():
                self.log(
                    f"{mode}/fusion_gate_{direction}",
                    gate,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )
            if active_task is not None:
                self.log(
                    f"train_task/{active_task}_loss",
                    loss,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.shape[0],
                )
                for task_index, task in enumerate(TASKS):
                    probability = self.task_schedule.probabilities_for_step(int(self.global_step))[task_index]
                    self.log(
                        f"train_task/prob_{task}",
                        probability,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=False,
                        batch_size=x.shape[0],
                    )
            return loss

        t, weights = self.schedule_sampler.sample(x.shape[0], self.device)
        losses = self.diffusion.training_losses(self.model, x, t, model_kwargs={"y": batch["y"]})
        if mode == "train" and self.schedule_sampler_type != "uniform":
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        loss = self._add_motion_expert_auxiliary_loss(loss, mode)

        # To monitor diffusion step wise loss
        assert losses["loss"].shape[0] == x.shape[0]
        diff_steps, bin_size = self.cfg.MODEL.DIFFUSION_STEPS, self.cfg.MODEL.DIFFUSION_STEPS // 10
        for idx, i in enumerate(range(0, diff_steps, bin_size)):
            mask = (t >= i) & (t < i + bin_size)
            if mask.any():
                range_loss = losses["loss"][mask].mean()
                self.log(
                    f"{mode}_loss/{idx}",
                    range_loss,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=mask.sum(),
                )

        for k, v in losses.items():
            self.log(f"{mode}/{k}", v.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.task_schedule is not None and self.task_schedule.mode == "adaptive":
            return self._adaptive_validation_step(batch, batch_idx)
        return self.training_step(batch, batch_idx, mode="val")

    def on_validation_epoch_start(self):
        if self.task_schedule is None or self.task_schedule.mode != "adaptive":
            return
        self._adaptive_val_sums = torch.zeros(len(TASKS), device=self.device, dtype=torch.float64)
        self._adaptive_val_counts = torch.zeros(len(TASKS), device=self.device, dtype=torch.float64)

    def _adaptive_validation_step(self, batch, batch_idx):
        max_batches = int(getattr(self.cfg.TRAIN.TASK_SAMPLER, "ADAPTIVE_VAL_MAX_BATCHES", 0))
        if max_batches > 0 and batch_idx >= max_batches:
            return None

        x = batch["misc"]["traj"] if self.learn_traj else batch["misc"]["motion"]
        batch_size = x.shape[0]
        validation_t = float(getattr(self.cfg.TRAIN.TASK_SAMPLER, "ADAPTIVE_VAL_T", 0.75))
        t = torch.full((batch_size,), validation_t, device=x.device, dtype=torch.float32)
        generator = torch.Generator(device=x.device)
        validation_seed = int(getattr(self.cfg.TRAIN.TASK_SAMPLER, "ADAPTIVE_VAL_SEED", 6200))
        generator.manual_seed(validation_seed + 100003 * int(self.global_rank) + batch_idx)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

        task_losses = []
        for task_index, task in enumerate(TASKS):
            conditioned_batch = self._batch_for_task(batch, task)
            terms = self.flow.training_losses(
                self.model,
                x,
                model_kwargs={"y": conditioned_batch["y"]},
                noise=noise,
                t=t,
            )
            task_loss = terms["loss"]
            self._adaptive_val_sums[task_index] += task_loss.detach().double().sum()
            self._adaptive_val_counts[task_index] += task_loss.numel()
            task_losses.append(task_loss.mean())
        return torch.stack(task_losses).mean()

    def on_validation_epoch_end(self):
        if self.task_schedule is None or self.task_schedule.mode != "adaptive":
            return
        if self._adaptive_val_sums is None or not bool((self._adaptive_val_counts > 0).any()):
            return

        sums = self._adaptive_val_sums.clone()
        counts = self._adaptive_val_counts.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        scores = sums / counts.clamp_min(1.0)

        for task_index, task in enumerate(TASKS):
            self.log(
                f"val_task/{task}_flow_loss",
                scores[task_index].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        changed = self.task_schedule.update_from_scores(scores.tolist(), step=int(self.global_step))
        for task_index, task in enumerate(TASKS):
            self.log(
                f"val_task/prob_{task}",
                self.task_schedule.current_probabilities[task_index],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        if changed and int(self.global_rank) == 0:
            logger.warning(
                "Updated adaptive task replay probabilities at step "
                f"{int(self.global_step)}: "
                + ", ".join(
                    f"{task}={probability:.3f}"
                    for task, probability in zip(TASKS, self.task_schedule.current_probabilities)
                )
            )

    def on_save_checkpoint(self, checkpoint):
        if self.task_schedule is not None:
            checkpoint["explicit_task_schedule"] = self.task_schedule.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.task_schedule is not None and "explicit_task_schedule" in checkpoint:
            self.task_schedule.load_state_dict(checkpoint["explicit_task_schedule"])

    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False, return_one_step_hidden=False):
        if self.is_lstm:
            x = torch.zeros(B, self.window, self.model.input_feats, device=self.device)
            x = self.model(x, y)
            return x

        # B = y["traj"].shape[0]
        for k, v in y.items():
            assert len(v) == B, f"y[{k}] has batch size {len(v)} but expected {B}"
        if self.generative_type in {"flow", "flow_matching"}:
            return self.flow.sample_loop(
                self.model,
                (B, self.window, self.model.input_feats),
                model_kwargs={"y": y, "cond_scale": cond_scale},
                noise=None,
                progress=False,
                return_all_pred_xstart=return_all_pred_xstart,
                return_one_step_hidden=return_one_step_hidden,
            )

        if return_one_step_hidden:
            raise ValueError("one-step hidden export is only implemented for Flow Matching models.")

        x = self.diffusion.p_sample_loop(
            self.model,
            (B, self.window, self.model.input_feats),
            model_kwargs={"y": y, "cond_scale": cond_scale, "diffusion": self.diffusion},
            clip_denoised=False,
            noise=None,
            progress=False,
            # skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            # init_image=None,
            # dump_steps=None,
            # const_noise=False,
            return_all_pred_xstart=return_all_pred_xstart,
        )
        if return_all_pred_xstart:
            return x[0], x[1]  # final sample, all predicted xstart
        return x


class UEM_Module_TwoStage(pl.LightningModule):
    """For two stage baseline."""

    def __init__(self, cfg):
        super().__init__()

        traj_ckpt_path = cfg.MODEL.TRAJ_CKPT_PATH
        traj_exp_path = os.path.dirname(traj_ckpt_path)
        traj_cfg = cfg.clone()
        traj_cfg.defrost()
        traj_cfg.merge_from_file(f"{traj_exp_path}/hparams.yaml")
        traj_cfg.freeze()

        motion_ckpt_path = cfg.MODEL.MOTION_CKPT_PATH
        motion_exp_path = os.path.dirname(motion_ckpt_path)
        motion_cfg = cfg.clone()
        motion_cfg.defrost()
        motion_cfg.merge_from_file(f"{motion_exp_path}/hparams.yaml")
        motion_cfg.freeze()

        logger.warning(f"Loading from {traj_ckpt_path}")
        self.traj_module = UEM_Module.load_from_checkpoint(traj_ckpt_path, cfg=traj_cfg, map_location="cpu")

        logger.warning(f"Loading from {motion_ckpt_path}")
        self.motion_module = UEM_Module.load_from_checkpoint(motion_ckpt_path, cfg=motion_cfg, map_location="cpu")

        self.motion_cfg = motion_cfg
        self.traj_cfg = traj_cfg

    def sample_traj(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        return self.traj_module.sample(y, B, cond_scale, return_all_pred_xstart)

    def sample_motion(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        return self.motion_module.sample(y, B, cond_scale, return_all_pred_xstart)

    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        assert self.motion_cfg.DATA.REPRE_TYPE == self.traj_cfg.DATA.REPRE_TYPE

        # prep for trajectory prediction
        traj = y.pop("traj", None)  # B x T x D
        traj_mask = y.pop("traj_mask", None)  # B x T

        assert traj is not None
        if traj_mask is None:
            traj_mask = torch.zeros_like(traj[:, :, 0])

        y["repaint_mask"] = 1 - traj_mask[..., None].expand_as(traj)
        y["repaint_value"] = traj
        traj_pred = self.sample_traj(y, B, cond_scale, return_all_pred_xstart=False)

        # prep for motion prediction
        y.pop("repaint_mask", None)
        y.pop("repaint_value", None)
        y["traj"] = traj_pred
        y["traj_mask"] = torch.zeros_like(traj_mask)

        motion = self.sample_motion(y, B, cond_scale, return_all_pred_xstart=False)
        return traj_pred, motion
