import math
import torch
from torch import nn
import torch.nn.functional as F
import ipdb

from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, scale=None, shift=None):
        if scale is None and shift is None:
            return self.net(x)
        x = self.net[0](x)
        x = _apply_film(x, scale, shift)
        for layer in self.net[1:]:
            x = layer(x)
        return x


class MoEFeedForward(nn.Module):
    """Chunk-routed MoE FFN with an always-active per-layer shared expert.

    ``num_experts`` denotes *routed* experts.  When ``shared_expert=True``
    (the Motion Expert configuration), every layer owns one independent
    ``shared`` FFN plus ``num_experts`` routed FFNs.  Routing decisions are
    made once per temporal chunk and broadcast to its frames.  The router can
    consume motion, pose, egovideo and flow-time hidden features without
    changing the ``[B, T, D] -> [B, T, D]`` FFN interface.

    The dispatch path intentionally uses ordinary PyTorch indexing.  This is
    slower than a fused MoE kernel, but keeps the experiment reproducible and
    compatible with the existing DDP setup.
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout,
        num_experts=11,
        top_k=2,
        router_jitter=0.0,
        shared_expert=True,
        chunk_size=4,
        router_conditioned=True,
        routed_gate_init=0.05,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"MoE requires at least two experts, got {num_experts}.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}.")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_jitter = float(router_jitter)
        self.shared_expert_enabled = bool(shared_expert)
        self.chunk_size = int(chunk_size)
        self.router_conditioned = bool(router_conditioned and self.shared_expert_enabled)
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")

        # The conditional router receives four D-dimensional features:
        # current motion hidden, pose input hidden, egovideo hidden and the
        # broadcast flow-time embedding.  Direct logits keep this adapter
        # small (~0.4M parameters for all 12 layers) and preserve the 400M
        # parameter target.
        router_input_dim = self.dim * 4 if self.router_conditioned else self.dim
        self.router = nn.Linear(router_input_dim, self.num_experts)
        self.shared = FeedForward(self.dim, self.hidden_dim, dropout) if self.shared_expert_enabled else None
        self.routed_experts = nn.ModuleList(
            [FeedForward(self.dim, self.hidden_dim, dropout) for _ in range(self.num_experts)]
        )
        # A small initial routed contribution lets the copied dense FFN in the
        # shared branch preserve the original K12 behavior during warm-up.
        gate_logit = math.log(float(routed_gate_init) / max(1.0 - float(routed_gate_init), 1e-6))
        self.routed_gate = nn.Parameter(torch.tensor(gate_logit))
        self.last_aux_loss = None
        self.last_router_probs = None
        self.last_top_indices = None
        # Evaluation-only routing override.  Training always uses the configured
        # top-k router, so checkpoint optimization is unaffected by the
        # ablations below.
        self.inference_routing_mode = "top2"
        self.random_routing_seed = 62
        self._random_routing_calls = 0

    def set_inference_routing_mode(self, mode="top2", seed=62):
        """Select an evaluation routing ablation without changing parameters."""
        aliases = {"topk": "top2", "random": "random_top2", "shared": "shared_only"}
        mode = aliases.get(str(mode).lower(), str(mode).lower())
        valid = {"top2", "top1", "random_top2", "shared_only"}
        if mode not in valid:
            raise ValueError(f"Unknown MoE inference routing mode {mode!r}; expected {sorted(valid)}.")
        self.inference_routing_mode = mode
        self.random_routing_seed = int(seed)
        self._random_routing_calls = 0

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _chunk_mean(self, x, mask):
        """Return ``[B, ceil(T/C), D]`` masked means."""
        b, t, d = x.shape
        n_chunks = (t + self.chunk_size - 1) // self.chunk_size
        padded_t = n_chunks * self.chunk_size
        if padded_t != t:
            x = F.pad(x, (0, 0, 0, padded_t - t))
            if mask is not None:
                mask = F.pad(mask, (0, padded_t - t), value=False)
        x = x.view(b, n_chunks, self.chunk_size, d)
        if mask is None:
            return x.mean(dim=2)
        mask = mask.view(b, n_chunks, self.chunk_size).to(dtype=x.dtype)
        return (x * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)

    def _router_logits(self, x, router_conditions=None, frame_mask=None):
        """Build one router logit vector per prefix/frame token."""
        # The UniEgoMotion sequence has a prepended timestep token.  Conditions
        # are frame-aligned, so x[:, 1:] is the physical motion sequence.
        has_prefix = router_conditions is not None and "pose" in router_conditions and x.shape[1] > router_conditions["pose"].shape[1]
        frame_x = x[:, 1:] if has_prefix else x
        if frame_mask is None and router_conditions is not None:
            frame_mask = router_conditions.get("valid_frames")

        if not self.router_conditioned:
            return self.router(x)

        if router_conditions is None:
            raise ValueError("Conditional MoE routing requires router_conditions.")
        pose = router_conditions["pose"]
        ego = router_conditions["ego"]
        timestep = router_conditions["timestep"]
        if timestep.ndim == 2:
            timestep = timestep[:, None, :].expand(-1, frame_x.shape[1], -1)
        motion = frame_x
        expected = frame_x.shape[:2]
        features = []
        for name, feature in (("motion", motion), ("pose", pose), ("ego", ego), ("timestep", timestep)):
            if feature.ndim != 3 or feature.shape[:2] != expected:
                raise ValueError(
                    f"Router condition {name!r} must be [B,T,D] aligned to {tuple(expected)}, "
                    f"got {tuple(feature.shape)}."
                )
            features.append(feature)
        condition = torch.cat(features, dim=-1)
        chunk_condition = self._chunk_mean(condition, frame_mask)
        chunk_logits = self.router(chunk_condition)
        frame_logits = chunk_logits.repeat_interleave(self.chunk_size, dim=1)[:, : frame_x.shape[1]]

        # Route the prefix/timestep token from the sequence-level mean.  It is
        # not part of a physical frame chunk but still passes through every
        # layer's FFN.
        prefix_logits = self.router(self._masked_mean(condition, frame_mask).unsqueeze(1))
        if has_prefix:
            return torch.cat((prefix_logits, frame_logits), dim=1)
        return frame_logits

    def initialize_routed_from_shared(self, noise_std=0.01):
        """Copy a dense/shared FFN into routed experts with small perturbations."""
        if self.shared is None:
            return
        with torch.no_grad():
            shared_state = self.shared.state_dict()
            for expert in self.routed_experts:
                expert.load_state_dict(shared_state)
                if noise_std > 0:
                    for parameter in expert.parameters():
                        parameter.add_(torch.randn_like(parameter) * float(noise_std))

    def forward(self, x, router_conditions=None, frame_mask=None):
        original_shape = x.shape
        flat_x = x.reshape(-1, self.dim)
        routing_mode = "top2" if self.training else self.inference_routing_mode
        if routing_mode == "shared_only":
            self.last_aux_loss = None
            self.last_router_probs = None
            self.last_top_indices = None
            if self.shared is None:
                raise RuntimeError("shared_only routing requires an enabled shared expert.")
            return self.shared(flat_x).reshape(original_shape)

        router_logits = self._router_logits(x, router_conditions=router_conditions, frame_mask=frame_mask)
        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter
        router_probs = torch.softmax(router_logits, dim=-1)
        if routing_mode == "random_top2":
            # Use a private, per-layer deterministic generator so random-route
            # ablations do not consume or perturb the Flow sampler's noise RNG.
            generator = torch.Generator(device=router_probs.device)
            generator.manual_seed(self.random_routing_seed + self._random_routing_calls)
            self._random_routing_calls += 1
            random_scores = torch.rand(
                router_probs.shape,
                device=router_probs.device,
                dtype=router_probs.dtype,
                generator=generator,
            )
            top_indices = torch.topk(random_scores, 2, dim=-1).indices
            top_weights = torch.full_like(top_indices, 0.5, dtype=router_probs.dtype)
        else:
            effective_top_k = 1 if routing_mode == "top1" else self.top_k
            top_values, top_indices = torch.topk(router_probs, effective_top_k, dim=-1)
            top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        effective_top_k = top_indices.shape[-1]
        flat_top_indices = top_indices.reshape(-1, effective_top_k)
        flat_top_weights = top_weights.reshape(-1, effective_top_k)

        flat_output = torch.zeros_like(flat_x)
        # Each selected token is evaluated only by the experts assigned to it.
        # The empty branch is valid for small batches and simply contributes no
        # output; DDP is configured with find_unused_parameters for this mode.
        for expert_id, expert in enumerate(self.routed_experts):
            token_index, slot_index = torch.where(flat_top_indices == expert_id)
            if token_index.numel() == 0:
                continue
            expert_output = expert(flat_x.index_select(0, token_index))
            expert_weight = flat_top_weights[token_index, slot_index].unsqueeze(-1)
            flat_output.index_add_(0, token_index, expert_output * expert_weight)

        # Switch-style load-balancing loss.  The loss is exposed to the outer
        # UEM module and added there, so it is not mixed into the motion output.
        if frame_mask is not None and router_probs.shape[1] > frame_mask.shape[1]:
            token_mask = torch.cat(
                (torch.ones(frame_mask.shape[0], 1, device=frame_mask.device, dtype=frame_mask.dtype), frame_mask),
                dim=1,
            )
        else:
            token_mask = frame_mask
        if token_mask is None:
            token_mask = torch.ones(router_probs.shape[:2], device=router_probs.device, dtype=torch.bool)
        token_mask = token_mask.reshape(-1).to(dtype=router_probs.dtype)
        denom = token_mask.sum().clamp_min(1.0)
        top1_load = torch.nn.functional.one_hot(
            top_indices[..., 0], num_classes=self.num_experts
        ).to(router_probs.dtype).reshape(-1, self.num_experts)
        top1_load = (top1_load * token_mask.unsqueeze(-1)).sum(dim=0) / denom
        mean_router_prob = (router_probs.reshape(-1, self.num_experts) * token_mask.unsqueeze(-1)).sum(dim=0) / denom
        self.last_aux_loss = self.num_experts * torch.sum(top1_load * mean_router_prob)
        self.last_router_probs = router_probs.detach()
        self.last_top_indices = top_indices.detach()
        if self.shared is not None:
            flat_output = self.shared(flat_x) + torch.sigmoid(self.routed_gate) * flat_output
        return flat_output.reshape(original_shape)


class MoEEncoderBlock(nn.Module):
    """Encoder block equivalent to EncoderBlock with an MoE FFN."""

    def __init__(self, dim, heads, dropout, ff_mult, num_experts=11, top_k=2, router_jitter=0.0, **moe_kwargs):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.ff = MoEFeedForward(
            dim,
            dim * ff_mult,
            dropout,
            num_experts=num_experts,
            top_k=top_k,
            router_jitter=router_jitter,
            **moe_kwargs,
        )

    def forward(self, x, mask=None, router_conditions=None):
        x = self.attn(x, kv_x=x, mask=mask) + x
        frame_mask = None if router_conditions is None else router_conditions.get("valid_frames")
        x = self.ff(x, router_conditions=router_conditions, frame_mask=frame_mask) + x
        return x


class MoEDecoderBlock(nn.Module):
    """Decoder block equivalent to DecoderBlock with an MoE FFN."""

    def __init__(self, dim, heads, dropout, ff_mult, num_experts=11, top_k=2, router_jitter=0.0, **moe_kwargs):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout)
        self.ff = MoEFeedForward(
            dim,
            dim * ff_mult,
            dropout,
            num_experts=num_experts,
            top_k=top_k,
            router_jitter=router_jitter,
            **moe_kwargs,
        )

    def forward(self, x, context, mask=None, context_mask=None, router_conditions=None):
        x = self.attn1(x, kv_x=x, mask=mask) + x
        x = self.attn2(x, kv_x=context, mask=context_mask) + x
        frame_mask = None if router_conditions is None else router_conditions.get("valid_frames")
        x = self.ff(x, router_conditions=router_conditions, frame_mask=frame_mask) + x
        return x


# class Attention(nn.Module):
#     def __init__(self, dim, heads, dropout):
#         super().__init__()
#         assert dim % heads == 0
#         self.heads = heads
#         self.dim = dim
#         self.dim_head = dim // heads
#         self.scale = self.dim_head**-0.5
#         self.norm = nn.LayerNorm(dim)

#         self.softmax = nn.Softmax(dim=-1)

#         self.q = nn.Linear(dim, dim, bias=False)
#         self.k = nn.Linear(dim, dim, bias=False)
#         self.v = nn.Linear(dim, dim, bias=False)
#         self.to_out = nn.Linear(dim, dim, bias=False)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.out_dropout = nn.Dropout(dropout)

#     def forward(self, x, kv_x, mask=None):
#         # mask B x 1 x 1 x N denotes which elements of kv_x are valid

#         x = self.norm(x)

#         qkv = (self.q(x), self.k(kv_x), self.v(kv_x))

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         if mask is not None:
#             dots.masked_fill_(mask == 0, -float("inf"))

#         attn = self.softmax(dots)
#         attn = self.attn_dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.to_out(out)
#         out = self.out_dropout(out)
#         return out


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_x, mask=None, scale=None, shift=None):
        # mask B x 1 x 1 x N denotes which elements of kv_x are valid

        x = self.norm(x)
        x = _apply_film(x, scale, shift)
        if mask is not None:
            mask = mask.squeeze([1, 2]).bool().logical_not()
        out = self.attn(x, kv_x, kv_x, key_padding_mask=mask)[0]
        out = self.out_dropout(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, x, mask=None):
        x = self.attn(x, kv_x=x, mask=mask) + x
        x = self.ff(x) + x
        return x


def _apply_film(x, scale=None, shift=None):
    """Apply identity-centered per-sample FiLM to a BxTxD tensor."""

    if scale is None and shift is None:
        return x
    if scale is None or shift is None:
        raise ValueError("FiLM scale and shift must be provided together.")
    return x * (1.0 + scale[:, None, :].to(dtype=x.dtype)) + shift[:, None, :].to(dtype=x.dtype)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult, task_film=False, num_tasks=3):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)
        self.task_film_enabled = bool(task_film)
        if self.task_film_enabled:
            # [task, self-attn/cross-attn/ffn, scale/shift, feature].
            # Zero initialization makes E18 exactly equal to E14 at step zero.
            self.task_film = nn.Parameter(torch.zeros(int(num_tasks), 3, 2, dim))
        else:
            self.register_parameter("task_film", None)

    def _film(self, task_id, sublayer):
        if not self.task_film_enabled:
            return None, None
        if task_id is None:
            raise ValueError("TaskFiLM requires a task_id for every sample.")
        values = self.task_film[task_id, sublayer]
        return values[:, 0], values[:, 1]

    def forward(self, x, context, mask=None, context_mask=None, task_id=None):
        scale, shift = self._film(task_id, 0)
        x = self.attn1(x, kv_x=x, mask=mask, scale=scale, shift=shift) + x
        scale, shift = self._film(task_id, 1)
        x = self.attn2(x, kv_x=context, mask=context_mask, scale=scale, shift=shift) + x
        scale, shift = self._film(task_id, 2)
        x = self.ff(x, scale=scale, shift=shift) + x
        return x


if __name__ == "__main__":
    dim = 768
    heads = 24
    num_enc_layers = 6
    num_dec_layers = 4
    b = 3

    ipdb.set_trace()
