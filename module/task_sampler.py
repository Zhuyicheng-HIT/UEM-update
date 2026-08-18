"""Deterministic explicit-task schedules for E13--E16."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import torch

from utils.task_conditioning import TASKS


def _normalize_probabilities(probabilities: Sequence[float], count: int) -> list[float]:
    if len(probabilities) != count:
        raise ValueError(f"Expected {count} probabilities, got {len(probabilities)}.")
    values = [float(value) for value in probabilities]
    if any(value < 0.0 for value in values):
        raise ValueError(f"Task probabilities must be non-negative, got {values}.")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("At least one task probability must be positive.")
    return [value / total for value in values]


def _largest_remainder_counts(total: int, probabilities: Sequence[float]) -> list[int]:
    raw = [total * probability for probability in probabilities]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


class ExplicitTaskSchedule:
    """Generate synchronized task IDs from optimizer step alone.

    Fixed and curriculum modes pre-generate schedules with exact counts.
    Adaptive mode follows the curriculum until ``adaptive_start_step`` and
    then draws deterministically from probabilities that may be updated after
    validation.  Because the random value is a pure function of step and seed,
    all DDP ranks select the same task without communication.
    """

    def __init__(
        self,
        *,
        mode: str,
        total_steps: int,
        seed: int,
        fixed_probabilities: Sequence[float],
        curriculum_fractions: Sequence[float],
        curriculum_probabilities: Sequence[Sequence[float]],
        adaptive_start_fraction: float = 0.60,
        adaptive_min_probability: float = 0.15,
        adaptive_max_probability: float = 0.55,
        adaptive_shift: float = 0.05,
        adaptive_threshold: float = 0.01,
        adaptive_update_interval: int = 5000,
    ) -> None:
        mode = str(mode).lower()
        if mode not in {"fixed", "curriculum", "adaptive"}:
            raise ValueError(f"Unsupported task schedule mode {mode!r}.")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        if not 0.0 <= adaptive_start_fraction <= 1.0:
            raise ValueError("adaptive_start_fraction must lie in [0, 1].")
        if not 0.0 <= adaptive_min_probability < adaptive_max_probability <= 1.0:
            raise ValueError("Adaptive probability bounds must satisfy 0 <= min < max <= 1.")
        if adaptive_shift <= 0.0:
            raise ValueError("adaptive_shift must be positive.")
        if adaptive_update_interval <= 0:
            raise ValueError("adaptive_update_interval must be positive.")

        self.mode = mode
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.fixed_probabilities = _normalize_probabilities(fixed_probabilities, len(TASKS))
        self.curriculum_fractions = [float(value) for value in curriculum_fractions]
        self.curriculum_probabilities = [
            _normalize_probabilities(probabilities, len(TASKS))
            for probabilities in curriculum_probabilities
        ]
        if len(self.curriculum_fractions) != len(self.curriculum_probabilities):
            raise ValueError("Curriculum fractions and probability rows must have the same length.")
        if any(value <= 0.0 for value in self.curriculum_fractions):
            raise ValueError("All curriculum phase fractions must be positive.")
        fraction_sum = sum(self.curriculum_fractions)
        if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"Curriculum fractions must sum to one, got {fraction_sum}.")

        self.adaptive_start_step = int(round(self.total_steps * adaptive_start_fraction))
        self.adaptive_min_probability = float(adaptive_min_probability)
        self.adaptive_max_probability = float(adaptive_max_probability)
        self.adaptive_shift = float(adaptive_shift)
        self.adaptive_threshold = float(adaptive_threshold)
        self.adaptive_update_interval = int(adaptive_update_interval)
        self.current_probabilities = list(
            self.fixed_probabilities if self.mode == "fixed" else self.curriculum_probabilities[-1]
        )
        self.previous_scores: list[float] | None = None
        self.last_adaptive_update_step = -1
        self.update_count = 0

        if self.mode == "fixed":
            self._schedule = self._build_schedule([1.0], [self.fixed_probabilities])
        else:
            self._schedule = self._build_schedule(
                self.curriculum_fractions,
                self.curriculum_probabilities,
            )

    @classmethod
    def from_config(cls, cfg: Any) -> "ExplicitTaskSchedule":
        task_cfg = cfg.TRAIN.TASK_SAMPLER
        rows = list(
            zip(
                task_cfg.CURRICULUM_RECON_PROBS,
                task_cfg.CURRICULUM_FORE_PROBS,
                task_cfg.CURRICULUM_GEN_PROBS,
            )
        )
        return cls(
            mode=task_cfg.MODE,
            total_steps=task_cfg.TOTAL_STEPS,
            seed=task_cfg.SEED,
            fixed_probabilities=task_cfg.FIXED_PROBS,
            curriculum_fractions=task_cfg.CURRICULUM_FRACTIONS,
            curriculum_probabilities=rows,
            adaptive_start_fraction=task_cfg.ADAPTIVE_START_FRACTION,
            adaptive_min_probability=task_cfg.ADAPTIVE_MIN_PROB,
            adaptive_max_probability=task_cfg.ADAPTIVE_MAX_PROB,
            adaptive_shift=task_cfg.ADAPTIVE_SHIFT,
            adaptive_threshold=task_cfg.ADAPTIVE_THRESHOLD,
            adaptive_update_interval=task_cfg.ADAPTIVE_UPDATE_INTERVAL,
        )

    def _build_schedule(
        self,
        fractions: Sequence[float],
        phase_probabilities: Sequence[Sequence[float]],
    ) -> list[int]:
        phase_lengths = _largest_remainder_counts(self.total_steps, fractions)
        schedule: list[int] = []
        for phase_index, (phase_length, probabilities) in enumerate(
            zip(phase_lengths, phase_probabilities)
        ):
            counts = _largest_remainder_counts(phase_length, probabilities)
            phase_tasks = [
                task_id
                for task_id, task_count in enumerate(counts)
                for _ in range(task_count)
            ]
            random.Random(self.seed + 1009 * phase_index).shuffle(phase_tasks)
            schedule.extend(phase_tasks)
        if len(schedule) != self.total_steps:
            raise RuntimeError(
                f"Built a {len(schedule)}-step task schedule for {self.total_steps} steps."
            )
        return schedule

    def task_id_for_step(self, step: int) -> int:
        step = int(step)
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")
        if self.mode != "adaptive" or step < self.adaptive_start_step:
            if step < len(self._schedule):
                return self._schedule[step]
            # Extension runs keep using the final phase instead of restarting
            # the easy curriculum from phase one.
            probabilities = (
                self.fixed_probabilities
                if self.mode == "fixed"
                else self.curriculum_probabilities[-1]
            )
        else:
            probabilities = self.current_probabilities

        draw = random.Random(self.seed + 104729 * step).random()
        cumulative = 0.0
        for task_id, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                return task_id
        return len(probabilities) - 1

    def task_for_step(self, step: int) -> str:
        return TASKS[self.task_id_for_step(step)]

    def task_ids_for_batch(
        self,
        step: int,
        local_batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return one deterministic task ID per sample for E20.

        Counts are computed for the complete DDP global batch, then shuffled
        and sliced by rank. Cumulative largest-remainder rounding removes the
        persistent one-sample bias that ordinary per-batch rounding creates
        for a 512-sample 40/30/30 batch.
        """

        if self.mode != "fixed":
            raise ValueError("Mixed-batch task sampling currently requires MODE='fixed'.")
        step = int(step)
        local_batch_size = int(local_batch_size)
        rank = int(rank)
        world_size = int(world_size)
        if step < 0 or local_batch_size <= 0:
            raise ValueError("step must be non-negative and local_batch_size must be positive.")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid DDP rank/world_size pair: rank={rank}, world_size={world_size}.")

        global_batch_size = local_batch_size * world_size
        # E20's declared global-512 super-cycle. Four batches give Recon the
        # rounding remainder while Fore/Gen alternate; the fifth batch closes
        # the 5-step budget exactly at 1024/768/768 samples.
        is_e20_ratio = all(
            math.isclose(actual, expected, abs_tol=1.0e-8)
            for actual, expected in zip(self.fixed_probabilities, (0.4, 0.3, 0.3))
        )
        if global_batch_size == 512 and is_e20_ratio:
            cycle = (
                (205, 154, 153),
                (205, 153, 154),
                (205, 154, 153),
                (205, 153, 154),
                (204, 154, 154),
            )
            counts = list(cycle[step % len(cycle)])
        else:
            counts_before = _largest_remainder_counts(
                step * global_batch_size, self.fixed_probabilities
            )
            counts_after = _largest_remainder_counts(
                (step + 1) * global_batch_size, self.fixed_probabilities
            )
            counts = [after - before for before, after in zip(counts_before, counts_after)]
        if any(count < 0 for count in counts) or sum(counts) != global_batch_size:
            raise RuntimeError(f"Invalid mixed-batch task counts at step {step}: {counts}.")

        global_ids = torch.tensor(
            [task_id for task_id, count in enumerate(counts) for _ in range(count)],
            dtype=torch.long,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 104729 * step)
        global_ids = global_ids[torch.randperm(global_batch_size, generator=generator)]
        start = rank * local_batch_size
        return global_ids[start : start + local_batch_size].to(device=device)

    def probabilities_for_step(self, step: int) -> list[float]:
        step = int(step)
        if self.mode == "fixed":
            return list(self.fixed_probabilities)
        if self.mode == "adaptive" and step >= self.adaptive_start_step:
            return list(self.current_probabilities)

        phase_lengths = _largest_remainder_counts(self.total_steps, self.curriculum_fractions)
        phase_end = 0
        for probabilities, phase_length in zip(self.curriculum_probabilities, phase_lengths):
            phase_end += phase_length
            if step < phase_end:
                return list(probabilities)
        return list(self.curriculum_probabilities[-1])

    def update_from_scores(self, scores: Sequence[float], *, step: int) -> bool:
        """Shift probability from the fastest-improving to the weakest task.

        Scores must be positive and lower-is-better.  Relative change from the
        preceding validation normalizes the otherwise incomparable task loss
        scales.  Returns whether probabilities changed.
        """

        values = [float(value) for value in scores]
        if len(values) != len(TASKS) or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"Expected {len(TASKS)} finite positive task scores, got {values}.")
        step = int(step)
        if self.mode != "adaptive" or step < self.adaptive_start_step:
            return False

        if self.previous_scores is None:
            self.previous_scores = values
            self.last_adaptive_update_step = step
            return False
        if step - self.last_adaptive_update_step < self.adaptive_update_interval:
            self.previous_scores = values
            return False

        relative_change = [
            current / max(previous, 1.0e-12)
            for current, previous in zip(values, self.previous_scores)
        ]
        self.previous_scores = values
        self.last_adaptive_update_step = step
        weakest = max(range(len(TASKS)), key=relative_change.__getitem__)
        strongest = min(range(len(TASKS)), key=relative_change.__getitem__)
        if relative_change[weakest] - relative_change[strongest] < self.adaptive_threshold:
            return False

        available = self.current_probabilities[strongest] - self.adaptive_min_probability
        capacity = self.adaptive_max_probability - self.current_probabilities[weakest]
        shift = min(self.adaptive_shift, available, capacity)
        if shift <= 0.0:
            return False

        self.current_probabilities[strongest] -= shift
        self.current_probabilities[weakest] += shift
        self.update_count += 1
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_probabilities": list(self.current_probabilities),
            "previous_scores": None if self.previous_scores is None else list(self.previous_scores),
            "last_adaptive_update_step": self.last_adaptive_update_step,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        probabilities = _normalize_probabilities(state["current_probabilities"], len(TASKS))
        if any(
            probability < self.adaptive_min_probability - 1.0e-8
            or probability > self.adaptive_max_probability + 1.0e-8
            for probability in probabilities
        ):
            raise ValueError(f"Checkpoint adaptive probabilities violate configured bounds: {probabilities}.")
        self.current_probabilities = probabilities
        previous_scores = state.get("previous_scores")
        self.previous_scores = None if previous_scores is None else [float(value) for value in previous_scores]
        self.last_adaptive_update_step = int(state.get("last_adaptive_update_step", -1))
        self.update_count = int(state.get("update_count", 0))


__all__ = ["ExplicitTaskSchedule"]
