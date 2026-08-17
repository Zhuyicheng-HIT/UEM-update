from collections import Counter

import torch

from module.task_sampler import ExplicitTaskSchedule
from utils.task_conditioning import TASKS, TASK_TO_ID, apply_task_conditioning


def make_conditioning():
    return {
        "valid_frames": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long),
        "traj": torch.randn(2, 4, 18),
        "img_embs": torch.randn(2, 4, 8),
        "valid_img_embs": torch.ones(2, 4, dtype=torch.long),
    }


def make_schedule(mode, total_steps=100):
    return ExplicitTaskSchedule(
        mode=mode,
        total_steps=total_steps,
        seed=62,
        fixed_probabilities=[0.4, 0.3, 0.3],
        curriculum_fractions=[0.15, 0.20, 0.25, 0.40],
        curriculum_probabilities=[
            [0.80, 0.15, 0.05],
            [0.55, 0.35, 0.10],
            [0.35, 0.35, 0.30],
            [0.20, 0.30, 0.50],
        ],
        adaptive_start_fraction=0.60,
        adaptive_update_interval=5,
    )


def test_reconstruction_uses_complete_conditions_and_padding_loss_mask():
    original = make_conditioning()
    conditioned = apply_task_conditioning(original, "recon", forecast_prefix=2)

    assert not conditioned["traj_mask"].bool().any()
    assert not conditioned["img_mask"].bool().any()
    torch.testing.assert_close(conditioned["valid_frames"], original["valid_frames"])
    torch.testing.assert_close(conditioned["loss_mask"], original["valid_frames"])
    assert conditioned["task_id"].tolist() == [TASK_TO_ID["recon"]] * 2
    assert "loss_mask" not in original


def test_generation_keeps_only_first_image_and_never_trains_on_padding():
    original = make_conditioning()
    conditioned = apply_task_conditioning(original, "gen", forecast_prefix=2)

    assert conditioned["traj_mask"].bool().all()
    assert not conditioned["img_mask"][:, 0].bool().any()
    assert conditioned["img_mask"][:, 1:].bool().all()
    assert conditioned["valid_frames"].bool().all()
    torch.testing.assert_close(conditioned["loss_mask"], original["valid_frames"])


def test_forecasting_masks_future_conditions_and_excludes_observed_prefix_from_loss():
    conditioned = apply_task_conditioning(make_conditioning(), "fore", forecast_prefix=2)

    assert not conditioned["traj_mask"][:, :2].bool().any()
    assert conditioned["traj_mask"][:, 2:].bool().all()
    assert not conditioned["img_mask"][:, :2].bool().any()
    assert conditioned["img_mask"][:, 2:].bool().all()
    assert conditioned["valid_frames"].bool().all()
    assert not conditioned["loss_mask"][:, :2].bool().any()
    assert conditioned["loss_mask"][0].tolist() == [0, 0, 1, 0]
    assert conditioned["loss_mask"][1].tolist() == [0, 0, 1, 1]


def test_single_sample_task_conditioning_collates_task_id_cleanly():
    conditioning = {key: value[0] for key, value in make_conditioning().items()}
    conditioned = apply_task_conditioning(conditioning, "gen", forecast_prefix=2)

    assert conditioned["task_id"].ndim == 0
    assert conditioned["traj_mask"].shape == (4,)
    assert conditioned["loss_mask"].shape == (4,)


def test_fixed_schedule_has_exact_task_budget_and_is_reproducible():
    first = make_schedule("fixed")
    second = make_schedule("fixed")
    first_ids = [first.task_id_for_step(step) for step in range(100)]
    second_ids = [second.task_id_for_step(step) for step in range(100)]

    assert first_ids == second_ids
    assert Counter(first_ids) == Counter({0: 40, 1: 30, 2: 30})
    assert first.probabilities_for_step(0) == [0.4, 0.3, 0.3]


def test_curriculum_preserves_total_budget_while_changing_phase_mix():
    schedule = make_schedule("curriculum")
    ids = [schedule.task_id_for_step(step) for step in range(100)]

    assert Counter(ids) == Counter({0: 40, 1: 30, 2: 30})
    assert schedule.probabilities_for_step(0) == [0.8, 0.15, 0.05]
    assert schedule.probabilities_for_step(99) == [0.2, 0.3, 0.5]


def test_adaptive_replay_moves_probability_within_bounds_and_round_trips_state():
    schedule = make_schedule("adaptive")
    assert schedule.update_from_scores([1.0, 1.0, 1.0], step=60) is False
    assert schedule.update_from_scores([1.2, 0.8, 1.0], step=65) is True
    assert schedule.current_probabilities == [0.25, 0.25, 0.5]

    restored = make_schedule("adaptive")
    restored.load_state_dict(schedule.state_dict())
    assert restored.state_dict() == schedule.state_dict()
    assert abs(sum(restored.current_probabilities) - 1.0) < 1.0e-8
    assert all(0.15 <= probability <= 0.55 for probability in restored.current_probabilities)


if __name__ == "__main__":
    test_reconstruction_uses_complete_conditions_and_padding_loss_mask()
    test_generation_keeps_only_first_image_and_never_trains_on_padding()
    test_forecasting_masks_future_conditions_and_excludes_observed_prefix_from_loss()
    test_single_sample_task_conditioning_collates_task_id_cleanly()
    test_fixed_schedule_has_exact_task_budget_and_is_reproducible()
    test_curriculum_preserves_total_budget_while_changing_phase_mix()
    test_adaptive_replay_moves_probability_within_bounds_and_round_trips_state()
    print(f"Explicit task sampling tests passed for {TASKS}")
