import torch
from torch import nn

from mydiffusion.flow_matching import FlowMatching


class ConstantVelocity(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, x, t, **model_kwargs):
        del t, model_kwargs
        return self.velocity.expand_as(x)


def test_oracle_velocity_sign_recovers_xstart_with_both_solvers():
    x_start = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    noise = torch.tensor([[[-0.5, 4.0], [2.0, -1.0]]])
    oracle = ConstantVelocity(noise - x_start)
    y = {"valid_frames": torch.ones(1, 2, dtype=torch.bool)}

    for solver in ("euler", "heun"):
        flow = FlowMatching(num_steps=4, solver=solver)
        sample = flow.sample_loop(
            oracle,
            x_start.shape,
            model_kwargs={"y": y},
            noise=noise,
        )
        torch.testing.assert_close(sample, x_start)


def test_clean_estimate_matches_xstart():
    flow = FlowMatching()
    x_start = torch.randn(3, 4, 5)
    noise = torch.randn_like(x_start)
    t = torch.tensor([0.1, 0.5, 0.9])
    x_t = flow.interpolate(x_start, noise, t)

    estimate = flow.estimate_xstart(x_t, t, noise - x_start)

    torch.testing.assert_close(estimate, x_start)


def test_training_loss_is_per_sample_and_ignores_padding():
    flow = FlowMatching()
    x_start = torch.zeros(2, 3, 2)
    noise = torch.zeros_like(x_start)
    valid_frames = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)

    class PaddingErrorModel(nn.Module):
        def forward(self, x, t, **model_kwargs):
            del t, model_kwargs
            prediction = torch.ones_like(x)
            prediction[0, 2] = 1000.0
            prediction[1] = 1000.0
            return prediction

    terms = flow.training_losses(
        PaddingErrorModel(),
        x_start,
        model_kwargs={"y": {"valid_frames": valid_frames}},
        noise=noise,
        t=torch.tensor([0.25, 0.75]),
    )

    # Each valid feature has squared error 1. Padded frames do not contribute;
    # an all-padding sample is explicitly assigned zero loss rather than NaN.
    torch.testing.assert_close(terms["loss"], torch.tensor([1.0, 0.0]))
    assert terms["loss"].shape == (2,)


def test_task_loss_mask_takes_precedence_over_sequence_attention_mask():
    flow = FlowMatching(prediction_type="x0", global_feature_start=0, global_feature_end=1)
    x_start = torch.zeros(1, 3, 2)
    noise = torch.zeros_like(x_start)

    class FrameErrorModel(nn.Module):
        def forward(self, x, t, **model_kwargs):
            del t, model_kwargs
            prediction = torch.ones_like(x)
            prediction[:, 1] = 2.0
            prediction[:, 2] = 1000.0
            return prediction

    terms = flow.training_losses(
        FrameErrorModel(),
        x_start,
        model_kwargs={
            "y": {
                "valid_frames": torch.ones(1, 3),
                "loss_mask": torch.tensor([[0, 1, 0]]),
            }
        },
        noise=noise,
        t=torch.tensor([0.5]),
    )

    torch.testing.assert_close(terms["loss"], torch.tensor([4.0]))


def test_split_global_weights_reallocate_equal_total_weight():
    feature_dim = 243
    model_kwargs = {"y": {"valid_frames": torch.ones(1, 1, dtype=torch.bool)}}
    x_start = torch.zeros(1, 1, feature_dim)
    noise = torch.zeros_like(x_start)
    t = torch.tensor([0.5])

    class FixedErrorModel(nn.Module):
        def __init__(self, error):
            super().__init__()
            self.register_buffer("error", error)

        def forward(self, x, timestep, **kwargs):
            del timestep, kwargs
            return self.error.expand_as(x)

    rotation_error = torch.zeros_like(x_start)
    rotation_error[..., 198:204] = 1.0
    translation_error = torch.zeros_like(x_start)
    translation_error[..., 204:207] = 1.0

    uniform = FlowMatching(prediction_type="x0", global_weight=8.0)
    split = FlowMatching(
        prediction_type="x0",
        global_weight=1.0,
        global_rotation_weight=6.0,
        global_translation_weight=12.0,
    )
    uniform_rotation = uniform.training_losses(
        FixedErrorModel(rotation_error), x_start, model_kwargs, noise=noise, t=t
    )["loss"]
    uniform_translation = uniform.training_losses(
        FixedErrorModel(translation_error), x_start, model_kwargs, noise=noise, t=t
    )["loss"]
    split_rotation = split.training_losses(
        FixedErrorModel(rotation_error), x_start, model_kwargs, noise=noise, t=t
    )["loss"]
    split_translation = split.training_losses(
        FixedErrorModel(translation_error), x_start, model_kwargs, noise=noise, t=t
    )["loss"]

    # Both schemes have total global weight 72 and denominator 234 + 72 = 306.
    torch.testing.assert_close(uniform_rotation, torch.tensor([48.0 / 306.0]))
    torch.testing.assert_close(uniform_translation, torch.tensor([24.0 / 306.0]))
    torch.testing.assert_close(split_rotation, torch.tensor([36.0 / 306.0]))
    torch.testing.assert_close(split_translation, torch.tensor([36.0 / 306.0]))


def test_split_global_weights_must_be_configured_together():
    try:
        FlowMatching(global_rotation_weight=6.0)
    except ValueError as error:
        assert "must be set together" in str(error)
    else:
        raise AssertionError("A partial split-weight configuration should be rejected")


def test_task_specific_global_weights_are_applied_per_sample():
    flow = FlowMatching(
        prediction_type="x0",
        global_feature_start=1,
        global_feature_end=2,
        task_global_weights=[2.0, 4.0, 8.0],
    )
    x_start = torch.zeros(3, 1, 3)
    prediction = torch.zeros_like(x_start)
    prediction[..., 1] = 1.0

    class FixedPrediction(nn.Module):
        def forward(self, x, timestep, **kwargs):
            del timestep, kwargs
            return prediction

    terms = flow.training_losses(
        FixedPrediction(),
        x_start,
        model_kwargs={
            "y": {
                "valid_frames": torch.ones(3, 1),
                "task_id": torch.tensor([0, 1, 2]),
            }
        },
        noise=torch.zeros_like(x_start),
        t=torch.full((3,), 0.5),
    )
    torch.testing.assert_close(terms["loss"], torch.tensor([2 / 4, 4 / 6, 8 / 10]))


def test_group_mse_diagnostics_are_unweighted_and_keep_the_objective_unchanged():
    feature_dim = 243
    x_start = torch.zeros(1, 1, feature_dim)
    noise = torch.zeros_like(x_start)
    prediction = torch.zeros_like(x_start)
    prediction[..., :198] = 2.0
    prediction[..., 198:207] = 3.0
    prediction[..., 207:] = 2.0

    class FixedPrediction(nn.Module):
        def forward(self, x, timestep, **kwargs):
            del timestep, kwargs
            return prediction.expand_as(x)

    flow = FlowMatching(prediction_type="x0", global_weight=8.0)
    terms = flow.training_losses(
        FixedPrediction(),
        x_start,
        model_kwargs={"y": {"valid_frames": torch.ones(1, 1)}},
        noise=noise,
        t=torch.tensor([0.5]),
    )

    torch.testing.assert_close(terms["local_mse"], torch.tensor([4.0]))
    torch.testing.assert_close(terms["global_mse"], torch.tensor([9.0]))
    torch.testing.assert_close(terms["loss"], torch.tensor([(234.0 * 4.0 + 72.0 * 9.0) / 306.0]))


def test_sampling_shape_and_intermediate_clean_estimates():
    shape = (2, 5, 3)
    noise = torch.randn(shape)
    model = ConstantVelocity(torch.zeros(1, 1, 1))
    for solver in ("euler", "heun"):
        flow = FlowMatching(num_steps=3, solver=solver)
        sample, estimates = flow.sample_loop(
            model,
            shape,
            model_kwargs={"y": {"valid_frames": torch.ones(2, 5)}},
            noise=noise,
            return_all_pred_xstart=True,
        )

        assert sample.shape == shape
        assert len(estimates) == 3
        assert all(estimate.shape == shape for estimate in estimates)


def test_repaint_is_rejected_explicitly():
    flow = FlowMatching()
    x_start = torch.zeros(1, 2, 3)
    y = {
        "valid_frames": torch.ones(1, 2),
        "repaint_mask": torch.ones_like(x_start),
    }

    try:
        flow.training_losses(
            ConstantVelocity(torch.zeros(1, 1, 1)),
            x_start,
            model_kwargs={"y": y},
            noise=torch.zeros_like(x_start),
            t=torch.tensor([0.5]),
        )
    except NotImplementedError as error:
        assert "does not support repaint" in str(error)
    else:
        raise AssertionError("Flow Matching should reject repaint masks")


if __name__ == "__main__":
    test_oracle_velocity_sign_recovers_xstart_with_both_solvers()
    test_clean_estimate_matches_xstart()
    test_training_loss_is_per_sample_and_ignores_padding()
    test_task_loss_mask_takes_precedence_over_sequence_attention_mask()
    test_split_global_weights_reallocate_equal_total_weight()
    test_split_global_weights_must_be_configured_together()
    test_group_mse_diagnostics_are_unweighted_and_keep_the_objective_unchanged()
    test_sampling_shape_and_intermediate_clean_estimates()
    test_repaint_is_rejected_explicitly()
    print("Flow Matching tests passed")
