import torch
from torch import nn

from mydiffusion.flow_matching import FlowMatching


class ConstantCleanTarget(nn.Module):
    def __init__(self, x_start):
        super().__init__()
        self.register_buffer("x_start", x_start)

    def forward(self, x, t, **model_kwargs):
        del t, model_kwargs
        return self.x_start.expand_as(x)


class FixedVelocityError(nn.Module):
    def forward(self, x, t, **model_kwargs):
        del t, model_kwargs
        return torch.tensor([[[1.0, 1.0, 2.0, 2.0]]], device=x.device).expand_as(x)


def test_target_predictive_training_and_sampling_are_exact_for_oracle():
    x_start = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    noise = torch.tensor([[[-0.5, 4.0], [2.0, -1.0]]])
    model = ConstantCleanTarget(x_start)
    y = {"valid_frames": torch.ones(1, 2, dtype=torch.bool)}

    for solver in ("euler", "heun"):
        flow = FlowMatching(
            num_steps=4,
            solver=solver,
            prediction_type="x0",
            global_feature_start=0,
            global_feature_end=1,
        )
        terms = flow.training_losses(
            model,
            x_start,
            model_kwargs={"y": y},
            noise=noise,
            t=torch.tensor([0.5]),
            return_diagnostics=True,
        )
        torch.testing.assert_close(terms["loss"], torch.zeros(1))
        torch.testing.assert_close(terms["pred_xstart"], x_start)
        torch.testing.assert_close(terms["pred_velocity"], noise - x_start)

        sample = flow.sample_loop(
            model,
            x_start.shape,
            model_kwargs={"y": y},
            noise=noise,
        )
        torch.testing.assert_close(sample, x_start)


def test_global_feature_weighting_preserves_scale_and_baseline():
    x_start = torch.zeros(1, 1, 4)
    noise = torch.zeros_like(x_start)
    y = {"valid_frames": torch.ones(1, 1, dtype=torch.bool)}
    kwargs = {
        "x_start": x_start,
        "model_kwargs": {"y": y},
        "noise": noise,
        "t": torch.tensor([0.5]),
    }

    baseline = FlowMatching(
        global_weight=1.0,
        global_feature_start=2,
        global_feature_end=4,
    ).training_losses(FixedVelocityError(), **kwargs)
    weighted = FlowMatching(
        global_weight=3.0,
        global_feature_start=2,
        global_feature_end=4,
    ).training_losses(FixedVelocityError(), **kwargs)

    torch.testing.assert_close(baseline["loss"], torch.tensor([2.5]))
    torch.testing.assert_close(weighted["loss"], torch.tensor([3.25]))


def test_target_to_velocity_rejects_clean_endpoint():
    flow = FlowMatching(prediction_type="x0")
    x = torch.zeros(1, 2, 3)
    try:
        flow.target_to_velocity(x, torch.zeros(1), x)
    except ValueError as error:
        assert "requires t > 0" in str(error)
    else:
        raise AssertionError("x0-to-velocity conversion must reject t=0")


if __name__ == "__main__":
    test_target_predictive_training_and_sampling_are_exact_for_oracle()
    test_global_feature_weighting_preserves_scale_and_baseline()
    test_target_to_velocity_rejects_clean_endpoint()
    print("Flow ablation tests passed")
