import torch

from model.uniegomotion import GlobalLocalOutputHead


def test_global_local_output_shape_and_noncontiguous_merge():
    head = GlobalLocalOutputHead(
        latent_dim=8,
        output_dim=12,
        global_start=3,
        global_end=5,
        mode="no_fusion",
        dropout=0.0,
    )
    shared = torch.randn(2, 4, 8)
    output = head(shared)

    assert output.shape == (2, 4, 12)
    with torch.no_grad():
        local = head.local_output(head.local_adapter(shared))
        global_part = head.global_output(head.global_adapter(shared))
        torch.testing.assert_close(output[..., :3], local[..., :3])
        torch.testing.assert_close(output[..., 3:5], global_part)
        torch.testing.assert_close(output[..., 5:], local[..., 3:])


def test_all_topologies_keep_an_identical_ddp_parameter_graph():
    parameter_names = None
    for mode in GlobalLocalOutputHead.MODES:
        head = GlobalLocalOutputHead(
            latent_dim=8,
            output_dim=12,
            global_start=3,
            global_end=5,
            mode=mode,
            dropout=0.0,
        )
        names_here = tuple(name for name, _ in head.named_parameters())
        if parameter_names is None:
            parameter_names = names_here
        else:
            assert names_here == parameter_names

        output = head(torch.randn(2, 4, 8))
        output.square().mean().backward()
        missing_gradients = [name for name, parameter in head.named_parameters() if parameter.grad is None]
        assert not missing_gradients, f"{mode} has unused parameters: {missing_gradients}"


def test_fusion_activity_masks_match_the_requested_topology():
    expected = {
        "no_fusion": (0.0, 0.0),
        "local_to_global": (1.0, 0.0),
        "global_to_local": (0.0, 1.0),
        "bidirectional": (1.0, 1.0),
    }
    for mode, activity in expected.items():
        head = GlobalLocalOutputHead(
            latent_dim=8,
            output_dim=12,
            global_start=3,
            global_end=5,
            mode=mode,
            dropout=0.0,
        )
        assert head.local_to_global_active.item() == activity[0]
        assert head.global_to_local_active.item() == activity[1]


if __name__ == "__main__":
    test_global_local_output_shape_and_noncontiguous_merge()
    test_all_topologies_keep_an_identical_ddp_parameter_graph()
    test_fusion_activity_masks_match_the_requested_topology()
    print("Global/Local output head tests passed")
