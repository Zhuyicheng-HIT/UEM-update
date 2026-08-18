import torch
import torch.nn as nn

from module.smplx_geometry_loss import SMPLXGeometryLoss, geometry_terms_from_joints


def test_geometry_terms_are_zero_for_identical_motion():
    joints = torch.randn(2, 4, 55, 3)
    contacts = torch.zeros(2, 4, 2)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    terms = geometry_terms_from_joints(joints, joints.clone(), contacts, mask, delta=0.01)
    for value in terms.values():
        torch.testing.assert_close(value, torch.zeros(2))


def test_geometry_loss_masks_padding_and_observed_forecast_prefix():
    target = torch.zeros(1, 4, 55, 3)
    pred = target.clone()
    pred[:, 0] = 100.0
    pred[:, 1] = 100.0
    pred[:, 2, 0, 0] = 0.1
    pred[:, 3] = 100.0
    mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    contacts = torch.zeros(1, 4, 2)
    terms = geometry_terms_from_joints(pred, target, contacts, mask, delta=0.01)

    assert terms["root"].item() > 0.0
    assert terms["joint"].item() > 0.0
    torch.testing.assert_close(terms["hand"], torch.zeros(1))
    torch.testing.assert_close(terms["foot_velocity"], torch.zeros(1))


def test_v4_decoder_is_differentiable_without_constructing_vertices():
    decoder = SMPLXGeometryLoss.__new__(SMPLXGeometryLoss)
    nn.Module.__init__(decoder)
    vertex_count = 60
    decoder.register_buffer("motion_mean", torch.zeros(243), persistent=False)
    decoder.register_buffer("motion_std", torch.ones(243), persistent=False)
    decoder.register_buffer("v_template", torch.randn(vertex_count, 3) * 0.01, persistent=False)
    decoder.register_buffer("shapedirs", torch.zeros(vertex_count, 3, 10), persistent=False)
    regressor = torch.zeros(55, vertex_count)
    regressor[torch.arange(55), torch.arange(55)] = 1.0
    decoder.register_buffer("joint_regressor", regressor, persistent=False)
    decoder.register_buffer("parents", torch.tensor([-1] + list(range(54))), persistent=False)
    decoder.register_buffer("left_hand_components", torch.zeros(12, 45), persistent=False)
    decoder.register_buffer("right_hand_components", torch.zeros(12, 45), persistent=False)

    motion = torch.zeros(2, 3, 243)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    local = motion[..., :198].reshape(2, 3, 22, 9)
    local[..., :6] = identity_6d
    motion[..., 198:204] = identity_6d
    motion = (motion / (1.0 + 1.0e-6)).requires_grad_()

    joints, contacts = decoder._decode(motion, torch.ones(2, 3, dtype=torch.bool))
    assert joints.shape == (2, 3, 55, 3)
    assert contacts.shape == (2, 3, 2)
    joints.square().mean().backward()
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all()
