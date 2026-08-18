import torch

from model.core import DecoderBlock


def test_task_film_is_identity_initialized_and_parameter_count_is_exact():
    torch.manual_seed(7)
    baseline = DecoderBlock(16, heads=4, dropout=0.0, ff_mult=2)
    modulated = DecoderBlock(16, heads=4, dropout=0.0, ff_mult=2, task_film=True, num_tasks=3)
    missing, unexpected = modulated.load_state_dict(baseline.state_dict(), strict=False)
    assert missing == ["task_film"]
    assert unexpected == []

    x = torch.randn(3, 5, 16)
    context = torch.randn(3, 4, 16)
    task_id = torch.tensor([0, 1, 2])
    torch.testing.assert_close(
        modulated(x, context, task_id=task_id),
        baseline(x, context),
    )
    assert modulated.task_film.numel() == 3 * 3 * 2 * 16


def test_task_film_can_separate_tasks_after_learning():
    block = DecoderBlock(8, heads=2, dropout=0.0, ff_mult=2, task_film=True, num_tasks=3)
    with torch.no_grad():
        block.task_film[1, 0, 1].fill_(0.5)
    x = torch.randn(1, 3, 8).expand(2, -1, -1).clone()
    context = torch.randn(1, 2, 8).expand(2, -1, -1).clone()
    output = block(x, context, task_id=torch.tensor([0, 1]))
    assert not torch.allclose(output[0], output[1])
