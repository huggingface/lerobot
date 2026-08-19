import torch

from lerobot.benchmarks.cig_bench.metrics import (
    action_response_magnitude,
    intervention_direction_alignment,
    intervention_response_ratio,
    motion_suppression,
)


def test_cig_bench_metrics():
    original = torch.zeros(1, 2, 3)
    changed = torch.ones(1, 2, 3)
    offset = torch.tensor([[1.0, 0.0, 0.0]])
    assert action_response_magnitude(original, changed).item() > 0
    assert torch.isfinite(intervention_response_ratio(original, changed, offset)).all()
    alignment = intervention_direction_alignment(torch.zeros(1, 3), offset, offset)
    torch.testing.assert_close(alignment, torch.ones(1))
    torch.testing.assert_close(motion_suppression(torch.ones(1), torch.zeros(1)), torch.ones(1))
