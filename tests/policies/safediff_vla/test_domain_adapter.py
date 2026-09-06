from __future__ import annotations

import torch

from lerobot.policies.safediff_vla.domain_adapter import LiberoBackboneDomainAdapter
from lerobot.utils.constants import ACTION, OBS_STATE


def stats(state_mean, state_std, action_mean, action_std):
    return {
        OBS_STATE: {"mean": state_mean, "std": state_std},
        ACTION: {"mean": action_mean, "std": action_std},
    }


def test_observation_is_unnormalized_in_target_and_renormalized_for_source() -> None:
    source = stats([1.0, 2.0], [2.0, 4.0], [0.0] * 7, [1.0] * 7)
    target = stats([5.0, 10.0], [5.0, 10.0], [0.0] * 7, [1.0] * 7)
    adapter = LiberoBackboneDomainAdapter(source, target, state_dim=2, action_dim=7)
    batch = {OBS_STATE: torch.tensor([[1.0, -1.0]]), "untouched": torch.tensor([3.0])}
    converted = adapter.observation_for_backbone(batch)
    # target raw = [10, 0], then source normalized = [(10-1)/2, (0-2)/4]
    torch.testing.assert_close(converted[OBS_STATE], torch.tensor([[4.5, -0.5]]))
    assert converted["untouched"] is batch["untouched"]
    torch.testing.assert_close(batch[OBS_STATE], torch.tensor([[1.0, -1.0]]))


def test_nominal_is_unnormalized_converted_and_target_normalized() -> None:
    source = stats([0.0, 0.0], [1.0, 1.0], [0.0] * 7, [1.0] * 7)
    target = stats([0.0, 0.0], [1.0, 1.0], [0.01] * 6 + [0.0], [0.005] * 6 + [1.0])
    adapter = LiberoBackboneDomainAdapter(source, target, state_dim=2, action_dim=7)
    nominal = torch.ones(1, 2, 7)
    converted = adapter.nominal_for_target(nominal)
    # LIBERO -> Safety per-step: translation .025, rotation .25, gripper unchanged.
    torch.testing.assert_close(converted[..., :3], torch.full((1, 2, 3), 3.0))
    torch.testing.assert_close(converted[..., 3:6], torch.full((1, 2, 3), 48.0), atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(converted[..., 6], torch.ones(1, 2))
