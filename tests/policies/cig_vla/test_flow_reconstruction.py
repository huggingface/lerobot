import pytest
import torch

from lerobot.policies.cig_vla.flow_matching import (
    compute_target_velocity,
    interpolate_actions,
    velocity_to_action_estimate,
)


@pytest.mark.parametrize("time", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_exact_reconstruction(time):
    clean, noise = torch.randn(3, 5, 7), torch.randn(3, 5, 7)
    timestep = torch.full((3,), time)
    recovered = velocity_to_action_estimate(
        interpolate_actions(clean, noise, timestep), compute_target_velocity(clean, noise), timestep
    )
    torch.testing.assert_close(recovered, clean)


def test_analytic_example_and_wrong_velocity():
    clean, noise, time = torch.tensor([[[2.0]]]), torch.tensor([[[6.0]]]), torch.tensor([0.25])
    noisy = interpolate_actions(clean, noise, time)
    assert noisy.item() == 3.0
    assert velocity_to_action_estimate(noisy, torch.tensor([[[4.0]]]), time).item() == 2.0
    assert velocity_to_action_estimate(noisy, torch.zeros_like(noisy), time).item() != 2.0
