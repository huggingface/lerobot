from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from tests.scripts.save_policy_to_safetensor import get_policy_stats


@pytest.mark.parametrize(
    "env_name,policy_name",
    [
        # ("xarm", "tdmpc"),
        ("pusht", "diffusion"),
        ("aloha", "act"),
    ],
)
def test_backward_compatibility(env_name, policy_name):
    env_policy_dir = Path("tests/data/save_policy_to_safetensors") / f"{env_name}_{policy_name}"
    saved_grad_stats = load_file(env_policy_dir / "grad_stats")
    saved_param_stats = load_file(env_policy_dir / "param_stats")
    saved_actions = load_file(env_policy_dir / "actions")

    grad_stats, param_stats, actions = get_policy_stats(env_name, policy_name)

    for key in saved_grad_stats:
        assert torch.isclose(grad_stats[key], saved_grad_stats[key]).all()
    for key in saved_param_stats:
        assert torch.isclose(param_stats[key], saved_param_stats[key]).all()
    for key in saved_actions:
        assert torch.isclose(actions[key], saved_actions[key]).all()
