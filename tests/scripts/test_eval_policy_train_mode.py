"""Regression tests for eval_policy() train-mode restore (issue #3668).

eval_policy() must leave policy.training unchanged after returning, whether
it succeeds or raises an exception.
"""
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import pytest

from lerobot.scripts.lerobot_eval import eval_policy


class _FakePolicy(nn.Module):
    """Minimal policy with Dropout so policy.training is observable."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def reset(self):
        pass


# Shape [1, 1] matches n_episodes=1, env.num_envs=1 used in all tests below.
_ROLLOUT_RETURN = {
    "done": torch.tensor([[True]], dtype=torch.bool),
    "reward": torch.tensor([[1.0]], dtype=torch.float32),
    "success": torch.tensor([[True]], dtype=torch.bool),
}


def _make_env():
    env = MagicMock()
    env.num_envs = 1
    return env


def test_eval_policy_restores_train_mode():
    """Policy started in train mode must be back in train mode after eval_policy()."""
    policy = _FakePolicy()
    policy.train()
    assert policy.training is True

    # Patch PreTrainedPolicy to nn.Module so _FakePolicy passes the isinstance guard.
    with patch("lerobot.scripts.lerobot_eval.PreTrainedPolicy", nn.Module):
        with patch("lerobot.scripts.lerobot_eval.rollout", return_value=_ROLLOUT_RETURN):
            eval_policy(
                env=_make_env(),
                policy=policy,
                env_preprocessor=None,
                env_postprocessor=None,
                preprocessor=None,
                postprocessor=None,
                n_episodes=1,
                max_episodes_rendered=0,
                videos_dir=None,
                return_episode_data=False,
                start_seed=None,
            )

    assert policy.training is True, (
        "eval_policy() left the policy in eval mode. "
        "Dropout is now disabled and BatchNorm stats are frozen for the rest of training."
    )


def test_eval_policy_preserves_eval_mode():
    """Policy already in eval mode must stay in eval mode after eval_policy()."""
    policy = _FakePolicy()
    policy.eval()
    assert policy.training is False

    with patch("lerobot.scripts.lerobot_eval.PreTrainedPolicy", nn.Module):
        with patch("lerobot.scripts.lerobot_eval.rollout", return_value=_ROLLOUT_RETURN):
            eval_policy(
                env=_make_env(),
                policy=policy,
                env_preprocessor=None,
                env_postprocessor=None,
                preprocessor=None,
                postprocessor=None,
                n_episodes=1,
                max_episodes_rendered=0,
                videos_dir=None,
                return_episode_data=False,
                start_seed=None,
            )

    assert policy.training is False


def test_eval_policy_restores_train_mode_on_exception():
    """Train mode must be restored even if eval_policy() raises mid-flight."""
    policy = _FakePolicy()
    policy.train()

    with patch("lerobot.scripts.lerobot_eval.PreTrainedPolicy", nn.Module):
        with patch("lerobot.scripts.lerobot_eval.rollout", side_effect=RuntimeError("simulated crash")):
            with pytest.raises(RuntimeError, match="simulated crash"):
                eval_policy(
                    env=_make_env(),
                    policy=policy,
                    env_preprocessor=None,
                    env_postprocessor=None,
                    preprocessor=None,
                    postprocessor=None,
                    n_episodes=1,
                    max_episodes_rendered=0,
                    videos_dir=None,
                    return_episode_data=False,
                    start_seed=None,
                )

    assert policy.training is True, (
        "eval_policy() left the policy in eval mode after raising an exception."
    )
