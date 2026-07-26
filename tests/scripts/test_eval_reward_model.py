from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.scripts.lerobot_eval_reward_model import (
    _binary_auc,
    _compute_advantages,
    _correlation,
    _held_out_episodes,
    _spearman,
)


def test_reward_evaluation_correlations():
    target = np.array([-1.0, -0.5, 0.0])
    prediction = np.array([-0.9, -0.5, -0.1])

    assert _correlation(prediction, target) == pytest.approx(1.0)
    assert _spearman(prediction, target) == pytest.approx(1.0)


def test_reward_evaluation_auc():
    labels = np.array([False, False, True, True])
    scores = np.array([-0.9, -0.7, -0.2, -0.1])

    assert _binary_auc(scores, labels) == pytest.approx(1.0)


def test_reward_evaluation_reproduces_training_episode_split():
    metadata = SimpleNamespace(
        total_episodes=5,
        episodes={"tasks": [["a"], ["a"], ["a"], ["b"], ["b"]]},
    )

    assert _held_out_episodes(metadata, 0.34) == [1, 2, 4]
    assert _held_out_episodes(metadata, 0.0) is None


def test_reward_evaluation_n_step_advantages():
    target = np.array([-0.9, -0.6, -0.3, -0.8, -0.4], dtype=np.float32)
    prediction = np.array([-0.8, -0.5, -0.2, -0.7, -0.3], dtype=np.float32)
    episodes = np.array([0, 0, 0, 1, 1])

    advantage = _compute_advantages(target, prediction, episodes, n_step=2)

    expected_first = target[0] - target[2] + prediction[2] - prediction[0]
    assert advantage[0] == pytest.approx(expected_first)
    assert advantage[1] == pytest.approx(target[1] - prediction[1])
    assert advantage[3] == pytest.approx(target[3] - prediction[3])
