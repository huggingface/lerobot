#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.datasets.compute_stats import (  # noqa: E402
    compute_relative_action_stats,
    compute_state_history_stats,
)
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import (  # noqa: E402
    Pi05FlattenStateHistoryProcessorStep,
    Pi05StateFromActionProcessorStep,
)
from lerobot.processor.relative_action_processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402


def _transition(action: torch.Tensor | None, state: torch.Tensor | None = None) -> dict:
    observation = {} if state is None else {OBS_STATE: state}
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
        TransitionKey.COMPLEMENTARY_DATA: {},
    }


def test_pi05_config_requests_action_history_prefix():
    config = PI05Config(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        state_from_action=True,
        proprioception_history_steps=2,
    )

    assert config.action_delta_indices == [-1, 0, 1, 2, 3]


def test_state_from_action_extracts_history_and_preserves_target_horizon():
    action = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    step = Pi05StateFromActionProcessorStep(enabled=True, history_steps=2)

    result = step(_transition(action))

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], action[:, :2])
    torch.testing.assert_close(result[TransitionKey.ACTION], action[:, 1:])


def test_relative_actions_use_newest_state_in_history_and_roundtrip():
    state_history = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]])
    absolute = torch.tensor([[[3.0, 30.0], [4.0, 40.0]]])
    relative_step = RelativeActionsProcessorStep(enabled=True)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    relative = relative_step(_transition(absolute, state_history))
    expected = torch.tensor([[[1.0, 10.0], [2.0, 20.0]]])
    torch.testing.assert_close(relative[TransitionKey.ACTION], expected)

    recovered = absolute_step(_transition(relative[TransitionKey.ACTION]))
    torch.testing.assert_close(recovered[TransitionKey.ACTION], absolute)


def test_flatten_state_history_preserves_chronological_order():
    state_history = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    step = Pi05FlattenStateHistoryProcessorStep(history_steps=2, max_state_dim=4)

    result = step(_transition(torch.zeros(1, 2, 2), state_history))

    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )


def test_state_history_can_be_relative_with_absolute_gripper():
    state_history = torch.tensor([[[1.0, 10.0, 0.2], [3.0, 20.0, 0.4]]])
    step = Pi05FlattenStateHistoryProcessorStep(
        history_steps=2,
        max_state_dim=6,
        relative=True,
        exclude_joints=["gripper"],
        state_names=["x", "y", "gripper_width"],
    )

    result = step(_transition(torch.zeros(1, 2, 3), state_history))

    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[-2.0, -10.0, 0.2, 0.0, 0.0, 0.4]]),
    )


def test_inference_state_history_is_rolled_and_reset():
    step = Pi05StateFromActionProcessorStep(enabled=True, history_steps=2)

    first = step(_transition(None, torch.tensor([[1.0, 2.0]])))
    second = step(_transition(None, torch.tensor([[3.0, 4.0]])))
    torch.testing.assert_close(
        first[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
    )
    torch.testing.assert_close(
        second[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    )

    step.reset()
    reset = step(_transition(None, torch.tensor([[5.0, 6.0]])))
    torch.testing.assert_close(
        reset[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[[5.0, 6.0], [5.0, 6.0]]])
    )


def test_flatten_state_history_checks_max_state_dim():
    step = Pi05FlattenStateHistoryProcessorStep(history_steps=2, max_state_dim=3)

    with pytest.raises(ValueError, match="above max_state_dim"):
        step(_transition(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)))


def test_relative_stats_can_use_absolute_action_as_state():
    actions = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float32)
    dataset = {"action": actions, "episode_index": np.zeros(4, dtype=np.int64)}
    features = {"action": {"shape": [2], "names": ["x", "y"]}}

    stats = compute_relative_action_stats(
        dataset,
        features,
        chunk_size=2,
        state_from_action=True,
    )

    np.testing.assert_allclose(stats["mean"], [0.5, 1.0])


def test_relative_state_history_stats_match_processor_representation():
    actions = np.asarray(
        [[0.0, 0.1], [1.0, 0.2], [3.0, 0.3]],
        dtype=np.float32,
    )
    dataset = {"action": actions, "episode_index": np.zeros(3, dtype=np.int64)}
    features = {"action": {"shape": [2], "names": ["x", "gripper_width"]}}

    stats = compute_state_history_stats(
        dataset,
        features,
        history_steps=2,
        exclude_joints=["gripper"],
        relative=True,
    )

    expected = np.asarray([[0.0, 0.1, 0.0, 0.1], [-1.0, 0.1, 0.0, 0.2], [-2.0, 0.2, 0.0, 0.3]])
    np.testing.assert_allclose(stats["mean"], expected.mean(axis=0))
