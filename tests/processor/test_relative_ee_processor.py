# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import torch

from lerobot.datasets.relative_ee_stats import compute_relative_ee_stats
from lerobot.processor import (
    RelativeEEActionsStep,
    RelativeEEDeriveStateStep,
    RelativeEEStateStep,
    TransitionKey,
    axis_angle_to_matrix,
    batch_to_transition,
    to_absolute_ee_actions,
    to_relative_ee_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _pose(xyz, axis_angle, gripper):
    return torch.tensor([*xyz, *axis_angle, gripper], dtype=torch.float32)


def test_relative_ee_roundtrip_uses_one_reference_per_chunk():
    references = torch.stack(
        [
            _pose([0.2, -0.1, 0.3], [0.0, 0.0, 0.0], 0.1),
            _pose([-0.4, 0.2, 0.1], [0.3, -0.2, 0.1], 0.5),
            _pose([0.0, 0.0, 0.0], [torch.pi - 1e-4, 0.0, 0.0], 0.9),
        ]
    )
    actions = torch.stack(
        [torch.stack([reference, reference + 0.05, reference - 0.03]) for reference in references]
    )
    actions[..., 6] = torch.tensor([0.2, 0.4, 0.6])

    relative = to_relative_ee_actions(actions, references)
    recovered = to_absolute_ee_actions(relative, references)

    assert relative.shape == (3, 3, 10)
    torch.testing.assert_close(recovered[..., :3], actions[..., :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered[..., 6], actions[..., 6])
    torch.testing.assert_close(
        axis_angle_to_matrix(recovered[..., 3:6]),
        axis_angle_to_matrix(actions[..., 3:6]),
        atol=2e-4,
        rtol=2e-4,
    )


def test_relative_ee_training_steps_derive_state_and_shift_padding():
    actions = torch.stack(
        [
            _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
            _pose([0.1, 0.0, 0.0], [0.0, 0.0, 0.1], 0.1),
            _pose([0.2, 0.0, 0.0], [0.0, 0.0, 0.2], 0.2),
        ]
    ).unsqueeze(0)
    transition = batch_to_transition(
        {
            ACTION: actions,
            "action_is_pad": torch.tensor([[True, False, False]]),
        }
    )

    derived = RelativeEEDeriveStateStep()(transition)
    processed = RelativeEEStateStep()(RelativeEEActionsStep()(derived))

    assert derived[TransitionKey.ACTION].shape == (1, 2, 7)
    assert derived[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 2, 7)
    assert processed[TransitionKey.ACTION].shape == (1, 2, 10)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 20)
    torch.testing.assert_close(processed[TransitionKey.ACTION][0, 0, :3], torch.zeros(3))


def test_relative_ee_state_step_tracks_previous_observation_and_resets():
    step = RelativeEEStateStep()
    first = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0).unsqueeze(0)
    second = _pose([0.1, 0.0, 0.0], [0.0, 0.0, 0.2], 0.5).unsqueeze(0)

    first_result = step(batch_to_transition({OBS_STATE: first}))
    second_result = step(batch_to_transition({OBS_STATE: second}))
    step.reset()
    reset_result = step(batch_to_transition({OBS_STATE: second}))

    first_state = first_result[TransitionKey.OBSERVATION][OBS_STATE]
    second_state = second_result[TransitionKey.OBSERVATION][OBS_STATE]
    reset_state = reset_result[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(first_state[:, :9], first_state[:, 10:19])
    assert not torch.allclose(second_state[:, :3], torch.zeros(1, 3))
    torch.testing.assert_close(reset_state[:, :9], reset_state[:, 10:19])


def test_relative_ee_stats_do_not_cross_episode_boundaries():
    actions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
            [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
        ],
        dtype=np.float32,
    )
    stats = compute_relative_ee_stats(
        {ACTION: actions, "episode_index": np.array([0, 0, 1, 1])},
        chunk_size=2,
    )

    np.testing.assert_allclose(stats[ACTION]["mean"][0], 1 / 3, atol=1e-6)
    np.testing.assert_allclose(stats[OBS_STATE]["mean"][0], -0.5, atol=1e-6)
    assert stats[ACTION]["mean"].shape == (10,)
    assert stats[OBS_STATE]["mean"].shape == (20,)
