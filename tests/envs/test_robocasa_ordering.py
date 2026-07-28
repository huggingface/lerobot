#!/usr/bin/env python

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

from lerobot.envs.robocasa import RoboCasaEnv, convert_action


def test_robocasa_action_uses_openpi_checkpoint_order():
    action = np.arange(12, dtype=np.float32)

    converted = convert_action(action)

    np.testing.assert_array_equal(converted["action.end_effector_position"], [0, 1, 2])
    np.testing.assert_array_equal(converted["action.end_effector_rotation"], [3, 4, 5])
    np.testing.assert_array_equal(converted["action.gripper_close"], [6])
    np.testing.assert_array_equal(converted["action.base_motion"], [7, 8, 9, 10])
    np.testing.assert_array_equal(converted["action.control_mode"], [11])


def test_robocasa_state_uses_openpi_checkpoint_order():
    env = object.__new__(RoboCasaEnv)
    env.obs_type = "pixels_agent_pos"
    env.camera_name = []
    raw_observation = {
        "state.end_effector_position_relative": np.arange(0, 3),
        "state.end_effector_rotation_relative": np.arange(3, 7),
        "state.base_position": np.arange(7, 10),
        "state.base_rotation": np.arange(10, 14),
        "state.gripper_qpos": np.arange(14, 16),
    }

    observation = env._format_raw_obs(raw_observation)

    np.testing.assert_array_equal(observation["agent_pos"], np.arange(16, dtype=np.float32))
