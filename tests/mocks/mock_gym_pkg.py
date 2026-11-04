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
"""Stand-in for a sim package such as gym_pusht: its env is registered only when this module is imported."""

import gymnasium as gym
import numpy as np

GYM_ID = "lerobot_mock_gym/MockTask-v0"


class MockEnv(gym.Env):
    """Like PushT, it has neither `task` nor `task_description`."""

    _max_episode_steps = 2
    observation_space = gym.spaces.Dict({"agent_pos": gym.spaces.Box(-1.0, 1.0, (6,), np.float32)})
    action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        return {"agent_pos": np.zeros(6, np.float32)}, {}

    def step(self, action):
        self.tick += 1
        return {"agent_pos": np.zeros(6, np.float32)}, 0.0, self.tick == self._max_episode_steps, False, {}


# A string entry point, like gym_pusht's: pickling the registry entry must not import this module.
if GYM_ID not in gym.registry:
    gym.register(id=GYM_ID, entry_point=f"{__name__}:MockEnv")
