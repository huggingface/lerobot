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

"""A toy environment config, served or in-process, for the env server tests."""

from dataclasses import dataclass

from lerobot.envs.configs import EnvConfig


@EnvConfig.register_subclass("toy_test")
@dataclass
class ToyEnvConfig(EnvConfig):
    fps: int = 20
    robot_type: str | None = None

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def sim_settings(self) -> dict:
        return {
            "type": "toy",
            "task": "reach",
            "fps": self.fps,
            "control": "delta",
            "semantics": "toy-delta-v1",
            "robot_type": self.robot_type,
        }
