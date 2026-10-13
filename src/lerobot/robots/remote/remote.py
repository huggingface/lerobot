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

"""Realtime Robot face of a simulator session."""

from typing import Any

import numpy as np

from lerobot.env_server.client import EnvClient
from lerobot.env_server.contracts import StepResult
from lerobot.robots.robot import Robot

from .configuration_remote import RemoteRobotConfig
from .world import EpisodeStatus


class RemoteRobot(Robot):
    config_class = RemoteRobotConfig
    name = "remote"

    def __init__(self, config: RemoteRobotConfig):
        super().__init__(config)
        self.config = config
        self.client = EnvClient(config.endpoint, config.deployment, config.timeout_s, config.setup_timeout_s)
        try:
            self.descriptor = self.client.describe()
            self.supports_position_hold = self.descriptor.hold == "position"
            self.supports_command_hold = self.descriptor.hold == "command"
            self.latest: StepResult | None = None
            state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
            if state is None or not state.names:
                raise ValueError("Robot face requires explicit state component names")
            self._state = state
            self._camera_keys = {
                f.name: f.name.removeprefix("observation.images.").removeprefix("observation.")
                for f in self.descriptor.features
                if f.kind == "rgb"
            }
        except BaseException:
            self.client.close()
            raise

    @property
    def cameras(self) -> dict[str, str]:
        """Simulator camera features keyed by descriptor name; sized like a real robot's cameras."""
        return self._camera_keys

    def feature_names(self, rename_map: dict[str, str]) -> dict[str, str]:
        """Map each descriptor feature to its policy key with ``rename_map``, as for any robot."""
        return {f.name: rename_map.get(f.name, f.name) for f in self.descriptor.features}

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        return {
            **dict.fromkeys(self._state.names, float),
            **{self._camera_keys[f.name]: f.shape for f in self.descriptor.features if f.kind == "rgb"},
        }

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.descriptor.action_feature.names, float)

    @property
    def is_connected(self) -> bool:
        # A failed session is still owned: teardown must reach disconnect() to release it.
        return bool(self.client.session)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.client.session:
            raise RuntimeError("Remote robot is already connected")
        self.latest = self.client.open(1, "realtime", self.config.task, self.config.task_id)

    def _snapshot(self) -> StepResult:
        self.latest = StepResult.from_dict(self.client.request("snapshot")["result"])
        return self.latest

    def get_observation(self) -> dict[str, Any]:
        result = self._snapshot()
        return {
            **{
                key: float(value)
                for key, value in zip(self._state.names, result.obs[self._state.name][0], strict=True)
            },
            **{camera: result.obs[name][0] for name, camera in self._camera_keys.items()},
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        names = self.descriptor.action_feature.names
        if set(action) != set(names):
            raise ValueError("Remote action must contain exactly the declared components")
        values = np.asarray([[action[name] for name in names]], dtype=np.float32)
        body = self.client.request("apply", actions=values)
        self.latest = StepResult.from_dict(body["result"])
        assert self.client.execution is not None
        applied = self.client.execution.applied[0]
        return {name: float(value) for name, value in zip(names, applied, strict=True)}

    def hold(self) -> None:
        self.client.request("hold")

    def reset_world(self, seed: int | None = None) -> None:
        self.latest = StepResult.from_dict(self.client.request("reset", seeds=[seed])["result"])

    @property
    def task_description(self) -> str:
        return self.latest.task[0] if self.latest is not None else ""

    @property
    def episode_status(self) -> EpisodeStatus:
        result = self.latest
        if result is None:
            raise RuntimeError("Connect before reading world status")
        return EpisodeStatus(
            bool(result.is_success[0]),
            bool(result.terminated[0]),
            bool(result.truncated[0]),
            int(result.step[0]),
            float(result.reward[0]),
        )

    def disconnect(self) -> None:
        self.client.close()
