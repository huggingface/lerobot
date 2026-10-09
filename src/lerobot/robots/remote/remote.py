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
from lerobot.env_server.contracts import ExecutionFeedback, StepResult
from lerobot.env_server.profiles import EvalProfile
from lerobot.robots.robot import Robot

from .configuration_remote import RemoteRobotConfig
from .world import EpisodeStatus, World


class RemoteRobot(Robot):
    config_class = RemoteRobotConfig
    name = "remote"

    def __init__(self, config: RemoteRobotConfig):
        super().__init__(config)
        self.config = config
        self.client = EnvClient(config.endpoint, config.deployment, config.timeout_s)
        try:
            self.descriptor = self.client.describe()
        except BaseException:
            self.client.close()
            raise
        try:
            self.profile = EvalProfile.load(config.profile) if config.profile else None
            if self.profile:
                self.profile.validate_descriptor(self.descriptor)
            self.supports_position_hold = self.descriptor.hold == "position"
            self.supports_command_hold = self.descriptor.hold == "command"
            self.latest: StepResult | None = None
            state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
            if state is None or not state.names:
                raise ValueError("Robot face requires explicit state component names")
            self._state = state
            self._camera_keys: dict[str, str] = {}
            for feature in self.descriptor.features:
                if feature.kind == "rgb":
                    canonical = (
                        self.profile.feature_mapping.get(feature.name, feature.name)
                        if self.profile
                        else feature.name
                    )
                    camera = canonical.removeprefix("observation.images.").removeprefix("observation.")
                    self._camera_keys[feature.name] = camera
        except BaseException:
            self.client.close()
            raise

    @property
    def world(self) -> World | None:
        return self if {"reset", "snapshot"}.issubset(self.descriptor.operations) else None

    @property
    def execution_feedback(self) -> ExecutionFeedback | None:
        return self.client.execution

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        return {
            **dict.fromkeys(self._state.names, float),
            **{self._camera_keys[f.name]: f.shape for f in self.descriptor.features if f.kind == "rgb"},
        }

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.policy_action_names, float)

    @property
    def policy_action_names(self) -> tuple[str, ...]:
        return (
            self.profile.policy_action_names(self.descriptor)
            if self.profile
            else self.descriptor.action_feature.names
        )

    @property
    def is_connected(self) -> bool:
        return bool(self.client.session) and not self.client.failed

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
        names = self.policy_action_names
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

    def reset_world(self, task: str | None = None, seed: int | None = None) -> None:
        if task is not None and task != (self.config.task or next(iter(self.descriptor.tasks))):
            raise ValueError("Changing task requires opening a new robot session")
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

    @property
    def sim_time(self) -> float:
        return self.latest.sim_time if self.latest is not None else 0.0

    def render(self) -> np.ndarray:
        return self.client.request("render")["frames"][0]

    def disconnect(self) -> None:
        self.client.close()
