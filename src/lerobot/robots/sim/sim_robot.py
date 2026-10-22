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

"""Expose one simulated robot through the ordinary Robot control and recording paths."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from lerobot.env_server.client import EnvClient
from lerobot.env_server.contracts import StepResult
from lerobot.env_server.in_process import InProcessEnvClient
from lerobot.envs.configs import check_served
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.transport.wire.features import FeatureSpec, validate_array

from .configuration_sim import SimRobotConfig


@dataclass(frozen=True)
class EpisodeStatus:
    """Outcome of the last simulator step, retained until the episode is reset."""

    success: bool
    terminated: bool
    truncated: bool
    step: int
    reward: float


class SimRobot(Robot):
    """Adapt one simulated robot in a task environment to LeRobot's Robot interface.

    The shared simulator backend owns observations and episode outcomes. Local calls run on the
    caller's thread; served calls use the same session on the server. Every action advances one
    control interval. Existing LeRobot loops own inference, processors, recording and pacing.
    """

    config_class = SimRobotConfig
    name = "sim"
    supports_command_hold = True

    def __init__(self, config: SimRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        env = config.env
        assert env is not None
        self._task_group = env.task if env.task and "," not in env.task else None
        self.latest: StepResult | None = None
        self.client: EnvClient | InProcessEnvClient
        if env.endpoint is None:
            self.client = InProcessEnvClient(env.backend_config())
        else:
            self.client = EnvClient(
                env.endpoint,
                env.type,
                config.timeout_s,
                config.setup_timeout_s,
                encoding=env.encoding,
                jpeg_quality=env.jpeg_quality,
            )
        try:
            descriptor = self.client.describe()
            if env.endpoint is not None:
                check_served(descriptor.config, env.sim_settings(), env.endpoint)
            self._source_features = descriptor.features
            # Robot datasets use observation.images.<camera>, including a single camera.
            self.descriptor = replace(
                descriptor,
                features=tuple(
                    replace(f, name="observation.images.image") if f.name == "observation.image" else f
                    for f in descriptor.features
                ),
            )
            unsupported = [
                f.name for f in self.descriptor.features if f.name != "observation.state" and f.kind != "rgb"
            ]
            if unsupported:
                raise ValueError(
                    f"SimRobot currently supports state and RGB observations only; unsupported: {unsupported}"
                )
            self.robot_type = descriptor.robot_type or self.name
            if set(self.cameras).intersection(self._state_feature.names):
                raise ValueError("Simulator camera and state component names overlap")
        finally:
            # Metadata probing never retains a native world or a transport connection.
            self.client.close()

    @property
    def _state_feature(self) -> FeatureSpec:
        state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
        if state is None or not state.names:
            raise ValueError("SimRobot requires named state components")
        return state

    @property
    def cameras(self) -> dict[str, tuple[int, ...]]:
        return {
            f.name.removeprefix("observation.images."): f.shape
            for f in self.descriptor.features
            if f.kind == "rgb"
        }

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        return {**dict.fromkeys(self._state_feature.names, float), **self.cameras}

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.descriptor.action_feature.names, float)

    @property
    def is_connected(self) -> bool:
        return bool(self.client.session)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("Sim robot is already connected")
        try:
            self.latest = self.client.open(
                1, "lockstep", self._task_group, self.config.task_id, seeds=[self.config.seed]
            )
            descriptor = self.client.descriptor
            assert descriptor is not None
            if (
                descriptor.features != self._source_features
                or descriptor.action_feature != self.descriptor.action_feature
                or descriptor.semantics != self.descriptor.semantics
                or descriptor.robot_type != self.descriptor.robot_type
            ):
                raise ValueError("Simulator schema changed since feature discovery")
            assert self.config.env is not None
            if self.config.env.endpoint is not None:
                check_served(descriptor.config, self.config.env.sim_settings(), self.config.env.endpoint)
        except BaseException:
            self.disconnect()
            raise

    def _result(self) -> StepResult:
        if not self.is_connected or self.latest is None:
            raise RuntimeError("Connect before using the sim robot")
        if self.client.failed:
            raise RuntimeError("Simulator execution failed; disconnect and reconnect before continuing")
        return self.latest

    def get_observation(self) -> RobotObservation:
        result = self._result()
        state = self._state_feature
        observation: RobotObservation = dict(
            zip(state.names, map(float, result.obs[state.name][0]), strict=True)
        )
        for source, feature in zip(self._source_features, self.descriptor.features, strict=True):
            if feature.kind == "rgb":
                image = result.obs[source.name][0]
                observation[feature.name.removeprefix("observation.images.")] = image.copy()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        """Submit one command and retain the paired observation and outcome."""
        status = self.episode_status
        if status.terminated or status.truncated:
            raise RuntimeError("Simulator episode ended; call reset_episode() before sending another action")
        names = self.descriptor.action_feature.names
        if set(action) != set(names):
            raise ValueError(
                f"This sim robot takes exactly the actions {list(names)}; missing "
                f"{sorted(set(names) - set(action))}, unknown {sorted(set(action) - set(names))}. "
                "Use the normal teleop/action processors to map the simulated robot's commands."
            )
        command = np.asarray([action[name] for name in names], dtype=np.float32)
        validate_array(command, self.descriptor.action_feature)
        self.latest = StepResult.from_dict(self.client.request("step", actions=command[None])["result"])
        return dict(zip(names, map(float, command), strict=True))

    def hold(self) -> None:
        """Leave the explicitly stepped world frozen without submitting another action."""
        self._result()

    def reset_episode(self, seed: int | None = None) -> None:
        """Reset the whole task; the workflow separately clears policy and processor state."""
        self._result()
        self.latest = StepResult.from_dict(self.client.request("reset", seeds=[seed])["result"])

    @property
    def task_description(self) -> str:
        """Task instruction supplied by the environment."""
        return self.latest.task[0] if self.latest is not None else ""

    @property
    def episode_status(self) -> EpisodeStatus:
        result = self._result()
        return EpisodeStatus(
            bool(result.is_success[0]),
            bool(result.terminated[0]),
            bool(result.truncated[0]),
            int(result.step[0]),
            float(result.reward[0]),
        )

    def disconnect(self) -> None:
        try:
            self.client.close()
        finally:
            self.latest = None


def check_control_rate(robot: Robot, fps: float) -> None:
    """Keep the loop's action rate equal to one simulator control interval."""
    if isinstance(robot, SimRobot) and not math.isclose(fps, robot.descriptor.fps):
        raise ValueError(
            f"The control loop runs at {fps} fps but the simulator control rate is {robot.descriptor.fps} Hz"
        )
