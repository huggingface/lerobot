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

"""Adapt one simulated robot and its cameras to the ordinary Robot interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from lerobot.env_server.contracts import EnvDescriptor
from lerobot.envs.configs import check_served
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.sims.adapters import canonical_observation, check_step_info, env_attribute
from lerobot.sims.backend import describe_env
from lerobot.transport.wire.features import FeatureSpec, validate_array

from .configuration_sim import SimRobotConfig

if TYPE_CHECKING:
    from lerobot.env_server.client import RemoteEnv


@dataclass(frozen=True)
class EpisodeStatus:
    """Task outcome of the last simulator step, retained until the episode is reset."""

    success: bool
    terminated: bool
    truncated: bool
    step: int
    reward: float


class SimRobot(Robot):
    """Adapt one simulated robot in a task environment to LeRobot's Robot interface.

    The local or remote environment owns physics, the scene and the task. ``send_action`` advances it
    once; ``reset_episode`` resets the whole task environment, not just the robot's pose.
    Existing LeRobot loops own inference, processors, recording and pacing. Feature inspection may
    construct an environment before ``connect``; ``disconnect`` also releases that environment.
    """

    config_class = SimRobotConfig
    name = "sim"
    supports_command_hold = True

    def __init__(self, config: SimRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self._env: gym.Env | RemoteEnv | None = None
        self._descriptor: EnvDescriptor | None = None
        self._source_features: tuple[FeatureSpec, ...] = ()
        self._connected = False
        self._failed = False
        self._observation: RobotObservation = {}
        self._episode_status: EpisodeStatus | None = None
        self._task = ""

    @property
    def descriptor(self) -> EnvDescriptor:
        """Reuse admission metadata without creating a local execution client."""
        if self._descriptor is None:
            config = self.config.env
            assert config is not None
            try:
                if config.endpoint is None:
                    self._env = config.create_env(task_id=self.config.task_id)
                    sim = config.backend_config()
                    group = config.task or sim.task
                    descriptor = describe_env(sim, self._env, {group: [str(self.config.task_id)]})
                else:
                    from lerobot.env_server.client import EnvClient, RemoteEnv

                    client = EnvClient(
                        config.endpoint,
                        config.type,
                        self.config.timeout_s,
                        self.config.setup_timeout_s,
                        encoding=config.encoding,
                        jpeg_quality=config.jpeg_quality,
                    )
                    remote_group = config.task if config.task and "," not in config.task else None
                    self._env = RemoteEnv(client, remote_group, self.config.task_id)
                    descriptor = self._env.descriptor
                    check_served(descriptor.config, config.sim_settings(), config.endpoint)
                self._source_features = descriptor.features
                # Robot recording uses observation.images.<camera>, including a single camera.
                features = tuple(
                    replace(f, name="observation.images.image") if f.name == "observation.image" else f
                    for f in descriptor.features
                )
                self._descriptor = replace(descriptor, features=features)
                unsupported = [f.name for f in features if f.name != "observation.state" and f.kind != "rgb"]
                if unsupported:
                    raise ValueError(
                        f"SimRobot currently supports state and RGB observations only; unsupported: {unsupported}"
                    )
                self.robot_type = descriptor.robot_type or self.name
                if set(self.cameras).intersection(self._state_feature.names):
                    raise ValueError("Simulator camera and state component names overlap")
            except BaseException:
                self.disconnect()
                raise
        return self._descriptor

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
        # Failed execution still owns resources until disconnect.
        return self._connected

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
            _ = self.descriptor
            self._connected = True
            self.reset_episode(seed=self.config.seed)
        except BaseException:
            self.disconnect()
            raise

    def _require_env(self) -> gym.Env | RemoteEnv:
        if not self.is_connected or self._env is None:
            raise RuntimeError("Connect before using the sim robot")
        if self._failed:
            raise RuntimeError("Simulator execution failed; disconnect and reconnect before continuing")
        return self._env

    def _cache(
        self,
        obs: dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        step: int,
    ) -> None:
        check_step_info(info)
        values = canonical_observation(obs)
        if set(values) != {f.name for f in self._source_features}:
            raise ValueError("Simulator observation keys differ from declared features")
        observation: RobotObservation = {}
        for source, feature in zip(self._source_features, self.descriptor.features, strict=True):
            value = values[source.name]
            validate_array(value, source)
            if feature.name == "observation.state":
                observation.update(zip(feature.names, map(float, value), strict=True))
            elif feature.kind == "rgb":
                observation[feature.name.removeprefix("observation.images.")] = value.copy()
        if not math.isfinite(reward):
            raise ValueError("Simulator returned a non-finite reward")
        self._observation = observation
        self._episode_status = EpisodeStatus(
            bool(info.get("is_success", info.get("success", info.get("succeed", False)))),
            bool(terminated),
            bool(
                truncated
                or (
                    self.descriptor.max_episode_steps is not None
                    and step >= self.descriptor.max_episode_steps
                )
            ),
            step,
            float(reward),
        )
        assert self._env is not None
        self._task = str(
            info.get(
                "task_description",
                env_attribute(
                    self._env,
                    "task_description",
                    env_attribute(self._env, "task", next(iter(self.descriptor.tasks))),
                ),
            )
        )

    def get_observation(self) -> RobotObservation:
        self._require_env()
        # Do not expose mutable native buffers to processors or asynchronous consumers.
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._observation.items()
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the simulated robot and advance its environment by one control interval."""
        env = self._require_env()
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
        try:
            obs, reward, terminated, truncated, info = env.step(command)
            self._cache(obs, reward, terminated, truncated, info, status.step + 1)
        except BaseException:
            self._failed = True
            raise
        # Submitted controller command, not measured motion completion.
        return dict(zip(names, map(float, command), strict=True))

    def hold(self) -> None:
        """An explicitly stepped world is already frozen; do not submit another action."""
        self._require_env()

    def reset_episode(self, seed: int | None = None) -> None:
        """Reset the task environment, including the robot and objects, and refresh cached results.

        The calling workflow is responsible for resetting inference and stateful processors.
        """
        env = self._require_env()
        try:
            obs, info = env.reset(seed=seed)
            self._cache(obs, 0.0, False, False, info, 0)
        except BaseException:
            self._failed = True
            raise

    @property
    def task_description(self) -> str:
        """Task instruction supplied by the environment."""
        return self._task

    @property
    def episode_status(self) -> EpisodeStatus:
        self._require_env()
        assert self._episode_status is not None
        return self._episode_status

    def disconnect(self) -> None:
        env, self._env = self._env, None
        self._connected = False
        self._descriptor = None
        self._source_features = ()
        self._observation = {}
        self._episode_status = None
        self._task = ""
        self._failed = False
        if env is not None:
            env.close()

    def __del__(self) -> None:
        # Feature inspection can create an environment before connect().
        try:
            if getattr(self, "_env", None) is not None:
                self.disconnect()
        except Exception:  # nosec B110
            pass


def check_control_rate(robot: Robot, fps: float) -> None:
    """Keep the loop's action rate equal to one simulator control interval."""
    if isinstance(robot, SimRobot) and not math.isclose(fps, robot.descriptor.fps):
        raise ValueError(
            f"The control loop runs at {fps} fps but the simulator control rate is {robot.descriptor.fps} Hz"
        )
