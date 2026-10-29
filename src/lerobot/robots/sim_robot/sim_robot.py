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

"""A sim env behind the ``Robot`` interface, run locally or on an env server."""

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.sim.types import EnvDescriptor, StepResult
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.feature_spec import FeatureSpec

from ..robot import Robot
from .config_sim_robot import SimRobotConfig

if TYPE_CHECKING:
    from lerobot.env_server.client import RemoteVectorEnv
    from lerobot.sim.vector_env import LocalVectorEnv


class SimRobot(Robot):
    """A simulated environment as a robot: swappable with hardware.

    Each action advances the environment by one step. When an episode ends, the robot stays in its final state
    and ignores actions until ``start_episode`` is called; ``latest`` keeps the final reward, termination and
    success. ``robot_type`` is the body the environment declares, or ``sim_robot`` if it declares none.
    """

    config_class = SimRobotConfig
    name = "sim_robot"
    supports_command_hold = True

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        self.config = config
        # Use the env's suite if ``task`` names exactly one; otherwise the sim's first suite.
        task = config.env.task
        self._task_group = task if task and "," not in task else None
        self._connected = False
        self.latest: StepResult | None = None
        # Build the env here, not in connect(): scripts read the features and descriptor before connecting.
        self._env: LocalVectorEnv | RemoteVectorEnv = self._make_env()

    def _make_env(self) -> "LocalVectorEnv | RemoteVectorEnv":
        env = self.config.env.create_env(self._task_group, self.config.task_id)
        descriptor = env.descriptor
        self.robot_type = descriptor.robot_type or self.name
        state = next(
            (feature for feature in descriptor.features if feature.name == "observation.state"), None
        )
        if state is None or not state.names:
            env.close()
            raise ValueError("A sim robot needs an observation.state feature with component names")
        self._state: FeatureSpec = state
        self._camera_keys: dict[str, str] = {
            feature.name: feature.name.removeprefix("observation.images.").removeprefix("observation.")
            for feature in descriptor.features
            if feature.kind == "rgb"
        }
        # Robot cameras use dataset names (observation.images.<camera>); eval envs keep benchmark names.
        self.descriptor: EnvDescriptor = replace(
            descriptor,
            features=tuple(
                replace(feature, name=f"observation.images.{self._camera_keys[feature.name]}")
                if feature.kind == "rgb"
                else feature
                for feature in descriptor.features
            ),
        )
        return env

    @property
    def cameras(self) -> dict[str, str]:
        """Map each camera name to its observation feature, keyed like a real robot's cameras."""
        return {camera: f"observation.images.{camera}" for camera in self._camera_keys.values()}

    def feature_names(self, rename_map: dict[str, str]) -> dict[str, str]:
        """Map each descriptor feature to its policy key with ``rename_map``, as for any robot."""
        return {f.name: rename_map.get(f.name, f.name) for f in self.descriptor.features}

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        return {
            **dict.fromkeys(self._state.names, float),
            **{
                f.name.removeprefix("observation.images."): f.shape
                for f in self.descriptor.features
                if f.kind == "rgb"
            },
        }

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.descriptor.action_feature.names, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self._env.closed:
            self._env = self._make_env()
        # Start the first episode so the robot has a state to observe.
        self._connected = True
        try:
            self.start_episode()
        except BaseException:
            self.disconnect()
            raise

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        result = self._env.latest
        return {
            **{
                key: float(value)
                for key, value in zip(self._state.names, result.obs[self._state.name][0], strict=True)
            },
            **{camera: result.obs[name][0] for name, camera in self._camera_keys.items()},
        }

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        names = self.descriptor.action_feature.names
        if set(action) != set(names):
            raise ValueError(
                f"This sim robot takes exactly the actions {list(names)}; missing "
                f"{sorted(set(names) - set(action))}, unknown {sorted(set(action) - set(names))}. A "
                "teleoperator must send the simulated robot's action names, as for the real follower."
            )
        values = np.asarray([[action[name] for name in names]], dtype=np.float32)
        if not self.episode_ended:
            self._env.step(values)
            self.latest = self._env.latest
        return {name: float(value) for name, value in zip(names, values[0], strict=True)}

    @check_if_not_connected
    def start_episode(self, seed: int | None = None) -> None:
        self._env.reset(seed=seed)
        self.latest = self._env.latest

    @property
    def episode_ended(self) -> bool:
        return self.latest is not None and bool(self.latest.terminated[0] or self.latest.truncated[0])

    def hold(self) -> None:
        """Nothing to do: the simulation advances only on send_action()."""

    def disconnect(self) -> None:
        # Not decorated: it must also close an env that was built but never connected.
        self._env.close()
        self._connected = False


def check_control_rate(robot: Robot, fps: float, flag: str | None = None, multiplier: int = 1) -> None:
    """Raise unless ``fps * multiplier`` commands per second match a sim robot's control rate."""
    if isinstance(robot, SimRobot) and not math.isclose(fps * multiplier, robot.descriptor.fps):
        rate = robot.descriptor.fps
        hint = f": pass {flag}={rate / multiplier:g}" if flag else ""
        raise ValueError(
            f"The control loop sends {fps * multiplier:g} commands per second but the simulator control rate "
            f"is {rate:g} Hz{hint}"
        )


def start_episode(robot: Robot) -> None:
    """Start a new episode. A sim robot resets its world; a real robot's scene is reset by hand."""
    if isinstance(robot, SimRobot):
        robot.start_episode()


def episode_ended(robot: Robot) -> bool:
    """Return whether a sim robot's episode just ended; always False for a real robot."""
    return isinstance(robot, SimRobot) and robot.episode_ended
