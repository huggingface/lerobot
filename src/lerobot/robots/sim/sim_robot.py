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

"""One simulated environment driven through the Robot interface, locally or served."""

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.sims.types import EnvDescriptor, StepResult
from lerobot.utils.feature_spec import FeatureSpec

from .config_sim import SimRobotConfig

if TYPE_CHECKING:
    from lerobot.env_server.client import BatchedEnv
    from lerobot.sims.vector_env import VectorEnv


class SimRobot(Robot):
    """A simulated environment as a robot: swappable with hardware, and like it, with no episode controls.

    Each sent action advances the environment exactly once; when its episode ends, the next one starts, as
    Gymnasium's autoreset does. ``latest`` keeps the result of the last step, so a finished episode's reward,
    termination and success stay readable while observations come from the next episode. Its ``robot_type``
    is the simulated body the environment declares, else ``sim``; ``name`` stays ``sim``.
    """

    config_class = SimRobotConfig
    name = "sim"

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        if config.env is None:
            raise ValueError("A sim robot needs --robot.env.type")
        self.config = config
        self._env_config = config.env
        # The env's suite, when it names exactly one; otherwise the simulator's first.
        task = config.env.task
        self._task_group = task if task and "," not in task else None
        self._connected = False
        self.latest: StepResult | None = None
        self._env: VectorEnv | BatchedEnv | None = self._make_env()

    def _make_env(self) -> "VectorEnv | BatchedEnv":
        env = self._env_config.create_env(self._task_group, self.config.task_id)
        descriptor = env.descriptor
        self.robot_type = descriptor.robot_type or self.name
        # Holding is a no-op, see hold().
        self.supports_position_hold = False
        self.supports_command_hold = True
        state = next(
            (feature for feature in descriptor.features if feature.name == "observation.state"), None
        )
        if state is None or not state.names:
            env.close()
            raise ValueError("Robot face requires explicit state component names")
        self._state: FeatureSpec = state
        self._camera_keys: dict[str, str] = {
            feature.name: feature.name.removeprefix("observation.images.").removeprefix("observation.")
            for feature in descriptor.features
            if feature.kind == "rgb"
        }
        # Robot cameras use the recorder's names; evaluation envs keep their benchmark names.
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
        """Camera names keyed by observation feature; recording sizes its image writers from them."""
        return {f"observation.images.{camera}": camera for camera in self._camera_keys.values()}

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

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise RuntimeError("Sim robot is already connected")
        if self._env is None:
            self._env = self._make_env()
        try:
            self._env.reset()
        except BaseException:
            self.disconnect()
            raise
        self.latest = self._env.latest
        self._connected = True

    def get_observation(self) -> dict[str, Any]:
        if not self._connected or self._env is None:
            raise RuntimeError("Connect before reading the environment")
        result = self._env.latest
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
            raise ValueError(
                f"This sim robot takes exactly the actions {list(names)}; missing "
                f"{sorted(set(names) - set(action))}, unknown {sorted(set(action) - set(names))}. A "
                "teleoperator must send the simulated robot's action names, as for the real follower."
            )
        values = np.asarray([[action[name] for name in names]], dtype=np.float32)
        if not self._connected or self._env is None:
            raise RuntimeError("Connect before sending an action")
        self._env.step(values)
        self.latest = self._env.latest
        if self.latest.terminated[0] or self.latest.truncated[0]:
            # As Gymnasium's autoreset does, a finished episode starts the next one.
            self._env.reset()
        return {name: float(value) for name, value in zip(names, values[0], strict=True)}

    def hold(self) -> None:
        """Nothing to do: the simulation advances only on send_action()."""

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._connected = False


def check_control_rate(robot: Robot, fps: float, flag: str | None = None) -> None:
    """A sim robot ticks at its env's rate: a loop at another fps would play or record at the wrong speed.

    ``flag`` is the command-line option that sets ``fps``, named in the error.
    """
    if isinstance(robot, SimRobot) and not math.isclose(fps, robot.descriptor.fps):
        rate = robot.descriptor.fps
        hint = f": pass {flag}={rate:g}" if flag else ""
        raise ValueError(
            f"The control loop runs at {fps:g} fps but the simulator control rate is {rate:g} Hz{hint}"
        )
