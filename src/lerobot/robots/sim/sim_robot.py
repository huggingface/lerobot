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

"""One simulated world driven through the Robot interface, in this process or served."""

import math
from typing import Any

import numpy as np

from lerobot.env_server.client import EnvClient
from lerobot.env_server.contracts import StepResult
from lerobot.env_server.in_process import InProcessEnvClient
from lerobot.envs.configs import check_served
from lerobot.robots.robot import Robot

from .configuration_sim import SimRobotConfig
from .world import EpisodeStatus


class SimRobot(Robot):
    """A simulated world as a robot: swappable with hardware, plus the ``World`` episode controls.

    The world runs in this process, or behind ``lerobot-env-server`` when ``env.endpoint`` is set; both run
    the same session code. In lockstep, each sent action advances the world exactly one step. Its
    ``robot_type`` is the simulated body the env declares, else ``sim``; ``name`` stays ``sim``.
    """

    config_class = SimRobotConfig
    name = "sim"

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        self.config = config
        env = config.env
        assert env is not None
        # The env's suite, when it names exactly one; otherwise the simulator's first.
        self._task_group = env.task if env.task and "," not in env.task else None
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
            self.descriptor = self.client.describe()
            self.robot_type = self.descriptor.robot_type or self.name
            if env.endpoint is not None:
                check_served(self.descriptor.config, env.sim_settings(), env.endpoint)
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
            raise RuntimeError("Sim robot is already connected")
        self.latest = self.client.open(1, self.config.clock, self._task_group, self.config.task_id)

    @property
    def lockstep(self) -> bool:
        return self.config.clock == "lockstep"

    def _snapshot(self) -> StepResult:
        if self.lockstep and self.latest is not None:
            # The world only moves on send_action, which already returned its observation.
            return self.latest
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
            raise ValueError(
                f"This sim robot takes exactly the actions {list(names)}; missing "
                f"{sorted(set(names) - set(action))}, unknown {sorted(set(action) - set(names))}. A "
                "teleoperator must send the simulated robot's action names, as for the real follower."
            )
        values = np.asarray([[action[name] for name in names]], dtype=np.float32)
        if self.lockstep:
            self.latest = StepResult.from_dict(self.client.request("step", actions=values)["result"])
            return {name: float(value) for name, value in zip(names, values[0], strict=True)}
        body = self.client.request("apply", actions=values)
        self.latest = StepResult.from_dict(body["result"])
        assert self.client.execution is not None
        applied = self.client.execution.applied[0]
        return {name: float(value) for name, value in zip(names, applied, strict=True)}

    def hold(self) -> None:
        # A lockstep world stays where it is until the next action.
        if not self.lockstep:
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


def check_control_rate(robot: Robot, fps: float) -> None:
    """A sim robot ticks at its env's rate: a loop at another fps would play or record at the wrong speed."""
    if isinstance(robot, SimRobot) and not math.isclose(fps, robot.descriptor.fps):
        raise ValueError(
            f"The control loop runs at {fps} fps but the simulator control rate is {robot.descriptor.fps} Hz"
        )
