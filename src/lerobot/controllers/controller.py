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

"""The interface of the controllers a ``ControlledRobot`` runs."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotController(Protocol):
    """A controller driving a robot's joints from its own thread, at its own rate.

    Each tick, ``ControlledRobot`` hands the controller the robot's latest low-level state plus a
    snapshot of the last command, and writes the joint targets it returns (G1: ``<joint>.q``).

    Controllers may also expose any of the following, which ``ControlledRobot`` picks up when present:

    - ``action_ft``: the command features callers send (SONIC's 64-D token, GR00T's joystick).
    - ``passthrough``: ``True`` when callers keep commanding the joints the controller leaves to them
      (the G1's arms under a locomotion controller); they then join the action features.
    - ``observation_ft`` / ``observation_state()``: a proprioceptive state that replaces the robot's
      own (SONIC echoes its last token), and its current values.
    - ``kp`` / ``kd``: PD gains written with the targets, and the robot's gains while it runs.
    - ``default_angles``: the home pose the robot eases into on connect and reset.
    - ``supports_position_hold``: ``True`` when re-sending the last command holds the robot (SONIC's
      token keeps the decoder balancing; joystick velocities would keep it walking).
    """

    control_dt: float
    """Control period in seconds; sets the rate of the controller thread."""

    def run_step(self, action: dict, lowstate: Any) -> dict:
        """Map one low-level state plus the last command into absolute joint targets."""
        ...

    def reset(self) -> None:
        """Drop per-episode state, e.g. history buffers and held commands."""
        ...
