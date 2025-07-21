# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("koch_screwdriver_follower")
@dataclass
class KochScrewdriverFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # No need to specify cameras here, this is just for typing.
    # You can set the cameras in your commands like this:
    # python -m lerobot.record \
    # --robot.type=koch_screwdriver_follower \
    # --robot.port=/dev/servo_5837053138 \
    # --robot.cameras="{ screwdriver: {type: opencv, index_or_path: /dev/video0, width: 800, height: 600, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, width: 800, height: 600, fps: 30}}"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    # See the [Hardware API Redesign PR](https://github.com/huggingface/lerobot/pull/777) for more details
    use_degrees: bool = False

    # Maximum current (raw Dynamixel units) allowed for the screwdriver motor.
    # Around 300 corresponds to ~0.8 A on an XL-330 which is plenty for M4 screws
    # yet well below the shutdown threshold.
    screwdriver_current_limit: int = 300

    # Ratio of the above limit at which the software clutch engages.
    # When |present_current| >= limit * clutch_ratio the follower will cut the
    # screwdriver velocity command to 0 and send haptic feedback to the leader.
    clutch_ratio: float = 0.5

    # Cool-down duration (seconds) during which any velocity command for the
    # screwdriver is forced to zero after the clutch engages.  This allows the
    # current to fall and prevents repeated brown-outs.  Adjust based on your
    # control-loop FPS (e.g. 1.0 s ≈ 30–60 frames).
    clutch_cooldown_s: float = 1.0
