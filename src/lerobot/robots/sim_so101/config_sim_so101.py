#!/usr/bin/env python

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

from dataclasses import dataclass, field

from ..config import RobotConfig


@dataclass
class SimCameraConfig:
    """A camera rendered offscreen from the MuJoCo scene.

    ``mujoco_name`` must match a ``<camera name=...>`` element in the scene XML.
    The dict key under which this config is stored becomes the observation key
    (e.g. ``camera1``) and must match the image keys the policy was trained with.
    """

    mujoco_name: str
    width: int = 640
    height: int = 480
    # Present only so RobotConfig.__post_init__ (which validates width/height/fps
    # on every camera) passes; the sim renders synchronously, so this is nominal.
    fps: int = 30


@RobotConfig.register_subclass("sim_so101")
@dataclass
class SimSO101Config(RobotConfig):
    """A MuJoCo-backed stand-in for the SO-101 follower.

    Implements the same ``Robot`` contract as ``so101_follower`` but drives a
    MuJoCo simulation instead of Feetech motors, so the async RTC rollout
    pipeline (``lerobot-rollout --inference.type=rtc``) can run without hardware.
    """

    # Path to the MuJoCo scene XML. Use the Menagerie SO-ARM100 ``scene.xml``,
    # extended with one ``<camera>`` element per entry in ``cameras``.
    mjcf_path: str

    # Offscreen-rendered observation cameras. Key -> spec.
    cameras: dict[str, SimCameraConfig] = field(default_factory=dict)

    # Control rate of the rollout loop. ``send_action`` advances the sim by
    # ~1/control_fps of simulated time. Keep equal to ``lerobot-rollout --fps``.
    control_fps: float = 30.0

    # so101 motor name -> MuJoCo actuator/joint name.
    # Defaults match the Menagerie SO-ARM100 model.
    joint_map: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "Rotation",
            "shoulder_lift": "Pitch",
            "elbow_flex": "Elbow",
            "wrist_flex": "Wrist_Pitch",
            "wrist_roll": "Wrist_Roll",
            "gripper": "Jaw",
        }
    )

    # Body joints are reported/commanded in degrees, matching so101 use_degrees=True.
    use_degrees: bool = True
