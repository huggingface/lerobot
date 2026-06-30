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


@dataclass
class SimLiftSuccessConfig:
    """Episode-success criterion read from privileged sim state.

    Two criteria, selected by ``criterion``:

    - ``"lift"`` (default) — the "Lift" criterion used by robosuite/LIBERO-style
      benchmarks: success is the tracked body's z position rising ``height_m``
      above where it rested at connect time, which only happens while it's
      grasped and held (resting contact alone can't raise it).
    - ``"place_in_box"`` — pick-and-place: success once the tracked body comes
      to rest *inside* the ``box_body_name`` body's cavity (within its geoms'
      horizontal footprint and below their top rim), moving slower than
      ``settle_speed_mps`` so a body merely passing over/through isn't counted.
    """

    # Name of the MJCF body to track (must have a freejoint, e.g. `scene_cube.xml`'s "cube").
    body_name: str = "cube"
    # Which criterion to score: "lift" or "place_in_box".
    criterion: str = "lift"
    # "lift": how far above its resting height (meters) counts as "lifted".
    height_m: float = 0.05
    # "place_in_box": MJCF body whose cavity the tracked body must end up in.
    box_body_name: str = "box"
    # "place_in_box": max tracked-body speed (m/s) to count as settled (not just
    # passing through the box volume).
    settle_speed_mps: float = 0.05


@RobotConfig.register_subclass("sim_so101")
@dataclass
class SimSO101Config(RobotConfig):
    """A MuJoCo-backed stand-in for the SO-101 follower.

    Implements the same ``Robot`` contract as ``so101_follower`` but drives a
    MuJoCo simulation instead of Feetech motors, so the async RTC rollout
    pipeline (``lerobot-rollout --inference.type=rtc``) can run without hardware.
    """

    # Path to the MuJoCo scene XML. Use the Menagerie SO-101
    # (``robotstudio_so101``) ``scene.xml``, extended with one ``<camera>``
    # element per entry in ``cameras`` beyond its built-in ``wrist_cam``.
    mjcf_path: str

    # Offscreen-rendered observation cameras. Key -> spec.
    cameras: dict[str, SimCameraConfig] = field(default_factory=dict)

    # Control rate of the rollout loop. ``send_action`` advances the sim by
    # ~1/control_fps of simulated time. Keep equal to ``lerobot-rollout --fps``.
    control_fps: float = 30.0

    # so101 motor name -> MuJoCo actuator/joint name.
    # Identity by default: the Menagerie SO-101 model's joint/actuator names
    # already match the so101_follower motor names. Override only if using a
    # differently-named MJCF (e.g. the older SO-ARM100 model).
    joint_map: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "shoulder_pan",
            "shoulder_lift": "shoulder_lift",
            "elbow_flex": "elbow_flex",
            "wrist_flex": "wrist_flex",
            "wrist_roll": "wrist_roll",
            "gripper": "gripper",
        }
    )

    # Body joints are reported/commanded in degrees, matching so101 use_degrees=True.
    use_degrees: bool = True

    # Optional task-success check, queried by the "eval" rollout strategy via
    # `check_success()`. Read directly off privileged sim state (not exposed
    # through `get_observation`), so it has no effect on what the policy sees.
    # None disables success tracking (e.g. scenes without a trackable object).
    success: SimLiftSuccessConfig | None = None

    # Conveyor belt speed in m/s, set once at connect() time (constant for the whole
    # rollout — not a per-step control). Applied as the ctrl of an MJCF actuator named
    # "belt_motor" if the MJCF has one (e.g. scene_cube.xml's belt); silently has no
    # effect on scenes without a belt. 0 by default so existing non-belt behavior is
    # unchanged unless explicitly set.
    belt_speed: float = 0.0

    # Distance in meters from the robot base (origin) to the belt's near (robot-side)
    # edge. connect() slides the whole pick-and-place layout (conveyor frame + moving
    # surface + drop-off box + the cube's start position) forward/back to honor it,
    # keeping the robot -> belt -> box arrangement intact (the box follows so it stays
    # just beyond the far edge). Default 0.14 reproduces the bundled scene_cube.xml
    # exactly. No effect on scenes without a "belt" body. Note the home keyframe's arm
    # pose is tuned for the default; very different distances may not frame the cube in
    # wrist_cam at start, and the cube must stay within the arm's ~0.40 m reach.
    belt_distance: float = 0.14
