#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Inference script for a pi0 model trained with UMI-style relative EE actions
on an OpenArm robot (single right arm, one wrist camera).

Training dataset layout:
  observation.images.cam0  [3, 720, 960]
  action                   [x, y, z, ax, ay, az, proximal, distal]  (shape 8)

The model uses ``derive_state_from_action=true``, so observation.state is
derived from the action column during training.  At inference the state must
be provided by the robot — this script uses FK to compute the current EE
pose and gripper position, which it exposes as ``observation.state``.

Pipeline:
  1. Read arm joints from robot → FK → observation.state [x,y,z,ax,ay,az,prox,dist]
  2. Read camera image → observation.images.cam0
  3. pi0 preprocessor (loaded from checkpoint):
     - DeriveStateFromActionStep: no-op at inference (state from robot)
     - RelativeActionsProcessorStep: caches current state
     - RelativeStateProcessorStep: buffers prev state, stacks [prev,cur],
       subtracts current → velocity info, flattens
     - NormalizerProcessorStep: normalizes
  4. pi0 predicts relative action chunk (30 steps)
  5. pi0 postprocessor: unnormalize, add cached state → absolute EE
  6. IK: absolute EE [x,y,z,ax,ay,az] → arm joint targets
  7. Gripper [proximal, distal] → gripper motor targets
  8. Send to robot

Usage:
    python evaluate.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.processor import RelativeStateProcessorStep
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ---------------------------------------------------------------------------
# Configuration — adapt these to your setup
# ---------------------------------------------------------------------------

FPS = 46
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "red cube"

HF_MODEL_ID = "pepijn223/grabette-umi-pi0"

# Latency compensation: skip this many predicted action steps to account for
# camera + inference + execution latency.  Formula: ceil(total_ms / (1000/FPS)).
# At 46 FPS (~22ms/step) with ~150ms total latency: ceil(150/22) ≈ 7.
# Start with 0 for a safe first test, then increase to match measured latency.
LATENCY_SKIP_STEPS = 0

URDF_PATH = "src/lerobot/robots/openarm_follower/urdf/openarm_bimanual_pybullet.urdf"
URDF_EE_FRAME = "openarm_right_ee_target"

IK_POSITION_WEIGHT = 1.0
IK_ORIENTATION_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Dataset features for inference
#
# The training dataset has only observation.images.cam0 and action.
# observation.state is derived from action during training
# (derive_state_from_action=true) but must be supplied by the robot at
# inference.  We define it here so build_dataset_frame can map FK output
# to the right feature.
# ---------------------------------------------------------------------------

DATASET_FEATURES: dict = {
    "observation.state": {
        "dtype": "float32",
        "shape": [8],
        "names": ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"],
    },
    "observation.images.cam0": {
        "dtype": "video",
        "shape": [3, 720, 960],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 720,
            "video.width": 960,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "action": {
        "dtype": "float32",
        "shape": [8],
        "names": ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"],
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


# ---------------------------------------------------------------------------
# FK / IK callables
# ---------------------------------------------------------------------------


class JointsToEE:
    """FK: raw robot observation → flat dict matching observation.state names.

    Arm joint positions → EE pose [x,y,z,ax,ay,az] via forward kinematics.
    Gripper motor positions → [proximal, distal].
    Camera images pass through unchanged.
    """

    def __init__(self, kinematics: RobotKinematics, arm_motor_names: list[str]):
        self.kin = kinematics
        self.arm = arm_motor_names

    def __call__(self, obs: RobotObservation) -> RobotObservation:
        q = np.array([float(obs[f"{m}.pos"]) for m in self.arm])
        t = self.kin.forward_kinematics(q)
        rot = Rotation.from_matrix(t[:3, :3]).as_rotvec()

        out: dict = {
            "x": float(t[0, 3]),
            "y": float(t[1, 3]),
            "z": float(t[2, 3]),
            "ax": float(rot[0]),
            "ay": float(rot[1]),
            "az": float(rot[2]),
            "proximal": float(obs["proximal.pos"]),
            "distal": float(obs["distal.pos"]),
        }
        for k, v in obs.items():
            if not k.endswith((".pos", ".vel", ".torque")):
                out[k] = v
        return out


class EEToJoints:
    """IK: policy action dict → motor position dict for the robot.

    Reads [x,y,z,ax,ay,az] from the action, runs IK for arm joint targets.
    Passes [proximal, distal] as direct gripper position commands.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        arm_motor_names: list[str],
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ):
        self.kin = kinematics
        self.arm = arm_motor_names
        self.pw = position_weight
        self.ow = orientation_weight
        self.q_curr: np.ndarray | None = None

    def __call__(self, args: tuple[RobotAction, RobotObservation]) -> RobotAction:
        action, obs = args

        q_raw = np.array([float(obs[f"{m}.pos"]) for m in self.arm])
        if self.q_curr is None:
            self.q_curr = q_raw

        t_des = np.eye(4)
        t_des[:3, :3] = Rotation.from_rotvec([action["ax"], action["ay"], action["az"]]).as_matrix()
        t_des[:3, 3] = [action["x"], action["y"], action["z"]]

        q_target = self.kin.inverse_kinematics(
            self.q_curr, t_des, position_weight=self.pw, orientation_weight=self.ow
        )
        self.q_curr = q_target

        out: dict = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(self.arm)}
        out["proximal.pos"] = float(action["proximal"])
        out["distal.pos"] = float(action["distal"])
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    camera_config = {
        "cam0": OpenCVCameraConfig(index_or_path=0, width=960, height=720, fps=FPS),
    }
    robot_config = OpenArmFollowerConfig(
        port="can0",
        id="right_openarm",
        side="right",
        cameras=camera_config,
        max_relative_target=8.0,
        gripper_port="/dev/ttyUSB0",
    )
    robot = OpenArmFollower(robot_config)

    policy = PI0Policy.from_pretrained(HF_MODEL_ID)
    policy.config.latency_skip_steps = LATENCY_SKIP_STEPS

    arm_motor_names = list(robot.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=URDF_EE_FRAME,
        joint_names=arm_motor_names,
    )

    fk = JointsToEE(kinematics, arm_motor_names)
    ik = EEToJoints(kinematics, arm_motor_names, IK_POSITION_WEIGHT, IK_ORIENTATION_WEIGHT)

    dataset = LeRobotDataset.create(
        repo_id="tmp/openarm_eval_scratch",
        fps=FPS,
        features=DATASET_FEATURES,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    relative_state_steps = [s for s in preprocessor.steps if isinstance(s, RelativeStateProcessorStep)]

    robot.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarm_umi_pi0_relative_ee_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        log_say("Starting policy execution")
        for step in relative_state_steps:
            step.reset()

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            robot_action_processor=ik,
            robot_observation_processor=fk,
        )
    finally:
        robot.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()
