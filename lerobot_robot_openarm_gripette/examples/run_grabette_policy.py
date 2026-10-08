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

"""Run a Grabette Cartesian policy on the OpenArm (right arm) + Gripette.

Policies trained on handheld Grabette data with `relative_action_mode="pose"` and no reference key
(e.g. CarolinePascal/grabette-pick-and-place-texture-act) predict chunks of `oak_l` camera poses
relative to the chunk's first pose, plus the gripper's grasp projection (strategy, closure). Each
chunk is anchored at the arm's current `oak_l` pose, turned into joint targets with placo IK on
Pollen's OpenArm + Gripette URDF, and streamed at the dataset frame rate.

python examples/run_grabette_policy.py --port can0 --gripette 172.18.7.107:50051 --dry_run
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from gripette.grasp_projection import GraspProjection
from huggingface_hub import snapshot_download
from openarm_gripette_model import OPENARM_RIGHT_DIR

from lerobot.configs import PreTrainedConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot_robot_openarm_gripette import OpenArmGripetteFollower, OpenArmGripetteFollowerConfig
from lerobot_robot_openarm_gripette.manual_start import prepare_manual_start
from lerobot_robot_openarm_gripette.prediction_logger import PredictionLogger

URDF_JOINTS = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]
ROBOT_JOINTS = [f"joint_{i}" for i in range(1, 8)]
CAMERA = "policy_cam"


def load_policy(args: argparse.Namespace, device: str):
    path = Path(args.policy)
    if not path.is_dir():
        subfolder = f"checkpoints/{args.checkpoint}/pretrained_model" if args.checkpoint else ""
        root = snapshot_download(
            args.policy, revision=args.revision, allow_patterns=f"{subfolder}/*" if subfolder else None
        )
        path = Path(root) / subfolder
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if not (
        getattr(cfg, "relative_action_mode", None) == "pose"
        and cfg.relative_reference_key is None
        and cfg.relative_pose_rotation_format == "rot6d"
    ):
        raise ValueError("Expected a rot6d pose-mode relative-action policy without a reference key.")
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return cfg, policy, preprocessor, postprocessor


def r6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """First two rotation-matrix rows -> rotation matrix (Gram-Schmidt), as in LeRobot's pose mode."""
    b1 = r6d[:3] / np.linalg.norm(r6d[:3])
    b2 = r6d[3:] - b1.dot(r6d[3:]) * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", default="CarolinePascal/grabette-pick-and-place-texture-act")
    parser.add_argument(
        "--checkpoint", default="070000", help="Hub checkpoint step; empty for the repo root."
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--port", default="can0")
    parser.add_argument("--gripette", default="172.18.7.107:50051")
    parser.add_argument("--id", default="my_follower")
    parser.add_argument("--device", default=None, help="Defaults to cuda > mps > cpu.")
    parser.add_argument("--task", default="")
    parser.add_argument("--fps", type=float, default=50.0, help="Dataset frame rate: one action per tick.")
    parser.add_argument("--n_action_steps", type=int, default=None, help="Defaults to the policy's.")
    parser.add_argument("--duration", type=float, default=60.0, help="Episode length in seconds.")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("outputs/openarm_gripette_policy"),
        help="Directory for timestamped policy prediction JSONL logs.",
    )
    parser.add_argument(
        "--joint_signs",
        type=float,
        nargs=7,
        default=[1.0] * 7,
        help="Robot joint = sign * URDF joint, for joint_1..joint_7.",
    )
    parser.add_argument(
        "--max_joint_step",
        type=float,
        default=5.0,
        help="Abort if IK moves any joint more than this many degrees in one tick (IK branch flip).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Run the policy and IK without moving.")
    args = parser.parse_args()

    device = args.device or auto_select_torch_device().type
    cfg, policy, preprocessor, postprocessor = load_policy(args, device)
    names = [n.lower() for n in cfg.action_feature_names]
    pos_idx = [names.index(n) for n in cfg.relative_pose_position_names]
    rot_idx = [names.index(n) for n in cfg.relative_pose_rotation_names]
    strategy_idx, closure_idx = names.index("strategy"), names.index("closure")
    (image_key, image_feature), *others = cfg.image_features.items()
    if others:
        raise ValueError(f"Expected a single camera input, got {list(cfg.image_features)}")
    n_exec = args.n_action_steps or cfg.n_action_steps

    kin = RobotKinematics(str(OPENARM_RIGHT_DIR), target_frame_name="oak_l", joint_names=URDF_JOINTS)
    for dof in ("proximal", "distal", "r_wrist_roll_mimic"):
        kin.solver.mask_dof(dof)
    kin.solver.enable_joint_limits(True)
    kin.solver.add_regularization_task(1e-4)
    projection = GraspProjection()
    signs = np.array(args.joint_signs)

    robot = OpenArmGripetteFollower(
        OpenArmGripetteFollowerConfig(
            id=args.id,
            arm_config=OpenArmFollowerConfigBase(port=args.port, side="right", max_relative_target=8.0),
            gripette_addr=args.gripette,
            gripper_camera_name=CAMERA,
            gripper_camera_height=image_feature.shape[1],
            gripper_camera_width=image_feature.shape[2],
        )
    )
    logger = PredictionLogger(args.log_dir)
    print(f"Prediction log: {logger.path}")
    connected = False
    try:
        robot.connect()
        connected = True
        q_cmd = prepare_manual_start(robot, signs)
        held = robot.arm.get_observation()
        q_cmd = signs * np.array([held[f"{joint}.pos"] for joint in ROBOT_JOINTS])
        print("Held joints (URDF deg):", np.round(q_cmd, 1))
        print("Held oak_l position (m):", np.round(kin.forward_kinematics(q_cmd)[:3, 3], 3))
        input(f"{'DRY RUN: the arm will not move. ' if args.dry_run else ''}Press ENTER to start...")

        logger.write(
            {
                "type": "metadata",
                "policy": args.policy,
                "checkpoint": args.checkpoint,
                "device": device,
                "fps": args.fps,
                "dry_run": args.dry_run,
                "n_action_steps": n_exec,
                "initial_joints_deg": dict(zip(ROBOT_JOINTS, q_cmd, strict=True)),
            }
        )
        run_started = time.perf_counter()
        chunk_index = 0
        t_end = time.perf_counter() + args.duration
        while time.perf_counter() < t_end:
            obs = robot.get_observation()
            strategy, closure = projection.encode(obs["proximal.pos"], obs["distal.pos"])
            frame = {
                image_key: obs[CAMERA],
                "observation.state": np.array([strategy, closure], dtype=np.float32),
            }
            with torch.inference_mode():
                batch = preprocessor(
                    prepare_observation_for_inference(frame, torch.device(device), args.task, robot.name)
                )
                chunk = postprocessor(policy.predict_action_chunk(batch))[0].cpu().numpy()

            poses = np.tile(np.eye(4), (len(chunk), 1, 1))
            for pose, action in zip(poses, chunk, strict=True):
                pose[:3, :3] = r6d_to_matrix(action[rot_idx])
                pose[:3, 3] = action[pos_idx]
            # The first predicted pose is only approximately the identity: anchoring on its inverse
            # chains the chunk's steps from the arm's current pose without a jump on the first tick.
            anchor = kin.forward_kinematics(q_cmd) @ np.linalg.inv(poses[0])
            q_chunk_start = q_cmd
            for action_index, (action, pose) in enumerate(zip(chunk[:n_exec], poses[:n_exec], strict=True)):
                tick = time.perf_counter()
                q_target = kin.inverse_kinematics(
                    q_cmd, anchor @ pose, position_weight=100.0, orientation_weight=10.0, max_iters=30
                )
                jump = np.abs(q_target - q_cmd)
                if jump.max() > args.max_joint_step:
                    raise RuntimeError(
                        f"IK moved {ROBOT_JOINTS[jump.argmax()]} by {jump.max():.1f} deg in one tick, aborting."
                    )
                q_cmd = q_target
                proximal, distal = projection.decode(action[strategy_idx], action[closure_idx])
                executed = False
                try:
                    if not args.dry_run:
                        robot.send_action(
                            {
                                **{
                                    f"{j}.pos": s * q
                                    for j, s, q in zip(ROBOT_JOINTS, signs, q_cmd, strict=True)
                                },
                                "proximal.pos": proximal,
                                "distal.pos": distal,
                            }
                        )
                        executed = True
                finally:
                    logger.write(
                        {
                            "type": "prediction",
                            "elapsed_s": time.perf_counter() - run_started,
                            "chunk_index": chunk_index,
                            "action_index": action_index,
                            "raw": dict(zip(cfg.action_feature_names, action, strict=True)),
                            "joint_targets_deg": dict(zip(ROBOT_JOINTS, q_cmd, strict=True)),
                            "gripper_targets_rad": {
                                "proximal": proximal,
                                "distal": distal,
                            },
                            "executed": executed,
                        }
                    )
                precise_sleep(max(0.0, 1 / args.fps - (time.perf_counter() - tick)))

            moved = 1000 * (
                kin.forward_kinematics(q_cmd)[:3, 3] - kin.forward_kinematics(q_chunk_start)[:3, 3]
            )
            print(
                f"chunk: oak_l moved {np.round(moved, 1)} mm, "
                f"max joint change {np.abs(q_cmd - q_chunk_start).max():.1f} deg, "
                f"gripper strategy={action[strategy_idx]:.2f} closure={action[closure_idx]:.2f}"
            )
            chunk_index += 1
            if args.dry_run:
                q_cmd = signs * np.array([robot.get_observation()[f"{j}.pos"] for j in ROBOT_JOINTS])
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if connected:
                input("Support the arm, then press ENTER to disable torque and disconnect...")
                robot.disconnect()
        finally:
            logger.close()
            print(f"Prediction log saved: {logger.path}")


if __name__ == "__main__":
    main()
