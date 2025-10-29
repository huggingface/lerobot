import argparse
import os
import shutil

import numpy as np

# RLBench
from rlbench import CameraConfig, Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
from rlbench.utils import name_to_task_class
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# You can define how end-effector actions and rotations are represented.
# The action represents the joint positions (joint1...joint7, gripper_open)
# or end-effector pose (position, rotation, gripper_open), absolute or relative (delta from current pose).
# The rotation can be represented as either Euler angles (3 values) or quaternions (4 values).
EULER_EEF = "euler"  # Actions have 7 values: [x, y, z, roll, pitch, yaw, gripper_state]
QUAT_EEF = "quat"  # Actions have 8 values: [x, y, z, qx, qy, qz, qw, gripper_state]
JOINTS = "joints"  # Actions have 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper_state]


def get_target_pose(demo: Demo, index: int):
    """Get the target pose (gripper position and open state) for a specific observation in the demo."""
    return np.array(
        [*demo._observations[max(0, index)].gripper_pose, demo._observations[max(0, index)].gripper_open]
    )


def get_target_joints(demo: Demo, index: int):
    """Get the target joint positions for a specific observation in the demo."""
    return np.array(
        [*demo._observations[max(0, index)].joint_positions, demo._observations[max(0, index)].gripper_open]
    )


def action_conversion(
    action: np.ndarray,
    to_representation: str = "euler",
    is_relative: bool = False,
    previous_action: np.ndarray = None,
):
    """Convert an action between Euler and quaternion representations.

    Args:
        action: np.ndarray of shape (7,) (Euler) or (8,) (quaternion).
                Euler format: [x, y, z, roll, pitch, yaw, gripper]
                Quaternion format: [x, y, z, qx, qy, qz, qw, gripper]
        to_representation: 'euler' or 'quat' target representation.
        is_relative: if True, compute the delta between `action` and
                     `previous_action` and return the delta (position and
                     rotation). `previous_action` must be provided in this case.
        previous_action: previous action (same format as `action`) used when
                         is_relative is True.

    Returns:
        np.ndarray converted action in the target representation. For relative
        mode the returned rotation represents the delta rotation (as Euler
        angles or as a unit quaternion depending on `to_representation`).

    Notes:
        - Quaternion ordering is (qx, qy, qz, qw) to match the rest of the
          codebase. Rotation objects from scipy are created/consumed with
          this ordering via as_quat(scalar_first=False).
        - When producing quaternions we always normalize to guard against
          numerical drift.
    """

    if to_representation not in ("euler", "quat"):
        raise ValueError("to_representation must be 'euler' or 'quat'")

    a = np.asarray(action, dtype=float)
    if a.size not in (7, 8):
        raise ValueError("action must be length 7 (Euler) or 8 (quaternion)")

    if is_relative and previous_action is None:
        raise ValueError("previous_action must be provided when is_relative is True")

    def _ensure_unit_quat(q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        if n == 0:
            raise ValueError("Zero quaternion encountered")
        return q / n

    # Helper: construct Rotation from either euler or quat stored in action array
    def _rot_from_action(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 7:
            return Rotation.from_euler("xyz", arr[3:6], degrees=False)
        else:
            return Rotation.from_quat(arr[3:7])  # (qx, qy, qz, qw)

    # Gripper state (keep as-is, demo code expects absolute gripper state even for deltas)
    gripper = a[-1]

    # Relative case: compute deltas
    if is_relative:
        prev = np.asarray(previous_action, dtype=float)
        if prev.size not in (7, 8):
            raise ValueError("previous_action must be length 7 or 8")

        delta_pos = a[:3] - prev[:3]

        # If both are Euler, simple subtraction of angles is fine
        if a.size == 7 and prev.size == 7:
            delta_ang = a[3:6] - prev[3:6]
            if to_representation == "euler":
                return np.array([*delta_pos, *delta_ang, gripper], dtype=float)
            else:
                # convert delta Euler to quaternion (and normalize)
                q = Rotation.from_euler("xyz", delta_ang, degrees=False).as_quat(scalar_first=False)
                q = _ensure_unit_quat(q)
                return np.array([*delta_pos, *q, gripper], dtype=float)

        # Otherwise use rotation algebra to compute the delta rotation
        r_cur = _rot_from_action(a)
        r_prev = _rot_from_action(prev)
        r_delta = r_cur * r_prev.inv()

        if to_representation == "euler":
            delta_ang = r_delta.as_euler("xyz", degrees=False)
            return np.array([*delta_pos, *delta_ang, gripper], dtype=float)
        else:
            q = r_delta.as_quat(scalar_first=False)
            q = _ensure_unit_quat(q)
            return np.array([*delta_pos, *q, gripper], dtype=float)

    # Absolute case: just convert representations
    if to_representation == "euler":
        if a.size == 7:
            return a.astype(float)
        else:
            euler = Rotation.from_quat(a[3:7]).as_euler("xyz", degrees=False)
            return np.array([*a[:3], *euler, gripper], dtype=float)
    else:  # to_representation == 'quat'
        if a.size == 8:
            q = _ensure_unit_quat(a[3:7])
            return np.array([*a[:3], *q, gripper], dtype=float)
        else:
            q = Rotation.from_euler("xyz", a[3:6], degrees=False).as_quat(scalar_first=False)
            q = _ensure_unit_quat(q)
            return np.array([*a[:3], *q, gripper], dtype=float)


# ------------------------
# Main
# ------------------------


def main(args):
    task_class = name_to_task_class(args.task)

    # RLBench setup
    camera_config = CameraConfig(image_size=(args.image_height, args.image_width))
    obs_config = ObservationConfig(
        left_shoulder_camera=camera_config,
        right_shoulder_camera=camera_config,
        overhead_camera=camera_config,
        wrist_camera=camera_config,
        front_camera=camera_config,
    )
    obs_config.set_all(True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=args.absolute_actions),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(task_class)

    # Remove the dataset root if already exists
    if os.path.exists(args.save_path):
        print(f"Dataset root {args.save_path} already exists. Removing it.")
        shutil.rmtree(args.save_path)

    camera_names = ["left_shoulder_rgb", "right_shoulder_rgb", "front_rgb", "wrist_rgb", "overhead_rgb"]

    action_feature = {}
    if args.action_repr == "euler":
        action_feature = {
            "shape": (7,),  # pos(3) + euler(3) + gripper(1)
            "dtype": "float32",
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "description": "End-effector position (x,y,z), orientation (roll,pitch,yaw) and gripper state (0.0 closed, 1.0 open).",
        }
    elif args.action_repr == "quat":
        action_feature = {
            "shape": (8,),  # pos(3) + quat(4) + gripper(1)
            "dtype": "float32",
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
            "description": "End-effector position (x,y,z), orientation (qx,qy,qz,qw) and gripper state (0.0 closed, 1.0 open).",
        }
    elif args.action_repr == "joints":
        action_feature = {
            "shape": (8,),  # joint_1 to joint_7 + gripper(1)
            "dtype": "float32",
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"],
            "description": "Robot joint positions (absolute rotations) and gripper state (0.0 closed, 1.0 open).",
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.save_path,
        robot_type="franka",
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),  # pos(3) + euler(3) + gripper(1)
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                "description": "End-effector position (x,y,z), orientation (roll,pitch,yaw) and gripper state (0.0 closed, 1.0 open).",
            },
            "observation.state.joints": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
                "description": "Robot joint positions (absolute rotations).",
            },
            "action": action_feature,
            # All camera images
            **{
                f"observation.images.{cam}": {
                    "dtype": "video",
                    "shape": (args.image_height, args.image_width, 3),
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": args.fps,
                        "video.height": args.image_height,
                        "video.width": args.image_width,
                        "video.channels": 3,
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
                for cam in camera_names
            },
        },
    )

    # Collect demonstrations and add them to the LeRobot dataset
    print(f"Generating {args.num_episodes} demos for task: {args.task}")
    for _ in tqdm(range(args.num_episodes), desc="Collecting demos"):
        # generate a new demo
        demo = task.get_demos(1, live_demos=True)[0]

        for frame_index, observation in enumerate(demo):
            action = None
            if args.action_repr in ["euler", "quat"]:
                action = action_conversion(
                    get_target_pose(demo, frame_index + 1 if frame_index < len(demo) - 1 else frame_index),
                    args.action_repr,
                    not args.absolute_actions,
                    get_target_pose(demo, frame_index),
                )
            elif args.action_repr == "joints":
                action = get_target_joints(
                    demo, frame_index + 1 if frame_index < len(demo) - 1 else frame_index
                )

            # Create the frame data, following the same structure as the features defined above
            frame_data = {
                "observation.state": action_conversion(get_target_pose(demo, frame_index)).astype(np.float32),
                "observation.state.joints": observation.joint_positions.astype(np.float32),
                "action": action.astype(np.float32),
                "task": task.get_name(),
            }
            for cam in camera_names:
                frame_data[f"observation.images.{cam}"] = getattr(observation, cam)

            # Save the frame
            dataset.add_frame(frame_data)
        dataset.save_episode()
    env.shutdown()

    # dataset.push_to_hub()
    print(f"\033[92mDataset saved to {args.save_path} and pushed to HuggingFace Hub: {args.repo_id}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect RLBench demonstrations and save to LeRobot dataset format."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(os.getcwd(), "datasets"),
        help="Path to save the LeRobot dataset.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., 'username/dataset-name').",
    )
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of demonstrations to record.")
    parser.add_argument("--task", type=str, default="put_rubbish_in_bin", help="Name of the RLBench task.")
    parser.add_argument(
        "--action_repr",
        type=str,
        choices=["euler", "quat", "joints"],
        default="euler",
        help="Action representation: 'euler' for Euler angles, 'quat' for quaternions, or 'joints' for joint positions.",
    )
    parser.add_argument(
        "--absolute_actions",
        action="store_true",
        default=False,
        help="Whether to use absolute actions (default: False). Valid only for 'euler' and 'quat' action representations.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frames per second for the dataset (default: 30)."
    )
    parser.add_argument("--image_width", type=int, default=256, help="Image width in pixels (default: 256).")
    parser.add_argument(
        "--image_height", type=int, default=256, help="Image height in pixels (default: 256)."
    )

    args = parser.parse_args()
    main(args)
