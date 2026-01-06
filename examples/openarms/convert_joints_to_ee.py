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
Convert a joint-space OpenArms dataset to end-effector space.

For each frame, converts joint positions to EE poses (x, y, z, wx, wy, wz) using FK.
Grippers are kept as-is. Applies to both observation.state and action.

Example usage:
    python examples/openarms/convert_joints_to_ee.py \
        --input-dataset lerobot-data-collection/rac_blackf0 \
        --output-repo-id my_user/rac_blackf0_ee \
        --output-dir ./outputs/rac_blackf0_ee
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import write_info, write_stats
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

DEFAULT_URDF = "src/lerobot/robots/openarms/urdf/openarm_bimanual_pybullet.urdf"
DEFAULT_LEFT_EE_FRAME = "openarm_left_hand_tcp"
DEFAULT_RIGHT_EE_FRAME = "openarm_right_hand_tcp"

LEFT_URDF_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_URDF_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
EE_COMPONENTS = ["x", "y", "z", "wx", "wy", "wz"]


def compute_fk_for_arm(kinematics: RobotKinematics, joint_values: np.ndarray) -> dict[str, float]:
    """Compute FK for one arm, returns EE pose as dict."""
    t = kinematics.forward_kinematics(joint_values)
    pos = t[:3, 3]
    rotvec = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "wx": float(rotvec[0]),
        "wy": float(rotvec[1]),
        "wz": float(rotvec[2]),
    }


def convert_joints_to_ee(
    values: np.ndarray,
    names: list[str],
    left_kin: RobotKinematics,
    right_kin: RobotKinematics,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert joint values to EE values.
    
    Args:
        values: Array of shape (N,) with joint values for one frame
        names: List of feature names corresponding to values
        left_kin: Left arm kinematics solver
        right_kin: Right arm kinematics solver
    
    Returns:
        (new_values, new_names) with joints replaced by EE poses
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    
    new_values = []
    new_names = []
    
    for prefix, kinematics in [("right", right_kin), ("left", left_kin)]:
        joint_vals = []
        for jname in JOINT_NAMES:
            key = f"{prefix}_{jname}.pos"
            if key in name_to_idx:
                joint_vals.append(values[name_to_idx[key]])
        
        if len(joint_vals) == 7:
            ee_pose = compute_fk_for_arm(kinematics, np.array(joint_vals, dtype=float))
            for comp in EE_COMPONENTS:
                new_names.append(f"{prefix}_ee.{comp}")
                new_values.append(ee_pose[comp])
        
        gripper_key = f"{prefix}_gripper.pos"
        if gripper_key in name_to_idx:
            new_names.append(f"{prefix}_ee.gripper_pos")
            new_values.append(values[name_to_idx[gripper_key]])
    
    return np.array(new_values, dtype=np.float32), new_names


def transform_feature_info(old_info: dict, new_names: list[str]) -> dict:
    """Create new feature info with EE names instead of joint names."""
    return {
        "dtype": old_info.get("dtype", "float32"),
        "shape": (len(new_names),),
        "names": new_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert joint-space dataset to EE-space")
    parser.add_argument("--input-dataset", type=str, required=True, help="Input dataset repo ID")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Output dataset repo ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Path to URDF file")
    parser.add_argument("--left-ee-frame", type=str, default=DEFAULT_LEFT_EE_FRAME)
    parser.add_argument("--right-ee-frame", type=str, default=DEFAULT_RIGHT_EE_FRAME)
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted dataset to HF Hub")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    urdf_path = args.urdf
    if not Path(urdf_path).is_absolute():
        urdf_path = str(Path(__file__).parent.parent.parent / urdf_path)

    print(f"Loading dataset: {args.input_dataset}")
    dataset = LeRobotDataset(args.input_dataset)
    
    print(f"Initializing kinematics from {urdf_path}")
    left_kin = RobotKinematics(urdf_path, args.left_ee_frame, LEFT_URDF_JOINTS)
    right_kin = RobotKinematics(urdf_path, args.right_ee_frame, RIGHT_URDF_JOINTS)

    action_info = dataset.meta.features.get("action", {})
    state_info = dataset.meta.features.get("observation.state", {})
    action_names = action_info.get("names", [])
    state_names = state_info.get("names", [])

    print(f"Original action names ({len(action_names)}): {action_names[:8]}...")
    print(f"Original state names ({len(state_names)}): {state_names[:8]}...")

    sample_action = np.zeros(len(action_names), dtype=np.float32)
    _, new_action_names = convert_joints_to_ee(sample_action, action_names, left_kin, right_kin)
    sample_state = np.zeros(len(state_names), dtype=np.float32)
    _, new_state_names = convert_joints_to_ee(sample_state, state_names, left_kin, right_kin)

    print(f"New action names ({len(new_action_names)}): {new_action_names}")
    print(f"New state names ({len(new_state_names)}): {new_state_names}")

    new_features = dataset.meta.features.copy()
    new_features["action"] = transform_feature_info(action_info, new_action_names)
    new_features["observation.state"] = transform_feature_info(state_info, new_state_names)

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=args.output_repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    data_dir = dataset.root / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    print(f"Processing {len(parquet_files)} parquet files...")

    all_actions = []
    all_states = []

    for src_path in tqdm(parquet_files, desc="Converting"):
        df = pd.read_parquet(src_path).reset_index(drop=True)
        
        new_actions = []
        new_states = []
        
        for idx in range(len(df)):
            action_vals = np.array(df.iloc[idx]["action"], dtype=np.float32)
            state_vals = np.array(df.iloc[idx]["observation.state"], dtype=np.float32)
            
            new_action, _ = convert_joints_to_ee(action_vals, action_names, left_kin, right_kin)
            new_state, _ = convert_joints_to_ee(state_vals, state_names, left_kin, right_kin)
            
            new_actions.append(new_action.tolist())
            new_states.append(new_state.tolist())
            all_actions.append(new_action)
            all_states.append(new_state)
        
        df["action"] = new_actions
        df["observation.state"] = new_states
        
        relative_path = src_path.relative_to(dataset.root)
        out_path = output_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)

    print("Computing statistics...")
    all_actions_arr = np.stack(all_actions)
    all_states_arr = np.stack(all_states)
    
    stats = {}
    stats["action"] = get_feature_stats(all_actions_arr, axis=0, keepdims=True)
    stats["observation.state"] = get_feature_stats(all_states_arr, axis=0, keepdims=True)
    write_stats(stats, output_dir)

    print("Updating metadata...")
    new_meta.info["total_episodes"] = dataset.meta.total_episodes
    new_meta.info["total_frames"] = dataset.meta.total_frames
    new_meta.info["total_tasks"] = dataset.meta.total_tasks
    write_info(new_meta.info, output_dir)

    print("Copying episode metadata...")
    src_episodes_dir = dataset.root / "meta" / "episodes"
    dst_episodes_dir = output_dir / "meta" / "episodes"
    if src_episodes_dir.exists():
        shutil.copytree(src_episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    print("Copying tasks metadata...")
    src_tasks = dataset.root / "meta" / "tasks.parquet"
    dst_tasks = output_dir / "meta" / "tasks.parquet"
    if src_tasks.exists():
        shutil.copy2(src_tasks, dst_tasks)

    if dataset.meta.video_keys:
        print("Copying videos...")
        src_videos = dataset.root / "videos"
        dst_videos = output_dir / "videos"
        if src_videos.exists():
            shutil.copytree(src_videos, dst_videos, dirs_exist_ok=True)

    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"Repo ID: {args.output_repo_id}")

    if args.push_to_hub:
        print("\nPushing to Hub...")
        output_dataset = LeRobotDataset(args.output_repo_id, root=output_dir)
        output_dataset.push_to_hub()
        print(f"Pushed to: https://huggingface.co/datasets/{args.output_repo_id}")


if __name__ == "__main__":
    main()

