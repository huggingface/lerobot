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
Distributed EgoDex dataset porting using SLURM and datatrove.

EgoDex is a large-scale dataset for egocentric dexterous manipulation collected
with ARKit on Apple Vision Pro. This script converts EgoDex data to LeRobot format.

Reference: https://arxiv.org/abs/2505.11709, https://github.com/apple/ml-egodex 
"""

import argparse
from pathlib import Path

import cv2
import h5py
import mediapy as mpy
import numpy as np
from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Image dimensions
DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_IMAGE_WIDTH = 1920

class PortEgoDexShards(PipelineStep):
    def __init__(
        self,
        raw_dir: Path | str,
        repo_id: str,
        local_dir: Path | str = None,
        percentage: float = 100.0,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.repo_id = repo_id
        self.local_dir = Path(local_dir) if local_dir else Path("data/local_datasets")
        self.percentage = percentage

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from pathlib import Path

        import cv2
        import h5py
        import mediapy as mpy
        import numpy as np

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.utils import init_logging

        def _get_state_for_single_frame(transforms_group, frame_idx):
            """
            Construct 48D hand state representation from EgoDex.
            
            State vector composition (per hand = 24D, total = 48D):
            - Wrist 3D position (3)
            - Wrist orientation in 6D representation (6)
            - 5 fingertip 3D positions (15)
            """
            state_vector = []
            fingertip_joints = {
                "left": [
                    "leftThumbTip",
                    "leftIndexFingerTip",
                    "leftMiddleFingerTip",
                    "leftRingFingerTip",
                    "leftLittleFingerTip",
                ],
                "right": [
                    "rightThumbTip",
                    "rightIndexFingerTip",
                    "rightMiddleFingerTip",
                    "rightRingFingerTip",
                    "rightLittleFingerTip",
                ],
            }

            for hand_side in ["left", "right"]:
                hand_key = f"{hand_side}Hand"
                hand_transform = transforms_group[hand_key][frame_idx]

                # 1. Wrist 3D position
                hand_position = hand_transform[:3, 3]
                state_vector.extend(hand_position)

                # 2. Wrist orientation in compact 6D representation
                rotation_matrix = hand_transform[:3, :3]
                rotation_6d = np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])
                state_vector.extend(rotation_6d)

                # 3. 3D positions of 5 fingertips
                for fingertip in fingertip_joints[hand_side]:
                    fingertip_transform = transforms_group[fingertip][frame_idx]
                    fingertip_pos = fingertip_transform[:3, 3]
                    state_vector.extend(fingertip_pos)

            # Also return camera extrinsics for optional coordinate frame transformations
            return np.array(state_vector, dtype=np.float32), transforms_group["camera"][frame_idx]

        def get_state_and_action_from_egodex_annotations(demo):
            """
            Convert EgoDex demo annotations into states and actions.
            
            The "action" is the state at time t+1 (next-pose prediction).
            """
            transforms_group = demo["transforms"]
            total_frames = list(transforms_group.values())[0].shape[0]

            states_list, extrinsics_list = [], []
            for frame_idx in range(total_frames):
                state_vector, extrinsics = _get_state_for_single_frame(transforms_group, frame_idx)
                states_list.append(state_vector)
                extrinsics_list.append(extrinsics.flatten())  # Flatten 4x4 to 16D

            state = np.array(states_list, dtype=np.float32)
            extrinsics = np.array(extrinsics_list, dtype=np.float32)

            # Shift by 1 timestep to convert state to action
            action = np.roll(state, -1, axis=0)

            return state, action, extrinsics

        def process_demo(hdf5_file_path, video_path):
            """Process a single EgoDex demo and return frames for LeRobot."""
            video = mpy.read_video(str(video_path))
            video = np.asarray(video)
            num_frames = video.shape[0]
            frames = []

            with h5py.File(hdf5_file_path, "r") as demo:
                state, action, extrinsics = get_state_and_action_from_egodex_annotations(demo)

                # Get natural language task description
                if demo.attrs.get("llm_type") == "reversible":
                    direction = demo.attrs.get("which_llm_description", "1")
                    lang_instruction = demo.attrs.get(
                        "llm_description" if direction == "1" else "llm_description2",
                        "manipulation task",
                    )
                else:
                    lang_instruction = demo.attrs.get("llm_description", "manipulation task")

                for step_idx in range(num_frames):
                    # Resize image to default dimensions
                    image_resized = cv2.resize(
                        video[step_idx],
                        (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                    frame = {
                        "task": lang_instruction,
                        "observation.image": image_resized,
                        "observation.state": state[step_idx],
                        "observation.extrinsics": extrinsics[step_idx],
                        "action": action[step_idx],
                    }
                    frames.append(frame)

            return frames

        init_logging()

        # Define EgoDex features
        EGODEX_FEATURES = {
            "observation.image": {
                "dtype": "video",
                "shape": (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (48,),
                "names": [
                    # Left hand wrist position (3)
                    "left_wrist_x",
                    "left_wrist_y",
                    "left_wrist_z",
                    # Left hand wrist rotation 6D (6)
                    "left_rot_0",
                    "left_rot_1",
                    "left_rot_2",
                    "left_rot_3",
                    "left_rot_4",
                    "left_rot_5",
                    # Left fingertips (15)
                    "left_thumb_x",
                    "left_thumb_y",
                    "left_thumb_z",
                    "left_index_x",
                    "left_index_y",
                    "left_index_z",
                    "left_middle_x",
                    "left_middle_y",
                    "left_middle_z",
                    "left_ring_x",
                    "left_ring_y",
                    "left_ring_z",
                    "left_little_x",
                    "left_little_y",
                    "left_little_z",
                    # Right hand wrist position (3)
                    "right_wrist_x",
                    "right_wrist_y",
                    "right_wrist_z",
                    # Right hand wrist rotation 6D (6)
                    "right_rot_0",
                    "right_rot_1",
                    "right_rot_2",
                    "right_rot_3",
                    "right_rot_4",
                    "right_rot_5",
                    # Right fingertips (15)
                    "right_thumb_x",
                    "right_thumb_y",
                    "right_thumb_z",
                    "right_index_x",
                    "right_index_y",
                    "right_index_z",
                    "right_middle_x",
                    "right_middle_y",
                    "right_middle_z",
                    "right_ring_x",
                    "right_ring_y",
                    "right_ring_z",
                    "right_little_x",
                    "right_little_y",
                    "right_little_z",
                ],
            },
            "observation.extrinsics": {
                "dtype": "float32",
                "shape": (16,),
                "names": [f"extrinsic_{i}" for i in range(16)],
            },
            "action": {
                "dtype": "float32",
                "shape": (48,),
                "names": [f"action_{i}" for i in range(48)],
            },
        }

        # 1. Discover all HDF5 files
        files = sorted(list(self.raw_dir.rglob("*.hdf5")))
        if not files:
            print(f"No HDF5 files found in {self.raw_dir}")
            return

        # 2. Apply percentage filter
        if self.percentage < 100:
            num_files = max(1, int(len(files) * self.percentage / 100))
            files = files[:num_files]
            print(f"Processing {self.percentage}% of dataset: {num_files} files")

        # 3. Assign files to this worker
        my_files = files[rank::world_size]
        if not my_files:
            print(f"Rank {rank} has no files to process.")
            return

        print(f"Rank {rank} processing {len(my_files)} files out of {len(files)} total.")

        # 4. Create a LeRobot dataset for this shard
        shard_repo_id = f"{self.repo_id}_world_{world_size}_rank_{rank}"
        shard_root = self.local_dir / shard_repo_id if self.local_dir else None

        dataset = LeRobotDataset.create(
            repo_id=shard_repo_id,
            fps=30,
            robot_type="hand",
            features=EGODEX_FEATURES,
            root=shard_root,
        )

        # 5. Process each file
        for input_h5 in my_files:
            try:
                # Derive corresponding video path
                video_path = input_h5.with_suffix(".mp4")
                if not video_path.exists():
                    print(f"Warning: Video file not found for {input_h5}, skipping.")
                    continue

                # Process demo and add frames
                frames = process_demo(input_h5, video_path)
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()

                # Clean up to avoid OOM
                del frames

            except Exception as e:
                print(f"Error processing {input_h5}: {e}")
                continue

        # 6. Finalize the dataset
        dataset.finalize()


def make_port_executor(
    raw_dir,
    repo_id,
    job_name,
    logs_dir,
    workers,
    partition,
    cpus_per_task,
    mem_per_cpu,
    local_dir,
    percentage,
    slurm=True,
):
    kwargs = {
        "pipeline": [
            PortEgoDexShards(raw_dir, repo_id, local_dir, percentage),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,
                "workers": workers,
                "time": "10:00:00",  # EgoDex is large, allow more time
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update(
            {
                "tasks": workers,
                "workers": 1,  # Run locally sequentially for debugging
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main():
    parser = argparse.ArgumentParser(
        description="Convert EgoDex dataset to LeRobot format using SLURM."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input EgoDex data (HDF5 + MP4 files).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier (e.g., user/egodex-lerobot).",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Path to logs directory.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="port_egodex",
        help="Job name used in SLURM.",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Launch over SLURM. Use --slurm 0 to launch sequentially (useful for debugging).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of SLURM workers.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="SLURM partition.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="Number of CPUs per worker.",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="4G",
        help="Memory per CPU.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of dataset to process (e.g., 1.0 for 1%%). Useful for testing.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory to save the LeRobot dataset. Defaults to data/local_datasets.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1

    port_executor = make_port_executor(**kwargs)
    port_executor.run()


if __name__ == "__main__":
    main()

