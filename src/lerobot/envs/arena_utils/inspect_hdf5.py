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
Inspect and analyze HDF5 dataset files from Isaac Lab Arena.

This module provides tools to inspect HDF5 files and extract their schema
for use in conversion to LeRobot format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetField:
    """Represents a single field (dataset) in the HDF5 file."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    is_image: bool = False
    is_video: bool = False

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """Get the shape for a single frame (without time dimension)."""
        # First dimension is typically time/frames
        return self.shape[1:] if len(self.shape) > 1 else self.shape


@dataclass
class HDF5Schema:
    """Schema extracted from an HDF5 dataset file."""

    # Metadata
    env_name: str = ""
    robot_type: str = "unknown"
    fps: int = 50  # Default Isaac Lab FPS

    # Episode info
    episode_names: list[str] = field(default_factory=list)
    total_frames: int = 0

    # Data fields
    action_fields: dict[str, DatasetField] = field(default_factory=dict)
    observation_fields: dict[str, DatasetField] = field(default_factory=dict)
    camera_fields: dict[str, DatasetField] = field(default_factory=dict)
    state_fields: dict[str, DatasetField] = field(default_factory=dict)

    # Primary action field (the main 'actions' dataset)
    primary_action_key: str = "actions"

    def get_lerobot_features(
        self,
        include_cameras: bool = True,
        state_key: str | None = None,
        action_key: str | None = None,
    ) -> dict[str, dict]:
        """
        Generate LeRobot features dictionary from the schema.

        Args:
            include_cameras: Whether to include camera observations
            state_key: Specific observation key to use as state (default: auto-detect)
            action_key: Specific action key to use (default: primary_action_key)

        Returns:
            Features dictionary compatible with LeRobotDataset
        """
        features = {}

        # Action feature
        action_key_to_use = action_key or self.primary_action_key
        if action_key_to_use in self.action_fields:
            action_field = self.action_fields[action_key_to_use]
            features["action"] = {
                "dtype": "float32",
                "shape": action_field.feature_shape,
                "names": None,
            }

        # Observation state feature
        # Auto-detect state key if not provided
        if state_key is None:
            # Priority order for state observation
            state_priorities = [
                "robot_joint_pos",
                "joint_position",
                "robot_state",
            ]
            for key in state_priorities:
                if key in self.observation_fields:
                    state_key = key
                    break
            # Fallback to first observation field
            if state_key is None and self.observation_fields:
                state_key = next(iter(self.observation_fields.keys()))

        if state_key and state_key in self.observation_fields:
            obs_field = self.observation_fields[state_key]
            features["observation.state"] = {
                "dtype": "float32",
                "shape": obs_field.feature_shape,
                "names": None,
            }

        # Camera features
        if include_cameras:
            for cam_name, cam_field in self.camera_fields.items():
                # Only include RGB cameras (uint8)
                if cam_field.dtype == "uint8" and len(cam_field.feature_shape) == 3:
                    feature_name = f"observation.images.{cam_name}"
                    features[feature_name] = {
                        "dtype": "video",
                        "shape": cam_field.feature_shape,
                        "names": None,
                    }

        return features

    def get_camera_names(self) -> list[str]:
        """Get list of RGB camera names."""
        return [
            name
            for name, field in self.camera_fields.items()
            if field.dtype == "uint8" and len(field.feature_shape) == 3
        ]


class HDF5Inspector:
    """
    Inspects HDF5 files from Isaac Lab Arena to extract schema and data.

    This class analyzes the structure of an HDF5 file and provides methods
    to extract data for conversion to LeRobot format.
    """

    def __init__(self, file_path: str | Path):
        """
        Initialize the inspector with an HDF5 file.

        Args:
            file_path: Path to the HDF5 file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        self._schema: HDF5Schema | None = None

    def get_schema(self) -> HDF5Schema:
        """
        Extract and return the schema from the HDF5 file.

        Returns:
            HDF5Schema object containing the file structure
        """
        if self._schema is not None:
            return self._schema

        schema = HDF5Schema()

        with h5py.File(self.file_path, "r") as f:
            # Extract metadata from data group attributes
            data_group = f.get("data")
            if data_group is None:
                raise ValueError("HDF5 file does not contain 'data' group")

            # Parse env_args attribute if present
            if "env_args" in data_group.attrs:
                env_args = json.loads(data_group.attrs["env_args"])
                schema.env_name = env_args.get("env_name", "")

                # Infer robot type from env name
                env_name_lower = schema.env_name.lower()
                if "gr1" in env_name_lower:
                    schema.robot_type = "GR1"
                elif "g1" in env_name_lower:
                    schema.robot_type = "G1"

                # Calculate FPS from sim args
                sim_args = env_args.get("sim_args", {})
                dt = sim_args.get("dt", 0.005)
                decimation = sim_args.get("decimation", 4)
                schema.fps = int(1.0 / (dt * decimation))

            if "total" in data_group.attrs:
                schema.total_frames = int(data_group.attrs["total"])

            # Get episode names
            schema.episode_names = self._get_episode_names(data_group)

            # Analyze first episode to get data structure
            if schema.episode_names:
                first_episode = schema.episode_names[0]
                episode_group = data_group[first_episode]
                self._analyze_episode_structure(episode_group, schema)

        self._schema = schema
        return schema

    def _get_episode_names(self, data_group: h5py.Group) -> list[str]:
        """Extract sorted episode names from data group."""
        episodes = []
        for key in data_group:
            if key.startswith("demo_"):
                episodes.append(key)
        episodes.sort(key=lambda x: int(x.split("_")[1]))
        return episodes

    def _analyze_episode_structure(self, episode_group: h5py.Group, schema: HDF5Schema) -> None:
        """Analyze the structure of an episode to populate schema."""

        # Check for main actions dataset
        if "actions" in episode_group:
            actions_ds = episode_group["actions"]
            schema.action_fields["actions"] = DatasetField(
                name="actions",
                shape=actions_ds.shape,
                dtype=str(actions_ds.dtype),
            )
            schema.primary_action_key = "actions"

        # Check for processed_actions
        if "processed_actions" in episode_group:
            processed_ds = episode_group["processed_actions"]
            schema.action_fields["processed_actions"] = DatasetField(
                name="processed_actions",
                shape=processed_ds.shape,
                dtype=str(processed_ds.dtype),
            )

        # Check for action group with sub-actions
        if "action" in episode_group:
            action_group = episode_group["action"]
            if isinstance(action_group, h5py.Group):
                for key in action_group:
                    ds = action_group[key]
                    if isinstance(ds, h5py.Dataset):
                        schema.action_fields[f"action/{key}"] = DatasetField(
                            name=f"action/{key}",
                            shape=ds.shape,
                            dtype=str(ds.dtype),
                        )

        # Analyze observations
        if "obs" in episode_group:
            obs_group = episode_group["obs"]
            for key in obs_group:
                ds = obs_group[key]
                if isinstance(ds, h5py.Dataset):
                    schema.observation_fields[key] = DatasetField(
                        name=key,
                        shape=ds.shape,
                        dtype=str(ds.dtype),
                    )

        # Analyze camera observations
        if "camera_obs" in episode_group:
            camera_group = episode_group["camera_obs"]
            for key in camera_group:
                ds = camera_group[key]
                if isinstance(ds, h5py.Dataset):
                    is_image = ds.dtype == np.uint8 and len(ds.shape) == 4 and ds.shape[-1] in [1, 3, 4]
                    schema.camera_fields[key] = DatasetField(
                        name=key,
                        shape=ds.shape,
                        dtype=str(ds.dtype),
                        is_image=is_image,
                        is_video=is_image,
                    )

        # Analyze states (for replay/reset)
        if "states" in episode_group:
            self._analyze_states_group(episode_group["states"], schema, "states")

        if "initial_state" in episode_group:
            self._analyze_states_group(episode_group["initial_state"], schema, "initial_state")

    def _analyze_states_group(self, group: h5py.Group, schema: HDF5Schema, prefix: str) -> None:
        """Recursively analyze states group."""

        def visit_item(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                full_name = f"{prefix}/{name}"
                schema.state_fields[full_name] = DatasetField(
                    name=full_name,
                    shape=obj.shape,
                    dtype=str(obj.dtype),
                )

        group.visititems(visit_item)

    def load_episode_data(
        self,
        episode_name: str,
        include_cameras: bool = True,
        camera_names: list[str] | None = None,
        action_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Load all data for a single episode.

        Args:
            episode_name: Name of the episode (e.g., 'demo_0')
            include_cameras: Whether to load camera data
            camera_names: Specific camera names to load (None for all)
            action_key: Specific action key to load (None uses schema.primary_action_key)

        Returns:
            Dictionary containing episode data
        """
        schema = self.get_schema()
        action_key_to_use = action_key or schema.primary_action_key

        with h5py.File(self.file_path, "r") as f:
            episode_group = f[f"data/{episode_name}"]

            episode_data: dict[str, Any] = {
                "num_frames": 0,
                "actions": None,
                "obs": {},
                "camera_obs": {},
            }

            # Load actions
            if action_key_to_use in episode_group:
                episode_data["actions"] = episode_group[action_key_to_use][:]
                episode_data["num_frames"] = len(episode_data["actions"])
            elif schema.primary_action_key in episode_group:
                # Fallback to primary action key if specified key not found
                logger.warning(
                    f"Action key '{action_key_to_use}' not found, "
                    f"falling back to '{schema.primary_action_key}'"
                )
                episode_data["actions"] = episode_group[schema.primary_action_key][:]
                episode_data["num_frames"] = len(episode_data["actions"])

            # Load observations
            if "obs" in episode_group:
                obs_group = episode_group["obs"]
                for key in obs_group:
                    episode_data["obs"][key] = obs_group[key][:]

            # Load camera observations
            if include_cameras and "camera_obs" in episode_group:
                camera_group = episode_group["camera_obs"]
                cameras_to_load = camera_names or list(camera_group.keys())

                for key in cameras_to_load:
                    if key in camera_group:
                        episode_data["camera_obs"][key] = camera_group[key][:]

            return episode_data

    def get_episode_count(self) -> int:
        """Get the number of episodes in the dataset."""
        return len(self.get_schema().episode_names)

    def print_summary(self) -> None:
        """Print a summary of the HDF5 file structure."""
        schema = self.get_schema()

        print(f"\n{'=' * 60}")
        print(f"HDF5 Dataset Summary: {self.file_path.name}")
        print(f"{'=' * 60}")
        print(f"Environment: {schema.env_name}")
        print(f"Robot Type: {schema.robot_type}")
        print(f"FPS: {schema.fps}")
        print(f"Total Episodes: {len(schema.episode_names)}")
        print(f"Total Frames: {schema.total_frames}")

        print(f"\n{'Action Fields':^60}")
        print("-" * 60)
        for name, fld in schema.action_fields.items():
            print(f"  {name}: shape={fld.feature_shape}, dtype={fld.dtype}")

        print(f"\n{'Observation Fields':^60}")
        print("-" * 60)
        for name, fld in schema.observation_fields.items():
            print(f"  {name}: shape={fld.feature_shape}, dtype={fld.dtype}")

        print(f"\n{'Camera Fields':^60}")
        print("-" * 60)
        for name, fld in schema.camera_fields.items():
            print(f"  {name}: shape={fld.feature_shape}, dtype={fld.dtype}")

        print(f"\n{'LeRobot Features':^60}")
        print("-" * 60)
        features = schema.get_lerobot_features()
        for name, feat in features.items():
            print(f"  {name}: shape={feat['shape']}, dtype={feat['dtype']}")


def print_hdf5_structure(file_path: str | Path, max_items: int = 10) -> None:
    """
    Load and display the structure and main content of an HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        max_items: Maximum number of items to display for arrays
    """
    print(f"Loading HDF5 file: {file_path}\n")

    with h5py.File(file_path, "r") as f:
        print("=" * 80)
        print("HDF5 File Structure")
        print("=" * 80)

        def print_structure(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            """Recursively print the structure of the HDF5 file."""
            prefix = "  " * name.count("/")
            if isinstance(obj, h5py.Group):
                print(f"{prefix}📁 Group: {name}/")
                # Print attributes
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}  └─ @{attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{prefix}📄 Dataset: {name}")
                print(f"{prefix}  ├─ Shape: {obj.shape}")
                print(f"{prefix}  ├─ Dtype: {obj.dtype}")
                print(f"{prefix}  └─ Size: {obj.size} elements")

                # Print attributes
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}    └─ @{attr_name}: {attr_value}")

        # Print root level info
        print(f"\nRoot group keys: {list(f.keys())}\n")

        # Root attributes
        if f.attrs:
            print("Root Attributes:")
            for attr_name, attr_value in f.attrs.items():
                print(f"  @{attr_name}: {attr_value}")
            print()

        # Recursively visit all items
        f.visititems(print_structure)

        print("\n" + "=" * 80)
        print("Sample Data Content")
        print("=" * 80)

        # Display sample data from datasets
        def print_data_samples(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"\n📊 {name}:")
                print(f"   Shape: {obj.shape}, Dtype: {obj.dtype}")

                data = obj[:]

                # Show sample based on dimensionality
                if len(obj.shape) == 0:  # Scalar
                    print(f"   Value: {data}")
                elif len(obj.shape) == 1:  # 1D array
                    if obj.shape[0] <= max_items:
                        print(f"   Data: {data}")
                    else:
                        print(f"   First {max_items} items: {data[:max_items]}")
                        print(f"   Last {max_items} items: {data[-max_items:]}")
                        print(f"   Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")
                else:  # Multi-dimensional
                    msg = f"   First item shape: {data[0].shape if len(data) > 0 else 'N/A'}"
                    print(msg)
                    if len(data) > 0:
                        print(f"   First item sample:\n{data[0]}")
                        if len(data) > 1:
                            print(f"   Last item sample:\n{data[-1]}")

                    # Stats for numeric data
                    if np.issubdtype(obj.dtype, np.number):
                        print(
                            f"   Stats - Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data):.4f}"
                        )

        f.visititems(print_data_samples)

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        # Count groups and datasets
        counts = {"groups": 0, "datasets": 0}

        def count_items(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Group):
                counts["groups"] += 1
            elif isinstance(obj, h5py.Dataset):
                counts["datasets"] += 1

        f.visititems(count_items)

        print(f"Total Groups: {counts['groups']}")
        print(f"Total Datasets: {counts['datasets']}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the structure and content of an HDF5 file.")
    parser.add_argument("--file-path", type=str, required=True, help="Path to the HDF5 file to inspect")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print concise summary using HDF5Inspector",
    )
    args = parser.parse_args()

    if args.summary:
        inspector = HDF5Inspector(args.file_path)
        inspector.print_summary()
    else:
        print_hdf5_structure(args.file_path)
