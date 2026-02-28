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
BehaviorLeRobotDatasetV3: A wrapper around LeRobotDataset v3.0 for loading BEHAVIOR-1K data.

This wrapper extends LeRobotDataset to support BEHAVIOR-1K specific features:
- Modality and camera selection (rgb, depth, seg_instance_id)
- Efficient chunk streaming mode with keyframe access
- Additional BEHAVIOR-1K metadata (cam_rel_poses, task_info, etc.)
"""

import logging
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
from behaviour_1k_constants import ROBOT_CAMERA_NAMES, ROBOT_TYPE
from torch.utils.data import Dataset, get_worker_info

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_safe_version,
    hf_transform_to_torch,
)
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)


class BehaviorLeRobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    Extended metadata class for BEHAVIOR-1K datasets.

    Adds support for:
    - Modality and camera filtering
    - Custom metainfo and annotation paths
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        metadata_buffer_size: int = 10,
        modalities: set[str] | None = None,
        cameras: set[str] | None = None,
    ):
        self.modalities = set(modalities) if modalities else {"rgb", "depth", "seg_instance_id"}
        self.camera_names = set(cameras) if cameras else {"head", "left_wrist", "right_wrist"}

        assert self.modalities.issubset({"rgb", "depth", "seg_instance_id"}), (
            f"Modalities must be subset of ['rgb', 'depth', 'seg_instance_id'], got {self.modalities}"
        )

        assert self.camera_names.issubset(set(ROBOT_CAMERA_NAMES[ROBOT_TYPE])), (
            f"Camera names must be subset of {list(ROBOT_CAMERA_NAMES[ROBOT_TYPE])}, got {self.camera_names}"
        )

        super().__init__(repo_id, root, revision, force_cache_sync, metadata_buffer_size)

    @property
    def filtered_features(self) -> dict[str, dict]:
        """Return only features matching selected modalities and cameras."""
        features = {}
        for name, feature_info in self.features.items():
            if not name.startswith("observation.images."):
                features[name] = feature_info
                continue

            parts = name.split(".")
            if len(parts) >= 4:
                modality = parts[2]
                camera = parts[3]
                if modality in self.modalities and camera in self.camera_names:
                    features[name] = feature_info

        return features

    @property
    def video_keys(self) -> list[str]:
        """Return only video keys for selected modalities and cameras."""
        all_video_keys = super().video_keys

        filtered_keys = []
        for key in all_video_keys:
            parts = key.split(".")
            if len(parts) >= 4:
                modality = parts[2]
                camera = parts[3]
                if modality in self.modalities and camera in self.camera_names:
                    filtered_keys.append(key)

        return filtered_keys

    def get_metainfo_path(self, ep_index: int) -> Path:
        """Get path to episode metainfo file."""
        if "metainfo_path" in self.info:
            fpath = self.info["metainfo_path"].format(episode_index=ep_index)
            return Path(fpath)
        return None

    def get_annotation_path(self, ep_index: int) -> Path:
        """Get path to episode annotation file."""
        if "annotation_path" in self.info:
            fpath = self.info["annotation_path"].format(episode_index=ep_index)
            return Path(fpath)
        return None


class BehaviorLeRobotDatasetV3(LeRobotDataset):
    """
    BEHAVIOR-1K wrapper for LeRobotDataset v3.0.

    Each BEHAVIOR-1K dataset contains a single task (e.g., behavior1k-task0000).
    See https://huggingface.co/collections/lerobot/behavior-1k for all available tasks.

    Key features:
    - Modality and camera selection
    - Efficient chunk streaming with keyframe access (recommended for B1K with GOP=250)
    - Support for BEHAVIOR-1K specific observations (cam_rel_poses, task_info, task_index)
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        # BEHAVIOR-1K specific arguments
        modalities: list[str] | None = None,
        cameras: list[str] | None = None,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize BEHAVIOR-1K dataset.

        Args:
            repo_id: HuggingFace repository ID (e.g., "lerobot/behavior1k-task0000")
            root: Local directory for dataset storage
            episodes: List of episode indices to load (for train/val split)
            image_transforms: Torchvision v2 transforms for images
            delta_timestamps: Temporal offsets for history/future frames
            tolerance_s: Tolerance for timestamp synchronization
            revision: Git revision/branch to load
            force_cache_sync: Force re-download from hub
            download_videos: Whether to download video files
            video_backend: Video decoder ('pyav' or 'torchcodec')
            batch_encoding_size: Batch size for video encoding
            modalities: List of modalities to load (None = all: rgb, depth, seg_instance_id)
            cameras: List of cameras to load (None = all: head, left_wrist, right_wrist)
            check_timestamp_sync: Verify timestamp synchronization (can be slow)
            chunk_streaming_using_keyframe: Use keyframe-based streaming (STRONGLY RECOMMENDED for B1K)
            shuffle: Shuffle chunks in streaming mode
            seed: Random seed for shuffling
        """
        Dataset.__init__(self)

        self.repo_id = repo_id
        if root:
            self.root = Path(root)
        else:
            dataset_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
            self.root = HF_LEROBOT_HOME / dataset_name

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.seed = seed

        self.image_writer = None
        self.episode_buffer = None
        self.writer = None
        self.latest_episode = None
        self._current_file_start_frame = None

        self.root.mkdir(exist_ok=True, parents=True)

        if modalities is None:
            modalities = ["rgb", "depth", "seg_instance_id"]
        if "seg_instance_id" in modalities:
            assert chunk_streaming_using_keyframe, (
                "For performance, seg_instance_id requires chunk_streaming_using_keyframe=True"
            )
        if "depth" in modalities:
            assert self.video_backend == "pyav", "Depth videos require video_backend='pyav'"
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]

        self.meta = BehaviorLeRobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            modalities=modalities,
            cameras=cameras,
        )

        if episodes is not None:
            self.episodes = sorted([i for i in episodes if i < len(self.meta.episodes)])
        else:
            self.episodes = list(range(len(self.meta.episodes)))

        logger.info(f"Total episodes: {len(self.episodes)}")

        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        if self._chunk_streaming_using_keyframe:
            if not shuffle:
                logger.warning("Chunk streaming enabled but shuffle=False. This may reduce randomness.")
            self.chunks = self._get_keyframe_chunk_indices()
            self.current_streaming_chunk_idx = None if shuffle else 0
            self.current_streaming_frame_idx = None if shuffle else self.chunks[0][0] if self.chunks else 0
            self.obs_loaders = {}
            self._should_obs_loaders_reload = True

        self._lazy_loading = False
        self._recorded_frames = self.meta.total_frames
        self._writer_closed_for_reading = False

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.meta.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.meta.fps)

    @property
    def fps(self) -> int:
        """Frames per second."""
        return self.meta.fps

    @property
    def features(self) -> dict:
        """Dataset features (filtered by modalities/cameras)."""
        return self.meta.filtered_features

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        return len(self.hf_dataset)

    def get_episodes_file_paths(self) -> list[str]:
        """
        Get download patterns for requested episodes.

        Returns glob patterns for download rather than specific file paths.

        Note: Unlike the base LeRobotDataset, this method cannot filter downloads to only
        requested episodes because:
        1. BEHAVIOR-1K episode indices are encoded (e.g., 10010 for task 1, episode 10)
        2. Episodes are chunked across multiple parquet/video files
        3. The parquet files are organized by chunk, not by episode

        Therefore, we download full data/meta/video directories and rely on
        `self.load_hf_dataset()` to filter to requested episodes from the loaded data.
        """
        allow_patterns = ["data/**", "meta/**"]

        # Filter by modalities and cameras for video patterns
        if len(self.meta.video_keys) > 0:
            if len(self.meta.modalities) != 3 or len(self.meta.camera_names) != 3:
                # Only download specific modality/camera combinations
                for modality in self.meta.modalities:
                    for camera in self.meta.camera_names:
                        allow_patterns.append(f"**/observation.images.{modality}.{camera}/**")
            else:
                # Download all videos (no filtering needed)
                allow_patterns.append("videos/**")

        return allow_patterns

    def download_episodes(self, download_videos: bool = True) -> None:
        """
        Download episodes with modality/camera filtering.

        Follows the same pattern as base LeRobotDataset.download() but uses
        get_episodes_file_paths() which returns patterns for modality/camera filtering.
        """
        ignore_patterns = None if download_videos else "videos/"
        files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """Pull dataset from HuggingFace Hub."""

        from huggingface_hub import snapshot_download

        logger.info(f"Pulling dataset {self.repo_id} from HuggingFace Hub...")
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def load_hf_dataset(self) -> datasets.Dataset:
        """Load dataset from parquet files."""
        from datasets import load_dataset

        path = str(self.root / "data")
        hf_dataset = load_dataset("parquet", data_dir=path, split="train")

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _get_keyframe_chunk_indices(self, chunk_size: int = 250) -> list[tuple[int, int, int]]:
        """
        Divide episodes into chunks based on GOP size (keyframe interval).

        For BEHAVIOR-1K, GOP size is 250 frames for efficient storage.

        Returns:
            List of (start_index, end_index, local_start_index) tuples
        """
        chunks = []
        offset = 0

        for ep_array_idx in self.episodes:
            # self.episodes contains array indices, so access directly
            ep = self.meta.episodes[ep_array_idx]
            length = ep["length"]
            local_starts = list(range(0, length, chunk_size))
            local_ends = local_starts[1:] + [length]

            for local_start, local_end in zip(local_starts, local_ends, strict=True):
                chunks.append((offset + local_start, offset + local_end, local_start))
            offset += length

        return chunks

    def __getitem__(self, idx: int) -> dict:
        """Get item by index, with optional chunk streaming."""
        if not self._chunk_streaming_using_keyframe:
            item = self.hf_dataset[idx]

            for key in self.meta.video_keys:
                if key in self.features:
                    ep_idx = item["episode_index"].item()
                    timestamp = item["timestamp"].item()
                    video_path = self.root / self.meta.get_video_file_path(ep_idx, key)
                    frames = decode_video_frames(
                        video_path, [timestamp], self.tolerance_s, self.video_backend
                    )
                    item[key] = frames.squeeze(0)

            if self.image_transforms is not None:
                for key in self.features:
                    if key.startswith("observation.images."):
                        item[key] = self.image_transforms(item[key])

            if "task_index" in item:
                task_idx = item["task_index"].item()
                try:
                    item["task"] = self.meta.tasks.iloc[task_idx].name
                except (IndexError, AttributeError):
                    item["task"] = f"task_{task_idx}"

            return item

        return self._get_item_streaming(idx)

    def _get_item_streaming(self, idx: int) -> dict:
        """Get item in chunk streaming mode."""
        if self.current_streaming_chunk_idx is None:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            rng = np.random.default_rng(self.seed + worker_id)
            rng.shuffle(self.chunks)
            self.current_streaming_chunk_idx = rng.integers(0, len(self.chunks)).item()
            self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]

        if self.current_streaming_frame_idx >= self.chunks[self.current_streaming_chunk_idx][1]:
            self.current_streaming_chunk_idx += 1
            if self.current_streaming_chunk_idx >= len(self.chunks):
                self.current_streaming_chunk_idx = 0
            self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
            self._should_obs_loaders_reload = True

        item = self.hf_dataset[self.current_streaming_frame_idx]
        ep_idx = item["episode_index"].item()

        if self._should_obs_loaders_reload:
            for loader in self.obs_loaders.values():
                if hasattr(loader, "close"):
                    loader.close()
            self.obs_loaders = {}
            self.current_streaming_episode_idx = ep_idx
            self._should_obs_loaders_reload = False

        for key in self.meta.video_keys:
            if key in self.features:
                timestamp = item["timestamp"].item()
                video_path = self.root / self.meta.get_video_file_path(ep_idx, key)
                frames = decode_video_frames(video_path, [timestamp], self.tolerance_s, self.video_backend)
                item[key] = frames.squeeze(0)

        if self.image_transforms is not None:
            for key in self.features:
                if key.startswith("observation.images."):
                    item[key] = self.image_transforms(item[key])

        if "task_index" in item:
            task_idx = item["task_index"].item()
            try:
                item["task"] = self.meta.tasks.iloc[task_idx].name
            except (IndexError, AttributeError):
                item["task"] = f"task_{task_idx}"

        self.current_streaming_frame_idx += 1
        return item

    def __len__(self) -> int:
        """Total number of frames."""
        return len(self.hf_dataset)
