#!/usr/bin/env python

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
import contextlib
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.utils import (
    DEFAULT_IMAGE_PATH,
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
)
from lerobot.datasets.video_utils import (
    StreamingVideoEncoder,
    get_safe_default_codec,
    resolve_vcodec,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v3.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v2.1 to v3.0, which you can find at
              lerobot/scripts/convert_dataset_v21_to_v30.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. Multiple episodes are
        consolidated into chunked files which improves storage efficiency and loading performance. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory where the dataset will be downloaded and
                stored. If set, all dataset files will be stored directly under this path. If not set, the
                dataset files will be stored under $HF_LEROBOT_HOME/repo_id (configurable via the
                HF_LEROBOT_HOME environment variable).
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            force_cache_sync (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            batch_encoding_size (int, optional): Number of episodes to accumulate before batch encoding videos.
                Set to 1 for immediate encoding (default), or higher for batched encoding. Defaults to 1.
            vcodec (str, optional): Video codec for encoding videos during recording. Options: 'h264', 'hevc',
                'libsvtav1', 'auto', or hardware-specific codecs like 'h264_videotoolbox', 'h264_nvenc'.
                Defaults to 'libsvtav1'. Use 'auto' to auto-detect the best available hardware encoder.
            streaming_encoding (bool, optional): If True, encode video frames in real-time during capture
                instead of writing PNG images first. This makes save_episode() near-instant. Defaults to False.
            encoder_queue_maxsize (int, optional): Maximum number of frames to buffer per camera when using
                streaming encoding. Defaults to 30 (~1s at 30fps).
            encoder_threads (int | None, optional): Number of threads per encoder instance. None lets the
                codec auto-detect (default). Lower values reduce CPU usage per encoder. Maps to 'lp' (via svtav1-params) for
                libsvtav1 and 'threads' for h264/hevc.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.batch_encoding_size = batch_encoding_size
        self.vcodec = resolve_vcodec(vcodec)
        self._encoder_threads = encoder_threads

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        # Create reader (hf_dataset loaded below)
        self._reader = DatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=episodes,
            tolerance_s=tolerance_s,
            video_backend=self.video_backend,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
        )

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self._reader.hf_dataset = self._reader.load_hf_dataset()
            if not self._reader._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            self.download(download_videos)
            self._reader.hf_dataset = self._reader.load_hf_dataset()

        # Build index mapping now that hf_dataset is loaded
        self._reader._build_index_mapping()

        # Detect write-mode params for backward compatibility
        _has_write_params = streaming_encoding or batch_encoding_size != 1
        if _has_write_params:
            import warnings

            warnings.warn(
                "Passing write-mode parameters (streaming_encoding, batch_encoding_size) to "
                "LeRobotDataset.__init__() is deprecated. Use LeRobotDataset.resume() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            streaming_enc = None
            if streaming_encoding and len(self.meta.video_keys) > 0:
                streaming_enc = StreamingVideoEncoder(
                    fps=self.meta.fps,
                    vcodec=self.vcodec,
                    pix_fmt="yuv420p",
                    g=2,
                    crf=30,
                    preset=None,
                    queue_maxsize=encoder_queue_maxsize,
                    encoder_threads=encoder_threads,
                )
            self._writer = DatasetWriter(
                meta=self.meta,
                root=self.root,
                vcodec=self.vcodec,
                encoder_threads=encoder_threads,
                batch_encoding_size=batch_encoding_size,
                streaming_encoder=streaming_enc,
            )
            self._writer._recorded_frames = self.meta.total_frames
        else:
            self._writer = None

        self._is_finalized = False

    # ── Writer guard ──────────────────────────────────────────────────

    def _require_writer(self, method_name: str) -> None:
        if self._writer is None:
            raise RuntimeError(
                f"Cannot call '{method_name}()' on a read-only dataset. "
                f"Use LeRobotDataset.create() for new recording or "
                f"LeRobotDataset.resume() for resume recording."
            )

    # ── Reader guard ──────────────────────────────────────────────────

    def _ensure_reader(self) -> DatasetReader:
        """Lazily create the reader on first access."""
        if self._reader is None:
            self._reader = DatasetReader(
                meta=self.meta,
                root=self.root,
                episodes=self.episodes,
                tolerance_s=self.tolerance_s,
                video_backend=self.video_backend,
                delta_timestamps=self.delta_timestamps,
                image_transforms=self.image_transforms,
            )
        return self._reader

    # ── Reader proxy properties ───────────────────────────────────────

    @property
    def hf_dataset(self):
        if self._reader is None:
            return None
        return self._reader.hf_dataset

    @hf_dataset.setter
    def hf_dataset(self, value):
        self._ensure_reader().hf_dataset = value

    @property
    def delta_indices(self):
        if self._reader is None:
            return None
        return self._reader.delta_indices

    @property
    def _absolute_to_relative_idx(self):
        if self._reader is None:
            return None
        return self._reader._absolute_to_relative_idx

    @property
    def hf_features(self):
        if self._reader is None:
            return get_hf_features_from_features(self.meta.features)
        return self._reader.hf_features

    # ── Writer proxy properties ───────────────────────────────────────

    @property
    def episode_buffer(self):
        return self._writer.episode_buffer if self._writer is not None else None

    @episode_buffer.setter
    def episode_buffer(self, value):
        if self._writer is not None:
            self._writer.episode_buffer = value

    @property
    def image_writer(self):
        return self._writer.image_writer if self._writer is not None else None

    @property
    def writer(self):
        return self._writer.writer if self._writer is not None else None

    @property
    def latest_episode(self):
        return self._writer.latest_episode if self._writer is not None else None

    @property
    def _current_file_start_frame(self):
        return self._writer._current_file_start_frame if self._writer is not None else None

    @property
    def _streaming_encoder(self):
        return self._writer._streaming_encoder if self._writer is not None else None

    @property
    def episodes_since_last_encoding(self):
        return self._writer.episodes_since_last_encoding if self._writer is not None else 0

    @property
    def _recorded_frames(self):
        return self._writer._recorded_frames if self._writer is not None else 0

    # ── Metadata properties ───────────────────────────────────────────

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self._reader is None:
            return self.meta.total_frames
        return self._reader.num_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        if self._reader is None:
            return self.meta.total_episodes
        return self._reader.num_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    # ── Reader-delegated methods ──────────────────────────────────────

    def load_hf_dataset(self):
        return self._ensure_reader().load_hf_dataset()

    def create_hf_dataset(self):
        return self._ensure_reader().create_hf_dataset()

    def get_episodes_file_paths(self):
        return self._ensure_reader().get_episodes_file_paths()

    def _check_cached_episodes_sufficient(self) -> bool:
        reader = self._ensure_reader()
        reader.episodes = self.episodes
        return reader._check_cached_episodes_sufficient()

    # ── Writer-delegated methods ──────────────────────────────────────

    def add_frame(self, frame: dict) -> None:
        self._require_writer("add_frame")
        self._writer.add_frame(frame)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        self._require_writer("save_episode")
        self._writer.save_episode(episode_data, parallel_encoding)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        self._require_writer("clear_episode_buffer")
        self._writer.clear_episode_buffer(delete_images)

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        self._require_writer("create_episode_buffer")
        return self._writer.create_episode_buffer(episode_index)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        self._require_writer("_save_episode_data")
        return self._writer._save_episode_data(episode_buffer)

    def _save_episode_video(self, video_key: str, episode_index: int, temp_path=None) -> dict:
        self._require_writer("_save_episode_video")
        return self._writer._save_episode_video(video_key, episode_index, temp_path)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None) -> None:
        self._require_writer("_batch_save_episode_video")
        self._writer._batch_save_episode_video(start_episode, end_episode)

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int):
        self._require_writer("_encode_temporary_episode_video")
        return self._writer._encode_temporary_episode_video(video_key, episode_index)

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        self._require_writer("start_image_writer")
        self._writer.start_image_writer(num_processes, num_threads)

    def stop_image_writer(self) -> None:
        if self._writer is not None:
            self._writer.stop_image_writer()

    def _wait_image_writer(self) -> None:
        if self._writer is not None:
            self._writer._wait_image_writer()

    def _save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1
    ) -> None:
        self._require_writer("_save_image")
        self._writer._save_image(image, fpath, compress_level)

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        if self._writer is not None:
            return self._writer._get_image_file_path(episode_index, image_key, frame_index)
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def finalize(self):
        """
        Close the parquet writers. This function needs to be called after data collection/conversion, else footer metadata won't be written to the parquet files.
        The dataset won't be valid and can't be loaded as ds = LeRobotDataset(repo_id=repo, root=HF_LEROBOT_HOME.joinpath(repo))
        """
        if self._writer is not None:
            self._writer.finalize()
            self._is_finalized = True

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        if self._writer is not None:
            self._writer.close_writer()

    def __del__(self):
        """Safety check: close the parquet writer on garbage collection."""
        if hasattr(self, "_writer") and self._writer is not None:
            self._writer.close_writer()

    # ── Core Dataset methods ──────────────────────────────────────────

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        if self._writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
        reader = self._ensure_reader()
        if reader.hf_dataset is None:
            # One-shot load after finalize()
            reader.hf_dataset = reader.load_hf_dataset()
            reader._build_index_mapping()
        return reader.get_item(idx)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    # ── Hub methods (stay on facade) ──────────────────────────────────

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, repo_id=self.repo_id, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version."""
        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    # ── Class constructors ────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        metadata_buffer_size: int = 10,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        vcodec = resolve_vcodec(vcodec)
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
            metadata_buffer_size=metadata_buffer_size,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.batch_encoding_size = batch_encoding_size
        obj.vcodec = vcodec
        obj._encoder_threads = encoder_threads

        # Reader is lazily created on first access (write-only mode)
        obj._reader = None

        # Create writer
        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = StreamingVideoEncoder(
                fps=fps,
                vcodec=vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                queue_maxsize=encoder_queue_maxsize,
                encoder_threads=encoder_threads,
            )
        obj._writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
        )

        if image_writer_processes or image_writer_threads:
            obj._writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._writer.episode_buffer = obj._writer.create_episode_buffer()

        obj._is_finalized = False

        return obj

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | Path | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ) -> "LeRobotDataset":
        """Resume recording on an existing dataset.

        Loads metadata from an existing dataset and creates a writer for appending new episodes.
        The hf_dataset is not loaded until finalize() is called and data is read.
        """
        vcodec = resolve_vcodec(vcodec)
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        obj.root.mkdir(exist_ok=True, parents=True)
        obj.revision = revision if revision else CODEBASE_VERSION
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj.video_backend = video_backend if video_backend else get_safe_default_codec()
        obj.batch_encoding_size = batch_encoding_size
        obj.vcodec = vcodec
        obj._encoder_threads = encoder_threads

        # Load metadata
        obj.meta = LeRobotDatasetMetadata(
            obj.repo_id, obj.root, obj.revision, force_cache_sync=force_cache_sync
        )

        # Reader is lazily created on first access (write-only mode)
        obj._reader = None

        # Create writer for appending
        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = StreamingVideoEncoder(
                fps=obj.meta.fps,
                vcodec=vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                queue_maxsize=encoder_queue_maxsize,
                encoder_threads=encoder_threads,
            )
        obj._writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
        )
        obj._writer._recorded_frames = obj.meta.total_frames

        if image_writer_processes or image_writer_threads:
            obj._writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._writer.episode_buffer = obj._writer.create_episode_buffer()

        obj._is_finalized = False

        return obj
