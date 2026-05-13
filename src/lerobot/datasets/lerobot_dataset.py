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
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .dataset_reader import DatasetReader
from .dataset_writer import DatasetWriter
from .utils import (
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
)
from .video_utils import (
    StreamingVideoEncoder,
    get_safe_default_codec,
    resolve_vcodec,
)

logger = logging.getLogger(__name__)


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        episode_filter: Callable[[dict], bool] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        return_uint8: bool = False,
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
            - data (backed by datasets.Dataset), which reads values from parquet files.
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
            root (Path | None, optional): Local directory where the dataset will be read from or downloaded
                into. If set, all dataset files are materialized directly under this path. If not set,
                existing local datasets are still looked up under ``$HF_LEROBOT_HOME/{repo_id}``, but Hub
                downloads use a revision-safe snapshot cache under
                ``$HF_LEROBOT_HOME/hub``.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            episode_filter (Callable[[dict], bool] | None, optional): Predicate over per-episode
                metadata rows used to select episodes. Evaluated against ``meta/`` without ``stats`` keys
                (e.g.``task_index``, ``episode_index``, ``length``, ``from_timestamp``, ``to_timestamp``).
                Intersected with ``episodes`` when both are set. Example: ``lambda ep: ep["length"] >= 100``.
                Defaults to None.
            image_transforms (Callable | None, optional):
                Transform applied to visual modalities inside `__getitem__` after image decoding / tensor
                conversion. This works for both image-backed and video-backed observations and can later be
                updated with `set_image_transforms()` or cleared with `clear_image_transforms()`.
                Defaults to None.
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

        Note:
            Write-mode parameters (``streaming_encoding``, ``batch_encoding_size``) passed to
            ``__init__`` are deprecated. Use :meth:`create` for new datasets or :meth:`resume`
            to append to existing ones.
        """
        super().__init__()
        self.repo_id = repo_id
        self._requested_root = Path(root) if root else None
        self.reader = None
        self.set_image_transforms(image_transforms)
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self._video_backend = video_backend if video_backend else get_safe_default_codec()
        self._return_uint8 = return_uint8
        self._batch_encoding_size = batch_encoding_size
        self._vcodec = resolve_vcodec(vcodec)
        self._encoder_threads = encoder_threads

        if self._requested_root is not None:
            self._requested_root.mkdir(exist_ok=True, parents=True)

        # Load metadata (sets self.root once from the resolved metadata root)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self._requested_root, self.revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.revision = self.meta.revision

        if episodes is not None and any(
            episode >= self.meta.total_episodes or episode < 0 for episode in episodes
        ):
            logger.warning(
                f"Some episodes in the provided episodes list are out of range for this dataset ({self.meta.total_episodes})."
            )

        if episode_filter is not None:
            resolved = self.meta.filter_episodes(episode_filter, candidates=episodes)
            if not resolved:
                raise ValueError(
                    "The episode filter did not match any episode. Make sure the filter and episodes list are valid and compatible."
                )
            logger.info(f"The episode filter matched {len(resolved)} episode(s).")
            episodes = resolved
        self.episodes = episodes

        # Create reader (hf_dataset loaded below)
        self.reader = DatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=episodes,
            tolerance_s=tolerance_s,
            video_backend=self._video_backend,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            return_uint8=self._return_uint8,
        )

        # Load actual data
        if force_cache_sync or not self.reader.try_load():
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            self._download(download_videos)
            self.reader.load_and_activate()

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
                streaming_enc = self._build_streaming_encoder(
                    self.meta.fps, self._vcodec, encoder_queue_maxsize, encoder_threads
                )
            self.writer = DatasetWriter(
                meta=self.meta,
                root=self.root,
                vcodec=self._vcodec,
                encoder_threads=encoder_threads,
                batch_encoding_size=batch_encoding_size,
                streaming_encoder=streaming_enc,
                initial_frames=self.meta.total_frames,
            )
        else:
            self.writer = None

        self._is_finalized = False

    # ── Writer guard ──────────────────────────────────────────────────

    def _require_writer(self, method_name: str) -> None:
        if self.writer is None:
            raise RuntimeError(
                f"Cannot call '{method_name}()' on a read-only dataset. "
                f"Use LeRobotDataset.create() for new recording or "
                f"LeRobotDataset.resume() for resume recording."
            )

    # ── Reader guard ──────────────────────────────────────────────────

    def _ensure_reader(self) -> DatasetReader:
        """Lazily create the reader on first access."""
        if self.reader is None:
            self.meta.ensure_readable()
            self.reader = DatasetReader(
                meta=self.meta,
                root=self.root,
                episodes=self.episodes,
                tolerance_s=self.tolerance_s,
                video_backend=self._video_backend,
                delta_timestamps=self.delta_timestamps,
                image_transforms=self.image_transforms,
                return_uint8=self._return_uint8,
            )
        return self.reader

    @staticmethod
    def _build_streaming_encoder(
        fps: int,
        vcodec: str,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> StreamingVideoEncoder:
        return StreamingVideoEncoder(
            fps=fps,
            vcodec=vcodec,
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=None,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )

    # ── Metadata properties ───────────────────────────────────────────

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        # Check directly instead of using _ensure_reader(): in write-only mode
        # (create/resume) we rely on metadata rather than initializing a reader.
        if self.reader is None:
            return self.meta.total_frames
        return self.reader.num_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        # Check directly instead of using _ensure_reader(): in write-only mode
        # (create/resume) we rely on metadata rather than initializing a reader.
        if self.reader is None:
            return self.meta.total_episodes
        return self.reader.num_episodes

    @property
    def features(self) -> dict[str, dict]:
        """Feature specification dict mapping feature names to their type/shape metadata."""
        return self.meta.features

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """The underlying Hugging Face Dataset object"""
        self.reader = self._ensure_reader()
        if self.reader.hf_dataset is None:
            self.reader.load_and_activate()
        return self.reader.hf_dataset

    # ── Writer-delegated methods ──────────────────────────────────────

    def add_frame(self, frame: dict) -> None:
        """Add a single frame to the current episode buffer.

        Delegates to :meth:`DatasetWriter.add_frame`. The dataset must be in
        write mode (created via :meth:`create` or :meth:`resume`).

        Args:
            frame: Dict mapping feature names to their values for this frame.
                Must include a ``'task'`` key. Torch tensors are converted to numpy.

        Raises:
            RuntimeError: If the dataset is read-only (no writer).
        """
        self._require_writer("add_frame")
        self.writer.add_frame(frame)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        """Save the current episode buffer to disk.

        Delegates to :meth:`DatasetWriter.save_episode`. Encodes videos, writes
        parquet data, and updates metadata. The episode buffer is reset afterward.

        Args:
            episode_data: Optional pre-built episode dict. If ``None``, uses the
                internal episode buffer populated by :meth:`add_frame`.
            parallel_encoding: If ``True`` and multiple cameras exist, encode
                videos in parallel using a process pool.

        Raises:
            RuntimeError: If the dataset is read-only (no writer).
        """
        self._require_writer("save_episode")
        self.writer.save_episode(episode_data, parallel_encoding)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """Discard the current episode buffer without saving.

        Delegates to :meth:`DatasetWriter.clear_episode_buffer`. Useful for
        discarding a failed or interrupted recording episode.

        Args:
            delete_images: If ``True``, also remove temporary image files written
                to disk for the current episode.

        Raises:
            RuntimeError: If the dataset is read-only (no writer).
        """
        self._require_writer("clear_episode_buffer")
        self.writer.clear_episode_buffer(delete_images)

    def has_pending_frames(self) -> bool:
        """Check if there are unsaved frames in the episode buffer."""
        if self.writer is None:
            return False
        return self.writer.episode_buffer is not None and self.writer.episode_buffer["size"] > 0

    def finalize(self):
        """Flush all pending work and close writers.

        Must be called after data collection/conversion, otherwise footer metadata
        won't be written to the parquet files and the dataset will be invalid.

        Idempotent — safe to call multiple times.  DatasetWriter.__del__ acts as a
        safety net if this is never called explicitly.
        """
        if self._is_finalized:
            return
        if self.writer is not None:
            self.writer.finalize()
        self._is_finalized = True

    # ── Core Dataset methods ──────────────────────────────────────────

    def __len__(self):
        """Return the number of frames in the selected episodes."""
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        """Return a single frame by index, with all transforms applied.

        Loads the frame from the underlying HF dataset, expands delta-timestamp
        windows, decodes video frames, and applies image transforms. Delegates
        the core logic to :meth:`DatasetReader.get_item`.

        Args:
            idx: Index into the (possibly episode-filtered) dataset.

        Returns:
            Dict mapping feature names to their tensor values for this frame.

        Raises:
            RuntimeError: If the dataset is currently being recorded and
                :meth:`finalize` has not been called yet.
        """
        if self.writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
        reader = self._ensure_reader()
        if reader.hf_dataset is None:
            # One-shot load after finalize()
            reader.load_and_activate()
        return reader.get_item(idx)

    def select_columns(self, column_names: str | list[str]):
        """Select specific columns from the underlying dataset.

        Useful for extracting action sequences during replay without loading all features.
        Returns a ``datasets.Dataset`` containing only the requested columns.
        """
        return self.hf_dataset.select_columns(column_names)

    def get_raw_item(self, idx) -> dict:
        """Get a raw frame without image transforms applied.

        Unlike ``__getitem__``, this returns the raw HF dataset row at the given
        index with no delta-timestamp expansion, video decoding, or image transforms.
        """
        return self.hf_dataset[idx]

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            f"}})"
        )

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform applied to visual observations."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        if self.reader is not None:
            self.reader._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Remove the transform applied to visual observations."""
        self.set_image_transforms(None)

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
        """Upload the dataset to the Hugging Face Hub.

        Creates the repository if it does not exist, uploads all dataset files
        (optionally excluding videos), generates a dataset card, and tags the
        revision with the current codebase version.

        Args:
            branch: Optional branch to push to. Created from the current
                revision if it does not exist.
            tags: Optional list of tags for the dataset card.
            license: License identifier for the dataset card.
            tag_version: If ``True``, create a Git tag for the current codebase
                version.
            push_videos: If ``False``, skip uploading the ``videos/`` directory.
            private: If ``True``, create a private repository.
            allow_patterns: Glob pattern(s) restricting which files to upload.
            upload_large_folder: If ``True``, use ``upload_large_folder`` instead
                of ``upload_folder`` for very large datasets.
            **card_kwargs: Additional keyword arguments forwarded to dataset card
                creation.
        """
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

    def _download(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version."""
        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:
            # Reader is guaranteed to exist here (created in __init__ before _download)
            files = self.reader.get_episodes_file_paths()

        if self._requested_root is None:
            self.meta.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=files,
                    ignore_patterns=ignore_patterns,
                )
            )
        else:
            self._requested_root.mkdir(exist_ok=True, parents=True)
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self._requested_root,
                allow_patterns=files,
                ignore_patterns=ignore_patterns,
            )
            self.meta.root = self._requested_root

        # Propagate resolved root from metadata (single source of truth)
        self.root = self.meta.root
        self.reader.root = self.meta.root

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
        video_files_size_in_mb: int | None = None,
        data_files_size_in_mb: int | None = None,
    ) -> "LeRobotDataset":
        """Create a new LeRobotDataset from scratch for recording data.

        Returns a write-mode dataset with an active :class:`DatasetWriter`. Use
        :meth:`add_frame` / :meth:`save_episode` to populate it, then
        :meth:`finalize` when done.

        Args:
            repo_id: Repository identifier, typically ``'{hf_user}/{dataset_name}'``.
            fps: Frames per second used during data collection.
            features: Feature specification dict mapping feature names to their
                type/shape metadata.
            root: Local directory for dataset storage. Defaults to
                ``$HF_LEROBOT_HOME/{repo_id}``.
            robot_type: Optional robot type string stored in metadata.
            use_videos: If ``True``, visual modalities are stored as MP4 videos.
                If ``False``, they are stored as images.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            image_writer_processes: Number of subprocesses for async image
                writing. ``0`` means use threads only.
            image_writer_threads: Number of threads for async image writing.
            video_backend: Video decoding backend (used when reading back).
            batch_encoding_size: Number of episodes to accumulate before
                batch-encoding videos. ``1`` means encode immediately.
            vcodec: Video codec for encoding. Options include ``'libsvtav1'``,
                ``'h264'``, ``'hevc'``, ``'auto'``.
            metadata_buffer_size: Number of episode metadata records to buffer
                before flushing to parquet.
            streaming_encoding: If ``True``, encode video frames in real-time
                during capture instead of writing images first.
            encoder_queue_maxsize: Max buffered frames per camera when using
                streaming encoding.
            encoder_threads: Threads per encoder instance. ``None`` for auto.

        Returns:
            A new :class:`LeRobotDataset` in write mode.
        """
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
            video_files_size_in_mb=video_files_size_in_mb,
            data_files_size_in_mb=data_files_size_in_mb,
        )
        obj.repo_id = obj.meta.repo_id
        obj._requested_root = obj.meta.root
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj._return_uint8 = False
        obj._batch_encoding_size = batch_encoding_size
        obj._vcodec = vcodec
        obj._encoder_threads = encoder_threads

        # Reader is lazily created on first access (write-only mode)
        obj.reader = None

        # Create writer
        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(fps, vcodec, encoder_queue_maxsize, encoder_threads)
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

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

        Loads metadata from an existing dataset (local or Hub) and creates a
        :class:`DatasetWriter` for appending new episodes. The underlying HF
        dataset is not loaded until :meth:`finalize` is called and data is
        subsequently read.

        Args:
            repo_id: Repository identifier of the existing dataset.
            root: Local directory of the dataset. When provided, Hub downloads
                are materialized directly into this directory. When omitted,
                Hub downloads use a revision-safe snapshot cache under
                ``$HF_LEROBOT_HOME/hub``.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            revision: Git revision (branch, tag, or commit hash). Defaults to
                current codebase version tag.
            force_cache_sync: If ``True``, re-download metadata from the Hub even
                if a local cache exists.
            video_backend: Video decoding backend for reading back data.
            batch_encoding_size: Number of episodes to accumulate before
                batch-encoding videos.
            vcodec: Video codec for encoding.
            image_writer_processes: Subprocesses for async image writing.
            image_writer_threads: Threads for async image writing.
            streaming_encoding: If ``True``, encode video in real-time during
                capture.
            encoder_queue_maxsize: Max buffered frames per camera for streaming.
            encoder_threads: Threads per encoder instance. ``None`` for auto.

        Returns:
            A :class:`LeRobotDataset` in write mode, ready to append episodes.
        """
        if not root:
            raise ValueError(
                "resume() requires an explicit 'root' directory because it creates a DatasetWriter. "
                "Writing into the revision-safe Hub snapshot cache (used when root=None) would corrupt "
                "the shared cache. Please provide a local directory path."
            )
        vcodec = resolve_vcodec(vcodec)
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root)
        obj.revision = revision if revision else CODEBASE_VERSION
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend else get_safe_default_codec()
        obj._return_uint8 = False
        obj._batch_encoding_size = batch_encoding_size
        obj._vcodec = vcodec
        obj._encoder_threads = encoder_threads

        if obj._requested_root is not None:
            obj._requested_root.mkdir(exist_ok=True, parents=True)

        # Load metadata (revision-safe when root is not provided)
        obj.meta = LeRobotDatasetMetadata(
            obj.repo_id, obj._requested_root, obj.revision, force_cache_sync=force_cache_sync
        )
        obj.root = obj.meta.root

        # Reader is lazily created on first access (write-only mode)
        obj.reader = None

        # Create writer for appending
        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                obj.meta.fps, vcodec, encoder_queue_maxsize, encoder_threads
            )
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
            initial_frames=obj.meta.total_frames,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj
