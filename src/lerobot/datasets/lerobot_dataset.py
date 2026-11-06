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
import json
import logging
import shutil
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast, overload
from uuid import uuid4

import datasets
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.configs import DEFAULT_DEPTH_UNIT, DepthEncoderConfig, RGBEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from lerobot.utils.import_utils import get_safe_default_video_backend

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .dataset_reader import BaseDatasetReader, DatasetReader
from .dataset_writer import DatasetWriter
from .distributed import (
    DistributedEpisodeResult,
    DistributedEpisodeSpec,
    DistributedShardSpec,
    DistributedWritePlan,
    DistributedWriteSession,
    commit_receipt,
    publish_distributed_metadata,
    recover_distributed_metadata,
    result_digest,
    validate_distributed_results,
    write_commit_receipt,
    write_distributed_metadata,
)
from .distributed_writer import write_distributed_shard
from .storage import (
    DEFAULT_STORAGE_FORMAT,
    dataset_location,
    is_remote_uri,
    localize_remote_root,
    make_dataset_reader,
)
from .utils import (
    DEFAULT_EPISODES_PATH,
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
    update_chunk_file_indices,
)
from .video_utils import StreamingVideoEncoder

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
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        batch_encoding_size: int = 1,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        *,
        repo_type: str = "dataset",
        token: str | bool | None = None,
        video_decoder_cache_size: int | None = None,
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
                ``$HF_LEROBOT_HOME/hub``. May also be an object-store URI (e.g. ``hf://datasets/{repo_id}``)
                for storage formats that read data in place; only ``meta/`` is then materialized locally.
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
            rgb_encoder (RGBEncoderConfig | None, optional): Video encoder settings for cameras
                (codec, quality, etc.). When ``None``, :func:`~lerobot.configs.video.rgb_encoder_defaults`
                is used by the writer.
            depth_encoder (DepthEncoderConfig | None, optional): Video encoder settings for depth cameras
                (codec, quality, etc.). When ``None``, :func:`~lerobot.configs.video.depth_encoder_defaults`
                is used by the writer.
            encoder_threads (int | None, optional): Number of encoder threads (global). ``None`` lets the
                codec decide.
            streaming_encoding (bool, optional): If True, encode video frames in real-time during capture
                instead of writing PNG images first. This makes save_episode() near-instant. Defaults to False.
            encoder_queue_maxsize (int, optional): Maximum number of frames to buffer per camera when using
                streaming encoding. Defaults to 30 (~1s at 30fps).
            repo_type (str, optional): "dataset" (default) or "bucket" for an HF
                Storage Bucket. With "bucket" and no ``root``, the dataset is read
                in place from ``hf://buckets/{repo_id}`` (map-style access requires
                a non-default storage format; the default format is streaming-only
                on buckets). An explicit ``root`` always wins over ``repo_type``.
            token: Authentication token used while downloading this dataset
                from the Hub. Pass a string token, ``True`` to require the
                locally stored token, ``False`` to disable authentication, or
                ``None`` to use the Hugging Face Hub default. The token is not
                retained on the dataset instance after initialization.
            video_decoder_cache_size (int, optional): For non-default storage formats only, the
                number of open video decoders each DataLoader worker keeps. Larger values cut
                re-reads under a shuffled sampler at the cost of RAM. Defaults to the reader's own
                default (256 for ``"lance"``).

        Note:
            Write-mode parameters (``streaming_encoding``, ``batch_encoding_size``) passed to
            ``__init__`` are deprecated. Use :meth:`create` for new datasets or :meth:`resume`
            to append to existing ones.
        """
        super().__init__()
        self.repo_id = repo_id
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")
        if root is None and repo_type == "bucket":
            root = dataset_location(repo_id, repo_type=repo_type).uri
        # Datasets can live at an object-store root (e.g. ``hf://datasets/...``): a
        # non-default reader reads the data in place and only ``meta/`` is localized.
        self._storage_root = root if root is not None and is_remote_uri(root) else None
        if self._storage_root is not None:
            root = localize_remote_root(
                repo_id, self._storage_root, revision, token=token, force_cache_sync=force_cache_sync
            )
        self._requested_root = Path(root) if root else None
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision: str | None = revision if revision else CODEBASE_VERSION
        self._video_backend = video_backend if video_backend else get_safe_default_video_backend()
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit
        self._batch_encoding_size = batch_encoding_size
        self._encoder_threads = encoder_threads

        if self._requested_root is not None:
            self._requested_root.mkdir(exist_ok=True, parents=True)

        # Load metadata (sets self.root once from the resolved metadata root)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self._requested_root,
            self.revision,
            # an object-store root already refreshed its meta/ at localization
            force_cache_sync=force_cache_sync and self._storage_root is None,
            token=token,
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        self.meta.rescale_depth_stats(self._depth_output_unit)

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

        if self._storage_root is not None and self.meta.storage_format == DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"The dataset at {self._storage_root!r} has the default {DEFAULT_STORAGE_FORMAT!r} "
                "storage format, which cannot be read in place from an object store. For HF Storage "
                "Buckets, use repo_type='bucket' with dataset.streaming=true."
            )

        self.image_transforms = image_transforms
        self.reader: BaseDatasetReader | None
        self.writer: DatasetWriter | None
        if self.meta.storage_format != DEFAULT_STORAGE_FORMAT:
            # non-default formats read the data in place at its root
            reader_kwargs: dict[str, Any] = {
                "meta": self.meta,
                "root": self._storage_root or root,
                "episodes": episodes,
                "delta_timestamps": delta_timestamps,
                "image_transforms": image_transforms,
                "tolerance_s": tolerance_s,
                "return_uint8": return_uint8,
                "depth_output_unit": depth_output_unit,
                "revision": revision,
                "token": token,
            }
            if video_decoder_cache_size is not None:
                reader_kwargs["video_decoder_cache_size"] = video_decoder_cache_size
            self.reader = make_dataset_reader(self.meta.storage_format, **reader_kwargs)
            self.episodes = self.reader.episodes
            self.writer = None
            self._is_finalized = False
            return

        if video_decoder_cache_size is not None:
            raise ValueError("video_decoder_cache_size only applies to non-default storage formats.")
        # The default format is always served by DatasetReader.
        reader = DatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=episodes,
            tolerance_s=tolerance_s,
            video_backend=self._video_backend,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            return_uint8=return_uint8,
            depth_output_unit=depth_output_unit,
        )
        self.reader = reader

        # Load actual data
        if force_cache_sync or not reader.try_load():
            if self.revision is not None and is_valid_version(self.revision):
                if token is None:
                    self.revision = get_safe_version(self.repo_id, self.revision)
                else:
                    self.revision = get_safe_version(self.repo_id, self.revision, token=token)
            self._download(download_videos, token=token)
            reader.load_and_activate()

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
                    self.meta.fps,
                    rgb_encoder,
                    depth_encoder,
                    encoder_queue_maxsize,
                    encoder_threads,
                )
            self.writer = DatasetWriter(
                meta=self.meta,
                root=self.root,
                rgb_encoder=rgb_encoder,
                depth_encoder=depth_encoder,
                encoder_threads=encoder_threads,
                batch_encoding_size=batch_encoding_size,
                streaming_encoder=streaming_enc,
                initial_frames=self.meta.total_frames,
            )
        else:
            self.writer = None

        self._is_finalized = False

    # ── Writer guard ──────────────────────────────────────────────────

    def _require_writer(self, method_name: str) -> DatasetWriter:
        """Return the writer, raising if the dataset is read-only or already finalized."""
        if self.writer is None:
            raise RuntimeError(
                f"Cannot call '{method_name}()' on a read-only dataset. "
                f"Use LeRobotDataset.create() for new recording or "
                f"LeRobotDataset.resume() for resume recording."
            )
        if self._is_finalized:
            raise RuntimeError(
                f"Cannot call '{method_name}()' after finalize(). "
                f"Use LeRobotDataset.resume() to append more episodes."
            )
        return self.writer

    def _require_distributed_write_storage(self) -> None:
        if self.meta.storage_format != DEFAULT_STORAGE_FORMAT or self._storage_root is not None:
            raise ValueError(
                "Distributed writes require a local dataset root with "
                f"storage_format={DEFAULT_STORAGE_FORMAT!r}; got {self.meta.storage_format!r}."
            )

    # ── Reader guard ──────────────────────────────────────────────────

    def _ensure_reader(self) -> BaseDatasetReader:
        """Return the reader, lazily creating the default one on first access.

        ``self.reader`` is only ``None`` in write mode (create/resume), which
        exists for the default format only — non-default formats construct
        their reader in ``__init__``.
        """
        if self.writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
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
                depth_output_unit=self._depth_output_unit,
            )
        return self.reader

    @staticmethod
    def _build_streaming_encoder(
        fps: int,
        rgb_encoder: RGBEncoderConfig | None,
        depth_encoder: DepthEncoderConfig | None,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> StreamingVideoEncoder:
        return StreamingVideoEncoder(
            fps=fps,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )

    # ── Metadata properties ───────────────────────────────────────────

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """Physical unit (``"m"`` or ``"mm"``) depth maps and statistics are returned in on read."""
        return self._depth_output_unit

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
            return len(self.episodes) if self.episodes is not None else self.meta.total_episodes
        return self.reader.num_episodes

    @property
    def features(self) -> dict[str, dict]:
        """Feature specification dict mapping feature names to their type/shape metadata."""
        return self.meta.features

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """The underlying Hugging Face Dataset object"""
        reader = self._ensure_reader()
        if not isinstance(reader, DatasetReader):
            raise AttributeError(
                f"hf_dataset is not available for storage_format={self.meta.storage_format!r}: "
                "data is read through its own dataset reader."
            )
        if reader.hf_dataset is None:
            reader.load_and_activate()
        return reader.hf_dataset

    @property
    def absolute_to_relative_idx(self) -> Mapping[int, int] | None:
        """Mapping from absolute frame indices to relative row positions.

        Non-None only for episode-filtered datasets where absolute indices
        (from metadata) differ from row positions in the filtered view.
        """
        return self._ensure_reader().absolute_to_relative_idx

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
        writer = self._require_writer("add_frame")
        if (
            getattr(self, "_distributed_write_session", None) is not None
            and getattr(self, "_distributed_episode_lock", None) is None
        ):
            raise RuntimeError("Call start_distributed_episode() before adding frames to a session episode.")
        if getattr(self, "_distributed_reused_result", None) is None:
            writer.add_frame(frame)

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
        self._require_writer("save_episode").save_episode(episode_data, parallel_encoding)

    def save_distributed_episode(
        self, spec: DistributedEpisodeSpec, parallel_encoding: bool = True
    ) -> DistributedEpisodeResult:
        """Write the worker-owned artifact for one preallocated episode."""
        writer = self._require_writer("save_distributed_episode")
        if not getattr(self, "_is_distributed_writer", False):
            raise RuntimeError(
                "save_distributed_episode() requires LeRobotDataset.open_distributed_writer()."
            )
        try:
            if getattr(self, "_distributed_active_spec", None) is None:
                raise RuntimeError("Call start_distributed_episode() before saving a distributed episode.")
            if getattr(self, "_distributed_active_spec", None) != spec:
                raise ValueError("Saved specification differs from the started distributed episode.")
            if (
                self._distributed_write_session is not None
                and getattr(self, "_distributed_episode_lock", None) is None
            ):
                raise RuntimeError("Call start_distributed_episode() before saving a session episode.")
            reused = getattr(self, "_distributed_reused_result", None)
            if reused is not None:
                return reused
            result = writer.save_distributed_episode(spec, parallel_encoding)
            if self._distributed_write_session is not None:
                self._distributed_write_session.record_result(result)
            return result
        finally:
            # A failed validation may leave queued image writes. Drain them before
            # another attempt can acquire ownership and remove the staging files.
            if getattr(self, "_distributed_episode_lock", None) is not None:
                writer._wait_image_writer()
            self._release_distributed_episode_lock()

    def start_distributed_episode(self, spec: DistributedEpisodeSpec) -> None:
        """Bind the worker's frame staging directory to a preallocated episode."""
        writer = self._require_writer("start_distributed_episode")
        if not getattr(self, "_is_distributed_writer", False):
            raise RuntimeError(
                "start_distributed_episode() requires LeRobotDataset.open_distributed_writer()."
            )
        if getattr(self, "_distributed_episode_lock", None) is not None:
            raise RuntimeError("Finish the active distributed episode before starting another.")
        session = self._distributed_write_session
        self._distributed_reused_result = None
        self._distributed_active_spec = spec
        if session is not None:
            lock = session.episode_lock(spec)
            lock.acquire()
            self._distributed_episode_lock = lock
            try:
                self._distributed_reused_result = session.reusable_result(spec)
                if self._distributed_reused_result is None:
                    session.cleanup_episode(spec)
                writer.start_distributed_episode(spec)
            except BaseException:
                self._release_distributed_episode_lock()
                raise
        else:
            writer.start_distributed_episode(spec)

    def _release_distributed_episode_lock(self) -> None:
        lock = getattr(self, "_distributed_episode_lock", None)
        if lock is not None:
            lock.release()
            self._distributed_episode_lock = None
        self._distributed_reused_result = None

    def create_distributed_write_plan(
        self,
        *,
        episode_lengths: list[int],
        episode_tasks: list[list[str]],
        max_frames_per_worker: int = 100_000,
        pack_episodes: bool = True,
    ) -> DistributedWritePlan:
        """Allocate a distributed batch after the dataset's currently published episodes."""
        self._require_writer("create_distributed_write_plan")
        self._require_distributed_write_storage()
        self.meta.ensure_readable()
        start_chunk_index, start_file_index = self._next_distributed_file_indices()
        return DistributedWritePlan.from_episode_lengths(
            episode_lengths=episode_lengths,
            episode_tasks=episode_tasks,
            chunks_size=self.meta.chunks_size,
            video_keys=list(self.meta.video_keys),
            start_episode_index=self.meta.total_episodes,
            start_dataset_index=self.meta.total_frames,
            start_chunk_index=start_chunk_index,
            start_file_index=start_file_index,
            task_to_index=self._distributed_task_mapping(),
            max_frames_per_worker=max_frames_per_worker,
            pack_episodes=pack_episodes,
            data_files_size_in_mb=self.meta.data_files_size_in_mb if pack_episodes else None,
            video_files_size_in_mb=self.meta.video_files_size_in_mb if pack_episodes else None,
        )

    def write_distributed_shard(
        self, shard: DistributedShardSpec, episodes: Iterable[Iterable[dict]]
    ) -> list[DistributedEpisodeResult]:
        """Write one task with a single Parquet and one MP4 per video feature.

        ``episodes`` supplies frame iterables in plan order. Size targets are
        checked after each complete episode; the input suffix is left untouched
        and exposed by session.pending_shards() as a new schedulable task. The
        returned results cover only this task's completed prefix. Retries reuse
        its durable boundary and verified results. Requires a persistent session.
        """
        writer = self._require_writer("write_distributed_shard")
        session = getattr(self, "_distributed_write_session", None)
        if session is None or not getattr(self, "_is_distributed_writer", False):
            raise ValueError("write_distributed_shard() requires open_distributed_writer(session=...).")
        if self.has_pending_frames() or getattr(self, "_distributed_episode_lock", None) is not None:
            raise RuntimeError("Finish the active episode before writing a distributed shard.")
        return write_distributed_shard(writer, session, shard, episodes)

    def _distributed_task_mapping(self) -> dict[str, int]:
        if self.meta.tasks is None:
            return {}
        return {str(task): int(row["task_index"]) for task, row in self.meta.tasks.iterrows()}

    def _next_distributed_file_indices(self) -> tuple[int, int]:
        if self.meta.total_episodes == 0:
            return 0, 0
        if self.meta.episodes is None:
            raise RuntimeError("Existing episode metadata is required to plan a distributed append.")

        column_pairs = [
            ("data/chunk_index", "data/file_index"),
            ("meta/episodes/chunk_index", "meta/episodes/file_index"),
            *[
                (f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index")
                for video_key in self.meta.video_keys
            ],
        ]
        occupied_slots = [
            (int(chunk_index), int(file_index))
            for chunk_column, file_column in column_pairs
            if chunk_column in self.meta.episodes.column_names
            and file_column in self.meta.episodes.column_names
            for chunk_index, file_index in zip(
                self.meta.episodes[chunk_column], self.meta.episodes[file_column], strict=True
            )
        ]
        if not occupied_slots:
            raise ValueError("Existing dataset metadata does not contain distributed output file indices.")
        return update_chunk_file_indices(*max(occupied_slots), self.meta.chunks_size)

    def _validate_distributed_plan_base(self, plan: DistributedWritePlan) -> None:
        if not plan.episodes:
            raise ValueError("Distributed plan must contain episodes.")
        if plan.shards:
            if tuple(spec for shard in plan.shards for spec in shard.episodes) != plan.episodes:
                raise ValueError("Distributed shards must cover the plan's episodes exactly once, in order.")
            if any(not shard.episodes for shard in plan.shards):
                raise ValueError("Distributed shards must not be empty.")
            if plan.chunks_size != self.meta.chunks_size:
                raise ValueError("Distributed plan chunks_size differs from the dataset.")
            for field in ("data_files_size_in_mb", "video_files_size_in_mb"):
                limit = getattr(plan, field)
                if limit is not None and limit != getattr(self.meta, field):
                    raise ValueError(f"Distributed plan {field} differs from the dataset.")
        first_spec = plan.episodes[0]
        if first_spec.episode_index != self.meta.total_episodes:
            raise ValueError(
                "Distributed plan starts at episode "
                f"{first_spec.episode_index}, but the dataset currently has {self.meta.total_episodes} episodes."
            )
        if first_spec.dataset_from_index != self.meta.total_frames:
            raise ValueError(
                "Distributed plan starts at frame "
                f"{first_spec.dataset_from_index}, but the dataset currently has {self.meta.total_frames} frames."
            )
        if set(plan.video_keys) != set(self.meta.video_keys):
            raise ValueError(
                "Distributed write plan video_keys must match the dataset video feature keys: "
                f"{sorted(plan.video_keys)!r} != {sorted(self.meta.video_keys)!r}."
            )
        for task, task_index in self._distributed_task_mapping().items():
            if plan.task_to_index.get(task) != task_index:
                raise ValueError(
                    f"Distributed plan must preserve existing task index {task_index} for {task!r}."
                )

        expected_episode_index = self.meta.total_episodes
        expected_dataset_index = self.meta.total_frames
        expected_chunk_index = first_spec.data_chunk_index
        expected_file_index = first_spec.data_file_index
        for spec in plan.episodes:
            if spec.episode_index != expected_episode_index:
                raise ValueError("Distributed plan episode indexes must be contiguous.")
            if spec.dataset_from_index != expected_dataset_index:
                raise ValueError("Distributed plan frame ranges must be contiguous.")
            if (spec.data_chunk_index, spec.data_file_index) != (
                expected_chunk_index,
                expected_file_index,
            ):
                raise ValueError("Distributed plan output file indexes must be contiguous.")
            if spec.task_indices != tuple(plan.task_to_index[task] for task in spec.tasks):
                raise ValueError(
                    f"Distributed plan has invalid task indices for episode {spec.episode_index}."
                )
            expected_episode_index += 1
            expected_dataset_index = spec.dataset_to_index
            expected_chunk_index, expected_file_index = update_chunk_file_indices(
                expected_chunk_index,
                expected_file_index,
                plan.chunks_size,
            )

    def _ensure_distributed_plan_outputs_are_available(self, plan: DistributedWritePlan) -> None:
        for spec in plan.episodes:
            paths = [
                self.root
                / self.meta.data_path.format(
                    chunk_index=spec.data_chunk_index,
                    file_index=spec.data_file_index,
                ),
                self.root
                / DEFAULT_EPISODES_PATH.format(
                    chunk_index=spec.data_chunk_index,
                    file_index=spec.data_file_index,
                ),
            ]
            if self.meta.video_path is not None:
                paths.extend(
                    self.root
                    / self.meta.video_path.format(
                        video_key=video_key,
                        chunk_index=spec.data_chunk_index,
                        file_index=spec.data_file_index,
                    )
                    for video_key in plan.video_keys
                )
            collisions = [path for path in paths if path.exists()]
            if collisions:
                raise FileExistsError(
                    f"Distributed plan output already exists for episode {spec.episode_index}: "
                    f"{collisions[0]}"
                )

    def create_distributed_write_session(self, plan: DistributedWritePlan) -> DistributedWriteSession:
        """Persist a new or append distributed write plan for explicit recovery."""
        self._require_writer("create_distributed_write_session")
        self._require_distributed_write_storage()
        self._validate_distributed_plan_base(plan)
        self._ensure_distributed_plan_outputs_are_available(plan)
        return DistributedWriteSession.create(
            self.root,
            plan,
            data_path_template=self.meta.data_path,
            video_path_template=self.meta.video_path,
            camera_keys=tuple(self.meta.camera_keys),
            depth_keys=tuple(self.meta.depth_keys),
        )

    @classmethod
    def resume_distributed_write_session(cls, root: str | Path) -> DistributedWriteSession:
        """Load persisted distributed progress and expose specs still requiring execution."""
        return DistributedWriteSession.resume(root)

    def commit_distributed_results(
        self, plan: DistributedWritePlan, results: list[DistributedEpisodeResult]
    ) -> None:
        """Validate worker artifacts and publish metadata with crash recovery."""
        self._require_writer("commit_distributed_results")
        self._require_distributed_write_storage()
        recover_distributed_metadata(self.root)
        receipt = commit_receipt(self.root, plan)
        if receipt.is_file():
            with receipt.open(encoding="utf-8") as file:
                committed = json.load(file)
            if committed["digest"] != result_digest(results):
                raise ValueError("Conflicting results for an already committed distributed plan.")
            validate_distributed_results(self.root, self.meta.data_path, self.meta.video_path, plan, results)
            self.meta._load_metadata()
            self._cleanup_committed_distributed_session(plan)
            return
        self._validate_distributed_plan_base(plan)
        ordered_results = validate_distributed_results(
            self.root,
            self.meta.data_path,
            self.meta.video_path,
            plan,
            results,
        )
        staging_root = self.root / f".distributed-metadata-{uuid4().hex}"
        try:
            write_distributed_metadata(
                staging_root,
                deepcopy(self.meta.info),
                plan,
                ordered_results,
                existing_metadata_root=self.root if self.meta.total_episodes else None,
                existing_stats=deepcopy(self.meta.stats),
            )
            write_commit_receipt(staging_root, plan, ordered_results)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        try:
            publish_distributed_metadata(self.root, staging_root)
        except Exception as error:
            raise RuntimeError(
                f"Failed to publish distributed metadata. Staged metadata remains at {staging_root / 'meta'}."
            ) from error
        shutil.rmtree(staging_root, ignore_errors=True)
        self.meta._load_metadata()
        self._cleanup_committed_distributed_session(plan)

    def _cleanup_committed_distributed_session(self, plan: DistributedWritePlan) -> None:
        try:
            session = DistributedWriteSession.resume(self.root)
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if session.plan == plan:
            for shard in session.task_specs():
                # A worker can die after its success record, before cleaning its
                # staging directory. Committed tasks can no longer write there.
                shutil.rmtree(session.shard_staging_root(shard), ignore_errors=True)
            shutil.rmtree(session.session_root, ignore_errors=True)

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
        self._require_writer("clear_episode_buffer").clear_episode_buffer(delete_images)

    def has_pending_frames(self) -> bool:
        """Check if there are unsaved frames in the episode buffer."""
        if self.writer is None:
            return False
        # save_episode pops "size", so a buffer abandoned mid-save has nothing pending.
        return self.writer.episode_buffer is not None and self.writer.episode_buffer.get("size", 0) > 0

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
        self._release_distributed_episode_lock()
        self._is_finalized = True

    # ── Core Dataset methods ──────────────────────────────────────────

    def __len__(self):
        """Return the number of frames in the selected episodes."""
        return self.num_frames

    @overload
    def __getitem__(self, idx: int) -> dict: ...

    @overload
    def __getitem__(self, idx: slice) -> list[dict]: ...

    def __getitem__(self, idx: int | slice) -> dict | list[dict]:
        """Return one frame or a slice of frames, with all transforms applied.

        Loads the frame from the underlying HF dataset, expands delta-timestamp
        windows, decodes video frames, and applies image transforms. Delegates
        the core logic to :class:`DatasetReader`.

        Args:
            idx: Integer index or slice into the possibly episode-filtered dataset.

        Returns:
            A frame dictionary for an integer index, or a list of frame
            dictionaries for a slice.

        Raises:
            RuntimeError: If the dataset is currently being recorded and
                :meth:`finalize` has not been called yet.
        """
        if isinstance(idx, slice):
            return [self[item_idx] for item_idx in range(*idx.indices(len(self)))]

        return self._ensure_reader().get_item(idx)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return self._ensure_reader().get_items(list(indices))

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
        self._ensure_reader().set_image_transforms(image_transforms)
        self.image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Remove the transform applied to visual observations."""
        if self.reader is not None:
            self.reader.set_image_transforms(None)
        self.image_transforms = None

    # ── Hub methods (stay on facade) ──────────────────────────────────

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool | None = None,
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
            private: If ``True``, create a private repository. If ``None``
                (default), defer to the org default on the Hub (only affects orgs).
            allow_patterns: Glob pattern(s) restricting which files to upload.
            upload_large_folder: Kept for backward compatibility and ignored.
                ``upload_folder`` handles datasets of all sizes.
            **card_kwargs: Additional keyword arguments forwarded to dataset card
                creation.
        """
        if self.meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise NotImplementedError(
                f"push_to_hub is not supported for storage_format={self.meta.storage_format!r}: "
                "the data files are not managed by LeRobotDataset."
            )
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

        hub_api.upload_folder(
            repo_id=self.repo_id,
            folder_path=self.root,
            repo_type="dataset",
            revision=branch,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, repo_id=self.repo_id, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def _download(self, download_videos: bool = True, *, token: str | bool | None = None) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version."""
        ignore_patterns = None if download_videos else "videos/"
        files = None
        token_kwargs = {} if token is None else {"token": token}
        # Only the default format downloads data; __init__ builds its DatasetReader before _download.
        reader = cast(DatasetReader, self.reader)
        if self.episodes is not None:
            files = reader.get_episodes_file_paths()

        if self._requested_root is None:
            self.meta.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=files,
                    ignore_patterns=ignore_patterns,
                    **token_kwargs,
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
                **token_kwargs,
            )
            self.meta.root = self._requested_root

        # Propagate resolved root from metadata (single source of truth)
        self.root = self.meta.root
        reader.root = self.meta.root

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
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
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
            rgb_encoder: Video encoder settings for cameras (codec, quality, etc.).
                When ``None``, :func:`~lerobot.configs.video.rgb_encoder_defaults` is used.
            depth_encoder: Video encoder settings for depth cameras (codec, quality, etc.).
                When ``None``, :func:`~lerobot.configs.video.depth_encoder_defaults` is used.
            encoder_threads: Number of encoder threads (global). ``None``
                lets the codec decide.
            metadata_buffer_size: Number of episode metadata records to buffer
                before flushing to parquet.
            streaming_encoding: If ``True``, encode video frames in real-time
                during capture instead of writing images first.
            encoder_queue_maxsize: Max buffered frames per camera when using
                streaming encoding.

        Returns:
            A new :class:`LeRobotDataset` in write mode.
        """
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
        obj._video_backend = video_backend if video_backend is not None else get_safe_default_video_backend()
        obj._return_uint8 = False
        obj._depth_output_unit = DEFAULT_DEPTH_UNIT
        obj._batch_encoding_size = batch_encoding_size
        obj._encoder_threads = encoder_threads
        obj._storage_root = None

        # Reader is lazily created on first access (write-only mode)
        obj.reader = None

        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
            )
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
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
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        *,
        token: str | bool | None = None,
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
            rgb_encoder: Video encoder settings for cameras (codec, quality, etc.).
                When ``None``, :func:`~lerobot.configs.video.rgb_encoder_defaults` is used.
            depth_encoder: Video encoder settings for depth cameras (codec, quality, etc.).
                When ``None``, :func:`~lerobot.configs.video.depth_encoder_defaults` is used.
            encoder_threads: Number of encoder threads (global). ``None``
                lets the codec decide.
            image_writer_processes: Subprocesses for async image writing.
            image_writer_threads: Threads for async image writing.
            streaming_encoding: If ``True``, encode video in real-time during
                capture.
            encoder_queue_maxsize: Max buffered frames per camera for streaming.
            token: Authentication token used if metadata must be downloaded
                from the Hub. The token is not retained on the dataset instance.

        Returns:
            A :class:`LeRobotDataset` in write mode, ready to append episodes.
        """
        if not root:
            raise ValueError(
                "resume() requires an explicit 'root' directory because it creates a DatasetWriter. "
                "Writing into the revision-safe Hub snapshot cache (used when root=None) would corrupt "
                "the shared cache. Please provide a local directory path."
            )
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root)
        obj.revision = revision if revision else CODEBASE_VERSION
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend else get_safe_default_video_backend()
        obj._return_uint8 = False
        obj._depth_output_unit = DEFAULT_DEPTH_UNIT
        obj._batch_encoding_size = batch_encoding_size

        if obj._requested_root is not None:
            obj._requested_root.mkdir(exist_ok=True, parents=True)

        # Load metadata (revision-safe when root is not provided)
        obj.meta = LeRobotDatasetMetadata(
            obj.repo_id,
            obj._requested_root,
            obj.revision,
            force_cache_sync=force_cache_sync,
            token=token,
        )

        obj._encoder_threads = encoder_threads
        obj._storage_root = None
        obj.root = obj.meta.root

        # Reader is lazily created on first access (write-only mode)
        obj.reader = None

        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                obj.meta.fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
            )
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
            initial_frames=obj.meta.total_frames,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj

    @classmethod
    def open_distributed_writer(
        cls,
        repo_id: str,
        root: str | Path,
        *,
        rgb_encoder: RGBEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        session: DistributedWriteSession | None = None,
    ) -> "LeRobotDataset":
        """Open a worker writer for a preallocated distributed batch."""
        if is_remote_uri(root):
            raise ValueError(
                "Distributed writes require a local dataset root; object-store URIs are not supported."
            )
        root = Path(root)
        recover_distributed_metadata(root)
        if session is not None and session.root != root:
            raise ValueError(f"Distributed session root {session.root} does not match writer root {root}.")
        dataset = cls.resume(
            repo_id,
            root=root,
            batch_encoding_size=1,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            encoder_threads=encoder_threads,
            streaming_encoding=False,
        )
        dataset._require_distributed_write_storage()
        if session is not None and not commit_receipt(root, session.plan).is_file():
            dataset._validate_distributed_plan_base(session.plan)
        dataset._is_distributed_writer = True
        dataset._distributed_write_session = session
        return dataset
