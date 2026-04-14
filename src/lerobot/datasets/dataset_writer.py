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
"""Private writer component for LeRobotDataset. Handles sequential recording (episode buffer, ParquetWriter, image writer, video encoding)."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import shutil
import tempfile
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import PIL.Image
import pyarrow.parquet as pq
import torch

from .compute_stats import compute_episode_stats
from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import (
    get_hf_features_from_features,
    validate_episode_buffer,
    validate_frame,
)
from .image_writer import AsyncImageWriter, write_image
from .io_utils import (
    embed_images,
    get_file_size_in_mb,
    load_episodes,
    write_info,
)
from .utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_IMAGE_PATH,
    update_chunk_file_indices,
)
from .video_utils import (
    StreamingVideoEncoder,
    concatenate_video_files,
    encode_video_frames,
    get_video_duration_in_s,
)

logger = logging.getLogger(__name__)


def _encode_video_worker(
    video_key: str,
    episode_index: int,
    root: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    encoder_threads: int | None = None,
) -> Path:
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    img_dir = (root / fpath).parent
    encode_video_frames(
        img_dir, temp_path, fps, vcodec=vcodec, overwrite=True, encoder_threads=encoder_threads
    )
    shutil.rmtree(img_dir)
    return temp_path


class DatasetWriter:
    """Encapsulates write-side state and methods for LeRobotDataset.

    Owns: episode_buffer, image_writer, _pq_writer (ParquetWriter), _latest_episode,
    _current_file_start_frame, _streaming_encoder, _episodes_since_last_encoding, _recorded_frames.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        vcodec: str,
        encoder_threads: int | None,
        batch_encoding_size: int,
        streaming_encoder: StreamingVideoEncoder | None = None,
        initial_frames: int = 0,
    ):
        """Initialize the writer with metadata, codec, and encoding config.

        Args:
            meta: Dataset metadata instance (used for feature schema, chunk
                settings, and episode persistence).
            root: Local dataset root directory.
            vcodec: Video codec for encoding (e.g. ``'libsvtav1'``, ``'h264'``).
            encoder_threads: Threads per encoder instance. ``None`` for auto.
            batch_encoding_size: Number of episodes to accumulate before
                batch-encoding videos.
            streaming_encoder: Optional pre-built :class:`StreamingVideoEncoder`
                for real-time encoding. ``None`` disables streaming mode.
            initial_frames: Starting frame count (non-zero when resuming).
        """
        self._meta = meta
        self._root = root
        self._vcodec = vcodec
        self._encoder_threads = encoder_threads
        self._batch_encoding_size = batch_encoding_size
        self._streaming_encoder = streaming_encoder

        # Writer state
        self.image_writer: AsyncImageWriter | None = None
        self.episode_buffer: dict = self._create_episode_buffer()
        self._pq_writer: pq.ParquetWriter | None = None
        self._latest_episode: dict | None = None
        self._current_file_start_frame: int | None = None
        self._episodes_since_last_encoding: int = 0
        self._recorded_frames: int = initial_frames
        self._finalized = False

    def _create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self._meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self._meta.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self._root / fpath

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def _save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1
    ) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath, compress_level=compress_level)
        else:
            self.image_writer.save_image(image=image, fpath=fpath, compress_level=compress_level)

    def add_frame(self, frame: dict) -> None:
        """
        Add a single frame to the current episode buffer.

        Apart from images written to a temporary directory, nothing is written to disk
        until ``save_episode()`` is called.

        The caller must provide all user-defined features plus ``"task"``, and must
        not provide ``"timestamp"`` or ``"frame_index"``; those are computed
        automatically.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self._meta.features)

        if self.episode_buffer is None:
            self.episode_buffer = self._create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame_index / self._meta.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))

        # Start streaming encoder on first frame of episode
        if frame_index == 0 and self._streaming_encoder is not None:
            self._streaming_encoder.start_episode(
                video_keys=list(self._meta.video_keys),
                temp_dir=self._root,
            )

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self._meta.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self._meta.features.keys()}'."
                )

            if self._meta.features[key]["dtype"] == "video" and self._streaming_encoder is not None:
                self._streaming_encoder.feed_frame(key, frame[key])
                self.episode_buffer[key].append(None)
            elif self._meta.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                compress_level = 1 if self._meta.features[key]["dtype"] == "video" else 6
                self._save_image(frame[key], img_path, compress_level)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(
        self,
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        """Save the current episode in self.episode_buffer to disk."""
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self._meta.total_episodes, self._meta.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self._meta.total_frames, self._meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self._meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self._meta.get_task_index(task) for task in tasks])

        for key, ft in self._meta.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()

        has_video_keys = len(self._meta.video_keys) > 0
        use_streaming = self._streaming_encoder is not None and has_video_keys
        use_batched_encoding = self._batch_encoding_size > 1

        if use_streaming:
            non_video_buffer = {
                k: v
                for k, v in episode_buffer.items()
                if self._meta.features.get(k, {}).get("dtype") not in ("video",)
            }
            non_video_features = {k: v for k, v in self._meta.features.items() if v["dtype"] != "video"}
            ep_stats = compute_episode_stats(non_video_buffer, non_video_features)
        else:
            ep_stats = compute_episode_stats(episode_buffer, self._meta.features)

        ep_metadata = self._save_episode_data(episode_buffer)

        if use_streaming:
            streaming_results = self._streaming_encoder.finish_episode()
            for video_key in self._meta.video_keys:
                temp_path, video_stats = streaming_results[video_key]
                if video_stats is not None:
                    ep_stats[video_key] = {
                        k: v if k == "count" else np.squeeze(v.reshape(1, -1, 1, 1) / 255.0, axis=0)
                        for k, v in video_stats.items()
                    }
                ep_metadata.update(self._save_episode_video(video_key, episode_index, temp_path=temp_path))
        elif has_video_keys and not use_batched_encoding:
            num_cameras = len(self._meta.video_keys)
            if parallel_encoding and num_cameras > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cameras) as executor:
                    future_to_key = {
                        executor.submit(
                            _encode_video_worker,
                            video_key,
                            episode_index,
                            self._root,
                            self._meta.fps,
                            self._vcodec,
                            self._encoder_threads,
                        ): video_key
                        for video_key in self._meta.video_keys
                    }

                    results = {}
                    for future in concurrent.futures.as_completed(future_to_key):
                        video_key = future_to_key[future]
                        try:
                            temp_path = future.result()
                            results[video_key] = temp_path
                        except Exception as exc:
                            logger.error(f"Video encoding failed for {video_key}: {exc}")
                            raise exc

                for video_key in self._meta.video_keys:
                    temp_path = results[video_key]
                    ep_metadata.update(
                        self._save_episode_video(video_key, episode_index, temp_path=temp_path)
                    )
            else:
                for video_key in self._meta.video_keys:
                    ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` need to be executed after encoding the videos
        self._meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            self._episodes_since_last_encoding += 1
            if self._episodes_since_last_encoding == self._batch_encoding_size:
                start_ep = self._meta.total_episodes - self._batch_encoding_size
                end_ep = self._meta.total_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self._episodes_since_last_encoding = 0

        if episode_data is None:
            self.clear_episode_buffer(delete_images=len(self._meta.image_keys) > 0)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None) -> None:
        """Batch save videos for multiple episodes."""
        if end_episode is None:
            end_episode = self._meta.total_episodes

        logger.info(
            f"Batch encoding {self._batch_encoding_size} videos for episodes {start_episode} to {end_episode - 1}"
        )

        chunk_idx = self._meta.episodes[start_episode]["data/chunk_index"]
        file_idx = self._meta.episodes[start_episode]["data/file_index"]
        episode_df_path = self._root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        episode_df = pd.read_parquet(episode_df_path)

        for ep_idx in range(start_episode, end_episode):
            logger.info(f"Encoding videos for episode {ep_idx}")

            if (
                self._meta.episodes[ep_idx]["data/chunk_index"] != chunk_idx
                or self._meta.episodes[ep_idx]["data/file_index"] != file_idx
            ):
                episode_df.to_parquet(episode_df_path)
                self._meta.episodes = load_episodes(self._root)

                chunk_idx = self._meta.episodes[ep_idx]["data/chunk_index"]
                file_idx = self._meta.episodes[ep_idx]["data/file_index"]
                episode_df_path = self._root / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                episode_df = pd.read_parquet(episode_df_path)

            video_ep_metadata = {}
            for video_key in self._meta.video_keys:
                video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
            video_ep_metadata.pop("episode_index")
            video_ep_df = pd.DataFrame(video_ep_metadata, index=[ep_idx]).convert_dtypes(
                dtype_backend="pyarrow"
            )

            episode_df = episode_df.combine_first(video_ep_df)
            episode_df.to_parquet(episode_df_path)
            self._meta.episodes = load_episodes(self._root)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        """Save episode data to a parquet file."""
        # Use metadata features as the authoritative schema
        hf_features = get_hf_features_from_features(self._meta.features)
        ep_dict = {key: episode_buffer[key] for key in hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_num_frames = len(ep_dataset)

        if self._latest_episode is None:
            chunk_idx, file_idx = 0, 0
            global_frame_index = 0
            self._current_file_start_frame = 0
            if self._meta.episodes is not None and len(self._meta.episodes) > 0:
                latest_ep = self._meta.episodes[-1]
                global_frame_index = latest_ep["dataset_to_index"]
                chunk_idx = latest_ep["data/chunk_index"]
                file_idx = latest_ep["data/file_index"]

                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self._meta.chunks_size)
                self._current_file_start_frame = global_frame_index
        else:
            latest_ep = self._latest_episode
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]
            global_frame_index = latest_ep["index"][-1] + 1

            latest_path = self._root / self._meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_file_size_in_mb(latest_path)

            frames_in_current_file = global_frame_index - self._current_file_start_frame
            av_size_per_frame = (
                latest_size_in_mb / frames_in_current_file if frames_in_current_file > 0 else 0
            )

            if latest_size_in_mb + av_size_per_frame * ep_num_frames >= self._meta.data_files_size_in_mb:
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self._meta.chunks_size)
                self.close_writer()
                self._current_file_start_frame = global_frame_index

        ep_dict["data/chunk_index"] = chunk_idx
        ep_dict["data/file_index"] = file_idx

        path = self._root / self._meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)

        table = ep_dataset.with_format("arrow")[:]
        if not self._pq_writer:
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )
        self._pq_writer.write_table(table)

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_index,
            "dataset_to_index": global_frame_index + ep_num_frames,
        }

        self._latest_episode = {**ep_dict, **metadata}
        self._recorded_frames += ep_num_frames

        return metadata

    def _save_episode_video(
        self,
        video_key: str,
        episode_index: int,
        temp_path: Path | None = None,
    ) -> dict:
        if temp_path is None:
            ep_path = self._encode_temporary_episode_video(video_key, episode_index)
        else:
            ep_path = temp_path

        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if (
            episode_index == 0
            or self._meta.latest_episode is None
            or f"videos/{video_key}/chunk_index" not in self._meta.latest_episode
        ):
            chunk_idx, file_idx = 0, 0
            if self._meta.episodes is not None and len(self._meta.episodes) > 0:
                old_chunk_idx = self._meta.episodes[-1][f"videos/{video_key}/chunk_index"]
                old_file_idx = self._meta.episodes[-1][f"videos/{video_key}/file_index"]
                chunk_idx, file_idx = update_chunk_file_indices(
                    old_chunk_idx, old_file_idx, self._meta.chunks_size
                )
            latest_duration_in_s = 0.0
            new_path = self._root / self._meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_path), str(new_path))
        else:
            latest_ep = self._meta.latest_episode
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"][0]
            file_idx = latest_ep[f"videos/{video_key}/file_index"][0]

            latest_path = self._root / self._meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_file_size_in_mb(latest_path)
            latest_duration_in_s = latest_ep[f"videos/{video_key}/to_timestamp"][0]

            if latest_size_in_mb + ep_size_in_mb >= self._meta.video_files_size_in_mb:
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self._meta.chunks_size)
                new_path = self._root / self._meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep_path), str(new_path))
                latest_duration_in_s = 0.0
            else:
                concatenate_video_files(
                    [latest_path, ep_path],
                    latest_path,
                )

        # Remove temporary directory
        shutil.rmtree(str(ep_path.parent))

        # Update video info (only needed when first episode is encoded)
        if episode_index == 0:
            self._meta.update_video_info(video_key)
            write_info(self._meta.info, self._meta.root)

        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }
        return metadata

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """Discard the current episode buffer and optionally delete temp images.

        Args:
            delete_images: If ``True``, remove temporary image directories
                written for the current episode.
        """
        # Cancel streaming encoder if active
        if self._streaming_encoder is not None:
            self._streaming_encoder.cancel_episode()

        if delete_images:
            if self.image_writer is not None:
                self._wait_image_writer()
            episode_index = self.episode_buffer["episode_index"]
            # episode_index is `int` when freshly created, but becomes `np.ndarray` after
            # save_episode() mutates the buffer. Handle both types here.
            if isinstance(episode_index, np.ndarray):
                episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
            for cam_key in self._meta.image_keys:
                img_dir = self._get_image_file_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        self.episode_buffer = self._create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        """Start an :class:`AsyncImageWriter` for background image persistence.

        Args:
            num_processes: Number of subprocesses. ``0`` means threads only.
            num_threads: Number of threads per process.
        """
        if isinstance(self.image_writer, AsyncImageWriter):
            logger.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """Stop the image writer (needed before pickling the dataset for DataLoader)."""
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        """Use ffmpeg to convert frames stored as png into mp4 videos."""
        return _encode_video_worker(
            video_key, episode_index, self._root, self._meta.fps, self._vcodec, self._encoder_threads
        )

    def close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None

    def flush_pending_videos(self) -> None:
        """Flush any pending video encoding (streaming or batch).

        For streaming encoding: closes the encoder.
        For batch encoding: encodes any remaining episodes that haven't been batch-encoded yet.
        """
        if self._streaming_encoder is not None:
            self._streaming_encoder.close()
        elif self._episodes_since_last_encoding > 0:
            start_ep = self._meta.total_episodes - self._episodes_since_last_encoding
            end_ep = self._meta.total_episodes
            logger.info(
                f"Encoding remaining {self._episodes_since_last_encoding} episodes, "
                f"from episode {start_ep} to {end_ep - 1}"
            )
            self._batch_save_episode_video(start_ep, end_ep)

    def cancel_pending_videos(self) -> None:
        """Cancel any in-progress streaming encoding without flushing."""
        if self._streaming_encoder is not None:
            self._streaming_encoder.cancel_episode()

    def cleanup_interrupted_episode(self, episode_index: int) -> None:
        """Remove temporary image directories for an interrupted episode."""
        for key in self._meta.video_keys:
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            if img_dir.exists():
                logger.debug(
                    f"Cleaning up interrupted episode images for episode {episode_index}, camera {key}"
                )
                shutil.rmtree(img_dir)

    def finalize(self) -> None:
        """Flush all pending work and release all resources.

        Idempotent — safe to call multiple times.
        """
        if getattr(self, "_finalized", False):
            return
        # 1. Wait for async image writes to complete, then stop
        if self.image_writer is not None:
            self.image_writer.wait_until_done()
            self.image_writer.stop()
            self.image_writer = None
        # 2. Flush pending video encoding (streaming or batch)
        self.flush_pending_videos()
        # 3. Close own parquet writer
        self.close_writer()
        # 4. Finalize metadata (idempotent)
        self._meta.finalize()
        self._finalized = True

    def __del__(self):
        """Safety net: release resources on garbage collection."""
        # During interpreter shutdown, referenced objects may already be collected.
        with contextlib.suppress(Exception):
            self.finalize()
