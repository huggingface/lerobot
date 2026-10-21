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

"""Convert to/from the ``"mcap"`` storage format.

There is no external ``lerobot-mcap`` converter package (unlike Lance, which has
``lerobot-lancedb``), so this ships in-tree under the ``mcap`` extra. Two entry points:

- :func:`convert_lerobot_dataset_to_mcap` re-exports an existing default-format
  ``LeRobotDataset`` as mcap (one ``.mcap`` file per episode, one channel per feature
  key, using the ``topic_for_key`` naming convention).
- :func:`ingest_mcap_episodes` onboards raw, independently-recorded ``.mcap`` files
  (arbitrary topic names, genuinely async/multi-rate channels) given an explicit
  :class:`ChannelSpec` map -- the scenario a real mcap recording pipeline produces,
  as opposed to a dataset that started out synced and was merely repackaged.

Both build the ``meta/episodes`` sync index (:mod:`lerobot.datasets.mcap_backend`)
that lets the reader map fixed-grid indices back onto real message timestamps.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import datasets
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from lerobot.utils.import_utils import _mcap_available, require_package

from .dataset_metadata import LeRobotDatasetMetadata
from .io_utils import load_image_as_numpy, write_episodes, write_info, write_stats, write_tasks
from .lerobot_dataset import LeRobotDataset
from .mcap_utils import (
    EPISODE_START_TIME_COLUMN,
    MCAP_CHUNK_INDEX_COLUMN,
    MCAP_FILE_INDEX_COLUMN,
    build_channel_time_index,
    channel_schema_json,
    episode_mcap_path,
    log_time_column,
    pack_tensor,
    topic_for_key,
    write_topic_map,
)
from .utils import DEFAULT_CHUNK_SIZE, DatasetInfo

if TYPE_CHECKING or _mcap_available:
    from mcap.reader import make_reader
    from mcap.writer import Writer

_COMPUTED_KEYS = ("timestamp", "frame_index", "episode_index", "index")


def _stored_keys(meta: LeRobotDatasetMetadata) -> tuple[list[str], list[str]]:
    """(tabular_keys, video_keys) -- the feature keys backed by real mcap messages."""
    tabular_keys = [
        key
        for key in meta.features
        if key not in _COMPUTED_KEYS and key not in meta.video_keys and key not in meta.image_keys
    ]
    return tabular_keys, list(meta.video_keys)


def _np_dtype_for(meta: LeRobotDatasetMetadata, key: str) -> np.dtype:
    return np.dtype(meta.features[key]["dtype"])


def _encode_png(frame_uint8_chw: np.ndarray) -> bytes:
    """Lossless encode -- keeps the mcap round-trip bit-exact against the source
    dataset for testing. The reader is codec-agnostic (PIL sniffs the format), so
    a later revision can switch to JPEG/H.264 purely as a converter-side choice."""
    hwc = np.transpose(frame_uint8_chw, (1, 2, 0))
    buf = io.BytesIO()
    PILImage.fromarray(hwc).save(buf, format="PNG")
    return buf.getvalue()


def _write_episode_mcap(
    path: Path,
    src: LeRobotDataset,
    ep_idx: int,
    ep_start: int,
    ep_end: int,
    tabular_keys: list[str],
    video_keys: list[str],
) -> None:
    """Write one episode's frames as per-message mcap channels, one per feature key."""
    fps = float(src.meta.fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()
        channel_ids = {}
        for key in tabular_keys:
            dtype = _np_dtype_for(src.meta, key)
            shape = tuple(src.meta.features[key].get("shape") or ())
            schema_id = writer.register_schema(
                name=key, encoding="jsonschema", data=channel_schema_json(str(dtype), shape)
            )
            channel_ids[key] = writer.register_channel(
                topic=topic_for_key(key), message_encoding="raw", schema_id=schema_id
            )
        for key in video_keys:
            channel_ids[key] = writer.register_channel(
                topic=topic_for_key(key), message_encoding="png", schema_id=0
            )

        for i, abs_idx in enumerate(range(ep_start, ep_end)):
            frame = src[abs_idx]
            log_time_ns = round(i / fps * 1e9)
            for key in tabular_keys:
                dtype = _np_dtype_for(src.meta, key)
                array = frame[key].numpy().astype(dtype, copy=False)
                writer.add_message(
                    channel_id=channel_ids[key],
                    log_time=log_time_ns,
                    data=pack_tensor(array),
                    publish_time=log_time_ns,
                )
            for key in video_keys:
                # src was opened with return_uint8=True, so frame[key] is already uint8 CHW.
                frame_uint8 = frame[key].numpy()
                writer.add_message(
                    channel_id=channel_ids[key],
                    log_time=log_time_ns,
                    data=_encode_png(frame_uint8),
                    publish_time=log_time_ns,
                )
        writer.finish()


def convert_lerobot_dataset_to_mcap(
    src: LeRobotDataset, dst_root: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Re-export ``src`` (an already-loaded default-format ``LeRobotDataset``) as an
    mcap dataset rooted at ``dst_root``.

    ``src`` is read with a plain, transform-free reader internally (its own
    ``delta_timestamps``/``image_transforms`` are ignored) so every frame comes
    back as raw per-key tensors ready to pack into mcap messages.
    """
    require_package("mcap", extra="mcap")
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    clean_src = LeRobotDataset(src.repo_id, root=src.root, return_uint8=True)
    meta = clean_src.meta
    if meta.depth_keys or meta.image_keys:
        raise NotImplementedError(
            f"mcap conversion only supports plain RGB video features in v1; found "
            f"depth_keys={meta.depth_keys}, image_keys={meta.image_keys}."
        )
    tabular_keys, video_keys = _stored_keys(meta)
    stored_keys = tabular_keys + video_keys
    topics = [topic_for_key(key) for key in stored_keys]

    episode_records = []
    for ep_idx in range(meta.total_episodes):
        ep = meta.episodes[ep_idx]
        ep_start, ep_end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        chunk_index, file_index = ep_idx // chunk_size, ep_idx % chunk_size
        path = episode_mcap_path(dst_root, chunk_index, file_index)
        _write_episode_mcap(path, clean_src, ep_idx, ep_start, ep_end, tabular_keys, video_keys)

        # Re-scan what was actually written (single source of truth for the sync
        # index -- see mcap_backend.py's module docstring) rather than trusting
        # the writer loop's own bookkeeping.
        time_index = build_channel_time_index(path, topics)
        episode_start_ns = min(int(time_index[topic][0]) for topic in topics if len(time_index[topic]))

        record = {
            "episode_index": ep_idx,
            "dataset_from_index": ep_start,
            "dataset_to_index": ep_end,
            MCAP_CHUNK_INDEX_COLUMN: chunk_index,
            MCAP_FILE_INDEX_COLUMN: file_index,
            EPISODE_START_TIME_COLUMN: episode_start_ns,
        }
        for key, topic in zip(stored_keys, topics, strict=True):
            record[log_time_column(key)] = time_index[topic].tolist()
        if "tasks" in meta.episodes.column_names:
            record["tasks"] = ep["tasks"]
        episode_records.append(record)

    write_info(
        _mcap_info(meta),
        dst_root,
    )
    write_tasks(meta.tasks, dst_root)
    if meta.stats is not None:
        write_stats(meta.stats, dst_root)
    write_episodes(datasets.Dataset.from_list(episode_records), dst_root)
    write_topic_map(dst_root, dict(zip(stored_keys, topics, strict=True)))


def _mcap_info(meta: LeRobotDatasetMetadata):
    from dataclasses import replace

    return replace(meta.info, storage_format="mcap", video_path=None)


@dataclass
class ChannelSpec:
    """Maps one raw mcap topic to one LeRobot feature key for :func:`ingest_mcap_episodes`.

    Raw mcap channels carry no inherent LeRobot dtype/shape -- unlike
    :func:`convert_lerobot_dataset_to_mcap`, which already has ``meta.features`` to read
    from, ingestion needs the caller to supply it (except for ``video`` channels, whose
    shape is inferred by decoding the first message).
    """

    key: str
    topic: str
    kind: Literal["tabular", "video"]
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


def _peek_first_message(mcap_path: Path, topic: str) -> bytes | None:
    require_package("mcap", extra="mcap")
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(topics=[topic]):
            return message.data
    return None


def ingest_mcap_episodes(
    mcap_paths: list[Path],
    channels: list[ChannelSpec],
    dst_root: Path,
    fps: int,
    robot_type: str | None = None,
    tasks: str | list[str] = "unspecified task",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Onboard raw, independently-recorded ``.mcap`` files (one per episode, arbitrary
    topic names) into a LeRobot mcap-format dataset at ``dst_root``.

    Files are copied byte-for-byte (no re-encode): the reader resolves each key's topic
    through ``meta/mcap_topics.json`` rather than assuming ``topic_for_key``'s naming
    convention, so nothing about the source recording needs to change.

    Episode boundaries are the INTERSECTION of every mapped channel's real time range
    (latest channel start to earliest channel end) -- the largest window where every
    channel the caller asked for was actually active. This is what makes ingestion
    robust to genuinely async sources (a camera warming up late, a sensor dropping out
    before the others stop): the grid simply doesn't extend into a region no channel
    can answer, rather than raising at read time for every frame in that region.

    ``task_index`` is not expected to be a real per-message channel here -- a raw
    recording pipeline doesn't publish task labels -- so it's stored as one constant
    value per episode instead (see ``McapDatasetReader._task_index_is_channel``).
    """
    require_package("mcap", extra="mcap")
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    if not mcap_paths:
        raise ValueError("mcap_paths must be non-empty.")
    if isinstance(tasks, str):
        tasks = [tasks] * len(mcap_paths)
    if len(tasks) != len(mcap_paths):
        raise ValueError(f"tasks must have one entry per episode ({len(mcap_paths)}), got {len(tasks)}.")

    topics = [c.topic for c in channels]
    features = {}
    for c in channels:
        if c.kind == "video":
            first = next((m for p in mcap_paths if (m := _peek_first_message(p, c.topic)) is not None), None)
            if first is None:
                raise ValueError(f"No messages found on topic '{c.topic}' for channel '{c.key}'.")
            shape = c.shape or load_image_as_numpy(first, dtype=np.uint8, is_depth=False).shape
            features[c.key] = {"dtype": "video", "shape": tuple(shape), "names": ["channels", "height", "width"]}
        else:
            if c.dtype is None or c.shape is None:
                raise ValueError(f"Tabular channel '{c.key}' needs an explicit dtype and shape.")
            features[c.key] = {"dtype": c.dtype, "shape": tuple(c.shape), "names": None}

    task_to_index: dict[str, int] = {}
    episode_records = []
    abs_idx = 0
    for ep_idx, (src_path, task) in enumerate(zip(mcap_paths, tasks, strict=True)):
        time_index = build_channel_time_index(src_path, topics)
        missing = [c.key for c in channels if len(time_index[c.topic]) == 0]
        if missing:
            raise ValueError(f"Episode {ep_idx} ({src_path}) has no messages for channel(s) {missing}.")

        ep_start_ns = max(int(time_index[c.topic][0]) for c in channels)
        ep_end_ns = min(int(time_index[c.topic][-1]) for c in channels)
        if ep_end_ns <= ep_start_ns:
            raise ValueError(
                f"Episode {ep_idx} ({src_path}): mapped channels never overlap in time "
                f"(start={ep_start_ns}ns, end={ep_end_ns}ns) -- nothing to ingest."
            )
        n_frames = max(1, round((ep_end_ns - ep_start_ns) / 1e9 * fps) + 1)

        chunk_index, file_index = ep_idx // chunk_size, ep_idx % chunk_size
        dst_path = episode_mcap_path(dst_root, chunk_index, file_index)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

        task_to_index.setdefault(task, len(task_to_index))
        record = {
            "episode_index": ep_idx,
            "dataset_from_index": abs_idx,
            "dataset_to_index": abs_idx + n_frames,
            MCAP_CHUNK_INDEX_COLUMN: chunk_index,
            MCAP_FILE_INDEX_COLUMN: file_index,
            EPISODE_START_TIME_COLUMN: ep_start_ns,
            "task_index": task_to_index[task],
            "tasks": [task],
        }
        for c in channels:
            record[log_time_column(c.key)] = time_index[c.topic].tolist()
        episode_records.append(record)
        abs_idx += n_frames

    info = DatasetInfo(
        codebase_version="v3.0",
        fps=fps,
        features={
            **features,
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            "index": {"dtype": "int64", "shape": (1,), "names": None},
            "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        },
        total_episodes=len(mcap_paths),
        total_frames=abs_idx,
        total_tasks=len(task_to_index),
        storage_format="mcap",
        video_path=None,
        robot_type=robot_type,
    )
    write_info(info, dst_root)
    tasks_df = pd.DataFrame(
        {"task_index": list(task_to_index.values())}, index=pd.Index(list(task_to_index), name="task")
    )
    write_tasks(tasks_df, dst_root)
    write_episodes(datasets.Dataset.from_list(episode_records), dst_root)
    write_topic_map(dst_root, {c.key: c.topic for c in channels})


def materialize_mcap_dataset(
    src_root: Path,
    dst_root: Path,
    repo_id: str = "materialized/mcap",
    use_videos: bool = True,
    batch_size: int = 256,
    tolerance_s: float = 1e-4,
) -> LeRobotDataset:
    """Materialize an mcap-format LeRobotDataset into the default parquet/mp4 format.

    The reverse of :func:`convert_lerobot_dataset_to_mcap`: reads every frame through
    ``McapDatasetReader`` (paying the nearest-match/tolerance resampling cost once,
    here, batched per episode) and writes it back out via the standard
    ``add_frame``/``save_episode`` path. Useful for comparing DataLoader throughput
    against the native async-in-place format on identical resampled data, or for
    handing an mcap dataset to tooling that only understands the default layout.

    ``tolerance_s`` must be generous enough to cover the source recording's real
    jitter/multi-rate gaps -- pass the same value you'd use to read ``src_root``
    directly, not the default reader's tight default.
    """
    require_package("mcap", extra="mcap")
    src = LeRobotDataset(repo_id, root=src_root, return_uint8=True, tolerance_s=tolerance_s)
    meta = src.meta
    if meta.storage_format != "mcap":
        raise ValueError(f"Expected an mcap-format dataset, got storage_format={meta.storage_format!r}.")

    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {k: v for k, v in meta.features.items() if k not in skip}

    dst = LeRobotDataset.create(
        repo_id=repo_id,
        fps=meta.fps,
        features=features,
        root=dst_root,
        use_videos=use_videos,
        robot_type=meta.robot_type,
    )
    # Shape-(1,) tabular features round-trip as 0-dim tensors on read (see
    # mcap_backend._feature_shapes), but add_frame/validate_frame expects the
    # declared shape literally -- reshape scalars back to (1,) before writing.
    tabular_shapes = {
        k: tuple(v.get("shape") or ()) for k, v in features.items() if v["dtype"] not in ("video", "image")
    }

    for ep_idx in range(meta.total_episodes):
        ep = meta.episodes[ep_idx]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        for batch_start in range(start, end, batch_size):
            indices = list(range(batch_start, min(batch_start + batch_size, end)))
            for item in src.__getitems__(indices):
                frame = {}
                for k in features:
                    shape = tabular_shapes.get(k)
                    frame[k] = item[k].reshape(shape) if shape else item[k]
                frame["task"] = item["task"]
                dst.add_frame(frame)
        dst.save_episode()
    dst.finalize()
    return LeRobotDataset(repo_id, root=dst_root)
