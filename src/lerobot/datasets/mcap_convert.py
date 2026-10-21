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

"""Convert an existing default-format :class:`LeRobotDataset` into the ``"mcap"``
storage format.

There is no external ``lerobot-mcap`` converter package (unlike Lance, which has
``lerobot-lancedb``), so this ships in-tree under the ``mcap`` extra.
:func:`convert_lerobot_dataset_to_mcap` re-exports a dataset's frames as one
``.mcap`` file per episode (one channel per feature key, using the
``topic_for_key`` naming convention) and builds the ``meta/episodes`` sync index
(:mod:`lerobot.datasets.mcap_backend`) that lets the reader map fixed-grid indices
back onto real message timestamps.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import numpy as np
from PIL import Image as PILImage

from lerobot.utils.import_utils import _mcap_available, require_package

from .dataset_metadata import LeRobotDatasetMetadata
from .io_utils import write_episodes, write_info, write_stats, write_tasks
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
from .utils import DEFAULT_CHUNK_SIZE

if TYPE_CHECKING or _mcap_available:
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
