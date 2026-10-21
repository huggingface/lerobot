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

"""Helpers for the mcap dataset reader.

Everything the ``"mcap"`` storage format needs outside the reader class: topic
naming, raw tensor (de)serialization, and the on-disk path convention (one
mcap file per episode, located the same way ``videos/{key}/chunk_index`` locates
an mp4 file in the default/lance formats). The one-time, conversion-time-only
per-channel timestamp index build also lives here since it's format plumbing,
not reader logic — the reader only ever reads the index back from
``meta/episodes``, never rebuilds it (see :mod:`lerobot.datasets.mcap_backend`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _mcap_available, require_package

from .utils import CHUNK_FILE_PATTERN, DATA_DIR

if TYPE_CHECKING or _mcap_available:
    from mcap.reader import make_reader

# One mcap file per episode, chunked the same way as the default/lance formats.
DEFAULT_MCAP_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".mcap"

# meta/episodes columns this format adds on top of dataset_from_index/dataset_to_index.
MCAP_CHUNK_INDEX_COLUMN = "mcap/chunk_index"
MCAP_FILE_INDEX_COLUMN = "mcap/file_index"
EPISODE_START_TIME_COLUMN = "episode_start_time_ns"

# Optional key -> physical mcap topic map. Datasets built by convert_lerobot_dataset_to_mcap
# don't need one (topic_for_key's convention is used directly); datasets ingested from
# externally-recorded .mcap files (arbitrary topic names, e.g. "/camera/rgb/image_raw")
# carry one so the raw files can be used as-is -- no re-encode just to rename topics.
TOPIC_MAP_PATH = "meta/mcap_topics.json"


def write_topic_map(root: Path, topics: dict[str, str]) -> None:
    path = root / TOPIC_MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topics, indent=2, sort_keys=True))


def load_topic_map(root: Path) -> dict[str, str]:
    path = root / TOPIC_MAP_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def topic_for_key(key: str) -> str:
    """Feature key -> mcap topic, e.g. ``observation.state`` -> ``/observation/state``."""
    return "/" + key.replace(".", "/")


def key_for_topic(topic: str) -> str:
    """Inverse of :func:`topic_for_key`."""
    return topic.lstrip("/").replace("/", ".")


def log_time_column(key: str) -> str:
    """The ``meta/episodes`` column holding a channel's per-episode message timestamps."""
    return f"mcap/{key}/log_time_ns"


def pack_tensor(array: np.ndarray) -> bytes:
    """Pack a numpy array into raw contiguous bytes for a tabular mcap message."""
    return np.ascontiguousarray(array).tobytes()


def unpack_tensor(data: bytes, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`pack_tensor`."""
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def channel_schema_json(dtype: str, shape: tuple[int, ...]) -> bytes:
    """Self-describing schema payload for a tabular channel (external inspectability only;
    the reader never parses this back — it trusts ``meta.features[key]`` instead)."""
    return json.dumps({"dtype": dtype, "shape": list(shape)}).encode("utf-8")


def episode_mcap_path(root: Path, chunk_index: int, file_index: int) -> Path:
    return root / DEFAULT_MCAP_PATH.format(chunk_index=chunk_index, file_index=file_index)


def build_channel_time_index(mcap_path: str | Path, topics: list[str]) -> dict[str, np.ndarray]:
    """One streaming pass over ``mcap_path``; returns ``{topic: sorted log_time_ns array}``.

    No ``decoder_factories`` are used, so message payloads are never deserialized —
    only ``message.log_time`` is read. Called once by the converter at dataset
    creation time; never called by the reader (see the module docstring).
    """
    require_package("mcap", extra="mcap")
    times: dict[str, list[int]] = {topic: [] for topic in topics}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=list(topics)):
            times[channel.topic].append(message.log_time)
    return {topic: np.array(sorted(values), dtype=np.int64) for topic, values in times.items()}


def localize_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> Path:
    """v1 is local-disk only: raise immediately so ``storage.py``'s format-probe loop
    (:func:`lerobot.datasets.storage.localize_remote_root`) skips this format and tries
    others, instead of implementing object-store support here.
    """
    raise FileNotFoundError("storage_format='mcap' does not support object-store roots yet.")
