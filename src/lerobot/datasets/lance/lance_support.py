#!/usr/bin/env python
"""
Lance support utilities: build an episode-level Lance schema from LeRobot v3.0 features, and assemble row data (sequence columns + video blob columns).

- Schema construction:
  - Base columns: episode_index, task_index, fps, length, timestamps: LargeList<float32>
  - Sequence columns: actions and observation.state as LargeList<FixedSizeList<float32>>
  - Video columns: one pa.large_binary() per camera_key, with field.metadata {"lance-encoding:blob": "true"}; also keep from/to timestamp columns for random access.

- Row assembly: convert numpy/Python lists to Arrow-friendly structures consistent with the schema.

Note: This module does not depend on lance at runtime; it uses PyArrow to build the schema and rows. Writing to Lance is handled in convert_dataset_v30_to_lance.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa


def _norm_cam_key(camera_key: str) -> str:
    """Normalize a visual feature key into a column-safe name.
    Example: 'observation.images.front' -> 'video_observation_images_front'.
    """
    safe = camera_key.replace(".", "_")
    return f"video_{safe}"


def build_lance_schema(features: Dict[str, Dict], camera_keys: List[str]) -> pa.Schema:
    """Generate an episode-level Lance schema based on LeRobot v3.0 features.

    Args:
        features: info['features'] dict containing dtype, shape, names, etc.
        camera_keys: list of visual modality keys (features with dtype == 'video' or 'image').

    Returns:
        pa.Schema: schema of the Lance dataset (one row per episode).
    """
    fields: List[pa.Field] = []

    # Base columns
    fields.append(pa.field("episode_index", pa.int64()))
    fields.append(pa.field("task_index", pa.int64()))
    fields.append(pa.field("fps", pa.int32()))
    fields.append(pa.field("length", pa.int64()))
    fields.append(pa.field("timestamps", pa.large_list(pa.float32())))

    # actions: LargeList<FixedSizeList<float32>>
    if "action" in features and isinstance(features["action"].get("shape"), tuple):
        act_shape = features["action"]["shape"]
        if len(act_shape) == 1:
            fields.append(
                pa.field(
                    "actions",
                    pa.large_list(pa.fixed_size_list(pa.float32(), act_shape[0])),
                )
            )

    # observation.state: LargeList<FixedSizeList<float32>>
    if "observation.state" in features and isinstance(features["observation.state"].get("shape"), tuple):
        st_shape = features["observation.state"]["shape"]
        if len(st_shape) == 1:
            fields.append(
                pa.field(
                    "obs_state",
                    pa.large_list(pa.fixed_size_list(pa.float32(), st_shape[0])),
                )
            )

    # For each camera_key, add a blob column + from/to timestamp columns
    for ck in camera_keys:
        col_name = _norm_cam_key(ck)
        fields.append(
            pa.field(
                col_name,
                pa.large_binary(),
                metadata={"lance-encoding:blob": "true"},
            )
        )
        fields.append(pa.field(f"{col_name}_from_ts", pa.float64()))
        fields.append(pa.field(f"{col_name}_to_ts", pa.float64()))

    return pa.schema(fields)


def build_episode_row(
    episode_index: int,
    task_index: int,
    fps: int,
    timestamps: List[float],
    actions: np.ndarray | List[List[float]] | None,
    obs_state: np.ndarray | List[List[float]] | None,
    camera_blobs: Dict[str, bytes],
    camera_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """Build a dictionary of episode row data for pa.Table.from_pydict.

    - actions/obs_state accept (T, D) numpy arrays or Python lists and are mapped to LargeList<FixedSizeList<float32>>.
    - camera_blobs: keys are feature camera_keys; internally mapped to normalized column names.
    - camera_ranges: keys are camera_keys, values are (from_ts, to_ts).
    """
    row: Dict[str, Any] = {}
    row["episode_index"] = episode_index
    row["task_index"] = task_index
    row["fps"] = fps
    row["length"] = len(timestamps)
    row["timestamps"] = [float(x) for x in timestamps]

    if actions is not None:
        arr = np.asarray(actions, dtype=np.float32)
        # Convert to list of lists so Arrow recognizes List<FixedSizeList>
        row["actions"] = arr.tolist()

    if obs_state is not None:
        arr = np.asarray(obs_state, dtype=np.float32)
        row["obs_state"] = arr.tolist()

    for ck, blob in camera_blobs.items():
        col_name = _norm_cam_key(ck)
        row[col_name] = blob
        if ck in camera_ranges:
            fts, tts = camera_ranges[ck]
            row[f"{col_name}_from_ts"] = float(fts)
            row[f"{col_name}_to_ts"] = float(tts)
        else:
            # Fallback when range is missing: [0, length/fps]
            row[f"{col_name}_from_ts"] = 0.0
            row[f"{col_name}_to_ts"] = float(len(timestamps)) / float(fps) if fps > 0 else 0.0

    return row


def build_frames_schema(features: Dict[str, Dict]) -> pa.Schema:
    """Generate a frame-level Lance schema.

    Columns:
    - episode_index: int64
    - frame_index: int64
    - timestamp: float32
    - index: int64 (global frame index)
    - task_index: int64
    - action: FixedSizeList<float32> (if present)
    - obs_state: FixedSizeList<float32> (if present)
    - embedding: LargeList<float32> (optional placeholder, if present in features)
    - reward: float32 (if present)
    """
    fields: List[pa.Field] = []
    fields.append(pa.field("episode_index", pa.int64()))
    fields.append(pa.field("frame_index", pa.int64()))
    fields.append(pa.field("timestamp", pa.float32()))
    fields.append(pa.field("index", pa.int64()))
    fields.append(pa.field("task_index", pa.int64()))

    if "action" in features and isinstance(features["action"].get("shape"), tuple):
        act_shape = features["action"]["shape"]
        if len(act_shape) == 1:
            fields.append(pa.field("action", pa.fixed_size_list(pa.float32(), act_shape[0])))
    if "observation.state" in features and isinstance(features["observation.state"].get("shape"), tuple):
        st_shape = features["observation.state"]["shape"]
        if len(st_shape) == 1:
            fields.append(pa.field("obs_state", pa.fixed_size_list(pa.float32(), st_shape[0])))
    if "embedding" in features:
        fields.append(pa.field("embedding", pa.large_list(pa.float32())))
    if "reward" in features and features["reward"].get("shape") == (1,):
        fields.append(pa.field("reward", pa.float32()))

    return pa.schema(fields)


def build_frame_row(
    episode_index: int,
    frame_index: int,
    timestamp: float,
    index: int,
    task_index: int,
    action: np.ndarray | List[float] | None = None,
    obs_state: np.ndarray | List[float] | None = None,
    embedding: np.ndarray | List[float] | None = None,
    reward: float | None = None,
) -> Dict[str, Any]:
    """Build a dictionary for a single frame row consistent with build_frames_schema."""
    row: Dict[str, Any] = {
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "index": int(index),
        "task_index": int(task_index),
    }
    if action is not None:
        arr = np.asarray(action, dtype=np.float32)
        row["action"] = arr.tolist()
    if obs_state is not None:
        arr = np.asarray(obs_state, dtype=np.float32)
        row["obs_state"] = arr.tolist()
    if embedding is not None:
        arr = np.asarray(embedding, dtype=np.float32)
        row["embedding"] = arr.tolist()
    if reward is not None:
        row["reward"] = float(reward)
    return row
