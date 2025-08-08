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

import os
import time
import logging
from typing import Any

import numpy as np
import rerun as rr

# Registry for per-stream depth scales: meters per unit
_DEPTH_METERS_PER_UNIT: dict[str, float] = {}


def register_depth_scale(stream_key: str, meters_per_unit: float) -> None:
    """
    Register the physical scale for a depth stream key.

    - stream_key: base camera key without suffix (e.g., "cam_kinect", "cam_low").
    - meters_per_unit: conversion factor from the stream's integer/float value to meters.

    Examples:
    - RealSense uint16 depth: meters_per_unit = device.depth_scale (e.g., 0.0005)
    - Kinect float32 depth in millimeters: meters_per_unit = 0.001
    - Generic meters float32: meters_per_unit = 1.0
    """
    _DEPTH_METERS_PER_UNIT[stream_key] = float(meters_per_unit)


def _depth_to_meters(key: str, depth: np.ndarray) -> np.ndarray:
    """Convert a raw depth array to meters using per-stream scale when available."""
    # Derive base stream key (remove trailing "_depth" when present)
    base = key[:-6] if key.endswith("_depth") else key
    if base in _DEPTH_METERS_PER_UNIT:
        meters_per_unit = _DEPTH_METERS_PER_UNIT[base]
    else:
        if np.issubdtype(depth.dtype, np.floating):
            meters_per_unit = 0.001  # assume float depths are in millimeters
        elif np.issubdtype(depth.dtype, np.integer):
            meters_per_unit = 1e-6  # assume uint16 like RealSense is micrometers
        else:
            meters_per_unit = 0.001
    return depth.astype(np.float32) * meters_per_unit


def _camera_entity_path_from_key(key: str) -> tuple[str, bool]:
    """Return (entity_path, is_depth) using dot-separated hierarchy.

    - "cam_kinect" -> ("observation.cam_kinect.rgb", False)
    - "cam_kinect_depth" -> ("observation.cam_kinect.depth", True)
    """
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1] == "depth":
        return f"observation.{parts[0]}.depth", True
    return f"observation.{key}.rgb", False

_RR_LAST_LOG_T: float | None = None
_RR_WIN_S: float = 5.0
_RR_COUNT: int = 0
_RR_SUM_MS: float = 0.0
_RR_SUM_SQ: float = 0.0
_RR_MIN_MS: float = float("inf")
_RR_MAX_MS: float = 0.0


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rerun_logger = logging.getLogger("rerun_performance")
    t0 = time.perf_counter()
    rr.spawn(memory_limit=memory_limit)
    rerun_logger.info(f"Rerun spawn took {(time.perf_counter() - t0)*1000:.1f}ms")
    global _RR_LAST_LOG_T, _RR_COUNT, _RR_SUM_MS, _RR_SUM_SQ, _RR_MIN_MS, _RR_MAX_MS
    _RR_LAST_LOG_T = time.perf_counter()
    _RR_COUNT = 0
    _RR_SUM_MS = 0.0
    _RR_SUM_SQ = 0.0
    _RR_MIN_MS = float("inf")
    _RR_MAX_MS = 0.0


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    rerun_logger = logging.getLogger("rerun_performance")
    t_start = time.perf_counter()
    # Log observations
    for key, val in observation.items():
        # Scalars
        if isinstance(val, float):
            rr.log(f"observation.{key}", rr.Scalar(val))
            continue

        # 1D vectors
        if isinstance(val, np.ndarray) and val.ndim == 1:
            for i, v in enumerate(val):
                rr.log(f"observation.{key}.{i}", rr.Scalar(float(v)))
            continue

        # Depth images (raw) → log as DepthImage with correct meter; no custom colorization
        if isinstance(val, np.ndarray) and val.ndim == 2 and (np.issubdtype(val.dtype, np.floating) or np.issubdtype(val.dtype, np.integer)):
            entity_path, _ = _camera_entity_path_from_key(key)
            depth_m = _depth_to_meters(key, val)
            rr.log(entity_path, rr.DepthImage(depth_m, meter=1.0))
            continue

        # Color images
        if isinstance(val, np.ndarray) and val.ndim == 3:
            entity_path, _ = _camera_entity_path_from_key(key)
            img = val
            # Kinect cameras emit BGR now; convert to RGB for visualization only
            if "kinect" in key:
                # BGR->RGB channel swap (cheap)
                if img.shape[-1] == 3:
                    img = img[..., ::-1].copy()
            rr.log(entity_path, rr.Image(img))
            continue

    # Log actions
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}.{i}", rr.Scalar(float(v)))
    # Aggregate rerun overhead every 5 seconds
    dur_ms = (time.perf_counter() - t_start) * 1000
    global _RR_LAST_LOG_T, _RR_COUNT, _RR_SUM_MS, _RR_SUM_SQ, _RR_MIN_MS, _RR_MAX_MS
    _RR_COUNT += 1
    _RR_SUM_MS += dur_ms
    _RR_SUM_SQ += dur_ms * dur_ms
    _RR_MIN_MS = min(_RR_MIN_MS, dur_ms)
    _RR_MAX_MS = max(_RR_MAX_MS, dur_ms)
    now_t = time.perf_counter()
    if _RR_LAST_LOG_T is None:
        _RR_LAST_LOG_T = now_t
    if (now_t - _RR_LAST_LOG_T) >= _RR_WIN_S and _RR_COUNT > 0:
        n = _RR_COUNT
        avg = _RR_SUM_MS / n
        mean_sq = _RR_SUM_SQ / n
        var = max(0.0, mean_sq - (avg ** 2))
        std = var ** 0.5
        rerun_logger.info(
            f"Rerun log 5s stats — log_ms(avg={avg:.1f}, std={std:.1f}, min={_RR_MIN_MS:.1f}, max={_RR_MAX_MS:.1f}), frames={n}"
        )
        # Reset window
        _RR_LAST_LOG_T = now_t
        _RR_COUNT = 0
        _RR_SUM_MS = 0.0
        _RR_SUM_SQ = 0.0
        _RR_MIN_MS = float("inf")
        _RR_MAX_MS = 0.0
