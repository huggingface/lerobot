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


def _depth_to_mm(key: str, depth: np.ndarray) -> np.ndarray:
    """Convert a raw depth array to millimeters using per-stream scale when available.

    Fallbacks when scale is not registered:
    - float arrays default to millimeters (meters_per_unit = 0.001)
    - uint16 arrays default to micrometers (meters_per_unit = 1e-6)
    """
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

    return depth.astype(np.float32) * (meters_per_unit * 1000.0)


def _camera_entity_path_from_key(key: str) -> tuple[str, bool]:
    """Return (entity_path, is_depth) using dot-separated hierarchy.

    - "cam_kinect" -> ("observation.cam_kinect.rgb", False)
    - "cam_kinect_depth" -> ("observation.cam_kinect.depth", True)
    """
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1] == "depth":
        return f"observation.{parts[0]}.depth", True
    return f"observation.{key}.rgb", False

def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
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

        # Depth images (raw)
        if isinstance(val, np.ndarray) and val.ndim == 2 and (np.issubdtype(val.dtype, np.floating) or np.issubdtype(val.dtype, np.integer)):
            entity_path, _ = _camera_entity_path_from_key(key)
            depth_mm = _depth_to_mm(key, val)
            rr.log(entity_path, rr.DepthImage(depth_mm, meter=1000.0), static=True)
            continue

        # Color images
        if isinstance(val, np.ndarray) and val.ndim == 3:
            entity_path, _ = _camera_entity_path_from_key(key)
            rr.log(entity_path, rr.Image(val), static=True)
            continue

    # Log actions
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}.{i}", rr.Scalar(float(v)))
