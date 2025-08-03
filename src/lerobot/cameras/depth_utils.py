#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Depth processing utilities for consistent visualization across all depth cameras.

Pure functions that process depth data independently of camera SDKs.
This separation allows any depth camera to use the same processing pipeline.
"""

from enum import Enum
from typing import Any, Dict, Tuple

import cv2
import numpy as np


class DepthColormap(str, Enum):
    """Supported depth colormaps for visualization."""

    JET = "jet"  # Traditional robotics colormap
    TURBO = "turbo"  # Better perceptual uniformity
    VIRIDIS = "viridis"  # Perceptually uniform
    PLASMA = "plasma"  # High contrast
    INFERNO = "inferno"  # Dark background


class DepthRange:
    """Standard depth ranges for different applications."""

    TABLETOP = (100, 2000)  # 10cm - 2m for manipulation
    INDOOR = (200, 5000)  # 20cm - 5m for indoor robotics
    OUTDOOR = (500, 50000)  # 50cm - 50m for outdoor robotics
    HUMAN_SCALE = (300, 8000)  # 30cm - 8m for human interaction


class DepthColorizer:
    """Centralized depth colorization for consistent visualization across all cameras."""

    def __init__(
        self,
        colormap: str = "jet",
        depth_range: tuple[int, int] = (200, 5000),
        invalid_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """Initialize colorizer with consistent settings.

        Args:
            colormap: Color scheme for visualization.
            depth_range: (min_mm, max_mm) for normalization.
            invalid_color: RGB color for invalid pixels (depth = 0).
        """
        self.colormap = colormap
        self.depth_range = depth_range
        self.invalid_color = invalid_color

    def colorize(self, raw_depth: np.ndarray) -> np.ndarray:
        """Convert raw uint16 depth to colorized RGB for storage/visualization.

        Args:
            raw_depth: Raw depth as uint16 millimeters, shape (H, W).

        Returns:
            np.ndarray: Colorized depth as uint8 RGB, shape (H, W, 3).
        """
        min_depth, max_depth = self.depth_range

        # Create mask for valid depth values
        valid_mask = raw_depth > 0

        # Clip and normalize depth values
        depth_clipped = np.clip(raw_depth, min_depth, max_depth)
        depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply colormap
        colormap_cv2 = getattr(cv2, f"COLORMAP_{self.colormap.upper()}")
        depth_colored_bgr = cv2.applyColorMap(depth_normalized, colormap_cv2)
        depth_colored_rgb = cv2.cvtColor(depth_colored_bgr, cv2.COLOR_BGR2RGB)

        # Set invalid pixels to specified color
        depth_colored_rgb[~valid_mask] = self.invalid_color
        return depth_colored_rgb


# Global depth colorizer - single consistent colorization for ALL cameras
GLOBAL_DEPTH_COLORIZER = DepthColorizer(
    colormap="jet",
    depth_range=DepthRange.INDOOR,  # 200-5000mm for indoor robotics
    invalid_color=(0, 0, 0),  # Black for invalid pixels
)


def colorize_depth_frame(raw_depth: np.ndarray) -> np.ndarray:
    """Global depth colorization function - all cameras use this for consistency.

    Args:
        raw_depth: Raw depth as uint16 millimeters, shape (H, W).

    Returns:
        np.ndarray: Colorized depth as uint8 RGB, shape (H, W, 3).

    Note:
        Uses global colorization settings: JET colormap, 200-5000mm range.
        All depth cameras produce identical visualization.
    """
    return GLOBAL_DEPTH_COLORIZER.colorize(raw_depth)


def is_raw_depth(array: np.ndarray) -> bool:
    """Detect if array contains raw depth data.

    Args:
        array: Input array to check.

    Returns:
        bool: True if array appears to be raw depth data.
    """
    return (
        array.dtype == np.uint16 and array.ndim == 2 and array.max() > 50  # Reasonable minimum depth in mm
    )


def depth_statistics(raw_depth: np.ndarray) -> dict[str, Any]:
    """Compute depth statistics for monitoring/debugging.

    Args:
        raw_depth: Raw depth as uint16 millimeters, shape (H, W).

    Returns:
        Dict[str, Any]: Statistics including min, max, mean, valid pixel count.
    """
    valid_mask = raw_depth > 0

    if not valid_mask.any():
        return {
            "min_mm": 0,
            "max_mm": 0,
            "mean_mm": 0.0,
            "median_mm": 0.0,
            "valid_pixels": 0,
            "total_pixels": int(raw_depth.size),
            "valid_percentage": 0.0,
        }

    valid_depth = raw_depth[valid_mask]
    return {
        "min_mm": int(valid_depth.min()),
        "max_mm": int(valid_depth.max()),
        "mean_mm": float(valid_depth.mean()),
        "median_mm": float(np.median(valid_depth)),
        "valid_pixels": int(valid_mask.sum()),
        "total_pixels": int(raw_depth.size),
        "valid_percentage": float(valid_mask.sum() / raw_depth.size * 100),
    }


def depth_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """Convert depth from millimeters to meters.

    Args:
        depth_mm: Depth in millimeters.

    Returns:
        np.ndarray: Depth in meters as float32.
    """
    return depth_mm.astype(np.float32) / 1000.0
