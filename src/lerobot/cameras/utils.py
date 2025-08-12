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

import platform
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation

IndexOrPath: TypeAlias = int | Path

 


class DepthColorizer:
    """
    Ultra-fast, vectorized, LUT-based depth colorizer.

    This utility pre-computes a lookup table (LUT) for depth colorization,
    which is significantly faster (~100x) than per-frame calculations. It supports
    both uint16 (e.g., RealSense) and float32 (e.g., Kinect) depth inputs.
    """

    COLORMAP_MAPPING = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "bone": cv2.COLORMAP_BONE,
    }

    def __init__(
        self,
        colormap: str = "jet",
        min_depth_m: float = 0.5,
        max_depth_m: float = 4.5,
        clipping: bool = True,
    ):
        self.colormap = self.COLORMAP_MAPPING.get(colormap.lower(), cv2.COLORMAP_JET)
        self.min_depth_mm = min_depth_m * 1000
        self.max_depth_mm = max_depth_m * 1000
        self.clipping = clipping

        # Build LUT for all possible 16-bit depth values (0-65535)
        self._build_lut()

    def _build_lut(self):
        """Pre-compute the color lookup table using vectorized NumPy operations."""
        depth_range = self.max_depth_mm - self.min_depth_mm
        if depth_range <= 0:
            self.lut = np.zeros((65536, 3), dtype=np.uint8)
            return

        # Create all possible uint16 values at once (vectorized)
        all_values = np.arange(65536, dtype=np.float32)
        
        # Map from uint16 range to actual depth values in mm
        depth_values = (all_values / 65535.0) * depth_range + self.min_depth_mm
        
        # Normalize to 0-255 for colormap
        normalized = ((depth_values - self.min_depth_mm) / depth_range * 255).astype(np.uint8)
        
        # Apply colormap in one vectorized operation
        # Note: normalized needs to be reshaped for cv2.applyColorMap
        normalized_2d = normalized.reshape(-1, 1)
        bgr_lut = cv2.applyColorMap(normalized_2d, self.colormap)
        
        # Convert BGR to RGB (most cameras expect RGB)
        self.lut = cv2.cvtColor(bgr_lut, cv2.COLOR_BGR2RGB).reshape(65536, 3)
        
        # Handle out-of-range values for better visualization
        if self.clipping:
            # Values below minimum depth - use minimum color
            min_idx = int(self.min_depth_mm * 65535 / depth_range) if depth_range > 0 else 0
            if min_idx > 0:
                self.lut[:min_idx] = self.lut[min_idx]
            
            # Values above maximum depth - use maximum color  
            max_idx = int((self.max_depth_mm - self.min_depth_mm) * 65535 / depth_range) if depth_range > 0 else 65535
            if max_idx < 65535:
                self.lut[max_idx + 1:] = self.lut[max_idx]

    def colorize(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Colorize a depth map using the pre-computed LUT.

        Args:
            depth_data: Depth map as either float32 (mm) or uint16 (mm).

        Returns:
            Colorized depth as an RGB numpy array.
        """
        if depth_data is None or depth_data.size == 0:
            return np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)

        # Convert float32 depth to uint16 for LUT indexing if necessary.
        if depth_data.dtype == np.float32:
            # Quantize float32 depth to uint16 for LUT indexing
            depth_range = self.max_depth_mm - self.min_depth_mm
            if depth_range > 0:
                depth_quantized = (depth_data - self.min_depth_mm) / depth_range * 65535
                if self.clipping:
                    depth_quantized = np.clip(depth_quantized, 0, 65535)
                depth_indices = depth_quantized.astype(np.uint16)
            else:
                depth_indices = np.zeros_like(depth_data, dtype=np.uint16)
        else:
            # Assume uint16 input
            depth_indices = depth_data.astype(np.uint16)

        # Direct LUT indexing - this is the key optimization.
        return self.lut[depth_indices]


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "kinect":
            from .kinect.camera_kinect import KinectCamera

            cameras[key] = KinectCamera(cfg)
        else:
            raise ValueError(f"The camera type '{cfg.type}' is not valid.")

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2

    if rotation == Cv2Rotation.ROTATE_90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rotation == Cv2Rotation.ROTATE_180:
        return cv2.ROTATE_180
    elif rotation == Cv2Rotation.ROTATE_270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return None


def get_cv2_backend() -> int:
    import cv2

    if platform.system() == "Windows":
        return cv2.CAP_MSMF  # Use MSMF for Windows instead of AVFOUNDATION
    # elif platform.system() == "Darwin":  # macOS
    #     return cv2.CAP_AVFOUNDATION
    else:  # Linux and others
        return cv2.CAP_ANY
