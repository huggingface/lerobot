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

"""
Script to colorize raw depth data saved as .npy files.

This script is called during video encoding to convert raw depth data
(float32 from Kinect or uint16 from RealSense) into colorized PNG images
that can be encoded into videos.

Usage:
    python colorize_depth.py <directory_path> [--colormap JET] [--min_depth 0.5] [--max_depth 4.5]
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.utils import DepthColorizer

logger = logging.getLogger(__name__)


def colorize_depth_directory(
    directory: Path,
    colormap: str = "JET",
    min_depth_m: float = 0.5,
    max_depth_m: float = 4.5,
    delete_npy: bool = True,
    depth_scale_m_per_unit: float | None = None,
) -> None:
    """
    Colorize all .npy depth files in a directory and save as .png files.
    
    Args:
        directory: Path to directory containing .npy depth files
        colormap: OpenCV colormap name (JET, VIRIDIS, etc.)
        min_depth_m: Minimum depth in meters for colorization
        max_depth_m: Maximum depth in meters for colorization
        delete_npy: Whether to delete the .npy files after colorization
    """
    npy_files = list(directory.glob("*.npy"))
    
    if not npy_files:
        logger.warning(f"No .npy files found in {directory}")
        return
    
    logger.info(f"Found {len(npy_files)} .npy files to colorize in {directory}")
    
    # Create colorizer
    colorizer = DepthColorizer(
        colormap=colormap,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        clipping=True,
    )
    
    for npy_path in npy_files:
        try:
            # Load raw depth data
            depth_data = np.load(npy_path)
            
            # Determine the depth unit based on dtype
            # Kinect uses float32 in millimeters
            # RealSense uses uint16 in micrometers
            if depth_data.dtype == np.float32:
                # Kinect: already in millimeters
                depth_mm = depth_data
            elif depth_data.dtype == np.uint16:
                # RealSense: use provided depth scale if available (meters per unit)
                if depth_scale_m_per_unit is not None and depth_scale_m_per_unit > 0:
                    depth_mm = depth_data.astype(np.float32) * (depth_scale_m_per_unit * 1000.0)
                else:
                    # Fallback: assume micrometers and convert to millimeters
                    depth_mm = depth_data.astype(np.float32) / 1000.0
            else:
                logger.warning(f"Unexpected depth dtype {depth_data.dtype} in {npy_path}, skipping")
                continue
            
            # Colorize the depth data
            depth_rgb = colorizer.colorize(depth_mm)
            
            # Save as PNG with the same name
            png_path = npy_path.with_suffix('.png')
            cv2.imwrite(str(png_path), depth_rgb)
            
            # Delete the .npy file if requested
            if delete_npy:
                npy_path.unlink()
            
        except Exception as e:
            logger.error(f"Failed to colorize {npy_path}: {e}")
            continue
    
    logger.info(f"Colorization complete for {directory}")


def main():
    # Ensure basic logging is configured for subprocess runs
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Colorize raw depth data from .npy files to .png images")
    parser.add_argument("directory", type=str, help="Directory containing .npy depth files")
    parser.add_argument("--colormap", type=str, default="JET", help="OpenCV colormap name (default: JET)")
    parser.add_argument("--min_depth", type=float, default=0.5, help="Minimum depth in meters (default: 0.5)")
    parser.add_argument("--max_depth", type=float, default=4.5, help="Maximum depth in meters (default: 4.5)")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=None,
        help="Depth scale in meters per unit (RealSense). If omitted, falls back to micrometer assumption.",
    )
    parser.add_argument("--keep_npy", action="store_true", help="Keep .npy files after colorization")
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    colorize_depth_directory(
        directory=directory,
        colormap=args.colormap,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        delete_npy=not args.keep_npy,
        depth_scale_m_per_unit=args.depth_scale,
    )


if __name__ == "__main__":
    main()
