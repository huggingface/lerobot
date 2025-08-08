#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Process Kinect color frames saved as PNGs:
 - Ensure 3-channel BGR (drop alpha if present)
 - Optional resize to a target width/height

Usage:
  python process_kinect_color.py <directory> [--width W --height H]
"""

import argparse
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def process_directory(directory: Path, width: int | None, height: int | None) -> None:
    png_files = list(directory.glob("*.png"))
    if not png_files:
        logger.info(f"No PNG files found in {directory}")
        return

    logger.info(
        f"Processing {len(png_files)} Kinect color frames in {directory}"
        + (f" -> resize to {width}x{height}" if width and height else "")
    )

    for png in png_files:
        try:
            img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # Safety: if image has alpha, drop to get BGR; otherwise pass-through
            if img.ndim == 3 and img.shape[-1] == 4:
                bgr = img[..., :3]
            else:
                bgr = img

            if width and height:
                interp = (
                    cv2.INTER_AREA if width < bgr.shape[1] or height < bgr.shape[0] else cv2.INTER_LINEAR
                )
                bgr = cv2.resize(bgr, (width, height), interpolation=interp)

            cv2.imwrite(str(png), bgr)
        except Exception as e:
            logger.debug(f"Failed to process {png}: {e}")
            continue

    logger.info("Kinect color processing complete")


def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Process Kinect color PNGs: drop alpha and optional resize")
    ap.add_argument("directory", type=str, help="Directory containing PNG frames")
    ap.add_argument("--width", type=int, default=None, help="Target width (optional)")
    ap.add_argument("--height", type=int, default=None, help="Target height (optional)")
    args = ap.parse_args()

    d = Path(args.directory)
    if not d.exists():
        raise ValueError(f"Directory {d} does not exist")

    process_directory(d, args.width, args.height)


if __name__ == "__main__":
    main()


