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
"""Capture video feed from a camera as raw images."""

import argparse
import datetime as dt
import os
import time
from pathlib import Path

import cv2
import rerun as rr

# see https://rerun.io/docs/howto/visualization/limit-ram
RERUN_MEMORY_LIMIT = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "5%")


def display_and_save_video_stream(output_dir: Path, fps: int, width: int, height: int, duration: int):
    rr.init("lerobot_capture_camera_feed")
    rr.spawn(memory_limit=RERUN_MEMORY_LIMIT)

    now = dt.datetime.now()
    capture_dir = output_dir / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}"
    if not capture_dir.exists():
        capture_dir.mkdir(parents=True, exist_ok=True)

    # Opens the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_index = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break
        rr.log("video/stream", rr.Image(frame), static=True)
        cv2.imwrite(str(capture_dir / f"frame_{frame_index:06d}.png"), frame)
        frame_index += 1

    # Release the capture
    cap.release()

    # TODO(Steven): Add a graceful shutdown via a close() method for the Viewer context, though not currently supported in the Rerun API.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cam_capture/"),
        help="Directory where the capture images are written. A subfolder named with the current date & time will be created inside it for each capture.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames Per Second of the capture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the captured images.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the captured images.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Duration in seconds for which the video stream should be captured.",
    )
    args = parser.parse_args()
    display_and_save_video_stream(**vars(args))
