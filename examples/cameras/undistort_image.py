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

"""Preview a real camera calibration on an image and save a reusable processor.

The calibration JSON contains camera_key, camera_matrix, distortion_coefficients
and image_size (width, height). See docs/source/processors_robots_teleop.mdx.

uv run examples/cameras/undistort_image.py --calibration calibration.json --input frame.png --output corrected.png --processor-dir camera_processor
"""

import argparse
import json
from pathlib import Path

import cv2

from lerobot.processor import (
    ImageUndistortProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    observation_to_transition,
    transition_to_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.calibration.open() as calibration_file:
        calibration = json.load(calibration_file)
    pipeline = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ImageUndistortProcessorStep(**calibration)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {args.input}")
    camera_key = calibration["camera_key"]
    corrected = pipeline({camera_key: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)})[camera_key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {args.output}")
    pipeline.save_pretrained(args.processor_dir, config_filename="robot_observation_processor.json")


if __name__ == "__main__":
    main()
