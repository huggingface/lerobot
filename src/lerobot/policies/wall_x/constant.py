#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
Wall-X Constants and Configuration Data.
"""

CAMERA_NAME_MAPPING = {
    "face_view": "front view",
    "left_wrist_view": "left wrist view",
    "right_wrist_view": "right wrist view",
    "move1_view": "move view",
    "move2_view": "move view",
    "wall_view": "wall view",
    "top_view": "top view",
}

RESOLUTION = 256

# Parameters for preprocessing
MAX_PIXELS = 16384 * 28 * 28
MIN_PIXELS = 4 * 28 * 28
IMAGE_FACTOR = 28
PRIORITY_ORDER = None
GENERATE_SUBTASK_RATIO = 0.0
MODEL_TYPE = "qwen2_5"

TOKENIZER_MAX_LENGTH = 768
