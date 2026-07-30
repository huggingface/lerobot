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
"""VLM camera-view curation for LeRobot datasets.

For each dataset, the first episode is inspected by a vision-language model to
(1) judge whether each camera view is blurry/unusable and (2) assign a canonical
view label (``top``/``wrist``/``front``/…). The labels can then be applied by
renaming the camera keys — for video datasets via a download-free, server-side
Hub commit. Exposed as the ``lerobot-curate-cameras`` CLI.
"""

from .config import DEFAULT_VIEW_VOCABULARY, CameraCurationConfig
from .curator import (
    CameraVerdict,
    build_name_mapping,
    build_report,
    curate_cameras,
    is_valid_view_label,
    rename_camera_keys_on_hub,
    write_report,
)

__all__ = [
    "DEFAULT_VIEW_VOCABULARY",
    "CameraCurationConfig",
    "CameraVerdict",
    "build_name_mapping",
    "build_report",
    "curate_cameras",
    "is_valid_view_label",
    "rename_camera_keys_on_hub",
    "write_report",
]
