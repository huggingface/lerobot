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

from lerobot.cameras.depth_utils import is_raw_depth


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    """Enhanced logging with optimized depth visualization.
    
    Handles:
    - Raw depth: Native rr.DepthImage() for 3D visualization
    - RGB images: Standard rr.Image() logging
    - Skips colorized depth to reduce clutter and improve performance
    """
    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation.{obs}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", rr.Scalar(float(v)))
            else:
                # Enhanced multi-dimensional array handling with depth support
                if obs.endswith("_depth_raw") and is_raw_depth(val):
                    # Raw depth data - use native Rerun depth visualization
                    rr.log(f"observation.{obs}", rr.DepthImage(val, meter=1.0/1000.0), static=True)
                elif obs.endswith("_depth"):
                    # Skip colorized depth images to reduce clutter and improve performance
                    continue
                else:
                    # Regular RGB images only
                    rr.log(f"observation.{obs}", rr.Image(val), static=True)
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", rr.Scalar(float(v)))
