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

import cv2
import numpy as np
import rerun as rr


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    """
    Log observation and action data to Rerun with hierarchical camera organization.
    
    Uses hierarchical paths like:
    - observation/cam_kinect/rgb
    - observation/cam_kinect/depth  
    - observation/cam_low/rgb
    - observation/cam_low/depth
    """
    # Separate camera frames from other observation data
    camera_frames = {}
    other_obs = {}
    
    for key, val in observation.items():
        if isinstance(val, np.ndarray) and val.ndim > 1:
            camera_frames[key] = val
        else:
            other_obs[key] = val

    # Log non-camera observations with simple paths
    for obs, val in other_obs.items():
        if isinstance(val, float):
            rr.log(f"observation/{obs}", rr.Scalar(val))
        elif isinstance(val, np.ndarray) and val.ndim == 1:
            for i, v in enumerate(val):
                rr.log(f"observation/{obs}/{i}", rr.Scalar(float(v)))

    # Log camera frames with hierarchical organization
    for key, frame in camera_frames.items():
        try:
            # Parse camera key to create hierarchical path
            # Examples: "cam_kinect" -> "cam_kinect/rgb"
            #          "cam_kinect_depth" -> "cam_kinect/depth"
            #          "cam_low_depth" -> "cam_low/depth"
            parts = key.rsplit('_', 1)
            if len(parts) == 2 and parts[1] == "depth":
                # This is a depth stream: cam_kinect_depth -> cam_kinect/depth
                entity_path = f"observation/{parts[0]}/depth"
            else:
                # This is a color stream: cam_kinect -> cam_kinect/rgb
                entity_path = f"observation/{key}/rgb"

            # Handle different frame types
            if frame.ndim == 2 and frame.dtype in [np.float32, np.uint16]:
                # Raw depth data - convert to millimeters and log as DepthImage
                if frame.dtype == np.uint16:
                    # RealSense (uint16) is in micrometers. Convert to millimeters.
                    depth_mm = frame.astype(np.float32) / 1000.0
                else:
                    # Kinect (float32) is already in millimeters.
                    depth_mm = frame

                # Rerun's `meter` argument specifies how many depth units are in a meter.
                # Since depth_mm is in millimeters, there are 1000 millimeters in a meter.
                rr.log(entity_path, rr.DepthImage(depth_mm, meter=1000.0), static=True)
                
            elif frame.ndim == 3:
                # Color image data - handle BGRA from Kinect
                if frame.shape[2] == 4:
                    # Convert BGRA to RGB for visualization
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    rr.log(entity_path, rr.Image(frame_rgb), static=True)
                else:
                    # Regular RGB/BGR image
                    rr.log(entity_path, rr.Image(frame), static=True)
            else:
                # Unexpected frame format - log with warning
                print(f"Warning: Unexpected frame format for {key}: shape={frame.shape}, dtype={frame.dtype}")
                rr.log(f"observation/{key}", rr.Image(frame), static=True)
                
        except Exception as e:
            # Log any errors to help with debugging
            print(f"Error logging {key} to Rerun: {e} (shape={frame.shape}, dtype={frame.dtype})")

    # Log actions with simple paths
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action/{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action/{act}/{i}", rr.Scalar(float(v)))
