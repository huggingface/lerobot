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

import contextlib
import numbers
import os
import sys
from typing import Any

import cv2
import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def init_rerun(
    session_name: str = "lerobot_control_loop",
    headless: bool = True,
    grpc_port: int = 9876,
    web_port: int = 9090,
    open_browser: bool = False,
    server_memory_limit: str = "25%",
) -> None:
    """Initializes the Rerun SDK for visualizing the control loop.
    
    Args:
        session_name: Name of the Rerun session.
        headless: If True, run in headless mode with gRPC server (default).
                  If False, spawn a local GUI viewer.
                  Can be overridden by RERUN_HEADLESS env var ("true"/"false").
        grpc_port: Port for gRPC server (default 9876).
        web_port: Port for web viewer (default 9090) - DEPRECATED, not used anymore.
        open_browser: Whether to attempt opening browser (default False for headless).
        server_memory_limit: Server-side buffer for late viewers (default "25%").
    
    Notes:
        In headless mode (default), only the gRPC server is started on the Jetson.
        To view data, run the web viewer on your external computer (with GPU):
            rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://JETSON_IP:9876/proxy"
        Then open http://localhost:9090 on your external computer's browser.
    """
    # Existing env tweaks
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    
    rr.init(session_name)
    
    # Check if headless mode is overridden by environment variable
    headless_env = os.getenv("RERUN_HEADLESS")
    if headless_env is not None:
        headless = headless_env.lower() in ("true", "1", "yes")
    
    if headless:
        # Start ONLY gRPC server on Jetson (headless logging endpoint)
        # The web viewer should be run separately on external GPU-capable machine
        # Suppress output messages from Rerun server startup
        with suppress_output():
            server_uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit=server_memory_limit)
        # Note: server_uri can be printed if debugging: print(f"gRPC server: {server_uri}")
    else:
        # Fallback to spawn a local viewer (for dev with GUI)
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def _downsample_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Downsample an image for visualization bandwidth reduction.
    
    Args:
        image: Input image (HWC or CHW format)
        scale_factor: Scaling factor (e.g., 0.5 for half size)
    
    Returns:
        Downsampled image in the same format as input
    """
    if scale_factor >= 1.0:
        return image
    
    # Check if CHW format (channels first)
    is_chw = image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)
    
    if is_chw:
        # Convert CHW to HWC for cv2.resize
        image = np.transpose(image, (1, 2, 0))
    
    # Calculate new dimensions
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    # Downsample using cv2 (fast and high quality)
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    if is_chw:
        # Convert back to CHW
        downsampled = np.transpose(downsampled, (2, 0, 1))
    
    return downsampled


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`. Images are downsampled for bandwidth reduction.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.
    
    Environment Variables:
        RERUN_DOWNSAMPLE_FACTOR: Image downsampling factor (default: 0.5 for half resolution)
                                 Set to 1.0 to disable downsampling.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
    """
    # Get downsample factor from environment (default: 0.5 for half resolution)
    downsample_factor = float(os.getenv("RERUN_DOWNSAMPLE_FACTOR", "0.5"))
    
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Check if this is an image (3D array with reasonable dimensions)
                is_image = arr.ndim == 3 or (arr.ndim == 2 and arr.shape[0] > 10 and arr.shape[1] > 10)
                
                if is_image and arr.ndim == 3:
                    # Downsample image before sending to Rerun
                    arr = _downsample_image(arr, downsample_factor)
                
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))