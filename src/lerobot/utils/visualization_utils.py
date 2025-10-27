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

import numbers
import os
from typing import Any
from uuid import uuid4

import numpy as np
import rerun as rr

from lerobot.datasets.utils import DEFAULT_AUDIO_CHUNK_DURATION
from lerobot.robots import Robot

from .constants import OBS_PREFIX, OBS_STR


def init_rerun(
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    robot: Robot | None = None,
    reset_time: bool = False,
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
        robot: A Robot object. If provided, Rerun will be initialized with a blueprint that includes the object's cameras and microphones.
        reset_time: Whether to reset the timer "episode_time" to 0.
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(
        application_id=session_name,
        recording_id=uuid4(),
        default_blueprint=build_rerun_blueprint(robot) if robot is not None else None,
    )
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)

    if reset_time:
        rr.set_time_seconds("episode_time", seconds=0.0)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def build_rerun_blueprint(robot: Robot) -> rr.blueprint.Grid:
    """ "
    Builds a Rerun blueprint for optimized visualization of the robot's observations and actions :
    -   Time series views for all scalar observations and actions (e.g. position, velocity, torque, etc.).
    -   Spatial 2D views for all camera observations.
    -   Time series views for all microphone observations.

    Args:
        robot: A Robot object.
    Returns:
        A Rerun blueprint.
    """
    contents = [
        rr.blueprint.TimeSeriesView(
            origin="states_actions",
            plot_legend=rr.blueprint.PlotLegend(visible=True),
        )
    ]
    if robot.microphones:
        contents += [
            rr.blueprint.TimeSeriesView(
                origin="microphones",
                plot_legend=rr.blueprint.PlotLegend(visible=True),
            )
        ]
    if robot.cameras:
        contents += [
            rr.blueprint.Spatial2DView(
                origin=camera_name,
            )
            for camera_name in robot.cameras
        ]

    return rr.blueprint.Grid(contents)


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    compress_images: bool = False,
    log_time: float | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
        log_time: The time to log the data in the "episode_time" timeline.
                  If None, the current time is used in Rerun's default timeline.
    """
    if log_time is not None:
        rr.set_time_seconds("episode_time", seconds=log_time)

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                # Convert channel x samples -> samples x channel when needed
                elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    arr = np.transpose(arr, (1, 0))

                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                elif arr.ndim == 2:
                    for i, channel_arr in enumerate(arr.T):
                        rr.send_columns(
                            "audio/"
                            + key
                            + f"_channel_{i}",  # TODO(CarolinePascal): Get actual channel number/name
                            indexes=[
                                rr.TimeSecondsColumn(
                                    "episode_time",
                                    times=log_time
                                    + np.linspace(
                                        -DEFAULT_AUDIO_CHUNK_DURATION,
                                        0,
                                        len(channel_arr),
                                        endpoint=False,
                                    ),
                                )
                            ],
                            columns=rr.Scalar.columns(scalar=channel_arr),
                        )
                elif arr.ndim == 3:
                    rr.log(key, rr.Image(arr), static=True)
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

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
