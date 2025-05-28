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
from uuid import uuid4

from lerobot.robots import Robot
from lerobot.datasets.utils import DEFAULT_AUDIO_CHUNK_DURATION

def _init_rerun(session_name: str = "lerobot_control_loop", robot: Robot | None = None, reset_time: bool = False) -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(
        application_id = session_name,
        recording_id = uuid4(),
        default_blueprint = build_rerun_blueprint(robot) if robot is not None else None
    )
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)

    if reset_time:
        rr.set_time_seconds("episode_time", seconds=0.0)

def build_rerun_blueprint(robot: Robot) -> rr.blueprint.Grid:
    contents=[
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

def log_rerun_data(observation: dict[str | Any], action: dict[str | Any], log_time: float | None = None):
    if log_time is not None:
        rr.set_time_seconds("episode_time", seconds=log_time)
    
    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"states_actions/observation.{obs}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"states_actions/observation.{obs}_{i}", rr.Scalar(float(v)))
            elif val.ndim == 2:
                rr.send_columns(
                    "audio/" + obs,
                    indexes=[
                        rr.TimeSecondsColumn(
                            "episode_time",
                            times=log_time
                            + np.linspace(
                                -DEFAULT_AUDIO_CHUNK_DURATION,
                                0,
                                len(observation[obs]),
                                endpoint=False,
                            ),
                        )
                    ],
                    columns=rr.Scalar.columns(scalar=np.mean(observation[obs], axis=1)),
                )
            else:
                rr.log(obs, rr.Image(val), static=True)
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"states_actions/action.{act}", rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"states_actions/action.{act}_{i}", rr.Scalar(float(v)))
