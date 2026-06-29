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

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    log_rerun_data.blueprint = None  # Reset blueprint cache for new session

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def _build_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]):
    """Build a Rerun blueprint laying out camera images, observation and action scalars in separate views.

    Camera images, observation and action scalars are arranged in a grid.
    """

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]

    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))

    return rrb.Blueprint(rrb.Grid(*views))


def _ensure_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]) -> None:
    """Build and send the blueprint once, from the first observation and action data."""
    if getattr(log_rerun_data, "blueprint", None) is not None:
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    blueprint = _build_blueprint(observation_paths, action_paths, image_paths)
    log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def _collect_observation(
    rr, observation: RobotObservation, image_paths: set[str], compress_images: bool
) -> tuple[list[str], list[float]]:
    """Log observation images individually and gather every observation scalar into one batch.

    Images (and depth maps) keep their own entity path; all scalar / vector values are flattened into a
    single ``(names, values)`` pair so they can be logged as one grouped ``rr.Scalars`` batch.
    """
    names: list[str] = []
    values: list[float] = []
    for k, v in observation.items():
        if v is None:
            continue
        key = str(k)
        label = key[len(OBS_PREFIX) :] if key.startswith(OBS_PREFIX) else key
        path = key if key.startswith(OBS_PREFIX) else f"{OBS_PREFIX}{key}"

        if _is_scalar(v):
            names.append(label)
            values.append(float(v))
        elif isinstance(v, np.ndarray):
            arr = v
            # Convert CHW -> HWC when needed
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim <= 1:
                _extend_scalars(names, values, label, arr)
            else:
                if arr.shape[-1] == 1:
                    img_entity = rr.DepthImage(arr, colormap=rr.components.Colormap.Viridis)
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                rr.log(path, entity=img_entity, static=True)
                image_paths.add(path)
    return names, values


def _collect_action(action: RobotAction) -> tuple[list[str], list[float]]:
    """Gather every action scalar / vector value into one ``(names, values)`` batch."""
    names: list[str] = []
    values: list[float] = []
    for k, v in action.items():
        if v is None:
            continue
        key = str(k)
        label = key[len(ACTION_PREFIX) :] if key.startswith(ACTION_PREFIX) else key

        if _is_scalar(v):
            names.append(label)
            values.append(float(v))
        elif isinstance(v, np.ndarray):
            _extend_scalars(names, values, label, v.reshape(-1))
    return names, values


def _extend_scalars(names: list[str], values: list[float], label: str, arr: np.ndarray) -> None:
    """Append a 0/1-D array to the grouped scalar batch, naming multi-element entries ``label[i]``."""
    flat = arr.reshape(-1).astype(float)
    if flat.size == 1:
        names.append(label)
        values.append(float(flat[0]))
    else:
        for i, val in enumerate(flat):
            names.append(f"{label}[{i}]")
            values.append(float(val))


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - All observation scalars/vectors are gathered and logged as a single `rr.Scalars` batch under the
      `observation` entity path; likewise every action value is logged as one `rr.Scalars` batch under the
      `action` entity path. This keeps a robot's joints grouped in one plot instead of split across one
      entity path (and one log call) per joint.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`,
      and (for single-channel depth) as `rr.DepthImage`. Each image keeps its own entity path.
    - Vector (1D / flattened) values contribute one named series per element (`label[i]`) within the batch.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    On the first call, the per-series names and a blueprint are sent so the observation and action scalars
    each get a single time-series view and each image gets its own spatial view.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    image_paths: set[str] = set()
    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    observation_names: list[str] = []
    action_names: list[str] = []

    if observation:
        observation_names, observation_values = _collect_observation(
            rr, observation, image_paths, compress_images
        )
        if observation_values:
            rr.log(OBS_STR, rr.Scalars(observation_values))
            observation_paths.add(OBS_STR)

    if action:
        action_names, action_values = _collect_action(action)
        if action_values:
            rr.log(ACTION, rr.Scalars(action_values))
            action_paths.add(ACTION)

    # On the first call, statically log the per-series names so the grouped batches show a readable
    # label (e.g. "shoulder_pan.pos") for each line, then send the blueprint.
    if getattr(log_rerun_data, "blueprint", None) is None:
        if observation_names:
            rr.log(OBS_STR, rr.SeriesLines(names=observation_names), static=True)
        if action_names:
            rr.log(ACTION, rr.SeriesLines(names=action_names), static=True)

    _ensure_blueprint(observation_paths, action_paths, image_paths)
