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

"""Rerun visualization backend.

Live control-loop streaming to the Rerun viewer (:func:`log_rerun_data`). Callers usually select a
backend at runtime through the dispatch in :mod:`lerobot.utils.visualization_utils` rather than
importing from here directly. Requires the ``viz`` extra (``pip install 'lerobot[viz]'``).
"""

import logging
import numbers
import os

import numpy as np

from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
from lerobot.lerobot_types import PolicyPrediction, RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_IMAGES, OBS_PREFIX, OBS_STR, PREDICTION
from .import_utils import require_package

logger = logging.getLogger(__name__)

# (height, width) of each image entity logged so far, so policy overlays given in image
# fractions can be scaled to pixels.
_IMAGE_SIZES: dict[str, tuple[int, int]] = {}


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


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

    # Reset the blueprint cache for the new session.
    log_rerun_data.blueprint = None  # type: ignore[attr-defined]
    log_rerun_data.blueprint_paths = None  # type: ignore[attr-defined]
    _IMAGE_SIZES.clear()

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


def _build_blueprint(
    observation_paths: set[str],
    action_paths: set[str],
    image_paths: set[str],
    text_paths: set[str] = frozenset(),  # type: ignore[assignment]
):
    """Build a Rerun blueprint laying out images, scalar series and text panels in separate views.

    Each image (camera or predicted frame) gets a spatial view, observation and action scalars each
    get a time-series view, and each predicted text gets a text panel. All arranged in a grid.
    """

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]

    if observation_paths:
        views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))
    views += [rrb.TextDocumentView(origin=path, name=path) for path in sorted(text_paths)]

    return rrb.Blueprint(rrb.Grid(*views))


def _ensure_blueprint(
    observation_paths: set[str],
    action_paths: set[str],
    image_paths: set[str],
    text_paths: set[str] = frozenset(),  # type: ignore[assignment]
) -> None:
    """Send the blueprint on the first data, and again whenever an entity that needs a view appears.

    Predictions can show up long after the first observation (e.g. on the first predicted chunk),
    so the layout grows with them instead of being frozen by the first call.
    """
    groups = (observation_paths, action_paths, image_paths, text_paths)
    known = getattr(log_rerun_data, "blueprint_paths", None)
    if getattr(log_rerun_data, "blueprint", None) is None or known is None:
        known = tuple(set() for _ in groups)
    elif all(new <= seen for new, seen in zip(groups, known, strict=True)):
        return

    merged = tuple(seen | new for seen, new in zip(known, groups, strict=True))
    if not any(merged):
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    blueprint = _build_blueprint(*merged)
    log_rerun_data.blueprint = blueprint  # type: ignore[attr-defined]
    log_rerun_data.blueprint_paths = merged  # type: ignore[attr-defined]
    rr.send_blueprint(blueprint)


def _log_image(rr, key: str, arr: np.ndarray, image_paths: set[str], compress_images: bool) -> None:
    """Log a 2D/3D array as an image (CHW -> HWC when needed; single-channel as depth)."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        # At record time, the depth unit is inferred from the frame type.
        depth_unit = infer_depth_unit(arr.dtype)
        img_entity = rr.DepthImage(
            arr,
            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
            colormap=rr.components.Colormap.Viridis,
        )
    else:
        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
    rr.log(key, entity=img_entity, static=True)
    image_paths.add(key)
    _IMAGE_SIZES[key] = (arr.shape[0], arr.shape[1])


def _camera_entity(camera: str) -> str | None:
    """Entity a camera's frames were logged under, for a policy-side feature key or a robot key.

    The control loop logs robot cameras as ``observation.<name>`` while a policy refers to them by
    feature key (``observation.images.<name>``); both resolve to whichever was actually logged.
    """
    name = camera.removeprefix(f"{OBS_IMAGES}.").removeprefix(OBS_PREFIX)
    for entity in (camera, f"{OBS_STR}.{name}", f"{OBS_IMAGES}.{name}"):
        if entity in _IMAGE_SIZES:
            return entity
    return None


def _log_prediction(
    rr,
    prediction: PolicyPrediction,
    observation_paths: set[str],
    image_paths: set[str],
    text_paths: set[str],
    compress_images: bool,
) -> None:
    """Log a policy's prediction, named after what it predicts.

    A predicted ``observation.images.top`` is logged as ``prediction.images.top`` and predicted
    language as ``prediction.<style>``. Boxes are logged under their camera's entity instead, so they
    draw over its image.
    """
    for key, value in prediction.get("observation", {}).items():
        path = f"{PREDICTION}.{key.removeprefix(OBS_PREFIX)}"
        arr = value.numpy(force=True)
        if arr.ndim == 1:
            rr.log(path, rr.Scalars(arr.astype(float)))
            observation_paths.add(path)
        else:
            _log_image(rr, path, arr, image_paths, compress_images)
    for style, text in prediction.get("language", {}).items():
        path = f"{PREDICTION}.{style}"
        rr.log(path, rr.TextDocument(text))
        text_paths.add(path)
    for camera_key, answer in prediction.get("boxes", {}).items():
        camera = _camera_entity(camera_key)
        if camera is None:
            logger.debug("Skipping predicted boxes: camera %r was not logged", camera_key)
            continue
        height, width = _IMAGE_SIZES[camera]
        detections = answer["detections"]
        boxes = np.asarray([d["bbox"] for d in detections], dtype=float).reshape(-1, 4)
        boxes = boxes * (width, height, width, height)
        # Under the camera's entity so they draw over its image; no detections clears them.
        entity = f"{camera}/{PREDICTION}"
        if len(boxes):
            labels = [d["label"] for d in detections]
            rr.log(
                entity, rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYXY, labels=labels), static=True
            )
        else:
            rr.log(entity, rr.Clear(recursive=False), static=True)


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    prediction: PolicyPrediction | None = None,
) -> None:
    """
    Logs observation, action and policy-prediction data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a single `rr.Scalars` batch under one entity path, so that every
      dimension shares the same view instead of being split across one view per element.
    - Multi-dimensional **action** arrays are flattened and logged as a single `rr.Scalars` batch.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    On the first call, a blueprint is built and sent so observation and action scalars get separate
    time-series views and each image gets its own spatial view.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
        prediction: An optional `PolicyPrediction`, named after what it predicts: a predicted
            ``observation.images.top`` as ``prediction.images.top``, language as ``prediction.<style>``
            in text panels, and boxes over the camera they are drawn on.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()
    text_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                observation_paths.add(key)
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    rr.log(key, rr.Scalars(v.astype(float)))
                    observation_paths.add(key)
                else:
                    _log_image(rr, key, v, image_paths, compress_images)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                action_paths.add(key)
            elif isinstance(v, np.ndarray):
                # Flatten any (incl. higher-dimensional) array into a single batched Scalars
                rr.log(key, rr.Scalars(v.reshape(-1).astype(float)))
                action_paths.add(key)

    if prediction:
        _log_prediction(rr, prediction, observation_paths, image_paths, text_paths, compress_images)

    _ensure_blueprint(observation_paths, action_paths, image_paths, text_paths)
