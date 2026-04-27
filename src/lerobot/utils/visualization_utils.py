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


def _derive_depth_obs_ranges(
    features: dict[str, dict] | None,
) -> dict[str, tuple[float, float] | None]:
    """Map observation keys of depth features to their ``(depth_min, depth_max)`` range.

    A feature is considered a depth map when its ``info`` dict carries
    ``video.is_depth_map=True`` (the marker set by ``hw_to_dataset_features``
    and persisted in ``info.json``). For each such feature, we record both
    the fully-namespaced dataset key (e.g. ``observation.depth.front``) and
    the corresponding raw observation key forms the robot is likely to emit
    (``front`` and ``front_depth``) so a single membership check covers all
    call sites.

    The mapped value is the ``(depth_min, depth_max)`` range stored on the
    feature (matching the quantization range used at encoding time), or
    ``None`` when the metadata doesn't expose a range — in which case the
    caller should let Rerun auto-normalize. Anchoring the colormap to a
    fixed range avoids per-frame re-normalization, which otherwise looks
    like flicker on near-static scenes.
    """
    ranges: dict[str, tuple[float, float] | None] = {}
    if not features:
        return ranges
    depth_prefix = f"{OBS_STR}.depth."
    for fk, fv in features.items():
        info = fv.get("info") if isinstance(fv, dict) else None
        if not isinstance(info, dict) or not info.get("video.is_depth_map", False):
            continue
        depth_min = info.get("video.depth_min")
        depth_max = info.get("video.depth_max")
        rng: tuple[float, float] | None = None
        if (
            isinstance(depth_min, (int, float))
            and isinstance(depth_max, (int, float))
            and depth_max > depth_min
        ):
            rng = (float(depth_min), float(depth_max))
        ranges[fk] = rng
        if fk.startswith(depth_prefix):
            bare = fk[len(depth_prefix) :]
            ranges[bare] = rng
            ranges[f"{bare}_depth"] = rng
    return ranges


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    features: dict[str, dict] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 2D NumPy arrays whose key matches a depth feature in ``features`` (i.e. carrying
      ``video.is_depth_map=True``) are logged as ``rr.DepthImage`` with the Viridis
      colormap and ``meter=1.0`` (depth values are expected in metric meters). When
      the feature exposes ``video.depth_min`` / ``video.depth_max`` (the encoder
      quantization range, persisted in ``info.json``), the colormap is anchored to
      that range via ``depth_range`` to keep the visualization stable across frames.
      Depth images are never JPEG-compressed regardless of ``compress_images``.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
        features: Optional dataset feature spec (e.g. ``LeRobotDataset.features``). When
            provided, observation entries matching a depth-map feature are rendered with
            ``rr.DepthImage`` instead of the generic ``rr.Image`` path.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    depth_obs_ranges = _derive_depth_obs_ranges(features)

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                is_depth = bool(depth_obs_ranges) and (k in depth_obs_ranges or key in depth_obs_ranges)
                if is_depth and arr.ndim == 2:
                    # Viridis-colormapped DepthImage; never JPEG-compress (lossy on float metric depth).
                    # Anchor the colormap to the encoder range when available, so the
                    # visualization doesn't flicker as per-frame min/max drift.
                    depth_range = depth_obs_ranges.get(k) or depth_obs_ranges.get(key)
                    depth_kwargs: dict = {
                        "meter": 1.0,
                        "colormap": rr.components.Colormap.Viridis,
                    }
                    if depth_range is not None:
                        depth_kwargs["depth_range"] = depth_range
                    rr.log(key, rr.DepthImage(arr, **depth_kwargs), static=True)
                    continue
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

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
