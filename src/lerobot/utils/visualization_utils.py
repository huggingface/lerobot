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
import time

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package

# Module-level Foxglove state. A single WebSocket server is shared for the
# process lifetime, and image channels are cached by topic (the Foxglove SDK
# requires reusing one channel per topic).
_foxglove_server = None
_foxglove_channels: dict = {}


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


def init_foxglove(host: str = "127.0.0.1", port: int | None = 8765) -> None:
    """
    Starts a Foxglove WebSocket server for visualizing the control loop.

    Connect to it from the Foxglove app at ``ws://<host>:<port>``. Calling this
    more than once is a no-op while a server is already running.

    Args:
        host: Host interface to bind the WebSocket server to.
        port: Port to bind the WebSocket server to (defaults to 8765).
    """

    require_package("foxglove-sdk", extra="foxglove", import_name="foxglove")
    import foxglove

    global _foxglove_server
    if _foxglove_server is not None:
        return
    _foxglove_server = foxglove.start_server(host=host, port=port or 8765)


def shutdown_foxglove() -> None:
    """Stops the Foxglove WebSocket server and clears cached channels."""

    global _foxglove_server
    if _foxglove_server is not None:
        _foxglove_server.stop()
        _foxglove_server = None
    _foxglove_channels.clear()


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

    if not (observation_paths or action_paths or image_paths):
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    blueprint = _build_blueprint(observation_paths, action_paths, image_paths)
    log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def _foxglove_safe_name(name: str) -> str:
    """Make a feature name usable as an unquoted Foxglove message path / topic segment.

    Foxglove message paths treat ``.`` as a field separator, so ``shoulder_pan.pos`` would have to be
    written as ``"shoulder_pan.pos"`` when plotting. Replacing ``.`` with ``_`` avoids the quoting.
    """

    return name.replace(".", "_")


# Static schema shared by all scalar topics. Each message carries a flat list of ``{label, value}``
# pairs rather than one field per feature, so the same schema fits any robot regardless of which
# observation/action features it reports. The ``label`` field name is what Foxglove looks for to name
# each series automatically, so a single filtered path plots every feature, e.g.
# ``/observation/state.scalars[:].value``.
_SCALARS_SCHEMA = {
    "type": "object",
    "title": "lerobot.Scalars",
    "properties": {
        "scalars": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                },
            },
        }
    },
}


def _log_foxglove_scalars(topic: str, values: dict[str, float]) -> None:
    """Log scalars on a typed JSON channel using the static :data:`_SCALARS_SCHEMA`.

    ``values`` is an ordered mapping of feature name to value; it is emitted as a ``scalars`` array of
    ``{label, value}`` objects. Insertion order is preserved so series stay stable across messages.
    """

    if not values:
        return

    import foxglove

    channel = _foxglove_channels.get(topic)
    if channel is None:
        channel = _foxglove_channels[topic] = foxglove.Channel(
            topic, schema=_SCALARS_SCHEMA, message_encoding="json"
        )
    channel.log({"scalars": [{"label": label, "value": value} for label, value in values.items()]})


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

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
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
                observation_paths.add(key)
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    rr.log(key, rr.Scalars(arr.astype(float)))
                    observation_paths.add(key)
                else:
                    if arr.shape[-1] == 1:
                        img_entity = rr.DepthImage(arr, colormap=rr.components.Colormap.Viridis)
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)
                    image_paths.add(key)

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

    _ensure_blueprint(observation_paths, action_paths, image_paths)


def log_foxglove_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to a Foxglove WebSocket server for real-time visualization.

    Mirrors :func:`log_rerun_data` but emits Foxglove messages over the server started by
    :func:`init_foxglove`. Data is mapped as follows:
    - Scalars (and elements of 1D arrays) are accumulated per source and logged on the
      ``/observation/state`` and ``/action/state`` topics as typed JSON messages using the static
      ``lerobot.Scalars`` schema: a ``scalars`` array of ``{label, value}`` objects (see
      :data:`_SCALARS_SCHEMA`). The ``label`` field lets Foxglove name each series automatically, so
      ``/observation/state.scalars[:].value`` plots every feature at once.
    - 3D NumPy arrays that resemble images are transposed from CHW to HWC when needed and logged on a
      per-source topic (e.g. ``/observation/images/front``) as a ``RawImage`` (or a JPEG
      ``CompressedImage`` when ``compress_images`` is True).

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to JPEG-compress images before logging to save bandwidth in exchange
            for CPU and quality.
    """

    require_package("foxglove-sdk", extra="foxglove", import_name="foxglove")
    from foxglove.channels import CompressedImageChannel, RawImageChannel
    from foxglove.messages import CompressedImage, RawImage, Timestamp

    if _foxglove_server is None:
        raise RuntimeError("init_foxglove() must be called before log_foxglove_data().")

    now = time.time_ns()
    timestamp = Timestamp(sec=now // 1_000_000_000, nsec=now % 1_000_000_000)

    def log_image(topic: str, frame_id: str, arr: np.ndarray) -> None:
        # Convert CHW -> HWC when needed (mirrors log_rerun_data).
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        height, width = arr.shape[0], arr.shape[1]
        channels = 1 if arr.ndim == 2 else arr.shape[2]

        if compress_images and channels == 3:
            import cv2

            # Camera frames are RGB; cv2.imencode assumes BGR, so swap to keep colors correct.
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            channel = _foxglove_channels.get(topic)
            if channel is None:
                channel = _foxglove_channels[topic] = CompressedImageChannel(topic=topic)
            channel.log(
                CompressedImage(timestamp=timestamp, frame_id=frame_id, data=buf.tobytes(), format="jpeg")
            )
            return

        encoding = {1: "mono8", 3: "rgb8", 4: "rgba8"}.get(channels)
        if encoding is None:
            return
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        channel = _foxglove_channels.get(topic)
        if channel is None:
            channel = _foxglove_channels[topic] = RawImageChannel(topic=topic)
        channel.log(
            RawImage(
                timestamp=timestamp,
                frame_id=frame_id,
                width=width,
                height=height,
                encoding=encoding,
                step=width * channels,
                data=arr.tobytes(),
            )
        )

    if observation:
        obs_scalars: dict[str, float] = {}
        for k, v in observation.items():
            if v is None:
                continue
            key = k[len(OBS_PREFIX) :] if str(k).startswith(OBS_PREFIX) else str(k)
            if _is_scalar(v):
                obs_scalars[key] = float(v)
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        obs_scalars[f"{key}_{i}"] = float(vi)
                else:
                    # Image topics still sanitize the name since it's used as a topic-path segment.
                    log_image(f"/{OBS_STR}/images/{_foxglove_safe_name(key)}", key, v)
        _log_foxglove_scalars(f"/{OBS_STR}/state", obs_scalars)

    if action:
        action_scalars: dict[str, float] = {}
        for k, v in action.items():
            if v is None:
                continue
            key = k[len(ACTION_PREFIX) :] if str(k).startswith(ACTION_PREFIX) else str(k)
            if _is_scalar(v):
                action_scalars[key] = float(v)
            elif isinstance(v, np.ndarray):
                for i, vi in enumerate(v.flatten()):
                    action_scalars[f"{key}_{i}"] = float(vi)
        _log_foxglove_scalars(f"/{ACTION}/state", action_scalars)


# ── Backend-agnostic dispatch ─────────────────────────────────────────────
# These let callers select a visualization backend at runtime via a string
# (e.g. a `--display_mode` CLI flag) without branching on the backend everywhere.

VISUALIZATION_MODES = ("rerun", "foxglove")


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``port`` is the local WebSocket server port (``ip`` is ignored; the server binds locally).
    """

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Logs observation/action data to the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        log_rerun_data(observation=observation, action=action, compress_images=compress_images)
    elif display_mode == "foxglove":
        log_foxglove_data(observation=observation, action=action, compress_images=compress_images)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """Shuts down the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")
