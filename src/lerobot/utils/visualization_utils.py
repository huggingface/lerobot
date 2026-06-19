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

from .constants import ACTION, ACTION_PREFIX, DONE, OBS_PREFIX, OBS_STATE, OBS_STR, REWARD
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
    """Replace ``.`` with ``_`` so a feature name is a single Foxglove topic-path segment.

    Foxglove treats ``.`` as a path separator, so an unsanitized name like ``observation.images.front``
    would split into nested segments instead of naming one topic.
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


def _log_foxglove_scalars(topic: str, values: dict[str, float], *, log_time: int | None = None) -> None:
    """Log scalars on a typed JSON channel using the static :data:`_SCALARS_SCHEMA`.

    ``values`` is an ordered mapping of feature name to value; it is emitted as a ``scalars`` array of
    ``{label, value}`` objects. Insertion order is preserved so series stay stable across messages.

    ``log_time`` is the message time in nanoseconds. When ``None`` the server's receive time is used
    (correct for live streaming); dataset playback passes the frame's dataset timestamp so the
    Foxglove timeline reflects the recorded episode.
    """

    if not values:
        return

    import foxglove

    channel = _foxglove_channels.get(topic)
    if channel is None:
        channel = _foxglove_channels[topic] = foxglove.Channel(
            topic, schema=_SCALARS_SCHEMA, message_encoding="json"
        )
    msg = {"scalars": [{"label": label, "value": value} for label, value in values.items()]}
    if log_time is None:
        channel.log(msg)
    else:
        channel.log(msg, log_time=log_time)


def _log_foxglove_image(
    topic: str, frame_id: str, arr: np.ndarray, *, compress_images: bool, time_ns: int
) -> None:
    """Log an image on a cached per-topic channel, stamped at ``time_ns`` (nanoseconds).

    ``arr`` may be HWC or CHW; CHW is transposed to HWC. ``time_ns`` sets both the message header
    timestamp and the channel ``log_time`` so the message lands at the right point on the Foxglove
    timeline (matching wall-clock for live streaming, or the dataset timestamp during playback).
    """

    from foxglove.channels import CompressedImageChannel, RawImageChannel
    from foxglove.messages import CompressedImage, RawImage, Timestamp

    timestamp = Timestamp(sec=time_ns // 1_000_000_000, nsec=time_ns % 1_000_000_000)

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
            CompressedImage(timestamp=timestamp, frame_id=frame_id, data=buf.tobytes(), format="jpeg"),
            log_time=time_ns,
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
        ),
        log_time=time_ns,
    )


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

    if _foxglove_server is None:
        raise RuntimeError("init_foxglove() must be called before log_foxglove_data().")

    now = time.time_ns()

    def log_image(topic: str, frame_id: str, arr: np.ndarray) -> None:
        _log_foxglove_image(topic, frame_id, arr, compress_images=compress_images, time_ns=now)

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


# ── Dataset playback over a Foxglove WebSocket server ─────────────────────
# A LeRobotDataset is random-access on disk, so rather than fire-and-forget a forward stream we
# advertise a seekable timeline and serve frames on demand for whatever time the user scrubs/plays
# to in the Foxglove app. This relies on the SDK's PlaybackControl capability.

_SUCCESS = "next.success"


def _feature_dim_names(feature: dict | None) -> list[str] | None:
    """Best-effort per-dimension series labels for a 1D feature, or ``None`` to fall back to indices.

    LeRobot records a feature's ``names`` inconsistently: a flat list (``["x", "y"]``), a category
    mapping (``{"motors": ["motor_0", "motor_1"]}``), or a name->index mapping
    (``{"delta_x": 0, "delta_y": 1}``). Each is handled, but labels are only returned when their count
    matches the feature's 1D shape, so a malformed/mismatched ``names`` can't silently mislabel series.
    """

    if not feature:
        return None
    shape = feature.get("shape")
    dim = shape[0] if shape and len(shape) == 1 else None
    names = feature.get("names")
    labels: list[str] | None = None
    if isinstance(names, dict):
        values = list(names.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            labels = [str(n) for group in values for n in group]
        elif values and all(isinstance(v, int) for v in values):
            labels = [name for name, _ in sorted(names.items(), key=lambda kv: kv[1])]
    elif isinstance(names, (list, tuple)):
        labels = [str(n) for n in names]
    if labels is not None and dim is not None and len(labels) == dim:
        return labels
    return None


def _frame_to_scalars(sample: dict, key: str, labels: list[str] | None = None) -> dict[str, float]:
    """Flatten a frame's vector/scalar feature ``key`` into ``{label: value}`` entries.

    ``labels`` provides one name per dimension (from the dataset's feature metadata); when absent or
    the wrong length, dimensions fall back to their index. A scalar feature becomes a single entry.
    Missing or ``None`` features yield an empty mapping.
    """

    v = sample.get(key)
    if v is None:
        return {}
    arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    flat = [float(arr)] if arr.ndim == 0 else [float(x) for x in arr.flatten()]
    if labels is None or len(labels) != len(flat):
        labels = [str(i) for i in range(len(flat))]
    return dict(zip(labels, flat, strict=True))


def serve_foxglove_dataset_playback(
    dataset,
    episode_index: int,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    compress_images: bool = False,
) -> None:
    """Serve a single dataset episode to Foxglove as a seekable, scrubbable timeline.

    Starts a Foxglove WebSocket server advertising the ``PlaybackControl`` capability over the
    episode's time range. The Foxglove app drives play/pause/seek/speed; a background thread and a
    ``ServerListener`` read frames from the on-disk ``dataset`` on demand and log them stamped at
    their dataset timestamps, so the user can scrub anywhere in the episode. Blocks until interrupted.

    Args:
        dataset: A ``LeRobotDataset`` loaded for the single episode to visualize.
        episode_index: Index of the episode being visualized (used only for the session name).
        host: Host interface to bind the WebSocket server to.
        port: Port to bind the WebSocket server to.
        compress_images: Whether to JPEG-compress camera frames before logging.
    """

    require_package("foxglove-sdk", extra="foxglove", import_name="foxglove")
    import bisect
    import threading

    import foxglove
    from foxglove.websocket import (
        Capability,
        PlaybackCommand,
        PlaybackControlRequest,
        PlaybackState,
        PlaybackStatus,
        ServerListener,
    )

    # Per-frame timestamps in nanoseconds (read straight from the table, no video decode).
    times_ns = [int(round(float(t) * 1e9)) for t in dataset.hf_dataset["timestamp"]]
    n_frames = len(times_ns)
    if n_frames == 0:
        raise ValueError("Cannot visualize an empty episode.")
    first_ns, last_ns = times_ns[0], times_ns[-1]
    camera_keys = list(dataset.meta.camera_keys)
    # Per-dimension series labels from the dataset metadata (e.g. joint names), computed once.
    scalar_labels = {
        OBS_STATE: _feature_dim_names(dataset.meta.features.get(OBS_STATE)),
        ACTION: _feature_dim_names(dataset.meta.features.get(ACTION)),
    }

    def topic_for(key: str) -> str:
        name = key[len(OBS_PREFIX) :] if str(key).startswith(OBS_PREFIX) else str(key)
        return f"/{OBS_STR}/images/{_foxglove_safe_name(name)}"

    def emit_frame(i: int) -> None:
        """Log every channel for frame ``i`` stamped at its dataset timestamp."""
        sample = dataset[i]
        log_time = times_ns[i]
        for key in camera_keys:
            arr = sample.get(key)
            if arr is None:
                continue
            arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            _log_foxglove_image(topic_for(key), key, arr, compress_images=compress_images, time_ns=log_time)
        _log_foxglove_scalars(
            f"/{OBS_STR}/state",
            _frame_to_scalars(sample, OBS_STATE, scalar_labels[OBS_STATE]),
            log_time=log_time,
        )
        _log_foxglove_scalars(
            f"/{ACTION}/state", _frame_to_scalars(sample, ACTION, scalar_labels[ACTION]), log_time=log_time
        )
        episode_scalars = {}
        for feat, label in ((DONE, "done"), (REWARD, "reward"), (_SUCCESS, "success")):
            v = sample.get(feat)
            if v is not None:
                episode_scalars[label] = float(v)
        _log_foxglove_scalars("/episode/state", episode_scalars, log_time=log_time)

    lock = threading.Lock()
    stop_event = threading.Event()
    # Shared playback state, guarded by ``lock``. ``seek_idx`` is a one-shot request set by the
    # listener and serviced by the playback loop, which is the *only* thread that emits frames (so
    # concurrent random access into the on-disk dataset / video decoder never overlaps).
    state = {
        "status": PlaybackStatus.Paused,
        "cursor": first_ns,
        "speed": 1.0,
        "last_idx": -1,
        "seek_idx": None,
    }

    def index_at(t_ns: int) -> int:
        return max(0, min(n_frames - 1, bisect.bisect_right(times_ns, t_ns) - 1))

    class _PlaybackListener(ServerListener):
        def on_playback_control_request(self, req: PlaybackControlRequest):
            # Only mutate state here; the playback loop performs all frame emission.
            with lock:
                did_seek = False
                if req.seek_time is not None:
                    cursor = max(first_ns, min(last_ns, req.seek_time))
                    state["cursor"] = cursor
                    state["last_idx"] = state["seek_idx"] = index_at(cursor)
                    did_seek = True
                if req.playback_speed and req.playback_speed > 0:
                    state["speed"] = req.playback_speed
                if req.playback_command == PlaybackCommand.Play:
                    # Restarting from the end replays from the beginning.
                    if state["cursor"] >= last_ns:
                        state["cursor"] = first_ns
                        state["last_idx"] = state["seek_idx"] = 0
                        did_seek = True
                    state["status"] = PlaybackStatus.Playing
                elif req.playback_command == PlaybackCommand.Pause:
                    state["status"] = PlaybackStatus.Paused
                status, cursor, speed = state["status"], state["cursor"], state["speed"]
                request_id = req.request_id or ""
            return PlaybackState(status, cursor, speed, did_seek, request_id)

    server = foxglove.start_server(
        name=f"{dataset.repo_id}/episode_{episode_index}",
        host=host,
        port=port,
        capabilities=[Capability.PlaybackControl, Capability.Time],
        server_listener=_PlaybackListener(),
        playback_time_range=(first_ns, last_ns),
    )

    def playback_loop() -> None:
        prev = time.monotonic()
        while not stop_event.is_set():
            time.sleep(1.0 / 60.0)
            ended = False
            speed = 1.0
            with lock:
                now = time.monotonic()
                dt = now - prev
                prev = now
                # A queued seek is always serviced, even while paused, so scrubbing updates the view.
                work = []
                seek_idx = state["seek_idx"]
                if seek_idx is not None:
                    state["seek_idx"] = None
                    work.append(seek_idx)
                if state["status"] == PlaybackStatus.Playing:
                    cursor = state["cursor"] + int(dt * 1e9 * state["speed"])
                    start_idx = state["last_idx"] + 1
                    if cursor >= last_ns:
                        cursor, target, ended = last_ns, n_frames - 1, True
                    else:
                        target = index_at(cursor)
                    state["cursor"] = cursor
                    work.extend(range(start_idx, target + 1))
                    state["last_idx"] = max(state["last_idx"], target)
                    if ended:
                        state["status"] = PlaybackStatus.Ended
                if not work:
                    continue
                cursor, speed = state["cursor"], state["speed"]
            # Emit outside the lock; this is the only thread that calls emit_frame.
            for i in work:
                emit_frame(i)
            server.broadcast_time(cursor)
            if ended:
                server.broadcast_playback_state(PlaybackState(PlaybackStatus.Ended, cursor, speed, False, ""))

    # Emit the first frame so channels are advertised (done before the loop starts, so emission stays
    # single-threaded). Late-connecting clients re-receive frames once they seek/play.
    emit_frame(0)
    with lock:
        state["last_idx"] = 0
    server.broadcast_time(first_ns)
    server.broadcast_playback_state(PlaybackState(PlaybackStatus.Paused, first_ns, 1.0, True, ""))

    thread = threading.Thread(target=playback_loop, name="foxglove-playback", daemon=True)
    thread.start()

    print(f"Foxglove server running. Connect the Foxglove app to ws://{host}:{port}")
    print("Use the playback controls in Foxglove to play/pause and scrub the episode. Ctrl-C to exit.")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Ctrl-C received. Exiting.")
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        server.stop()
        _foxglove_channels.clear()


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
