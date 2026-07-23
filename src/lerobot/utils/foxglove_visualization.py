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

"""Foxglove visualization backend.

Live control-loop streaming (:func:`log_foxglove_data`) and seekable dataset playback
(:func:`serve_foxglove_dataset_playback`) over a Foxglove WebSocket server. Callers usually select a
backend at runtime through the dispatch in :mod:`lerobot.utils.visualization_utils` rather than
importing from here directly. Requires the ``viz`` extra (``pip install 'lerobot[viz]'``).
"""

import logging
import numbers
import time

import cv2
import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .constants import (
    ACTION,
    ACTION_PREFIX,
    DONE,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    OBS_STR,
    REWARD,
    SUCCESS,
    TRUNCATED,
)
from .import_utils import require_package

# Static schema shared by all scalar topics. Each message carries a flat list of ``{label, value}``
# pairs rather than one field per feature, so the same schema fits any robot regardless of which
# observation/action features it reports. The ``label`` field name is what Foxglove looks for to name
# each series automatically, so a single filtered path plots every feature, e.g.
# ``/observation/state.scalars[:]``.
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


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def init_foxglove(host: str = "127.0.0.1", port: int | None = 8765) -> None:
    """
    Starts a Foxglove WebSocket server for visualizing the control loop.

    Connect to it from the Foxglove app at ``ws://<host>:<port>``. Calling this
    more than once is a no-op while a server is already running.

    Args:
        host: Host interface to bind the WebSocket server to.
        port: Port to bind the WebSocket server to (defaults to 8765).
    """

    require_package("foxglove-sdk", extra="viz", import_name="foxglove")
    import foxglove

    # Live-stream state lives as attributes on ``log_foxglove_data``:
    # ``.server`` is the shared WebSocket server and
    # ``.channels`` caches one Foxglove channel per topic
    if getattr(log_foxglove_data, "server", None) is not None:
        return
    log_foxglove_data.server = foxglove.start_server(host=host, port=port or 8765)
    log_foxglove_data.channels = {}


def shutdown_foxglove() -> None:
    """Stops the Foxglove WebSocket server and clears cached channels."""

    server = getattr(log_foxglove_data, "server", None)
    if server is not None:
        server.stop()
    log_foxglove_data.server = None
    log_foxglove_data.channels = {}


def _foxglove_safe_name(name: str) -> str:
    """Replace ``.`` with ``_`` so a feature name is a single Foxglove topic-path segment.

    Foxglove treats ``.`` as a path separator, so an unsanitized name like ``observation.images.front``
    would split into nested segments instead of naming one topic.
    """

    return name.replace(".", "_")


def _foxglove_topic(key: str, *, is_image: bool = False) -> str:
    """Build the Foxglove topic for a feature ``key``.

    Camera features map to a per-source image topic (``/observation/images/<name>``); scalar features
    share one aggregate topic per source: ``/observation/state`` for observations, ``/action/state``
    for actions.
    """

    if is_image:
        name = str(key)
        for prefix in (f"{OBS_IMAGES}.", OBS_PREFIX):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return f"/{OBS_STR}/images/{_foxglove_safe_name(name)}"
    source = ACTION if (str(key).startswith(ACTION_PREFIX) or str(key) == ACTION) else OBS_STR
    return f"/{source}/state"


def _log_foxglove_scalars(
    topic: str, values: dict[str, float], *, channels: dict | None = None, log_time: int | None = None
) -> None:
    """Log scalars on a typed JSON channel using the static :data:`_SCALARS_SCHEMA`.

    ``values`` is an ordered mapping of feature name to value; it is emitted as a ``scalars`` array of
    ``{label, value}`` objects. Insertion order is preserved so series stay stable across messages.

    ``channels`` is the per-topic channel cache to reuse (defaults to the live-stream cache on
    :func:`log_foxglove_data`; dataset playback passes its own local cache to stay self-contained).
    ``log_time`` is the message time in nanoseconds; when ``None`` the server's receive time is used.
    """

    if not values:
        return

    import foxglove

    if channels is None:
        channels = log_foxglove_data.channels
    channel = channels.get(topic)
    if channel is None:
        channel = channels[topic] = foxglove.Channel(topic, schema=_SCALARS_SCHEMA, message_encoding="json")
    msg = {"scalars": [{"label": label, "value": value} for label, value in values.items()]}
    if log_time is None:
        channel.log(msg)
    else:
        channel.log(msg, log_time=log_time)


def _labeled_scalars(name: str, values, labels: list[str] | None = None) -> dict[str, float]:
    """Expand a 1D sequence into ``{label: value}`` entries with a consistent fallback."""

    flat = [float(v) for v in values]
    if labels is None or len(labels) != len(flat):
        labels = [f"{name}_{i}" for i in range(len(flat))]
    return dict(zip(labels, flat, strict=True))


def _log_foxglove_image(
    topic: str,
    frame_id: str,
    arr: np.ndarray,
    *,
    compress_images: bool,
    channels: dict | None = None,
    log_time: int | None = None,
    depth_range: tuple[float, float] | None = None,
    raw_depth_values: bool = False,
) -> None:
    """Log an image on a cached per-topic channel.

    The encoding is chosen from the channel count and dtype: a single-channel ``float`` or ``uint16``
    frame is a depth map (``32FC1``/``16UC1``), single-channel ``uint8`` is ``mono8``, 3 => ``rgb8``
    (float input assumed in [0, 1], cast to uint8), 4 => ``rgba8``; other counts are skipped with a
    warning. When ``compress_images`` is set, ``rgb8`` is JPEG-encoded instead.

    Args:
        topic: Foxglove topic to log on.
        frame_id: Frame id stamped on the message.
        arr: Image as HWC or CHW (CHW is transposed to HWC), any dtype.
        compress_images: JPEG-encode ``rgb8`` frames; ignored for other encodings.
        channels: Per-topic channel cache to reuse (see :func:`_log_foxglove_scalars`).
        log_time: Message time in nanoseconds, also written to the header timestamp; when ``None``
            the server's receive time is used.
        depth_range: ``(lo, hi)`` clip bounds in a depth frame's own input units. Depth frames
            (``32FC1``/``16UC1``) are rescaled onto Foxglove's default display max for their encoding
            (``1.0`` / ``10000``) so they show with sensible contrast; ``depth_range`` sets the source
            range, else the frame's own min/max is used. Ignored for ``mono8``/``rgb8``/``rgba8``.
        raw_depth_values: If True, depth values are not rescaled and are logged as is.
    """

    from foxglove.channels import CompressedImageChannel, RawImageChannel
    from foxglove.messages import CompressedImage, RawImage, Timestamp

    if channels is None:
        channels = log_foxglove_data.channels
    time_ns = time.time_ns() if log_time is None else log_time
    timestamp = Timestamp(sec=time_ns // 1_000_000_000, nsec=time_ns % 1_000_000_000)
    log_kwargs = {} if log_time is None else {"log_time": log_time}

    # Convert CHW -> HWC when needed (mirrors log_rerun_data).
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    height, width = arr.shape[0], arr.shape[1]
    n_channels = 1 if arr.ndim == 2 else arr.shape[2]

    if n_channels == 1 and arr.dtype != np.uint8:
        # Depth map: infer the encoding from the dtype.
        encoding, target_dtype, value_max = (
            ("32FC1", np.float32, 1.0)
            if np.issubdtype(arr.dtype, np.floating)
            else ("16UC1", np.uint16, 10000.0)
        )
        if not raw_depth_values:
            # Rescale onto the encoding's display max with respect to the given depth_range.
            lo, hi = depth_range if depth_range is not None else (float(arr.min()), float(arr.max()))
            arr = arr.clip(lo, hi).astype(np.float32)
            arr = (arr - lo) / ((hi - lo) if hi > lo else 1.0) * value_max
        arr = np.ascontiguousarray(arr, dtype=target_dtype)
    else:
        if n_channels == 3 and np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255)
        arr = np.ascontiguousarray(arr, dtype=np.uint8)

        if compress_images and n_channels == 3:
            buf_src = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", buf_src)
            channel = channels.get(topic)
            if channel is None:
                channel = channels[topic] = CompressedImageChannel(topic=topic)
            channel.log(
                CompressedImage(timestamp=timestamp, frame_id=frame_id, data=buf.tobytes(), format="jpeg"),
                **log_kwargs,
            )
            return

        encoding = {1: "mono8", 3: "rgb8", 4: "rgba8"}.get(n_channels)
        if encoding is None:
            logging.warning(
                "Foxglove: skipping image on topic '%s' with unsupported shape %s (%d channels); "
                "expected 1 (mono8/16UC1/32FC1), 3 (rgb8), or 4 (rgba8) channels.",
                topic,
                tuple(arr.shape),
                n_channels,
            )
            return

    channel = channels.get(topic)
    if channel is None:
        channel = channels[topic] = RawImageChannel(topic=topic)
    channel.log(
        RawImage(
            timestamp=timestamp,
            frame_id=frame_id,
            width=width,
            height=height,
            encoding=encoding,
            step=width * n_channels * arr.itemsize,
            data=arr.tobytes(),
        ),
        **log_kwargs,
    )


def log_foxglove_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to a Foxglove WebSocket server for real-time visualization.

    Mirrors ``log_rerun_data`` but emits Foxglove messages over the server started by
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

    require_package("foxglove-sdk", extra="viz", import_name="foxglove")

    if getattr(log_foxglove_data, "server", None) is None:
        raise RuntimeError("init_foxglove() must be called before log_foxglove_data().")

    now = time.time_ns()

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
                    obs_scalars.update(_labeled_scalars(key, v))
                else:
                    _log_foxglove_image(
                        _foxglove_topic(k, is_image=True),
                        key,
                        v,
                        compress_images=compress_images,
                        log_time=now,
                    )
        _log_foxglove_scalars(_foxglove_topic(OBS_STATE), obs_scalars, log_time=now)

    if action:
        action_scalars: dict[str, float] = {}
        for k, v in action.items():
            if v is None:
                continue
            key = k[len(ACTION_PREFIX) :] if str(k).startswith(ACTION_PREFIX) else str(k)
            if _is_scalar(v):
                action_scalars[key] = float(v)
            elif isinstance(v, np.ndarray):
                action_scalars.update(_labeled_scalars(key, v.flatten()))
        _log_foxglove_scalars(_foxglove_topic(ACTION), action_scalars, log_time=now)


# ── Dataset playback over a Foxglove WebSocket server ─────────────────────
# A LeRobotDataset is random-access on disk, so rather than fire-and-forget a forward stream we
# advertise a seekable timeline and serve frames on demand for whatever time the user scrubs/plays
# to in the Foxglove app. This relies on the SDK's PlaybackControl capability.


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
        elif values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            labels = [name for name, _ in sorted(names.items(), key=lambda kv: kv[1])]
    elif isinstance(names, (list, tuple)):
        labels = [str(n) for n in names]
    if labels is not None and dim is not None and len(labels) == dim:
        return labels
    return None


def _frame_to_scalars(sample: dict, key: str, labels: list[str] | None = None) -> dict[str, float]:
    """Flatten a frame's vector/scalar feature ``key`` into ``{label: value}`` entries.

    ``labels`` provides one name per dimension (from the dataset's feature metadata); when absent or
    the wrong length, dimensions fall back to ``{name}_{i}`` (the short feature name), matching the
    live stream so series names agree. A scalar feature becomes a single entry. Missing or ``None``
    features yield an empty mapping.
    """

    v = sample.get(key)
    if v is None:
        return {}
    arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    if key.startswith(OBS_PREFIX):
        name = key[len(OBS_PREFIX) :]
    elif key.startswith(ACTION_PREFIX):
        name = key[len(ACTION_PREFIX) :]
    else:
        name = key
    if arr.ndim == 0:
        return {name: float(arr)}
    return _labeled_scalars(name, arr.flatten(), labels)


def serve_foxglove_dataset_playback(
    dataset,
    episode_index: int,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    compress_images: bool = False,
    autoplay: bool = True,
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
        autoplay: If True, start playing automatically as soon as a client connects, instead of
            waiting for the user to press play in the Foxglove app.
    """

    require_package("foxglove-sdk", extra="viz", import_name="foxglove")
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
    # Dataset-wide q01/q99 depth bounds (fallback min/max) used to normalize depth to [0, 1].
    depth_ranges: dict[str, tuple[float, float]] = {}
    for key in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(key)
        if not stats:
            continue
        lo = stats["q01"] if "q01" in stats else stats["min"]
        hi = stats["q99"] if "q99" in stats else stats["max"]
        depth_ranges[key] = (float(np.asarray(lo).item()), float(np.asarray(hi).item()))
    # Per-dimension series labels from the dataset metadata (e.g. joint names), computed once.
    scalar_labels = {
        OBS_STATE: _feature_dim_names(dataset.meta.features.get(OBS_STATE)),
        ACTION: _feature_dim_names(dataset.meta.features.get(ACTION)),
    }
    # Local channel cache so the playback server is self-contained and doesn't touch the live-stream cache.
    channels: dict = {}

    def emit_frame(i: int) -> None:
        """Log every channel for frame ``i`` stamped at its dataset timestamp."""
        sample = dataset[i]
        log_time = times_ns[i]
        for key in camera_keys:
            arr = sample.get(key)
            if arr is None:
                continue
            arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
            _log_foxglove_image(
                _foxglove_topic(key, is_image=True),
                key,
                arr,
                compress_images=compress_images,
                channels=channels,
                log_time=log_time,
                depth_range=depth_ranges.get(key),
                raw_depth_values=True,
            )
        _log_foxglove_scalars(
            _foxglove_topic(OBS_STATE),
            _frame_to_scalars(sample, OBS_STATE, scalar_labels[OBS_STATE]),
            channels=channels,
            log_time=log_time,
        )
        _log_foxglove_scalars(
            _foxglove_topic(ACTION),
            _frame_to_scalars(sample, ACTION, scalar_labels[ACTION]),
            channels=channels,
            log_time=log_time,
        )
        episode_scalars = {}
        for feat, label in (
            (DONE, "done"),
            (TRUNCATED, "truncated"),
            (REWARD, "reward"),
            (SUCCESS, "success"),
        ):
            v = sample.get(feat)
            if v is not None:
                episode_scalars[label] = float(v)
        _log_foxglove_scalars("/episode/state", episode_scalars, channels=channels, log_time=log_time)

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

    # One-shot latch so autoplay fires only on the first client subscription.
    autoplay_started = threading.Event()

    class _PlaybackListener(ServerListener):
        def on_subscribe(self, client, channel):
            # Start playing automatically once a client actually connects (subscribes). Using the
            # subscribe hook, rather than starting in Playing up front, means the timeline doesn't
            # advance before anyone is watching. Fires once; the user can still pause/seek after.
            if not autoplay:
                return
            with lock:
                if autoplay_started.is_set() or state["status"] != PlaybackStatus.Paused:
                    return
                autoplay_started.set()
                state["status"] = PlaybackStatus.Playing
                cursor, speed = state["cursor"], state["speed"]
            server.broadcast_playback_state(PlaybackState(PlaybackStatus.Playing, cursor, speed, False, ""))

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
        # Cap how far the cursor may advance in a single tick. A slow frame decode (or any stall)
        # would otherwise make ``dt`` huge and produce one enormous catch-up batch; clamping it makes
        # playback trail wall-clock under a slow decoder while each tick emits a bounded frame range.
        max_tick_dt_s = 0.25
        prev = time.monotonic()
        while not stop_event.is_set():
            time.sleep(1.0 / 60.0)
            ended = False
            speed = 1.0
            with lock:
                now = time.monotonic()
                dt = min(now - prev, max_tick_dt_s)
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
                    # cursor only grows while playing (seeks reset last_idx in the listener), so
                    # target >= last_idx here; a plain assignment is correct and clearer than max().
                    state["last_idx"] = target
                    if ended:
                        state["status"] = PlaybackStatus.Ended
                if not work:
                    continue
                cursor, speed = state["cursor"], state["speed"]
            # Emit outside the lock; this is the only thread that calls emit_frame. Re-check
            # stop_event between frames so shutdown stays responsive even mid-batch.
            for i in work:
                if stop_event.is_set():
                    break
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
        channels.clear()
