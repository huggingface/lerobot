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

import numbers
import os
import queue
import threading
import time

import numpy as np

from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


_last_image_log = 0.0


def _images_due() -> bool:
    """Whether to send camera frames on this pass. Scalars are never skipped.

    The display call sits INSIDE the control loop, so the link's throughput is
    the loop's problem: when rerun cannot get frames out fast enough its sink
    back-pressures and the send blocks where the robot is being driven. Over
    wifi that shows up as laggy video and jittery motion at the same time -
    one fault, two symptoms, and the lag grows without bound because TCP
    buffers rather than drops.

    Nobody needs 30 Hz to see what the robot is looking at. Set

        LEROBOT_RERUN_IMAGE_FPS=10

    to send frames at 10 Hz while the loop keeps running at 30, cutting the
    video off the critical path without touching control or the dataset - the
    dataset is written from the originals on the robot and never passes
    through here. Unset means every frame, as before. 0 means none.

    Timed rather than every-Nth so it does not need to know the loop rate.
    """
    raw = os.getenv("LEROBOT_RERUN_IMAGE_FPS")
    if not raw:
        return True
    try:
        fps = float(raw)
    except ValueError:
        return True
    if fps <= 0:
        return False

    global _last_image_log
    now = time.perf_counter()
    if now - _last_image_log < 1.0 / fps:
        return False
    _last_image_log = now
    return True


# --------------------------------------------------------------- async sink
#
# The display call sits inside the control loop. rerun's sink back-pressures:
# when it cannot get data out, the send BLOCKS where the robot is being
# driven. Measured on this robot over wifi, three 640x480 cameras at 30 Hz:
#
#   telemetry  mean 55.24 ms · worst 692.16 ms · 89.0% of work
#   observe     mean 5.41 ms
#   teleop      mean 0.05 ms
#   send        mean 1.30 ms
#
# against a 33.3 ms budget. The robot's own work is under 7 ms; the loop was
# running at 15.6 Hz because it spent the rest waiting on a screen.
#
# So hand the frames to a worker and return. When the link cannot keep up the
# queue overflows and the OLDEST pending frame is dropped, which is the right
# trade for a live view: a dropped frame is invisible, a stalled arm is not.
# The dataset never comes through here - it is written from the originals on
# the robot - so nothing dropped here is lost from a recording.
#
# Measured on a loopback server, 3x640x480 compressed, cost to the CALLER:
#
#   sync    mean 11.177 ms · worst 34.236 ms
#   async   mean  0.005 ms · worst  0.078 ms
#
# The frames are queued by reference, not copied. If a camera backend ever
# reuses its buffers, a queued frame could be overwritten before the worker
# encodes it; the cost of that is a torn picture on screen, and the cost of
# defending against it is a 2.7 MB memcpy per frame in the control loop.
#
#   LEROBOT_RERUN_ASYNC=0    log inline instead (the old behaviour)

_queue: queue.Queue | None = None
_worker: threading.Thread | None = None
_dropped = 0
_sent = 0


def _async_enabled() -> bool:
    return os.getenv("LEROBOT_RERUN_ASYNC", "1") not in ("0", "false", "False", "")


def _worker_loop() -> None:
    while True:
        item = _queue.get()  # type: ignore[union-attr]
        if item is None:
            return
        observation, action, compress_images = item
        try:
            _log_now(observation, action, compress_images)
        except Exception:
            # A display failure must never take the robot down with it.
            pass


def _start_worker() -> None:
    global _queue, _worker
    if _worker is not None and _worker.is_alive():
        return
    _queue = queue.Queue(maxsize=2)
    _worker = threading.Thread(target=_worker_loop, name="rerun-log", daemon=True)
    _worker.start()


def _enqueue(observation, action, compress_images: bool) -> None:
    global _dropped, _sent
    assert _queue is not None
    try:
        _queue.put_nowait((observation, action, compress_images))
    except queue.Full:
        # Drop the oldest, keep the newest: a live view wants current, not complete.
        try:
            _queue.get_nowait()
            _dropped += 1
        except queue.Empty:
            pass
        try:
            _queue.put_nowait((observation, action, compress_images))
        except queue.Full:
            _dropped += 1
            return
    _sent += 1


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
    global _last_image_log, _dropped, _sent
    _last_image_log = 0.0
    _dropped = 0
    _sent = 0

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)

    if _async_enabled():
        _start_worker()


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    global _worker
    if _worker is not None and _worker.is_alive():
        assert _queue is not None
        try:
            _queue.put(None, timeout=2.0)
            _worker.join(timeout=5.0)
        except Exception:
            pass
    _worker = None

    if _dropped:
        pct = 100.0 * _dropped / max(_sent + _dropped, 1)
        print(
            f"rerun: dropped {_dropped} of {_sent + _dropped} display frames "
            f"({pct:.0f}%) to keep the control loop on time."
        )

    rr.rerun_shutdown()


def _ordered_image_paths(image_paths: set[str]) -> list[str]:
    """Camera order, left to right.

    Alphabetical by default, which on a two-wrist robot already puts the
    wrists next to each other. Override when that is not the order you want
    to look at:

        LEROBOT_RERUN_IMAGE_ORDER=left_wrist,right_wrist,head

    Matching is by substring, so the short camera name is enough rather than
    the full entity path. Anything unmatched keeps its alphabetical place at
    the end, so a typo loses you the ordering, not the camera.
    """
    paths = sorted(image_paths)
    raw = os.getenv("LEROBOT_RERUN_IMAGE_ORDER")
    if not raw:
        return paths

    wanted = [w.strip() for w in raw.split(",") if w.strip()]
    ordered: list[str] = []
    for w in wanted:
        for p in paths:
            if w in p and p not in ordered:
                ordered.append(p)
    ordered += [p for p in paths if p not in ordered]
    return ordered


def _build_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]):
    """Lay the cameras out in one row, with the time series underneath.

    A flat Grid of everything lets rerun mix cameras and plots into the same
    rows, so the two wrist views end up wherever the cell count puts them.
    Cameras are what you actually watch while driving, so they get their own
    row, in a known order, with two thirds of the height.
    """

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun.blueprint as rrb

    image_views = [
        rrb.Spatial2DView(origin=path, name=path.split(".")[-1])
        for path in _ordered_image_paths(image_paths)
    ]

    scalar_views = []
    if observation_paths:
        scalar_views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
    if action_paths:
        scalar_views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))

    if image_views and scalar_views:
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*image_views),
                rrb.Horizontal(*scalar_views),
                row_shares=[2, 1],
            )
        )
    if image_views:
        return rrb.Blueprint(rrb.Horizontal(*image_views))
    return rrb.Blueprint(rrb.Grid(*scalar_views))


def _ensure_blueprint(observation_paths: set[str], action_paths: set[str], image_paths: set[str]) -> None:
    """Build and send the blueprint once, from the first observation and action data."""
    if getattr(log_rerun_data, "blueprint", None) is not None:
        return

    if not (observation_paths or action_paths or image_paths):
        return

    # Safe + zero-overhead: `log_rerun_data` already ran the `require_package` guard and imported rerun.
    import rerun as rr

    blueprint = _build_blueprint(observation_paths, action_paths, image_paths)
    log_rerun_data.blueprint = blueprint  # type: ignore[attr-defined]
    rr.send_blueprint(blueprint)


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

    if _async_enabled():
        if _worker is None or not _worker.is_alive():
            _start_worker()
        _enqueue(observation, action, compress_images)
        return

    _log_now(observation, action, compress_images)


def _log_now(
    observation: RobotObservation | None,
    action: RobotAction | None,
    compress_images: bool,
) -> None:
    """The actual logging. Runs on the worker thread unless async is off."""
    import rerun as rr

    observation_paths: set[str] = set()
    action_paths: set[str] = set()
    image_paths: set[str] = set()

    # Decided once per pass, not per camera: otherwise only the first camera
    # of a set would ever clear the interval and the rest would starve.
    send_images = _images_due()

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
                        # At record time, the depth unit is inferred from the frame type.
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    # NOT static. A static entity has no position on any
                    # timeline: it is for things that do not change, like a
                    # calibration or a fixed mesh. Camera frames are the
                    # opposite, and logging them static wedges the rerun
                    # viewer outright - reproduced on 0.33.1 with a single
                    # 160x120 frame logged once a second, which rules out
                    # throughput, JPEG decode and frame size. The same run
                    # with this one argument removed is fine.
                    #
                    # It also costs memory that nothing can reclaim. The
                    # viewer evicts oldest-BY-TIME, so entities with no time
                    # are never eviction candidates however long a session
                    # runs. On the timeline they are.
                    #
                    # It arrived with this file in #3902 and was carried
                    # through #3899; no commit gives a reason for it.
                    if send_images:
                        rr.log(key, entity=img_entity)
                    # Added either way: the blueprint needs to know the view
                    # exists even on a pass whose frames were skipped.
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
