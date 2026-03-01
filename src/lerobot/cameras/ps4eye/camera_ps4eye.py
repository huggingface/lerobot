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

"""
Provides the PS4EyeCamera class for capturing frames from Sony PS4 Eye cameras.

The PS4 Eye is a stereo USB camera that enumerates as a standard UVC/V4L2 device.
This driver wraps OpenCV's VideoCapture and adds stereo frame slicing (left / right / both eyes).

A single physical PS4 Eye can be used as **two independent cameras** in the lerobot
pipeline by creating two PS4EyeCamera instances with the same ``index_or_path`` but
different ``eye`` values. A class-level shared VideoCapture ensures the device is
opened only once:

    cameras = {
        "left":  PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="left",  width=3448, height=808, fps=30)),
        "right": PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="right", width=3448, height=808, fps=30)),
    }
    for cam in cameras.values():
        cam.connect()

Supported raw frame modes (width × height as reported by the device):
  3448 × 808  @ 60 fps  — full dual-eye panoramic (default)
  1748 × 408  @ 120 fps — half-resolution dual-eye

Eye slices (for 3448 × 808):
  left  eye: frame[0:800, 64:1328]   → 1264 × 800
  right eye: frame[0:800, 1328:2592] → 1264 × 800
"""

import logging
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_ps4eye import ColorMode, EyeSelection, PS4EyeCameraConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stereo crop geometry for 3448 × 808 (full-res) and 1748 × 408 (half-res)
# Tuple layout: (row_start, row_end, col_l0, col_l1, col_r0, col_r1)
# ---------------------------------------------------------------------------
_STEREO_CROPS: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {
    (3448, 808): (0, 800,  64, 1328, 1328, 2592),
    (1748, 408): (0, 400,  32,  668,  668, 1304),
}


# ---------------------------------------------------------------------------
# Shared VideoCapture registry
# ---------------------------------------------------------------------------

@dataclass
class _SharedCapture:
    """Shared state for a single physical PS4 Eye device."""

    videocapture: cv2.VideoCapture
    ref_count: int = 0

    # Raw panoramic frame (BGR, as delivered by OpenCV)
    raw_frame: NDArray[Any] | None = None
    raw_frame_lock: Lock = field(default_factory=Lock)
    new_raw_frame_event: Event = field(default_factory=Event)

    # Background reader thread
    reader_thread: Thread | None = None
    stop_event: Event | None = None

    # Actual capture dimensions (set after VideoCapture.open)
    capture_width: int = 0
    capture_height: int = 0


# Maps normalised index_or_path → shared capture state.
# Protected by _REGISTRY_LOCK for thread-safe connect/disconnect.
_SHARED_CAPTURES: dict[str, _SharedCapture] = {}
_REGISTRY_LOCK: Lock = Lock()


def _registry_key(index_or_path: int | str | Path) -> str:
    """Normalise a device index or path to a stable dict key."""
    return str(index_or_path)


class PS4EyeCamera(Camera):
    """
    Manages camera interactions with a Sony PS4 Eye stereo camera using OpenCV.

    The PS4 Eye enumerates as a standard UVC device and is accessed via OpenCV's
    VideoCapture. A single physical device can be used as two independent cameras
    by creating two instances with the same ``index_or_path`` but different ``eye``
    values (``"left"`` and ``"right"``). The underlying VideoCapture is shared
    transparently — the device is opened only once regardless of how many instances
    connect to it.

    Use the ``eye`` configuration field to select which eye is returned by
    :meth:`read` and :meth:`async_read`:

    - ``"left"``  → left image slice
    - ``"right"`` → right image slice
    - ``"both"``  → full unsplit panoramic frame

    Use the provided utility to discover available camera indices:
    ```bash
    lerobot-find-cameras opencv
    ```

    Example (two-camera pipeline):
        ```python
        from lerobot.cameras.ps4eye import PS4EyeCamera, PS4EyeCameraConfig

        cameras = {
            "left":  PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="left",  width=3448, height=808, fps=30)),
            "right": PS4EyeCamera(PS4EyeCameraConfig(index_or_path=1, eye="right", width=3448, height=808, fps=30)),
        }
        for cam in cameras.values():
            cam.connect()

        left_img  = cameras["left"].read()   # shape: (800, 1264, 3)
        right_img = cameras["right"].read()  # shape: (800, 1264, 3)

        for cam in cameras.values():
            cam.disconnect()
        ```
    """

    def __init__(self, config: PS4EyeCameraConfig) -> None:
        """
        Initializes the PS4EyeCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path
        self._key = _registry_key(config.index_or_path)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.eye = config.eye

        self._connected: bool = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # capture_* track the raw panoramic dimensions on the device.
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path}, eye={self.eye})"

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True if this instance has successfully called connect()."""
        return self._connected

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect(self, warmup: bool = True) -> None:
        """
        Connects this eye-instance to the PS4 Eye camera.

        If no other instance has opened this device yet, a new VideoCapture is
        created and a background reader thread is started. Subsequent instances
        sharing the same ``index_or_path`` reuse the existing VideoCapture.

        Raises:
            DeviceAlreadyConnectedError: If this instance is already connected.
            ConnectionError: If the camera cannot be opened.
            RuntimeError: If the camera opens but fails to apply the requested settings.
        """
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        with _REGISTRY_LOCK:
            shared = _SHARED_CAPTURES.get(self._key)

            if shared is None:
                # First instance to connect — open the device.
                cv2.setNumThreads(1)
                cap = cv2.VideoCapture(self.index_or_path)

                if not cap.isOpened():
                    cap.release()
                    raise ConnectionError(
                        f"Failed to open {self}. "
                        f"Run `lerobot-find-cameras opencv` to find available cameras."
                    )

                shared = _SharedCapture(videocapture=cap)
                _SHARED_CAPTURES[self._key] = shared

                self._apply_capture_settings(shared)
                self._start_shared_reader(shared, self._key)
            else:
                # Reuse existing capture; update our own width/height from it.
                self._sync_settings_from_shared(shared)

            shared.ref_count += 1

        self._connected = True
        self.capture_width = shared.capture_width
        self.capture_height = shared.capture_height

        if self.fps is None:
            self.fps = shared.videocapture.get(cv2.CAP_PROP_FPS)

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read()
                except Exception:
                    pass
                time.sleep(0.1)

        # Start the per-instance read thread now so read_latest() is immediately available.
        # This mirrors the OpenCV camera pattern where the thread is running by the time
        # the robot calls get_observation() / read_latest() on the first loop iteration.
        self._start_read_thread()

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """
        Disconnects this eye-instance.

        Decrements the shared reference count. When the last instance disconnects,
        the background reader thread is stopped and VideoCapture is released.

        Raises:
            DeviceNotConnectedError: If this instance is not connected.
        """
        if not self._connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        with _REGISTRY_LOCK:
            shared = _SHARED_CAPTURES.get(self._key)
            if shared is not None:
                shared.ref_count -= 1
                if shared.ref_count <= 0:
                    self._stop_shared_reader(shared)
                    shared.videocapture.release()
                    del _SHARED_CAPTURES[self._key]

        self._connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Shared VideoCapture helpers
    # ------------------------------------------------------------------

    def _apply_capture_settings(self, shared: _SharedCapture) -> None:
        """Configure FPS / width / height on a freshly opened VideoCapture."""
        cap = shared.videocapture

        default_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            target_w, target_h = default_width, default_height
        else:
            target_w = self.capture_width if self.capture_width else self.width
            target_h = self.capture_height if self.capture_height else self.height

        if (target_w, target_h) != (default_width, default_height):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(target_w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(target_h))

            actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            if actual_w != target_w or actual_h != target_h:
                raise RuntimeError(
                    f"{self} failed to set resolution {target_w}×{target_h}; "
                    f"device reports {actual_w}×{actual_h}."
                )
            shared.capture_width = actual_w
            shared.capture_height = actual_h
        else:
            shared.capture_width = default_width
            shared.capture_height = default_height

        if self.fps:
            success = cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
                raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _sync_settings_from_shared(self, shared: _SharedCapture) -> None:
        """Align our own width/height with what the shared capture is already set to."""
        if self.width is not None and self.height is not None:
            rw = self.capture_width if self.capture_width else self.width
            rh = self.capture_height if self.capture_height else self.height
            if rw != shared.capture_width or rh != shared.capture_height:
                raise RuntimeError(
                    f"{self}: Requested resolution {rw}×{rh} conflicts with the "
                    f"already-opened shared capture resolution "
                    f"{shared.capture_width}×{shared.capture_height}. "
                    f"All instances sharing the same device must use the same resolution."
                )

    @staticmethod
    def _shared_reader_loop(key: str) -> None:
        """Background thread that continuously reads raw panoramic frames."""
        shared = _SHARED_CAPTURES.get(key)
        if shared is None or shared.stop_event is None:
            return

        while not shared.stop_event.is_set():
            ret, frame = shared.videocapture.read()
            if not ret or frame is None:
                logger.warning(f"PS4EyeCamera[{key}]: raw frame read failed.")
                continue

            with shared.raw_frame_lock:
                shared.raw_frame = frame
            shared.new_raw_frame_event.set()

    @staticmethod
    def _start_shared_reader(shared: _SharedCapture, key: str) -> None:
        """Start the shared background reader thread."""
        shared.stop_event = Event()
        t = Thread(
            target=PS4EyeCamera._shared_reader_loop,
            args=(key,),
            daemon=True,
            name=f"ps4eye_shared_reader_{key}",
        )
        shared.reader_thread = t
        t.start()

    @staticmethod
    def _stop_shared_reader(shared: _SharedCapture) -> None:
        """Signal and join the shared background reader thread."""
        if shared.stop_event is not None:
            shared.stop_event.set()
        if shared.reader_thread is not None and shared.reader_thread.is_alive():
            shared.reader_thread.join(timeout=2.0)
        shared.reader_thread = None
        shared.stop_event = None

    # ------------------------------------------------------------------
    # Camera discovery
    # ------------------------------------------------------------------

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available PS4 Eye cameras connected to the system.

        Scans all OpenCV-visible capture devices and filters by the known
        panoramic resolutions (3448×808 or 1748×408).

        On Linux, ``/dev/video*`` paths are scanned directly.
        On macOS and Windows, indices 0–59 are probed.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys
            ``"name"``, ``"type"``, ``"id"``, ``"backend_api"``, and
            ``"default_stream_profile"``.
        """
        known_widths = {w for w, _ in _STEREO_CROPS}
        found: list[dict[str, Any]] = []

        targets_to_scan: list[str | int]
        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = list(range(60))

        for target in targets_to_scan:
            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                cap.release()
                continue

            default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            default_fps = cap.get(cv2.CAP_PROP_FPS)

            if default_width not in known_widths:
                cap.release()
                continue

            camera_info: dict[str, Any] = {
                "name": f"PS4 Eye @ {target}",
                "type": "PS4Eye",
                "id": target,
                "backend_api": cap.getBackendName(),
                "default_stream_profile": {
                    "width": default_width,
                    "height": default_height,
                    "fps": default_fps,
                },
            }
            found.append(camera_info)
            cap.release()

        return found

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads the latest panoramic frame and returns the configured eye slice.

        Fetches the most recent raw frame from the shared reader thread
        (non-blocking if a frame is already available), then applies color
        conversion, stereo crop, and rotation.

        Args:
            color_mode: Overrides the default color mode for this call.

        Returns:
            np.ndarray:
                - ``eye="left"`` or ``"right"``: ``(eye_h, eye_w, 3)``
                - ``eye="both"``: ``(capture_height, capture_width, 3)``

        Raises:
            DeviceNotConnectedError: If not connected.
            RuntimeError: If no frame is available within 500 ms or dimensions mismatch.
            ValueError: If ``color_mode`` is invalid.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        shared = _SHARED_CAPTURES.get(self._key)
        if shared is None:
            raise DeviceNotConnectedError(f"{self}: shared capture not found.")

        start_time = time.perf_counter()

        # Ensure the shared reader is running
        if shared.reader_thread is None or not shared.reader_thread.is_alive():
            self._restart_shared_reader(shared)

        # Wait for the next raw frame (500 ms hard timeout)
        if not shared.new_raw_frame_event.wait(timeout=0.5):
            raise RuntimeError(f"{self} timed out waiting for a raw frame from the shared reader.")

        with shared.raw_frame_lock:
            raw = shared.raw_frame
            shared.new_raw_frame_event.clear()

        if raw is None:
            raise RuntimeError(f"{self}: shared raw frame is None after event was set.")

        processed = self._postprocess_image(raw, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed

    def _restart_shared_reader(self, shared: _SharedCapture) -> None:
        """Restart the shared reader thread (e.g. if it died unexpectedly)."""
        key = self._key
        shared.stop_event = Event()
        t = Thread(
            target=lambda: PS4EyeCamera._shared_reader_loop(key),
            daemon=True,
            name=f"ps4eye_shared_reader_{key}",
        )
        shared.reader_thread = t
        t.start()

    def _postprocess_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, stereo crop, and rotation.

        Args:
            image: Raw BGR panoramic frame from OpenCV.
            color_mode: Target color mode (defaults to ``self.color_mode``).

        Returns:
            np.ndarray: Processed frame.

        Raises:
            ValueError: If ``color_mode`` is invalid.
            RuntimeError: If raw frame dimensions do not match expected capture dimensions.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. "
                f"Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(
                f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR)."
            )

        # Color conversion (OpenCV delivers BGR)
        processed: NDArray[Any] = image
        if requested_color_mode == ColorMode.RGB:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Stereo crop
        processed = self._crop_stereo(processed)

        # Rotation
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        return processed

    def _crop_stereo(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Slices the panoramic frame into the configured eye view.

        Args:
            image: Full panoramic frame (after color conversion).

        Returns:
            np.ndarray: The selected eye slice, or the full frame when ``eye="both"``.
        """
        if self.eye == EyeSelection.BOTH:
            return image

        h, w = image.shape[:2]
        crop_params = _STEREO_CROPS.get((w, h))

        if crop_params is None:
            logger.warning(
                f"{self}: Unknown panoramic resolution {w}\u00d7{h}. "
                f"Returning full frame without stereo crop. "
                f"Known resolutions: {list(_STEREO_CROPS.keys())}"
            )
            return image

        row_start, row_end, col_l0, col_l1, col_r0, col_r1 = crop_params

        if self.eye == EyeSelection.LEFT:
            return image[row_start:row_end, col_l0:col_l1]
        else:  # RIGHT
            return image[row_start:row_end, col_r0:col_r1]

    # ------------------------------------------------------------------
    # Per-instance async read (for lerobot async pipeline)
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """
        Per-instance background loop for async_read().

        Calls read() (which fetches from the shared frame buffer) and stores
        the processed result in this instance's latest_frame.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialized.")

        while not self.stop_event.is_set():
            try:
                frame = self.read()
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error in async read loop for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the per-instance async read thread."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the per-instance async read thread to stop."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame for this eye immediately (non-blocking).

        Returns whatever is currently in the per-instance buffer. The frame
        may be slightly stale but is guaranteed to have been processed through
        this eye's stereo crop and color conversion.

        Args:
            max_age_ms: If the buffered frame is older than this many milliseconds,
                a TimeoutError is raised. Defaults to 500ms.

        Returns:
            NDArray[Any]: The latest processed eye frame.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the read thread is not running or no frames captured yet.
            TimeoutError: If the latest frame is older than ``max_age_ms``.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Returns the latest available processed frame for this eye asynchronously.

        Starts the per-instance read thread on first call. The thread fetches
        frames from the shared device reader and applies this instance's eye
        crop, so left and right instances operate independently.

        Args:
            timeout_ms: Maximum wait time in milliseconds. Defaults to 200ms.

        Returns:
            np.ndarray: Latest eye-cropped frame.

        Raises:
            DeviceNotConnectedError: If not connected.
            TimeoutError: If no frame is available within ``timeout_ms``.
            RuntimeError: On unexpected errors.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame
