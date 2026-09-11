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
"""Shared machinery for the Stereolabs backends.

The ZED SDK exposes stereo cameras (``sl.Camera``) and single-sensor cameras (``sl.CameraOne``)
through two similar but distinct APIs. Everything that does not touch the SDK — the background
capture thread, the read families, colour/depth post-processing and teardown — lives here; the
concrete backends only implement the handful of calls that actually differ.
"""

import abc
import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _pyzed_available

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation

if TYPE_CHECKING:
    from .configuration_zed import ZedCameraConfig
    from .configuration_zed_one import ZedOneCameraConfig

_INSTALL_HINT = (
    "The ZED Python API (`pyzed`) is not installed. It ships with the ZED SDK (not PyPI). "
    "Install the ZED SDK 5.4, then run:  python /usr/local/zed/get_python_api.py"
)

_pyzed_import_error: BaseException | None = None
if TYPE_CHECKING or _pyzed_available:
    try:
        import pyzed.sl as sl
    except Exception as e:
        # Installed but not importable — usually an incomplete CUDA setup or a missing
        # LD_LIBRARY_PATH. Deferring the error to connect() keeps `lerobot-record` usable
        # for the other cameras instead of failing at import time.
        sl = None
        _pyzed_import_error = e
else:
    sl = None  # type: ignore[assignment]


__all__ = ["ZedCameraBase", "_require_sl", "logger", "sl"]


def _require_sl() -> None:
    """Raise the install hint if the ZED SDK is missing, or its import error if it is present but broken."""
    if sl is not None:
        return
    if _pyzed_import_error is not None:
        raise ImportError(
            f"The ZED Python API (`pyzed`) is installed but failed to import: {_pyzed_import_error}. "
            "Check that the ZED SDK and CUDA it was built against are installed and on the library path."
        ) from _pyzed_import_error
    raise ImportError(_INSTALL_HINT)


logger = logging.getLogger(__name__)


class ZedCameraBase(Camera, abc.ABC):
    """Common implementation shared by :class:`ZedCamera` and :class:`ZedOneCamera`."""

    def __init__(self, config: "ZedCameraConfig | ZedOneCameraConfig"):
        _require_sl()
        super().__init__(config)

        self.config = config
        # re-declared (as RealSenseCamera does): mypy uses follow_imports = "skip"
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height
        self.serial_number = config.serial_number
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.resolution_name = config.resolution

        # Subclasses narrow these: a monocular camera has no depth stream.
        self.use_rgb = getattr(config, "use_rgb", True)
        self.use_depth = getattr(config, "use_depth", False)
        self.depth_mode_name = getattr(config, "depth_mode", "NONE")

        self.cam: Any = None
        self.runtime: Any = None
        self._img_mat: Any = None
        self._depth_mat: Any = None
        self._color_view: Any = None  # resolved in connect() once `sl` is known to be importable

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None  # perf_counter, for read_latest aging
        self.latest_hw_timestamp_ns: int | None = None  # ZED IMAGE timestamp, for sync
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.capture_width: int | None = None
        self.capture_height: int | None = None
        self.intrinsics: dict[str, float] | None = None
        self._reset_connection_settings()

    # ---- hooks the concrete backends implement ----
    @abc.abstractmethod
    def _open_device(self) -> None:
        """Open the SDK camera, assigning it to ``self.cam`` (raise ConnectionError on failure)."""

    @abc.abstractmethod
    def _grab(self) -> Any:
        """Advance the camera one frame; return the SDK ``ERROR_CODE`` (negative = warning with a frame)."""

    @abc.abstractmethod
    def _retrieve_color(self) -> NDArray[Any]:
        """Retrieve the configured colour view as a raw BGRA array."""

    @abc.abstractmethod
    def _read_calibration(self, camera_configuration: Any) -> Any:
        """Return the per-camera calibration matching the stream being handed out."""

    def _retrieve_depth(self) -> NDArray[Any]:
        raise NotImplementedError(f"{self} has no depth stream.")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.serial_number if self.serial_number is not None else 'auto'})"
        )

    def _reset_connection_settings(self) -> None:
        """Restore settings that may have been auto-detected during a failed connection."""
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.capture_width = self.capture_height = None
        self.intrinsics = None

    @property
    def is_connected(self) -> bool:
        """True while the SDK camera handle is open."""
        return self.cam is not None and self.cam.is_opened()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Open the camera, validate the stream settings and start the background read thread.

        Args:
            warmup: Wait ``warmup_s`` seconds and require at least one frame before returning.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the SDK cannot open the camera, or no frame arrives during warmup.
            ValueError: If ``resolution``/``depth_mode`` is unknown, or the camera opened in a mode
                that does not match the requested ``fps``/``width``/``height``.
        """
        try:
            self._open_device()
            # LEFT is rectified (lens distortion removed); LEFT_UNRECTIFIED is the raw sensor image.
            self._color_view = sl.VIEW.LEFT if self.config.rectified else sl.VIEW.LEFT_UNRECTIFIED
            self._img_mat = sl.Mat()
            self._depth_mat = sl.Mat()
            self._configure_capture_settings()
            self._start_read_thread()
            if warmup and self.warmup_s > 0:
                self._run_warmup()
        except BaseException:
            try:
                self._cleanup_resources()
            except Exception:
                logger.exception(f"Failed to fully clean up {self} after connect() failed.")
            self._reset_connection_settings()
            raise

        logger.info(f"{self} connected.")

    def _run_warmup(self) -> None:
        """Block until ``warmup_s`` has elapsed and at least one frame has been captured.

        Raises:
            ConnectionError: If no frame arrives before ``warmup_s`` elapses.
        """
        warmup_read = self.async_read if self.use_rgb else self.async_read_depth
        start_time = time.time()
        while time.time() - start_time < self.warmup_s:
            warmup_read(timeout_ms=self.warmup_s * 1000)
            time.sleep(0.1)
        with self.frame_lock:
            if (self.use_rgb and self.latest_color_frame is None) or (
                self.use_depth and self.latest_depth_frame is None
            ):
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

    def _resolve_resolution(self) -> Any:
        try:
            return getattr(sl.RESOLUTION, self.resolution_name)
        except AttributeError as e:
            valid = [r.name for r in sl.RESOLUTION if r.name != "LAST"]
            raise ValueError(f"{self}: unknown resolution {self.resolution_name!r}; valid: {valid}") from e

    def _configure_capture_settings(self) -> None:
        info = self.cam.get_camera_information()
        conf = info.camera_configuration
        w, h = int(conf.resolution.width), int(conf.resolution.height)
        actual_fps = int(conf.fps)
        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            w, h = h, w  # the frames this camera will hand out are rotated

        # A requested resolution/fps that the camera did not honour must be an error, not a
        # silent override: robots publish observation shapes from the *config* before connecting,
        # so a mismatch would surface much later as a confusing frame-shape failure.
        requested = {"width": self.width, "height": self.height, "fps": self.fps}
        actual = {"width": w, "height": h, "fps": actual_fps}
        mismatched = {
            k: (v, actual[k]) for k, v in requested.items() if v is not None and int(v) != actual[k]
        }
        if mismatched:
            detail = ", ".join(
                f"{k}: requested {req}, camera opened {act}" for k, (req, act) in mismatched.items()
            )
            raise ValueError(
                f"{self}: camera did not honour the requested stream settings ({detail}). "
                f"Set resolution={self.resolution_name!r} together with matching width/height/fps, or "
                "leave width/height/fps unset to accept the camera's own mode."
            )

        self.width, self.height, self.fps = w, h, actual_fps
        self.capture_width, self.capture_height = (
            (h, w) if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) else (w, h)
        )

        cal = self._read_calibration(conf)
        fx, fy, cx, cy = float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy)
        # calibration describes the sensor frame; rotate it with the image (capture_width/height)
        cw, ch = self.capture_width, self.capture_height
        if self.rotation == cv2.ROTATE_90_CLOCKWISE:
            fx, fy, cx, cy = fy, fx, ch - 1 - cy, cx
        elif self.rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            fx, fy, cx, cy = fy, fx, cy, cw - 1 - cx
        elif self.rotation == cv2.ROTATE_180:
            cx, cy = cw - 1 - cx, ch - 1 - cy
        self.intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    # ---- frame post-processing ----
    def _check_frame_size(self, frame: NDArray[Any]) -> None:
        h, w = frame.shape[:2]
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )

    def _postprocess_color(self, bgra: NDArray[Any]) -> NDArray[Any]:
        self._check_frame_size(bgra)
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise RuntimeError(f"{self} frame shape={bgra.shape} is not the BGRA image the SDK returns.")
        # single pass off the reused Mat; slicing to BGR first costs 35 ms/frame vs 1.4
        code = cv2.COLOR_BGRA2RGB if self.color_mode == ColorMode.RGB else cv2.COLOR_BGRA2BGR
        img = cv2.cvtColor(bgra, code)
        if self.rotation is not None:
            img = cv2.rotate(img, self.rotation)
        return img

    def _postprocess_depth(self, depth_mm: NDArray[Any]) -> NDArray[Any]:
        self._check_frame_size(depth_mm)
        # DEPTH_U16_MM is already uint16 mm; rotate allocates, so only the other path copies
        # off the reused Mat. SDK returns (H, W), LeRobot expects (H, W, 1).
        d = cv2.rotate(depth_mm, self.rotation) if self.rotation is not None else depth_mm.copy()
        return d[..., np.newaxis] if d.ndim == 2 else d

    # ---- background read thread ----
    def _read_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        # CORRUPTED_FRAME = invalid colours (green/purple), CAMERA_REBOOTING = no new image at all
        no_frame_codes = (sl.ERROR_CODE.CORRUPTED_FRAME, sl.ERROR_CODE.CAMERA_REBOOTING)
        failure_count = 0
        last_warning = None
        while not stop_event.is_set():
            try:
                status = self._grab()
                if status > sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"{self} grab() failed: {status}")
                if status != sl.ERROR_CODE.SUCCESS:  # negative codes are warnings
                    if status != last_warning:  # once per streak, not once per frame
                        logger.warning(f"{self} grab() warning: {status}")
                    last_warning = status
                    if status in no_frame_codes:
                        continue
                else:
                    last_warning = None

                color = depth = None
                if self.use_rgb:
                    color = self._postprocess_color(self._retrieve_color())
                if self.use_depth:
                    depth = self._postprocess_depth(self._retrieve_depth())

                hw_ts = int(self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
                capture_time = time.perf_counter()

                with self.frame_lock:
                    # Under the lock, so a late frame cannot resurrect the buffer _stop_read_thread() cleared.
                    if stop_event.is_set():
                        break
                    if self.use_rgb:
                        self.latest_color_frame = color
                    if self.use_depth:
                        self.latest_depth_frame = depth
                    self.latest_timestamp = capture_time
                    self.latest_hw_timestamp_ns = hw_ts
                self.new_frame_event.set()
                failure_count = 0

            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.latest_hw_timestamp_ns = None  # else a reconnect can serve last session's clock
            self.new_frame_event.clear()

    def _cleanup_resources(self) -> None:
        """Stop background reads and close the SDK camera, including after partial setup."""
        read_thread = self.thread
        cam = self.cam

        try:
            self._stop_read_thread()
        finally:
            self.cam = None
            self.runtime = None
            self._img_mat = None
            self._depth_mat = None
            try:
                if cam is not None:
                    cam.close()
            finally:
                # Closing the camera may unblock a grab() that outlived the first bounded join.
                if read_thread is not None and read_thread.is_alive():
                    read_thread.join(timeout=2.0)
                    if read_thread.is_alive():  # pragma: no cover
                        logger.warning(f"{self} read thread remained alive after closing the camera.")

    # ---- reads ----
    def _async_read(self, timeout_ms: float, read_depth: bool = False) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms} ms.")
        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: event set but no frame available for {self}.")
        return frame

    def _read(self, read_depth: bool = False) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()  # a *new* grab, not the buffered one
        return self._async_read(timeout_ms=10000, read_depth=read_depth)

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """Block until the next colour frame and return it.

        Returns:
            np.ndarray: ``(height, width, 3)`` ``uint8`` frame in the configured ``color_mode``
            and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_rgb=False`` or the read thread is not running.
            TimeoutError: If no frame arrives within 10 s.
        """
        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — configured with use_rgb=False.")
        return self._read()

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 0) -> NDArray[np.uint16]:
        """Block until the next depth frame and return it.

        Returns:
            np.ndarray: ``(height, width, 1)`` ``uint16`` depth in millimetres, ``0`` where invalid.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_depth=False`` or the read thread is not running.
            TimeoutError: If no frame arrives within 10 s.
        """
        if timeout_ms:
            logger.warning(
                f"{self} read_depth() timeout_ms parameter is deprecated and will be removed in future versions."
            )
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — configured with use_depth=False.")
        return self._read(read_depth=True)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the most recent colour frame not yet consumed by ``async_read``/``read``.

        Args:
            timeout_ms: How long to wait for a frame the read thread has not handed out yet.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_rgb=False`` or the read thread is not running.
            TimeoutError: If no new frame arrives within ``timeout_ms``.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — configured with use_rgb=False.")
        return self._async_read(timeout_ms=timeout_ms)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        """Depth counterpart of :meth:`async_read`: ``(height, width, 1)`` ``uint16`` millimetres.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_depth=False`` or the read thread is not running.
            TimeoutError: If no new frame arrives within ``timeout_ms``.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — configured with use_depth=False.")
        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    def _read_latest(self, max_age_ms: int, read_depth: bool = False) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame too old: {age_ms:.1f} ms (max {max_age_ms} ms).")
        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the latest colour frame without waiting, as long as it is at most ``max_age_ms`` old.

        Unlike :meth:`async_read` this never consumes the frame, so :meth:`read_latest` and
        :meth:`read_latest_depth` called back to back come from the same grab.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_rgb=False``, the read thread is not running or no frame has
                been captured yet.
            TimeoutError: If the latest frame is older than ``max_age_ms``.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — configured with use_rgb=False.")
        return self._read_latest(max_age_ms=max_age_ms)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[np.uint16]:
        """Depth counterpart of :meth:`read_latest`: ``(height, width, 1)`` ``uint16`` millimetres.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_depth=False``, the read thread is not running or no frame has
                been captured yet.
            TimeoutError: If the latest frame is older than ``max_age_ms``.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — configured with use_depth=False.")
        return self._read_latest(max_age_ms=max_age_ms, read_depth=True)

    def disconnect(self) -> None:
        """Stop the read thread and close the SDK camera.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )
        self._cleanup_resources()
        logger.info(f"{self} disconnected.")
