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
Provides the OrbbecCamera class for capturing frames from Orbbec cameras.

Uses the ``pyorbbecsdk`` (2.0.18) Python bindings which wrap the Orbbec SDK v2.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        FormatConvertFilter,
        FrameSet,
        OBAlignMode,
        OBConvertFormat,
        OBError,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBSensorType,
        OBStreamType,
        Pipeline,
        VideoFrame,
        VideoStreamProfile,
    )
except Exception as e:
    logging.info(f"Could not import pyorbbecsdk: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import D2CMode, OrbbecCameraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-format conversion helper
# ---------------------------------------------------------------------------

# Maps Orbbec pixel formats to the SDK's FormatConvertFilter target.
# Mirrors ``determine_convert_format`` in pyorbbecsdk/examples/utils.py.
_CONVERT_FORMAT_MAP: dict = {}
_CONVERT_FILTER_CACHE: dict = {}


def _build_convert_map() -> None:
    """Populate _CONVERT_FORMAT_MAP once the SDK is imported."""
    global _CONVERT_FORMAT_MAP
    _CONVERT_FORMAT_MAP = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
        OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
        OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
    }


def _frame_to_rgb_image(frame: "VideoFrame") -> "NDArray[np.uint8] | None":
    """Convert an Orbbec ``VideoFrame`` to an RGB ``np.ndarray`` (H, W, 3).

    Follows the same pattern as ``frame_to_bgr_image`` in
    ``pyorbbecsdk/examples/utils.py``, but returns RGB instead of BGR.
    Returns ``None`` when conversion fails.
    """
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return data.reshape((height, width, 3))

    if fmt == OBFormat.BGR:
        return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_BGR2RGB)

    # All other formats: delegate to FormatConvertFilter (SDK-recommended path)
    if not _CONVERT_FORMAT_MAP:
        _build_convert_map()

    convert_fmt = _CONVERT_FORMAT_MAP.get(fmt)
    if convert_fmt is None:
        logger.warning(f"Unsupported Orbbec color format: {fmt}")
        return None

    try:
        f = _CONVERT_FILTER_CACHE.get(convert_fmt)
        if f is None:
            f = FormatConvertFilter()
            f.set_format_convert_format(convert_fmt)
            _CONVERT_FILTER_CACHE[convert_fmt] = f
        rgb_frame = f.process(frame)
        if rgb_frame is None:
            return None
        rgb_data = np.frombuffer(rgb_frame.get_data(), dtype=np.uint8)
        return rgb_data.reshape((height, width, 3))
    except Exception as e:
        logger.debug(f"FormatConvertFilter failed for {fmt}: {e}")
        return None


# ===========================================================================
# OrbbecCamera
# ===========================================================================


class OrbbecCamera(Camera):
    """Manages interactions with Orbbec depth cameras via ``pyorbbecsdk`` (2.0.18).

    * A **background thread** continuously reads FrameSets from the pipeline.
    * Color and depth are extracted from the **same** FrameSet for frame-level sync.
    * Depth output shape is ``(H, W, 1)`` uint16 (millimetres).
    * When ``align_depth=True``, hardware D2C alignment is attempted first
      (``OBAlignMode.HW_MODE`` via ``get_d2c_depth_profile_list``); falls back
      to software ``AlignFilter`` if hardware mode is unavailable.

    Example::

        config = OrbbecCameraConfig(
            index_or_serial_number=0, fps=30, width=640, height=480, use_depth=True, align_depth=True
        )
        cam = OrbbecCamera(config)
        cam.connect()
        color = cam.async_read()  # (480, 640, 3) RGB uint8
        depth = cam.async_read_depth()  # (480, 640, 1) uint16 mm
        cam.disconnect()
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config

        self.index_or_serial_number: int | str = config.index_or_serial_number
        self.serial_number: str | None = None

        self.fps = config.fps
        self.color_mode: ColorMode = config.color_mode
        self.use_depth: bool = config.use_depth
        self.align_depth: bool = config.align_depth
        self.d2c_mode: D2CMode = config.d2c_mode
        self.warmup_s: int = config.warmup_s

        # SDK objects (populated in connect())
        self._pipeline: Pipeline | None = None
        self._config: Config | None = None
        self._align_filter: AlignFilter | None = None  # software fallback only

        # Background read thread
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self.new_depth_frame_event: Event = Event()

        # Rotation & capture dimensions
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.capture_width: int | None = config.width
        self.capture_height: int | None = config.height

    def __str__(self) -> str:
        tag = self.serial_number or self.index_or_serial_number
        return f"{self.__class__.__name__}({tag})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect all connected Orbbec cameras.

        Probes each device's sensor list to report color/depth availability
        and default stream profiles — useful when configuring multi-camera setups.

        Returns:
            List of dicts, one per device, with keys:
            ``type``, ``index``, ``name``, ``id``, ``pid``, ``vid``,
            ``connection_type``, ``has_color_sensor``, ``has_depth_sensor``,
            ``default_color_profile``, ``default_depth_profile``.
        """
        ctx = Context()
        device_list = ctx.query_devices()
        found: list[dict[str, Any]] = []

        for idx in range(device_list.get_count()):
            device = device_list[idx]
            info = device.get_device_info()

            cam: dict[str, Any] = {
                "type": "Orbbec",
                "index": idx,
                "name": info.get_name(),
                "id": info.get_serial_number(),
                "pid": info.get_pid(),
                "vid": info.get_vid(),
                "connection_type": info.get_connection_type(),
                "has_color_sensor": False,
                "has_depth_sensor": False,
                "default_color_profile": None,
                "default_depth_profile": None,
            }

            sensor_list = device.get_sensor_list()
            pipeline = Pipeline(device)

            for si in range(sensor_list.get_count()):
                stype = sensor_list.get_sensor_by_index(si).get_type()

                if stype == OBSensorType.COLOR_SENSOR:
                    cam["has_color_sensor"] = True
                    try:
                        dp: VideoStreamProfile = pipeline.get_stream_profile_list(
                            OBSensorType.COLOR_SENSOR
                        ).get_default_video_stream_profile()
                        cam["default_color_profile"] = {
                            "width": dp.get_width(),
                            "height": dp.get_height(),
                            "fps": dp.get_fps(),
                            "format": str(dp.get_format()),
                        }
                    except Exception as e:
                        logger.debug(f"Could not query color profile for device {idx}: {e}")

                elif stype == OBSensorType.DEPTH_SENSOR:
                    cam["has_depth_sensor"] = True
                    try:
                        dp = pipeline.get_stream_profile_list(
                            OBSensorType.DEPTH_SENSOR
                        ).get_default_video_stream_profile()
                        cam["default_depth_profile"] = {
                            "width": dp.get_width(),
                            "height": dp.get_height(),
                            "fps": dp.get_fps(),
                            "format": str(dp.get_format()),
                        }
                    except Exception as e:
                        logger.debug(f"Could not query depth profile for device {idx}: {e}")

            found.append(cam)

        return found

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        """Return the ``Device`` matching ``self.index_or_serial_number``."""
        ctx = Context()
        device_list = ctx.query_devices()
        count = device_list.get_count()

        if count == 0:
            raise ConnectionError("No Orbbec device found. Run `lerobot-find-cameras orbbec` to check.")

        if isinstance(self.index_or_serial_number, int):
            if self.index_or_serial_number >= count:
                raise ConnectionError(
                    f"Device index {self.index_or_serial_number} out of range ({count} device(s) connected)."
                )
            device = device_list[self.index_or_serial_number]
            self.serial_number = device.get_device_info().get_serial_number()
            return device

        for i in range(count):
            device = device_list[i]
            sn = device.get_device_info().get_serial_number()
            if sn == self.index_or_serial_number:
                self.serial_number = sn
                return device

        raise ConnectionError(
            f"No Orbbec device with serial '{self.index_or_serial_number}'. "
            f"Use `lerobot-find-cameras orbbec` to list available devices."
        )

    def _configure_pipeline(self, device: Any) -> None:
        """Build ``Pipeline`` + ``Config`` for *device* and resolve stream dimensions.

        Color stream:
            Tries the requested (width, height, fps) in RGB format first,
            then any format, then falls back to the sensor default.

        Depth stream (when ``use_depth=True``):
            When ``align_depth=True``:
                1. Queries hardware-aligned depth profiles via
                   ``get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)``.
                2. If HW profiles are found, sets ``config.set_align_mode(HW_MODE)``.
                3. Otherwise falls back to software ``AlignFilter``.
            Frame sync is enabled with ``pipeline.enable_frame_sync()`` and
            ``OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE``.
        """
        pipeline = Pipeline(device)
        config = Config()

        # --- Color stream ---
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if self.fps is not None and self.capture_width is not None and self.capture_height is not None:
                try:
                    color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(
                        self.capture_width, self.capture_height, OBFormat.RGB, self.fps
                    )
                except OBError:
                    try:
                        color_profile = profile_list.get_video_stream_profile(
                            self.capture_width, self.capture_height, OBFormat.UNKNOWN_FORMAT, self.fps
                        )
                    except OBError:
                        color_profile = profile_list.get_default_video_stream_profile()
                        logger.warning(
                            f"{self}: requested color profile "
                            f"{self.capture_width}x{self.capture_height}@{self.fps} not available; "
                            f"using default {color_profile.get_width()}x"
                            f"{color_profile.get_height()}@{color_profile.get_fps()}."
                        )
            else:
                color_profile = profile_list.get_default_video_stream_profile()

            config.enable_stream(color_profile)

            # Store actual stream dimensions
            self.capture_width = color_profile.get_width()
            self.capture_height = color_profile.get_height()
            if self.fps is None:
                self.fps = color_profile.get_fps()
            if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                self.width, self.height = self.capture_height, self.capture_width
            else:
                self.width, self.height = self.capture_width, self.capture_height

        except OBError as e:
            raise ConnectionError(f"{self}: no color sensor – {e}") from e

        # --- Depth stream (optional) ---
        if self.use_depth:
            try:
                depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

                if self.align_depth:
                    if self.d2c_mode == D2CMode.SOFTWARE:
                        depth_profile = depth_profile_list.get_default_video_stream_profile()
                        self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                        logger.info(f"{self}: software D2C alignment enabled.")
                    else:
                        # Prefer hardware D2C alignment by default
                        try:
                            hw_profiles = pipeline.get_d2c_depth_profile_list(
                                color_profile, OBAlignMode.HW_MODE
                            )
                            if hw_profiles and len(hw_profiles) > 0:
                                depth_profile = hw_profiles[0]
                                config.set_align_mode(OBAlignMode.HW_MODE)
                                logger.info(f"{self}: hardware D2C alignment enabled.")
                            else:
                                raise RuntimeError("No HW D2C profiles available.")
                        except Exception as hw_err:
                            logger.warning(
                                f"{self}: hardware D2C unavailable ({hw_err}); "
                                f"falling back to software AlignFilter."
                            )
                            depth_profile = depth_profile_list.get_default_video_stream_profile()
                            self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                else:
                    depth_profile = depth_profile_list.get_default_video_stream_profile()

                config.enable_stream(depth_profile)
                config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                try:
                    pipeline.enable_frame_sync()
                except Exception as e:
                    logger.debug(f"{self}: enable_frame_sync not supported: {e}")

            except OBError as e:
                raise ConnectionError(f"{self}: use_depth=True but no depth sensor – {e}") from e

        self._pipeline = pipeline
        self._config = config

    def connect(self, warmup: bool = True) -> None:
        """Open the device, configure streams, start pipeline and read thread.

        Args:
            warmup: Discard frames for ``warmup_s`` seconds so auto-exposure settles.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            ConnectionError: If the device cannot be found or pipeline fails to start.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        device = self._resolve_device()
        self._configure_pipeline(device)

        assert self._pipeline is not None and self._config is not None
        try:
            self._pipeline.start(self._config)
        except Exception as e:
            self._pipeline = None
            self._config = None
            raise ConnectionError(f"Failed to start pipeline for {self}.") from e

        self._start_read_thread()

        try:
            if warmup:
                self.warmup_s = max(self.warmup_s, 1)
                deadline = time.time() + self.warmup_s
                while time.time() < deadline:
                    self.new_frame_event.wait(timeout=0.1)
                    self.new_frame_event.clear()

                with self.frame_lock:
                    if self.latest_color_frame is None or (
                        self.use_depth and self.latest_depth_frame is None
                    ):
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")
        except Exception:
            self._stop_read_thread()
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception as stop_err:
                    logger.warning(f"{self} pipeline.stop() error after failed connect: {stop_err}")
            self._pipeline = None
            self._config = None
            self._align_filter = None
            raise

        logger.info(f"{self} connected (depth={self.use_depth}, align={self.align_depth}).")

    # ------------------------------------------------------------------
    # Hardware read
    # ------------------------------------------------------------------

    def _read_from_hardware(self, timeout_ms: int = 200) -> "FrameSet | None":
        """Read a FrameSet from the pipeline; apply software AlignFilter if set."""
        if self._pipeline is None:
            raise DeviceNotConnectedError(f"{self}: pipeline not started.")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None

        if self._align_filter is not None:
            frames = self._align_filter.process(frames)
            if frames is not None:
                frames = frames.as_frame_set()

        return frames

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess_color(self, rgb: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
        """Validate, colour-convert, and rotate a color frame."""
        h, w, c = rgb.shape
        if c != 3:
            raise RuntimeError(f"{self}: unexpected channel count {c}.")

        if (
            self.capture_height
            and self.capture_width
            and (h != self.capture_height or w != self.capture_width)
        ):
            logger.debug(f"{self}: resize color {w}x{h} → {self.capture_width}x{self.capture_height}")
            rgb = cv2.resize(rgb, (self.capture_width, self.capture_height))

        if self.color_mode == ColorMode.BGR:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            rgb = cv2.rotate(rgb, self.rotation)

        return rgb

    def _postprocess_depth(self, depth_mm: "NDArray[np.uint16]") -> "NDArray[np.uint16]":
        """Rotate and expand dims of a depth frame: (H, W) → (H, W, 1)."""
        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            depth_mm = cv2.rotate(depth_mm, self.rotation)
        return np.expand_dims(depth_mm, axis=2)

    # ------------------------------------------------------------------
    # Background read thread
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Continuously read FrameSets, process color + depth, cache results."""
        assert self.stop_event is not None
        stop = self.stop_event  # local reference avoids race when self.stop_event is cleared
        while not stop.is_set():
            try:
                frames = self._read_from_hardware(timeout_ms=200)
                if frames is None:
                    continue

                color_frame_raw = frames.get_color_frame()
                if color_frame_raw is None:
                    continue

                rgb = _frame_to_rgb_image(color_frame_raw)
                if rgb is None:
                    logger.warning(
                        f"{self}: could not convert color frame (fmt={color_frame_raw.get_format()})"
                    )
                    continue

                processed_color = self._postprocess_color(rgb)
                processed_depth = None

                if self.use_depth:
                    depth_frame_raw = frames.get_depth_frame()
                    if depth_frame_raw is not None:
                        w = depth_frame_raw.get_width()
                        h = depth_frame_raw.get_height()
                        scale = depth_frame_raw.get_depth_scale()
                        depth_data = np.frombuffer(depth_frame_raw.get_data(), dtype=np.uint16).reshape(
                            (h, w)
                        )
                        if abs(scale - 1.0) < 1e-6:
                            depth_mm = depth_data
                        else:
                            depth_mm = (depth_data.astype(np.float32) * scale).astype(np.uint16)
                        processed_depth = self._postprocess_depth(depth_mm)

                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_color_frame = processed_color
                    if self.use_depth and processed_depth is not None:
                        self.latest_depth_frame = processed_depth
                    self.latest_timestamp = capture_time

                self.new_frame_event.set()
                if self.use_depth and processed_depth is not None:
                    self.new_depth_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"{self} _read_loop error: {e}")

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"{self}: read thread did not stop within 2 s.")
        self.thread = None
        if self.stop_event is not None and not self.stop_event.is_set():
            self.stop_event.set()
        self.stop_event = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()
            self.new_depth_frame_event.clear()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def read(self, timeout_ms: int = 0) -> "NDArray[Any]":
        """Read a color frame synchronously (blocks until a new frame arrives).

        The output color ordering is determined by ``config.color_mode`` (RGB by default).
        To change color ordering, configure it via ``OrbbecCameraConfig(color_mode=...)``.

        Args:
            timeout_ms: Maximum wait in milliseconds. Defaults to 10 000 ms.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        wait_timeout_ms = timeout_ms if timeout_ms > 0 else 10000
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=wait_timeout_ms)

    def read_depth(self, timeout_ms: int = 200) -> "NDArray[Any]":
        """Read a depth frame synchronously.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 1)`` dtype ``uint16`` (values in mm).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(f"{self}: depth stream not enabled. Set use_depth=True.")
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        wait_timeout_ms = timeout_ms if timeout_ms > 0 else 10000
        self.new_depth_frame_event.clear()
        depth = self.async_read_depth(timeout_ms=wait_timeout_ms)
        if depth is None:
            raise RuntimeError(f"{self}: no depth frame available.")
        return depth

    def async_read(self, timeout_ms: float = 200) -> "NDArray[Any]":
        """Return the latest color frame from the background thread.

        Args:
            timeout_ms: Max wait in milliseconds for a new frame.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 3)`` dtype ``uint8``.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for color frame from {self} after {timeout_ms} ms.")
        with self.frame_lock:
            frame = self.latest_color_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"{self}: frame event fired but frame is None.")
        return frame

    def async_read_depth(self, timeout_ms: float = 200) -> "NDArray[Any] | None":
        """Return the latest depth frame from the background thread.

        Returns ``None`` if ``use_depth=False``.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 1)`` dtype ``uint16``, or ``None``.
        """
        if not self.use_depth:
            return None
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self.new_depth_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for depth frame from {self} after {timeout_ms} ms.")
        with self.frame_lock:
            depth = self.latest_depth_frame
            self.new_depth_frame_event.clear()
        if depth is None:
            raise RuntimeError(f"{self}: depth event fired but frame is None.")
        return depth

    def read_latest(self, max_age_ms: int = 500) -> "NDArray[Any]":
        """Return the most recent color frame without blocking (peek buffer).

        Raises:
            TimeoutError: If the cached frame is older than ``max_age_ms``.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self.frame_lock:
            frame = self.latest_color_frame
            timestamp = self.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is {age_ms:.1f} ms old (max {max_age_ms} ms).")
        return frame

    # ------------------------------------------------------------------
    # Disconnection
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Stop the pipeline, background thread, and release all resources."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"Attempted to disconnect {self}, but it is already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self._pipeline is not None:
            pipeline = self._pipeline
            self._pipeline = None  # mark disconnected before blocking stop

            def _stop_pipeline() -> None:
                try:
                    pipeline.stop()
                except Exception as e:
                    logger.warning(f"{self} pipeline.stop() error: {e}")

            t = Thread(target=_stop_pipeline, daemon=True, name=f"{self}_pipeline_stop")
            t.start()
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning(f"{self}: pipeline.stop() did not finish within 3 s.")

        self._config = None
        self._align_filter = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()
            self.new_depth_frame_event.clear()

        logger.info(f"{self} disconnected.")
