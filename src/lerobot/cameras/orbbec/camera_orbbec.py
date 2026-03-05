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

Uses the ``pyorbbecsdk2`` (2.0.18) Python bindings which wrap the Orbbec SDK v2.

Architecture:
- ``read()`` calls ``_read_from_hardware()`` directly when no background thread is running.
  If a background thread is running it delegates to ``async_read()`` to avoid hardware races.
- Background thread is **lazy**: started only on the first ``async_read()`` / ``async_read_depth()``
  call.  ``connect()`` does **not** start it.
- ``read_latest()`` peeks the buffer; it works regardless of whether the thread is running
  (buffer may be populated by ``read()`` or the background thread).
- Color and depth are extracted from the **same** FrameSet for frame-level synchronisation.
- Depth output shape is ``(H, W, 1)`` uint16, matching RealSense conventions.
- Independent ``new_depth_frame_event`` for depth consumers.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore
from numpy.typing import NDArray  # type: ignore

try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        FormatConvertFilter,
        FrameSet,
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
    logging.info(f"Could not import pyorbbecsdk2: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-format conversion helpers
# ---------------------------------------------------------------------------


def _frame_to_rgb_image(frame: "VideoFrame") -> NDArray[np.uint8] | None:
    """Convert an Orbbec ``VideoFrame`` to an RGB ``np.ndarray`` (H, W, 3).

    Handles the most common Orbbec color formats:
        RGB, BGR, YUYV, UYVY, MJPG, I420, NV12, NV21.

    Returns ``None`` when the format is unsupported.
    """
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return data.reshape((height, width, 3))

    if fmt == OBFormat.BGR:
        bgr = data.reshape((height, width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.YUYV:
        yuyv = data.reshape((height, width, 2))
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.UYVY:
        uyvy = data.reshape((height, width, 2))
        bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.NV12:
        y = data[0:height, :]
        uv = data[height : height + height // 2].reshape(height // 2, width)
        yuv = cv2.merge([y, uv])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.NV21:
        y = data[0:height, :]
        uv = data[height : height + height // 2].reshape(height // 2, width)
        yuv = cv2.merge([y, uv])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if fmt == OBFormat.I420:
        y = data[0:height, :]
        u = data[height : height + height // 4].reshape(height // 2, width // 2)
        v = data[height + height // 4 :].reshape(height // 2, width // 2)
        yuv = cv2.merge([y, u, v])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Try using SDK's built-in FormatConvertFilter as a last resort
    try:
        convert_filter = FormatConvertFilter()
        convert_filter.set_format_convert_format(OBConvertFormat.MJPG_TO_RGB888)
        rgb_frame = convert_filter.process(frame)
        if rgb_frame is not None:
            rgb_data = np.frombuffer(rgb_frame.get_data(), dtype=np.uint8)
            return rgb_data.reshape((height, width, 3))
    except Exception as e:
        logger.debug(f"FormatConvertFilter fallback failed: {e}")

    logger.warning(f"Unsupported Orbbec color format: {fmt}")
    return None


# ===========================================================================
# OrbbecCamera
# ===========================================================================


class OrbbecCamera(Camera):
    """Manages interactions with Orbbec depth cameras via ``pyorbbecsdk2``.

    Architecture mirrors :class:`RealSenseCamera`:

    * A **background thread** continuously reads FrameSets from the pipeline.
    * Color and depth frames are extracted from the **same** FrameSet,
      guaranteeing frame-level temporal synchronisation.
    * ``read()`` and ``read_depth()`` delegate to the background cache
      (via ``async_read()``), not directly to the pipeline.
    * Depth output has shape ``(H, W, 1)`` uint16 (millimetres),
      matching RealSense conventions and dataset feature-shape expectations.
    * An independent ``new_depth_frame_event`` notifies depth consumers
      without coupling to colour frame timing.

    Orbbec-specific additions over RealSense:
    * ``align_depth`` — optional depth-to-color alignment via ``AlignFilter``.
    * Multi-format color conversion (RGB, BGR, YUYV, MJPG, NV12, …).

    Example::

        from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig

        config = OrbbecCameraConfig(
            index_or_serial_number=0, fps=30, width=640, height=480, use_depth=True, align_depth=True
        )
        cam = OrbbecCamera(config)
        cam.connect()

        color = cam.async_read()  # (480, 640, 3) RGB  uint8
        depth = cam.async_read_depth()  # (480, 640, 1) uint16 mm

        cam.disconnect()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config

        # Device identification ------------------------------------------------
        self.index_or_serial_number: int | str = config.index_or_serial_number
        self.serial_number: str | None = None  # resolved during connect()

        # Stream parameters ----------------------------------------------------
        self.fps = config.fps
        self.color_mode: ColorMode = config.color_mode
        self.use_depth: bool = config.use_depth
        self.align_depth: bool = config.align_depth
        self.warmup_s: int = config.warmup_s

        # SDK objects (populated in connect()) ---------------------------------
        self._pipeline: Pipeline | None = None
        self._config: Config | None = None
        self._align_filter: AlignFilter | None = None
        self._has_color_sensor: bool = False

        # Async read infrastructure (mirrors RealSense naming) -----------------
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self.new_depth_frame_event: Event = Event()

        # Rotation handling ----------------------------------------------------
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.capture_width: int | None = None
        self.capture_height: int | None = None
        # Native depth-sensor resolution (may differ from color when align_depth=False)
        self.capture_depth_width: int | None = None
        self.capture_depth_height: int | None = None
        self._depth_profile_size: tuple[int, int] | None = None  # set in _configure_pipeline

        if config.height and config.width:
            self.capture_width, self.capture_height = config.width, config.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = config.height, config.width

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        tag = self.serial_number if self.serial_number else self.index_or_serial_number
        return f"{self.__class__.__name__}({tag})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` when the Orbbec pipeline has been started successfully."""
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect all connected Orbbec cameras.

        Returns:
            A list of dicts, each containing at least::

                {
                    "type": "Orbbec",
                    "index": <int>,
                    "name": <str>,
                    "id": <serial_number_str>,
                    ...
                }
        """
        found: list[dict[str, Any]] = []
        ctx = Context()
        device_list = ctx.query_devices()
        count = device_list.get_count()

        for idx in range(count):
            device = device_list.get_device_by_index(idx)
            info = device.get_device_info()

            cam_info: dict[str, Any] = {
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

            # Probe sensors
            sensor_list = device.get_sensor_list()
            for si in range(sensor_list.get_count()):
                sensor = sensor_list.get_sensor_by_index(si)
                stype = sensor.get_type()
                if stype == OBSensorType.COLOR_SENSOR:
                    cam_info["has_color_sensor"] = True
                    try:
                        tmp_pipeline = Pipeline(device)
                        plist = tmp_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                        dp: VideoStreamProfile = plist.get_default_video_stream_profile()
                        cam_info["default_color_profile"] = {
                            "width": dp.get_width(),
                            "height": dp.get_height(),
                            "fps": dp.get_fps(),
                            "format": str(dp.get_format()),
                        }
                    except Exception as e:
                        logger.debug(f"Could not query color profile for device {idx}: {e}")
                elif stype == OBSensorType.DEPTH_SENSOR:
                    cam_info["has_depth_sensor"] = True
                    try:
                        tmp_pipeline = Pipeline(device)
                        plist = tmp_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                        dp = plist.get_default_video_stream_profile()
                        cam_info["default_depth_profile"] = {
                            "width": dp.get_width(),
                            "height": dp.get_height(),
                            "fps": dp.get_fps(),
                            "format": str(dp.get_format()),
                        }
                    except Exception as e:
                        logger.debug(f"Could not query depth profile for device {idx}: {e}")

            found.append(cam_info)

        return found

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        """Return the ``Device`` object that matches ``self.index_or_serial_number``."""
        ctx = Context()
        device_list = ctx.query_devices()
        count = device_list.get_count()

        if count == 0:
            raise ConnectionError("No Orbbec device found. Run `lerobot-find-cameras orbbec` to check.")

        # --- by integer index ---
        if isinstance(self.index_or_serial_number, int):
            if self.index_or_serial_number >= count:
                raise ConnectionError(
                    f"Requested device index {self.index_or_serial_number}, "
                    f"but only {count} Orbbec device(s) connected."
                )
            device = device_list.get_device_by_index(self.index_or_serial_number)
            self.serial_number = device.get_device_info().get_serial_number()
            return device

        # --- by serial number ---
        for i in range(count):
            device = device_list.get_device_by_index(i)
            sn = device.get_device_info().get_serial_number()
            if sn == self.index_or_serial_number:
                self.serial_number = sn
                return device

        available_sns = []
        for i in range(count):
            d = device_list.get_device_by_index(i)
            available_sns.append(d.get_device_info().get_serial_number())
        raise ConnectionError(
            f"No Orbbec device with serial number '{self.index_or_serial_number}'. "
            f"Available: {available_sns}. Use `lerobot-find-cameras orbbec`."
        )

    def _configure_pipeline(self, device: Any) -> None:
        """Create ``Pipeline`` + ``Config`` for the resolved *device*."""
        self._pipeline = Pipeline(device)
        self._config = Config()

        # --- Color stream ---
        try:
            profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if self.fps and self.capture_width and self.capture_height:
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
                            f"{self}: Requested colour profile "
                            f"{self.capture_width}x{self.capture_height}@{self.fps} not available; "
                            f"using default {color_profile.get_width()}x"
                            f"{color_profile.get_height()}@{color_profile.get_fps()}."
                        )
            else:
                color_profile = profile_list.get_default_video_stream_profile()

            self._config.enable_stream(color_profile)
            self._has_color_sensor = True
        except OBError as e:
            logger.warning(f"{self}: No color sensor available – {e}")
            self._has_color_sensor = False

        # --- Depth stream (optional) ---
        if self.use_depth:
            try:
                profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                if self.fps and self.capture_width and self.capture_height:
                    try:
                        depth_profile = profile_list.get_video_stream_profile(
                            self.capture_width, self.capture_height, OBFormat.UNKNOWN_FORMAT, self.fps
                        )
                    except OBError:
                        depth_profile = profile_list.get_default_video_stream_profile()
                        logger.warning(f"{self}: Requested depth profile not available; using default.")
                else:
                    depth_profile = profile_list.get_default_video_stream_profile()
                self._depth_profile_size = (depth_profile.get_width(), depth_profile.get_height())
                self._config.enable_stream(depth_profile)
            except OBError as e:
                raise ConnectionError(f"{self}: use_depth=True but no depth sensor found – {e}") from e

            # Require full frame set so colour & depth are synchronised
            try:
                self._config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            except Exception as e:
                logger.debug(f"{self}: Could not set FULL_FRAME_REQUIRE mode (FW may not support it): {e}")
        else:
            # Explicitly disable depth stream to save USB bandwidth
            try:
                self._config.disable_stream(OBStreamType.DEPTH_STREAM)
            except Exception as e:
                logger.debug(f"{self}: Could not disable depth stream (FW may not support it): {e}")
            logger.info(f"{self}: depth stream disabled (use_depth=False).")

        # --- Depth alignment ---
        if self.align_depth:
            try:
                self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            except Exception as e:
                logger.warning(f"{self}: Could not create AlignFilter – {e}")
                self._align_filter = None

    def _configure_capture_settings(self) -> None:
        """Infer ``fps``, ``width``, ``height`` from the active stream when not set."""
        if self._pipeline is None:
            raise DeviceNotConnectedError(f"Cannot configure capture settings: {self} not connected.")

        if self._has_color_sensor:
            try:
                profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                vp: VideoStreamProfile = profile_list.get_default_video_stream_profile()
                actual_w, actual_h, actual_fps = vp.get_width(), vp.get_height(), vp.get_fps()
            except Exception as e:
                logger.warning(f"{self}: Could not query color stream profile, using defaults: {e}")
                actual_w, actual_h, actual_fps = 640, 480, 30
        else:
            try:
                profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                vp = profile_list.get_default_video_stream_profile()
                actual_w, actual_h, actual_fps = vp.get_width(), vp.get_height(), vp.get_fps()
            except Exception as e:
                logger.warning(f"{self}: Could not query depth stream profile, using defaults: {e}")
                actual_w, actual_h, actual_fps = 640, 480, 30

        if self.fps is None:
            self.fps = actual_fps

        if self.width is None or self.height is None:
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_h, actual_w
            else:
                self.width, self.height = actual_w, actual_h
            self.capture_width, self.capture_height = actual_w, actual_h

        # Resolve capture dimensions for depth validation in _postprocess_image.
        # When align_depth=True, depth is reprojected to color space (same size).
        # Otherwise use the depth sensor's native profile size.
        if self.use_depth:
            if self.align_depth:
                self.capture_depth_width = self.capture_width
                self.capture_depth_height = self.capture_height
            elif self._depth_profile_size is not None:
                self.capture_depth_width, self.capture_depth_height = self._depth_profile_size
            else:
                self.capture_depth_width = self.capture_width
                self.capture_depth_height = self.capture_height

    def connect(self, warmup: bool = True) -> None:
        """Open the Orbbec device, configure streams, and start the pipeline.

        Background thread is **not** started here; it is started lazily on the
        first ``async_read()`` / ``async_read_depth()`` call.  Warmup uses direct
        ``read()`` calls so no thread is required.

        Args:
            warmup: If *True* (default), read & discard frames for ``warmup_s``
                seconds so auto-exposure / white-balance can settle.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            ConnectionError: If the device cannot be found or started.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        device = self._resolve_device()
        self._configure_pipeline(device)
        self._configure_capture_settings()

        try:
            assert self._pipeline is not None and self._config is not None
            self._pipeline.start(self._config)
        except Exception as e:
            self._pipeline = None
            self._config = None
            raise ConnectionError(
                f"Failed to start pipeline for {self}. "
                f"Run `lerobot-find-cameras orbbec` to list available devices."
            ) from e

        # Warmup — at least 1 second; use direct hardware reads (no thread needed)
        if warmup:
            self.warmup_s = max(self.warmup_s, 1)
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read()
                except Exception as e:
                    logger.debug(f"{self} warmup color read failed (expected during settling): {e}")
                if self.use_depth:
                    try:
                        self.read_depth()
                    except Exception as e:
                        logger.debug(f"{self} warmup depth read failed (expected during settling): {e}")
                time.sleep(0.1)

            with self.frame_lock:
                warmup_ok = True
                warmup_msg = ""
                if self.latest_color_frame is None:
                    warmup_ok = False
                    warmup_msg = f"{self} failed to capture color frames during warmup."
                elif self.use_depth and self.latest_depth_frame is None:
                    warmup_ok = False
                    warmup_msg = f"{self} failed to capture depth frames during warmup."

            if not warmup_ok:
                # Clean up pipeline & thread so device is properly released
                try:
                    self.disconnect()
                except Exception as e:
                    logger.warning(f"{self} cleanup disconnect failed: {e}")
                raise ConnectionError(warmup_msg)

        logger.info(f"{self} connected (color={self._has_color_sensor}, depth={self.use_depth}).")

    # ------------------------------------------------------------------
    # Hardware read (called by background thread or directly by read())
    # ------------------------------------------------------------------

    def _read_from_hardware(self, timeout_ms: int = 10000) -> "FrameSet":
        """Read a FrameSet from the Orbbec pipeline.

        Applies ``AlignFilter`` if configured.

        Returns:
            Aligned (or raw) FrameSet containing colour and/or depth frames.

        Raises:
            RuntimeError: If no frames are returned within *timeout_ms*.
            DeviceNotConnectedError: If the pipeline is not started.
        """
        if self._pipeline is None:
            raise DeviceNotConnectedError(f"{self}: pipeline not started.")

        frames: FrameSet = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError(f"{self} _read_from_hardware(): no frames within {timeout_ms} ms.")

        # Depth-to-color alignment
        if self._align_filter is not None:
            frames = self._align_filter.process(frames)
            if frames is not None:
                frames = frames.as_frame_set()
            if frames is None:
                raise RuntimeError(f"{self} _read_from_hardware(): align filter returned None.")

        return frames

    # ------------------------------------------------------------------
    # Post-processing (mirrors RealSense _postprocess_image)
    # ------------------------------------------------------------------

    def _postprocess_image(
        self,
        image: NDArray[Any],
        depth_frame: bool = False,
    ) -> NDArray[Any]:
        """Apply colour conversion, dimension validation and rotation.

        For depth frames (``depth_frame=True``):
        - Validates ``(H, W)`` matches ``capture_height × capture_width``.
        - Applies rotation.
        - Expands dims: ``(H, W)`` → ``(H, W, 1)`` (matches RealSense convention).

        For colour frames:
        - Validates 3-channel input.
        - Validates ``(H, W)`` matches ``capture_height × capture_width``.
        - Applies colour-mode conversion (RGB → BGR if configured).
        - Applies rotation.
        """
        if self.color_mode and self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"Invalid color_mode '{self.color_mode}'.")

        if depth_frame:
            h, w = image.shape[:2]

            if (
                self.capture_depth_height
                and self.capture_depth_width
                and (h != self.capture_depth_height or w != self.capture_depth_width)
            ):
                raise RuntimeError(
                    f"{self} depth frame width={w} or height={h} do not match "
                    f"configured depth width={self.capture_depth_width} or height={self.capture_depth_height}."
                )

            processed_image = image
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                processed_image = cv2.rotate(processed_image, self.rotation)

            # (H, W) → (H, W, 1) — matches RealSense / dataset feature shape
            processed_image = np.expand_dims(processed_image, axis=2)
        else:
            h, w, c = image.shape
            if c != 3:
                raise RuntimeError(f"{self}: unexpected channel count {c}.")

            if (
                self.capture_height
                and self.capture_width
                and (h != self.capture_height or w != self.capture_width)
            ):
                raise RuntimeError(
                    f"{self} frame width={w} or height={h} do not match "
                    f"configured width={self.capture_width} or height={self.capture_height}."
                )

            processed_image = image
            if self.color_mode == ColorMode.BGR:
                processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    # ------------------------------------------------------------------
    # Background read thread (mirrors RealSense _read_loop)
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Background loop: reads FrameSets from hardware, postprocesses, caches.

        Key changes from the original Orbbec implementation:
        1. Reads **one** FrameSet per iteration (colour + depth from the same frame).
        2. Sets independent ``new_depth_frame_event``.
        3. Failure counter — raises after 10 consecutive failures.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialised before _read_loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                # --- Read a single FrameSet containing both streams ---
                frames = self._read_from_hardware(timeout_ms=10000)

                # --- Process colour ---
                color_frame_raw = frames.get_color_frame()
                if color_frame_raw is None:
                    raise RuntimeError(f"{self} _read_loop: FrameSet has no color frame.")

                rgb_image = _frame_to_rgb_image(color_frame_raw)
                if rgb_image is None:
                    raise RuntimeError(
                        f"{self} _read_loop: could not convert color frame "
                        f"(format={color_frame_raw.get_format()}) to RGB."
                    )
                processed_color_frame = self._postprocess_image(rgb_image)

                # --- Process depth (if enabled) ---
                processed_depth_frame = None
                if self.use_depth:
                    depth_frame_raw = frames.get_depth_frame()
                    if depth_frame_raw is not None:
                        width_d = depth_frame_raw.get_width()
                        height_d = depth_frame_raw.get_height()
                        scale = depth_frame_raw.get_depth_scale()

                        depth_data = np.frombuffer(depth_frame_raw.get_data(), dtype=np.uint16)
                        depth_data = depth_data.reshape((height_d, width_d))
                        depth_mm = (depth_data.astype(np.float32) * scale).astype(np.uint16)

                        processed_depth_frame = self._postprocess_image(depth_mm, depth_frame=True)

                # --- Store results (thread-safe) ---
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_color_frame = processed_color_frame
                    if self.use_depth and processed_depth_frame is not None:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time

                self.new_frame_event.set()
                if self.use_depth and processed_depth_frame is not None:
                    self.new_depth_frame_event.set()

                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"{self} _read_loop error ({failure_count}/10): {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """Start (or restart) the background frame-read thread."""
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signal the background thread to stop and wait for it to join."""
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
            self.new_frame_event.clear()
        self.new_depth_frame_event.clear()

    # ------------------------------------------------------------------
    # Synchronous read (delegate to async — mirrors RealSense)
    # ------------------------------------------------------------------

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """直接从硬件读取一帧彩色图像（阻塞）。

        Returns:
            ``np.ndarray`` of shape ``(H, W, 3)`` with dtype ``uint8``.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        frames = self._read_from_hardware(timeout_ms=60000)

        color_frame_raw = frames.get_color_frame()
        if color_frame_raw is None:
            raise RuntimeError(f"{self} read(): FrameSet has no color frame.")
        rgb_image = _frame_to_rgb_image(color_frame_raw)
        if rgb_image is None:
            raise RuntimeError(
                f"{self} read(): could not convert color frame "
                f"(format={color_frame_raw.get_format()}) to RGB."
            )
        processed_color = self._postprocess_image(rgb_image)

        capture_time = time.perf_counter()
        with self.frame_lock:
            self.latest_color_frame = processed_color
            self.latest_timestamp = capture_time
        self.new_frame_event.set()

        return processed_color

    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """直接从硬件读取一帧深度图像（阻塞）。

        Returns:
            ``np.ndarray`` of shape ``(H, W, 1)`` with dtype ``uint16`` (values in mm).

        Raises:
            RuntimeError: If depth is not enabled or no depth frame available.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"{self} read_depth(): depth stream is not enabled. Set use_depth=True in OrbbecCameraConfig."
            )

        frames = self._read_from_hardware(timeout_ms=60000)

        depth_frame_raw = frames.get_depth_frame()
        if depth_frame_raw is None:
            raise RuntimeError(f"{self}: No depth frame available. Ensure camera is streaming.")

        width_d = depth_frame_raw.get_width()
        height_d = depth_frame_raw.get_height()
        scale = depth_frame_raw.get_depth_scale()
        depth_data = np.frombuffer(depth_frame_raw.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height_d, width_d))
        depth_mm = (depth_data.astype(np.float32) * scale).astype(np.uint16)
        processed_depth = self._postprocess_image(depth_mm, depth_frame=True)

        with self.frame_lock:
            self.latest_depth_frame = processed_depth
        self.new_depth_frame_event.set()

        return processed_depth

    # ------------------------------------------------------------------
    # Asynchronous read (mirrors RealSense)
    # ------------------------------------------------------------------

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest colour frame captured by the background thread.

        Args:
            timeout_ms: Max wait (ms) for a frame to become available.

        Returns:
            ``np.ndarray`` — the most recent colour frame ``(H, W, 3)``.

        Raises:
            DeviceNotConnectedError: If not connected.
            TimeoutError: If no frame arrived within *timeout_ms*.
        """
        if not self.is_connected:
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
            frame = self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self}: async_read event fired but frame is None.")

        return frame

    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[Any] | None:
        """Return the latest depth frame captured by the background thread.

        Returns ``None`` if depth is not enabled (mirrors RealSense behaviour).

        Args:
            timeout_ms: Max wait (ms) for a depth frame.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 1)`` uint16, or ``None``.
        """
        if not self.use_depth:
            return None

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_depth_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for depth frame from {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            depth_frame = self.latest_depth_frame
            self.new_depth_frame_event.clear()

        if depth_frame is None:
            raise RuntimeError(f"{self}: async_read_depth event fired but depth frame is None.")

        return depth_frame

    # ------------------------------------------------------------------
    # Read latest (mirrors RealSense)
    # ------------------------------------------------------------------

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent (color) frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer. The frame may be stale.

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self.frame_lock:
            frame = self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

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
            try:
                self._pipeline.stop()
            except Exception as e:
                logger.warning(f"{self} pipeline.stop() error: {e}")
            self._pipeline = None

        self._config = None
        self._align_filter = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()
        self.new_depth_frame_event.clear()

        logger.info(f"{self} disconnected.")
