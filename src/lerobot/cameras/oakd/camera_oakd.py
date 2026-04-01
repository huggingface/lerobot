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
Provides the OAKDCamera class for capturing frames from Luxonis OAK-D cameras.

Written for **depthai v3** (tested on 3.3.0).  The v3 API replaces
``XLinkOut`` with ``output.createOutputQueue()`` and replaces
``ColorCamera`` with the unified ``Camera`` node.
"""

import contextlib
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore
from numpy.typing import NDArray  # type: ignore

try:
    import depthai as dai

    _dai_available = True
except ImportError:
    dai = None  # type: ignore[assignment]
    _dai_available = False

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_oakd import OAKDCameraConfig

logger = logging.getLogger(__name__)


class OAKDCamera(Camera):
    """Manages interactions with Luxonis OAK-D cameras via DepthAI v3.

    Provides RGB capture and optional on-device stereo depth aligned to
    the color camera.  The interface mirrors ``RealSenseCamera`` so it
    can be used as a drop-in replacement inside LeRobot.

    Use the provided utility to discover connected devices::

        lerobot-find-cameras oakd

    Example::

        from lerobot.cameras.oakd import OAKDCamera, OAKDCameraConfig

        config = OAKDCameraConfig(fps=30, width=640, height=480, use_depth=True)
        camera = OAKDCamera(config)
        camera.connect()

        color = camera.read()
        depth = camera.read_depth()
        intrinsics = camera.get_depth_intrinsics()

        camera.disconnect()
    """

    def __init__(self, config: OAKDCameraConfig):
        super().__init__(config)

        self.config = config
        self.device_id = config.device_id
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self._device: Any | None = None  # dai.Device
        self._pipeline: Any | None = None  # dai.Pipeline
        self._color_queue: Any | None = None  # dai.MessageQueue
        self._depth_queue: Any | None = None  # dai.MessageQueue

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.device_id or 'auto'})"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect connected OAK devices."""
        if not _dai_available or dai is None:
            return []
        found: list[dict[str, Any]] = []
        for dev_info in dai.Device.getAllAvailableDevices():
            found.append(
                {
                    "type": "OAK-D",
                    "id": dev_info.getMxId(),
                    "name": dev_info.name,
                    "state": dev_info.state.name,
                    "protocol": dev_info.protocol.name if hasattr(dev_info, "protocol") else "unknown",
                }
            )
        return found

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Connect to the OAK-D device and start the capture pipeline."""
        if not _dai_available or dai is None:
            raise ImportError(
                "depthai is not installed. Install it with: pip install depthai"
            )

        try:
            if self.device_id:
                self._device = dai.Device(self.device_id)
            else:
                self._device = dai.Device()
        except RuntimeError as e:
            available = dai.Device.getAllAvailableDevices()
            if not available:
                raise ConnectionError(
                    "No OAK-D devices found. Check USB connection and run: lerobot-find-cameras oakd"
                ) from e
            raise

        pipeline = dai.Pipeline(self._device)

        # ---- RGB camera (CAM_A) ----
        rgb_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        w = self.capture_width or 640
        h = self.capture_height or 480
        rgb_out = rgb_cam.requestOutput((w, h), dai.ImgFrame.Type.BGR888i)
        # Keep only the most recent frame to reduce stale RGB/depth pairing.
        self._color_queue = rgb_out.createOutputQueue(maxSize=1, blocking=False)

        # ---- Stereo depth (optional) ----
        if self.use_depth:
            left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            stereo = pipeline.create(dai.node.StereoDepth)
            preset_name = (
                getattr(self.config, "stereo_preset", "FAST_ACCURACY") or "FAST_ACCURACY"
            ).strip().upper().replace("-", "_")
            pm = dai.node.StereoDepth.PresetMode
            if not hasattr(pm, preset_name):
                logger.warning(
                    "Unknown OAK-D stereo_preset %r; using FAST_ACCURACY.",
                    getattr(self.config, "stereo_preset", None),
                )
            preset_mode = getattr(pm, preset_name, pm.FAST_ACCURACY)
            stereo.setDefaultProfilePreset(preset_mode)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(w, h)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setExtendedDisparity(False)
            # Median filter supports max 1024 disparity levels; presets use 5 subpixel bits (~3040
            # levels) which triggers "Maximum disparity exceeds 1024" and disables the filter.
            # Use 3 fractional bits so median filter stays enabled for less noisy depth.
            try:
                stereo.initialConfig.setSubpixelFractionalBits(3)
            except Exception:
                pass
            try:
                stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            except Exception:
                logger.debug("OAK-D median filter configuration not supported on this device/SDK.")
            conf_thr = int(getattr(self.config, "stereo_confidence_threshold", 200))
            if conf_thr >= 0:
                try:
                    stereo.setConfidenceThreshold(conf_thr)
                except Exception:
                    logger.debug("OAK-D confidence threshold configuration not supported on this device/SDK.")

            left_out = left_cam.requestFullResolutionOutput()
            right_out = right_cam.requestFullResolutionOutput()
            left_out.link(stereo.left)
            right_out.link(stereo.right)

            # Keep only the most recent depth frame to reduce stale RGB/depth pairing.
            self._depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

        self._pipeline = pipeline
        pipeline.start()

        self._auto_detect_resolution()
        self._start_read_thread()

        if warmup:
            warmup_time = max(self.warmup_s, 1)
            start_time = time.time()
            while time.time() - start_time < warmup_time:
                with contextlib.suppress(TimeoutError):
                    self.async_read(timeout_ms=warmup_time * 1000)
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_color_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")
                if self.use_depth and self.latest_depth_frame is None:
                    raise ConnectionError(f"{self} failed to capture depth frames during warmup.")

        logger.info(f"{self} connected.")

    def _auto_detect_resolution(self) -> None:
        """Auto-detect width/height from the first captured frame when not configured."""
        if self.width is not None and self.height is not None and self.fps is not None:
            return

        frame_packet = self._color_queue.get()
        frame = frame_packet.getCvFrame()
        fh, fw = frame.shape[:2]
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = fh, fw
            self.capture_width, self.capture_height = fw, fh
        else:
            self.width, self.height = fw, fh
            self.capture_width, self.capture_height = fw, fh
        if self.fps is None:
            self.fps = 30

    # ------------------------------------------------------------------
    # Depth intrinsics
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_depth_intrinsics(self) -> dict[str, float | int]:
        """Return depth camera intrinsics aligned to RGB after rotation.

        Returns:
            Dict with keys: fx, fy, cx, cy, width, height, depth_scale.
            depth_scale converts raw uint16 values to meters (typically 0.001).
        """
        if not self.use_depth:
            raise RuntimeError(
                "Depth stream must be enabled (use_depth=True) to get depth intrinsics."
            )

        calib = self._device.readCalibration()
        intrinsics_matrix = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            self.capture_width,
            self.capture_height,
        )
        fx = float(intrinsics_matrix[0][0])
        fy = float(intrinsics_matrix[1][1])
        cx = float(intrinsics_matrix[0][2])
        cy = float(intrinsics_matrix[1][2])
        w, h = self.capture_width, self.capture_height
        depth_scale = 0.001  # OAK depth is in millimeters

        if self.rotation == cv2.ROTATE_90_CLOCKWISE:
            fx, fy = fy, fx
            cx, cy = h - 1 - cy, cx
            w, h = h, w
        elif self.rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            fx, fy = fy, fx
            cx, cy = cy, w - 1 - cx
            w, h = h, w
        elif self.rotation == cv2.ROTATE_180:
            cx, cy = w - 1 - cx, h - 1 - cy

        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": w,
            "height": h,
            "depth_scale": depth_scale,
        }

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """Read the latest depth frame (uint16, millimeters).

        Returns:
            np.ndarray of shape (H, W) with dtype uint16.
        """
        if not self.use_depth:
            raise RuntimeError(
                f"Depth stream is not enabled for {self}. Set use_depth=True."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        _ = self.async_read(timeout_ms=10000)

        with self.frame_lock:
            depth_map = self.latest_depth_frame

        if depth_map is None:
            raise RuntimeError("No depth frame available. Ensure camera is streaming.")

        return depth_map

    @check_if_not_connected
    def read(self, timeout_ms: int = 0) -> NDArray[Any]:
        """Read a single color frame (blocking)."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed color frame, blocking up to *timeout_ms*."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms} ms."
            )

        with self.frame_lock:
            frame = self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent color frame immediately (non-blocking peek)."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max: {max_age_ms} ms)."
            )

        return frame

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _postprocess_image(self, image: NDArray[Any], is_depth: bool = False) -> NDArray[Any]:
        """Apply color conversion and rotation."""
        if not is_depth:
            if self.color_mode == ColorMode.BGR:
                pass  # getCvFrame already returns BGR
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            image = cv2.rotate(image, self.rotation)

        return image

    def _read_loop(self) -> None:
        """Background loop that reads frames from the OAK device queues."""
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized.")

        def _drain_latest_packet(queue):
            """Drain queue and return the most recent packet."""
            latest = None
            while True:
                pkt = queue.tryGet()
                if pkt is None:
                    break
                latest = pkt
            return latest

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                color_packet = _drain_latest_packet(self._color_queue)
                if color_packet is None:
                    time.sleep(0.001)
                    continue

                color_frame = color_packet.getCvFrame()
                if color_frame.ndim == 2:
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_GRAY2BGR)
                processed_color = self._postprocess_image(color_frame)

                processed_depth = None
                if self.use_depth and self._depth_queue is not None:
                    depth_packet = _drain_latest_packet(self._depth_queue)
                    if depth_packet is None:
                        # Do not publish unsynchronized RGB-only updates when depth is enabled.
                        time.sleep(0.001)
                        continue
                    depth_frame = depth_packet.getFrame()
                    processed_depth = self._postprocess_image(depth_frame, is_depth=True)

                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_color_frame = processed_color
                    if processed_depth is not None:
                        self.latest_depth_frame = processed_depth
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

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
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect from the OAK-D device and release resources."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it is already disconnected."
            )

        self._stop_read_thread()

        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.stop()
            self._pipeline = None

        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.close()
            self._device = None

        self._color_queue = None
        self._depth_queue = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
