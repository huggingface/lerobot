"""Luxonis OAK camera implementation for LeRobot using DepthAI v3."""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import depthai as dai
except ImportError as e:
    dai = None
    _dai_import_error = e

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_oak import OAKCameraConfig

logger = logging.getLogger(__name__)


def _ensure_depthai() -> None:
    if dai is None:
        raise ImportError(
            "depthai is required for OAK cameras. "
            "Install it with: pip install depthai"
        ) from _dai_import_error


class OAKCamera(Camera):
    """Camera implementation for Luxonis OAK devices (OAK-D, OAK-D Lite, OAK-D Pro).

    Uses DepthAI v3 API with a background thread that reads from on-device queues.
    Supports color (RGB) and optional aligned stereo depth.

    Example:
        ```python
        from lerobot.cameras.oak import OAKCamera, OAKCameraConfig

        config = OAKCameraConfig(mxid_or_ip="", fps=30, width=640, height=400, use_depth=True)
        with OAKCamera(config) as cam:
            frame = cam.read()
            depth = cam.read_depth()
        ```
    """

    def __init__(self, config: OAKCameraConfig):
        _ensure_depthai()
        super().__init__(config)
        self.config = config
        self.mxid_or_ip = config.mxid_or_ip
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self._pipeline: Any | None = None
        self._device: Any | None = None
        self._rgb_queue: Any | None = None
        self._depth_queue: Any | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width = config.width or 640
        self.capture_height = config.height or 400
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.capture_height, self.capture_width

    def __str__(self) -> str:
        ident = self.mxid_or_ip or "auto"
        return f"OAKCamera({ident})"

    @property
    def is_connected(self) -> bool:
        return self._device is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect connected OAK devices."""
        _ensure_depthai()
        found = []
        for info in dai.DeviceBase.getAllAvailableDevices():
            found.append({
                "name": info.name,
                "type": "OAK",
                "id": info.deviceId,
                "state": str(info.state),
                "protocol": str(info.protocol),
                "platform": str(info.platform),
            })
        return found

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Start the DepthAI pipeline and background capture thread."""
        pipeline = dai.Pipeline()

        fps = self.fps or 30
        w, h = self.capture_width, self.capture_height

        cam_rgb = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps
        )

        color_type = (
            dai.ImgFrame.Type.BGR888i
            if self.color_mode == ColorMode.BGR
            else dai.ImgFrame.Type.RGB888i
        )
        rgb_out = cam_rgb.requestOutput((w, h), type=color_type)
        self._rgb_queue = rgb_out.createOutputQueue(maxSize=4, blocking=False)

        if self.use_depth:
            left = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B, sensorFps=fps
            )
            right = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C, sensorFps=fps
            )

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)

            left.requestOutput((w, h)).link(stereo.left)
            right.requestOutput((w, h)).link(stereo.right)

            rgb_align = cam_rgb.requestOutput(
                (w, h), dai.ImgFrame.Type.RGB888i, dai.ImgResizeMode.CROP, fps, True
            )
            rgb_align.link(stereo.inputAlignTo)

            self._depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        self._pipeline = pipeline
        self._device = True  # pipeline-managed device in v3 API

        self._start_read_thread()

        if warmup:
            start = time.time()
            while time.time() - start < self.warmup_s:
                try:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                except TimeoutError:
                    pass
                time.sleep(0.1)

            with self.frame_lock:
                if self.latest_color_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialized.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                if self._rgb_queue is None:
                    break

                rgb_msg = self._rgb_queue.tryGet()
                if rgb_msg is not None:
                    color_frame = rgb_msg.getCvFrame()
                    color_frame = self._postprocess_image(color_frame)

                    depth_frame = None
                    if self.use_depth and self._depth_queue is not None:
                        depth_msg = self._depth_queue.tryGet()
                        if depth_msg is not None:
                            depth_frame = depth_msg.getCvFrame()
                            depth_frame = self._postprocess_image(depth_frame, is_depth=True)

                    capture_time = time.perf_counter()
                    with self.frame_lock:
                        self.latest_color_frame = color_frame
                        if depth_frame is not None:
                            self.latest_depth_frame = depth_frame
                        self.latest_timestamp = capture_time
                    self.new_frame_event.set()
                    failure_count = 0
                else:
                    time.sleep(0.001)

            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning(f"Error in {self} read loop: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded max consecutive read failures.") from e

    def _postprocess_image(self, image: NDArray[Any], is_depth: bool = False) -> NDArray[Any]:
        processed = image
        # getCvFrame() always returns BGR; convert to RGB if requested
        if not is_depth and self.color_mode == ColorMode.RGB:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)
        return processed

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

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Blocking read of the latest color frame."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed color frame, waiting up to timeout_ms."""
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
            raise RuntimeError(f"Event set but no frame available for {self}.")

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

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """Read the latest depth frame. Returns (H, W) uint16 array in millimeters.

        Raises:
            RuntimeError: If depth is not enabled or no depth frame is available.
        """
        if not self.use_depth:
            raise RuntimeError(
                f"Depth not enabled for {self}. Set use_depth=True in config."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        _ = self.async_read(timeout_ms=timeout_ms)

        with self.frame_lock:
            depth = self.latest_depth_frame

        if depth is None:
            raise RuntimeError(f"No depth frame available from {self}.")

        return depth

    def disconnect(self) -> None:
        """Stop the pipeline and release the device."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it's already disconnected."
            )

        self._stop_read_thread()

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

        self._device = None
        self._rgb_queue = None
        self._depth_queue = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
