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
Provides the KinectCamera class for capturing frames from Microsoft Kinect v2 cameras.
"""

import gc
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

try:
    import pylibfreenect2
    from pylibfreenect2 import (
        FrameMap,
        FrameType,
        Freenect2,
        Freenect2Device,
        LoggerLevel,
        Registration,
        SyncMultiFrameListener,
        createConsoleLogger,
        setGlobalLogger,
    )

    KINECT_AVAILABLE = True

    # Set libfreenect2 logging to only show errors (suppress verbose output)
    logger_freenect = createConsoleLogger(LoggerLevel.Error)
    setGlobalLogger(logger_freenect)

except ImportError as e:
    KINECT_AVAILABLE = False
    logging.info(f"Could not import pylibfreenect2: {e}")

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_kinect import KinectCameraConfig, KinectPipeline

logger = logging.getLogger(__name__)


class KinectCamera(Camera):
    """
    Manages interactions with Microsoft Kinect v2 cameras for frame, depth, and IR recording.

    This class provides an interface similar to `RealSenseCamera` but tailored for
    Kinect v2 devices, leveraging the `pylibfreenect2` library. It supports multiple
    GPU-accelerated processing pipelines (CUDA, OpenCL, OpenGL) with automatic fallback
    to CPU processing.

    The Kinect v2 provides:
    - Color stream: 1920x1080 @ 30 FPS
    - Depth stream: 512x424 @ 30 FPS
    - IR stream: 512x424 @ 30 FPS

    Use the provided utility script to find available Kinect devices:
    ```bash
    python -m lerobot.find_cameras kinect
    ```

    Example:
        ```python
        from lerobot.cameras.kinect import KinectCamera, KinectCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with auto-detection
        config = KinectCameraConfig(device_index=0)
        camera = KinectCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)  # (1080, 1920, 3)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect
        camera.disconnect()

        # Example with depth capture and CUDA acceleration
        cuda_config = KinectCameraConfig(
            device_index=0,
            fps=30,
            color_mode=ColorMode.BGR,
            use_depth=True,
            use_ir=True,
            pipeline=KinectPipeline.CUDA,
            rotation=Cv2Rotation.NO_ROTATION
        )
        depth_camera = KinectCamera(cuda_config)
        depth_camera.connect()

        # Read depth and IR frames
        depth_map = depth_camera.read_depth()
        ir_image = depth_camera.read_ir()

        depth_camera.disconnect()
        ```
    """

    def __init__(self, config: KinectCameraConfig):
        """
        Initializes the KinectCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        if not KINECT_AVAILABLE:
            raise ImportError(
                "pylibfreenect2 is not installed. Please install it to use Kinect cameras. "
                "See: https://github.com/r9y9/pylibfreenect2"
            )

        super().__init__(config)

        self.config = config
        self.device_index = config.device_index
        self.serial_number = config.serial_number
        self.fps = config.fps or 30  # Kinect v2 default
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.use_ir = config.use_ir
        self.pipeline_type = config.pipeline
        self.warmup_s = config.warmup_s
        self.enable_bilateral_filter = config.enable_bilateral_filter
        self.enable_edge_filter = config.enable_edge_filter
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth

        # Kinect v2 fixed resolutions
        self.color_width = 1920
        self.color_height = 1080
        self.depth_width = 512
        self.depth_height = 424

        # Set output dimensions based on requested stream
        if config.width is None or config.height is None:
            self.width = self.color_width
            self.height = self.color_height
        else:
            self.width = config.width
            self.height = config.height

        # libfreenect2 objects
        self.fn2: Freenect2 | None = None
        self.device: Freenect2Device | None = None
        self.listener: SyncMultiFrameListener | None = None
        self.registration: Registration | None = None
        self.pipeline = None

        # Threading for async reads
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_ir: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # Handle rotation for capture dimensions
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        identifier = self.serial_number if self.serial_number else f"index_{self.device_index}"
        return f"{self.__class__.__name__}({identifier})"

    @property
    def is_connected(self) -> bool:
        """Checks if the Kinect device is connected and started."""
        return self.device is not None and self.listener is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the Kinect v2 camera specified in the configuration.

        Initializes the libfreenect2 pipeline, configures the required streams (color,
        depth, and/or IR), starts the device, and validates the actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If no Kinect devices are found or connection fails.
            RuntimeError: If the device starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Initialize Freenect2
        self.fn2 = Freenect2()

        # Check for available devices (may need to retry after previous instance release)
        num_devices = self.fn2.enumerateDevices()
        if num_devices == 0:
            # Wait a bit and try again (device might need time after previous release)
            time.sleep(0.5)
            num_devices = self.fn2.enumerateDevices()
            if num_devices == 0:
                raise ConnectionError(
                    "No Kinect v2 devices found. Ensure device is connected to USB 3.0 port."
                )

        # Select pipeline
        self.pipeline = self._create_pipeline()

        # Open device
        if self.serial_number:
            # Try with string first, then bytes if that fails
            serial_str = self.serial_number
            if isinstance(serial_str, bytes):
                serial_str = serial_str.decode("utf-8")

            try:
                self.device = self.fn2.openDevice(serial_str, pipeline=self.pipeline)
            except Exception:
                # Try with bytes if string failed
                try:
                    serial_bytes = (
                        serial_str.encode("utf-8") if isinstance(serial_str, str) else self.serial_number
                    )
                    self.device = self.fn2.openDevice(serial_bytes, pipeline=self.pipeline)
                except Exception:
                    # Fall back to index-based opening
                    if num_devices > 0:
                        self.device = self.fn2.openDevice(
                            self.fn2.getDeviceSerialNumber(0), pipeline=self.pipeline
                        )
                    else:
                        self.device = None
        elif self.device_index is not None:
            if self.device_index >= num_devices:
                raise ConnectionError(
                    f"Device index {self.device_index} not found. "
                    f"Only {num_devices} Kinect device(s) available."
                )
            device_serial = self.fn2.getDeviceSerialNumber(self.device_index)
            self.device = self.fn2.openDevice(device_serial, pipeline=self.pipeline)
        else:
            # Open first available device
            self.device = self.fn2.openDefaultDevice(pipeline=self.pipeline)

        if self.device is None:
            raise ConnectionError(f"Failed to open {self}. Ensure Kinect v2 is connected to a USB 3.0 port.")

        # Store actual serial number
        if not self.serial_number:
            self.serial_number = self.device.getSerialNumber()
            logger.debug(f"{self} serial number: {self.serial_number}")

        # Configure frame listener
        frame_types = FrameType.Color
        if self.use_depth:
            frame_types |= FrameType.Depth
            logger.debug(f"{self} depth stream enabled")
        if self.use_ir:
            frame_types |= FrameType.Ir
            logger.debug(f"{self} IR stream enabled")

        self.listener = SyncMultiFrameListener(frame_types)
        self.device.setColorFrameListener(self.listener)
        if self.use_depth or self.use_ir:
            self.device.setIrAndDepthFrameListener(self.listener)

        # Start device
        logger.debug(f"{self} starting device streams...")
        self.device.start()

        # Setup registration for depth alignment if needed
        if self.use_depth:
            self.registration = Registration(
                self.device.getIrCameraParams(), self.device.getColorCameraParams()
            )
            logger.debug(f"{self} depth registration configured")

        # Start async thread for better performance
        self._start_read_thread()

        if warmup:
            time.sleep(0.5)  # Initial warmup
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    # Use async_read during warmup to populate the buffer
                    self.async_read(timeout_ms=100)
                except Exception:
                    pass  # nosec
                time.sleep(0.1)

        logger.info(f"{self} connected with {self.pipeline.__class__.__name__}")

    def _create_pipeline(self):
        """Creates the appropriate processing pipeline based on configuration."""
        pipeline_type = self.pipeline_type

        if pipeline_type == KinectPipeline.AUTO:
            # Try pipelines in order of performance
            pipelines_to_try = [
                (KinectPipeline.CUDA, self._try_cuda_pipeline),
                (KinectPipeline.OPENCL, self._try_opencl_pipeline),
                (KinectPipeline.OPENGL, self._try_opengl_pipeline),
                (KinectPipeline.CPU, self._try_cpu_pipeline),
            ]

            for name, create_func in pipelines_to_try:
                pipeline = create_func()
                if pipeline is not None:
                    logger.info(f"Using {name.value} pipeline")
                    return pipeline

            # Should never reach here as CPU pipeline always works
            raise RuntimeError("Failed to create any pipeline")

        # Specific pipeline requested
        pipeline_creators = {
            KinectPipeline.CUDA: self._try_cuda_pipeline,
            KinectPipeline.OPENCL: self._try_opencl_pipeline,
            KinectPipeline.OPENGL: self._try_opengl_pipeline,
            KinectPipeline.CPU: self._try_cpu_pipeline,
        }

        pipeline = pipeline_creators[pipeline_type]()
        if pipeline is None:
            raise RuntimeError(f"Requested pipeline {pipeline_type.value} is not available")

        return pipeline

    def _try_cuda_pipeline(self):
        """Attempts to create CUDA pipeline if available."""
        try:
            if hasattr(pylibfreenect2, "CudaPacketPipeline"):
                return pylibfreenect2.CudaPacketPipeline()
        except Exception as e:
            logger.debug(f"CUDA pipeline not available: {e}")
        return None

    def _try_opencl_pipeline(self):
        """Attempts to create OpenCL pipeline if available."""
        try:
            if hasattr(pylibfreenect2, "OpenCLPacketPipeline"):
                return pylibfreenect2.OpenCLPacketPipeline()
        except Exception as e:
            logger.debug(f"OpenCL pipeline not available: {e}")
        return None

    def _try_opengl_pipeline(self):
        """Attempts to create OpenGL pipeline if available."""
        try:
            if hasattr(pylibfreenect2, "OpenGLPacketPipeline"):
                return pylibfreenect2.OpenGLPacketPipeline()
        except Exception as e:
            logger.debug(f"OpenGL pipeline not available: {e}")
        return None

    def _try_cpu_pipeline(self):
        """Creates CPU pipeline (always available)."""
        return pylibfreenect2.CpuPacketPipeline()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Kinect v2 cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing camera information.

        Raises:
            ImportError: If pylibfreenect2 is not installed.
        """
        if not KINECT_AVAILABLE:
            raise ImportError("pylibfreenect2 is not installed")

        found_cameras = []
        fn2 = Freenect2()
        try:
            num_devices = fn2.enumerateDevices()

            for i in range(num_devices):
                serial = fn2.getDeviceSerialNumber(i)
                # Decode serial number if it's bytes
                if isinstance(serial, bytes):
                    serial = serial.decode("utf-8")
                camera_info = {
                    "type": "Kinect v2",
                    "id": serial,
                    "index": i,
                    "name": f"Kinect v2 #{i}",
                    "serial_number": serial,
                    "color_resolution": "1920x1080",
                    "depth_resolution": "512x424",
                    "ir_resolution": "512x424",
                    "max_fps": 30,
                    "available_pipelines": [],
                }

                # Check available pipelines
                if hasattr(pylibfreenect2, "CudaPacketPipeline"):
                    camera_info["available_pipelines"].append("CUDA")
                if hasattr(pylibfreenect2, "OpenCLPacketPipeline"):
                    camera_info["available_pipelines"].append("OpenCL")
                if hasattr(pylibfreenect2, "OpenGLPacketPipeline"):
                    camera_info["available_pipelines"].append("OpenGL")
                camera_info["available_pipelines"].append("CPU")

                found_cameras.append(camera_info)
        finally:
            # Important: Clean up the Freenect2 instance to release USB device
            del fn2
            gc.collect()  # Force garbage collection to ensure device is released
            time.sleep(0.1)  # Give USB device time to reset

        return found_cameras

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single depth frame synchronously from the camera.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The depth map as a NumPy array (424, 512) of type float32
                        with values in millimeters, processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If depth stream is not enabled or reading fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.use_depth:
            raise RuntimeError(f"Failed to capture depth frame. Depth stream is not enabled for {self}.")

        frames = FrameMap()

        # Wait for new frame
        if not self.listener.waitForNewFrame(frames, timeout_ms):
            self.listener.release(frames)
            raise RuntimeError(f"{self} read_depth timeout after {timeout_ms}ms")

        try:
            depth_frame = frames[FrameType.Depth]
            depth_data = depth_frame.asarray().copy()

            # Apply depth filters
            if self.enable_bilateral_filter:
                depth_data = cv2.bilateralFilter(depth_data.astype(np.float32), 5, 50, 50)

            if self.enable_edge_filter:
                # Edge-aware filtering on normalized depth
                # Normalize to 0-255 range for filtering, then scale back
                depth_normalized = np.clip(depth_data / 8000.0 * 255, 0, 255).astype(np.uint8)
                depth_filtered = cv2.medianBlur(depth_normalized, 5)
                depth_data = depth_filtered.astype(np.float32) * 8000.0 / 255.0

            # Apply depth range limits
            depth_data = np.clip(depth_data, self.min_depth * 1000, self.max_depth * 1000)

            # Apply rotation if needed
            if self.rotation is not None:
                depth_data = cv2.rotate(depth_data, self.rotation)

            return depth_data

        finally:
            self.listener.release(frames)

    def read_ir(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single IR frame synchronously from the camera.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The IR image as a NumPy array (424, 512) of type float32.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If IR stream is not enabled or reading fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.use_ir:
            raise RuntimeError(f"Failed to capture IR frame. IR stream is not enabled for {self}.")

        frames = FrameMap()

        # Wait for new frame
        if not self.listener.waitForNewFrame(frames, timeout_ms):
            self.listener.release(frames)
            raise RuntimeError(f"{self} read_ir timeout after {timeout_ms}ms")

        try:
            ir_frame = frames[FrameType.Ir]
            ir_data = ir_frame.asarray().copy()

            # Apply rotation if needed
            if self.rotation is not None:
                ir_data = cv2.rotate(ir_data, self.rotation)

            return ir_data

        finally:
            self.listener.release(frames)

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single color frame synchronously from the camera.

        For performance, this method uses the async thread to get frames instead of
        blocking on waitForNewFrame. This matches the pattern used by other cameras
        in LeRobot and avoids the ~1 second delays seen with direct synchronous reads.

        Args:
            color_mode: Desired color mode for the output frame.
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The captured color frame as a NumPy array (1080, 1920, 3),
                        processed according to color_mode and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames fails.
            ValueError: If an invalid color_mode is requested.
        """
        # Use async_read for better performance - this is what all robots use anyway
        return self.async_read(timeout_ms=timeout_ms)

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.
        """
        while not self.stop_event.is_set():
            try:
                frames = FrameMap()
                # Use shorter timeout for more responsive shutdown
                if self.listener.waitForNewFrame(frames, 100):
                    try:
                        # Read color frame
                        color_frame = frames[FrameType.Color]
                        color_raw = color_frame.asarray()

                        # Convert BGRX to BGR using OpenCV
                        if color_raw.shape[2] == 4:
                            color_data = cv2.cvtColor(color_raw, cv2.COLOR_BGRA2BGR)
                        else:
                            color_data = color_raw.copy()

                        if self.color_mode == ColorMode.RGB:
                            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

                        if self.rotation is not None:
                            color_data = cv2.rotate(color_data, self.rotation)

                        # Read depth if enabled
                        depth_data = None
                        if self.use_depth:
                            depth_frame = frames[FrameType.Depth]
                            depth_data = depth_frame.asarray().copy()

                            if self.enable_bilateral_filter:
                                depth_data = cv2.bilateralFilter(depth_data.astype(np.float32), 5, 50, 50)

                            if self.enable_edge_filter:
                                # Edge-aware filtering on normalized depth
                                depth_normalized = np.clip(depth_data / 8000.0 * 255, 0, 255).astype(np.uint8)
                                depth_filtered = cv2.medianBlur(depth_normalized, 5)
                                depth_data = depth_filtered.astype(np.float32) * 8000.0 / 255.0

                            if self.rotation is not None:
                                depth_data = cv2.rotate(depth_data, self.rotation)

                        # Read IR if enabled
                        ir_data = None
                        if self.use_ir:
                            ir_frame = frames[FrameType.Ir]
                            ir_data = ir_frame.asarray().copy()

                            if self.rotation is not None:
                                ir_data = cv2.rotate(ir_data, self.rotation)

                        # Store frames thread-safely
                        with self.frame_lock:
                            self.latest_frame = color_data
                            self.latest_depth = depth_data
                            self.latest_ir = ir_data
                        self.new_frame_event.set()

                        # Frame stored successfully

                    finally:
                        self.listener.release(frames)

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        # If thread is already running, nothing to do
        if self.thread is not None and self.thread.is_alive():
            return

        # Clean up any previous thread
        if self.thread is not None:
            self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

        # Give thread time to start capturing
        time.sleep(0.1)

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available color frame asynchronously.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The latest captured color frame.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within timeout.
            RuntimeError: If the background thread died unexpectedly.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms}ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self):
        """
        Disconnects from the camera and releases all resources.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.device is not None:
            self.device.stop()
            self.device.close()
            self.device = None

        self.listener = None
        self.registration = None
        self.pipeline = None

        # Clean up Freenect2 instance
        if self.fn2 is not None:
            del self.fn2
            self.fn2 = None
            gc.collect()  # Force garbage collection

        logger.info(f"{self} disconnected.")
