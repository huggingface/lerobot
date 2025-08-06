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


class DepthColorizer:
    """Converts depth data to RGB using OpenCV colormaps."""

    COLORMAP_MAPPING = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "bone": cv2.COLORMAP_BONE,
    }

    def __init__(
        self, colormap: str = "jet", min_depth_m: float = 0.5, max_depth_m: float = 4.5, clipping: bool = True
    ):
        """
        Initialize the depth colorizer.

        Args:
            colormap: Name of the colormap to use
            min_depth_m: Minimum depth in meters for normalization
            max_depth_m: Maximum depth in meters for normalization
            clipping: Whether to clip values outside the min/max range
        """
        self.colormap = self.COLORMAP_MAPPING.get(colormap, cv2.COLORMAP_JET)
        self.min_depth_mm = min_depth_m * 1000
        self.max_depth_mm = max_depth_m * 1000
        self.clipping = clipping

    def colorize(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Convert depth data to RGB using the configured colormap.

        Args:
            depth_data: Depth map in millimeters as float32

        Returns:
            RGB image as uint8 with shape (H, W, 3)
        """
        # Handle empty or invalid data
        if depth_data is None or depth_data.size == 0:
            return np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)

        # Create a copy to avoid modifying the original
        depth_normalized = depth_data.copy()

        if self.clipping:
            # Clip to the valid range
            depth_normalized = np.clip(depth_normalized, self.min_depth_mm, self.max_depth_mm)

        # Normalize to 0-255 range
        # Handle case where min and max are the same
        depth_range = self.max_depth_mm - self.min_depth_mm
        if depth_range > 0:
            depth_normalized = (depth_normalized - self.min_depth_mm) / depth_range * 255
        else:
            depth_normalized = np.zeros_like(depth_normalized)

        # Convert to uint8
        depth_uint8 = depth_normalized.astype(np.uint8)

        # Apply colormap
        depth_colorized = cv2.applyColorMap(depth_uint8, self.colormap)

        # OpenCV returns BGR, but we want to maintain consistency with the color_mode
        # The camera class will handle the color mode conversion if needed
        return depth_colorized


class KinectCamera(Camera):
    """
    Manages interactions with Microsoft Kinect v2 cameras for frame, depth, and IR recording.

    This class provides an interface similar to `RealSenseCamera` but tailored for
    Kinect v2 devices, leveraging the `pylibfreenect2` library. It supports multiple
    GPU-accelerated processing pipelines (CUDA, OpenCL, OpenGL) with automatic fallback
    to CPU processing.

    The Kinect v2 provides:
    - Color stream: 1920x1080 @ 30 FPS
    - Depth stream: 512x424 @ 30 FPS (with colorization support)
    - IR stream: 512x424 @ 30 FPS

    Key features:
    - Depth colorization: Converts depth data to RGB using configurable colormaps
    - Multi-stream capture: Read color and colorized depth simultaneously
    - GPU acceleration: Automatic pipeline selection for optimal performance

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

        # Example with colorized depth capture
        depth_config = KinectCameraConfig(
            device_index=0,
            fps=30,
            color_mode=ColorMode.RGB,
            use_depth=True,
            depth_colormap="jet",
            depth_min_meters=0.5,
            depth_max_meters=4.0,
            pipeline=KinectPipeline.CUDA,
            rotation=Cv2Rotation.NO_ROTATION
        )
        depth_camera = KinectCamera(depth_config)
        depth_camera.connect()

        # Read all streams at once
        frames = depth_camera.async_read_all()
        color = frames["color"]          # (1080, 1920, 3)
        depth_rgb = frames["depth_rgb"]  # (424, 512, 3) colorized

        # Or read individually
        depth_colorized = depth_camera.read_depth_rgb()
        raw_depth = depth_camera.read_depth()  # Raw depth in mm
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
        self.latest_depth_rgb: np.ndarray | None = None  # Colorized depth
        self.latest_ir: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        # Initialize depth colorizer if depth is enabled
        self.depth_colorizer: DepthColorizer | None = None
        if self.use_depth:
            self.depth_colorizer = DepthColorizer(
                colormap=config.depth_colormap,
                min_depth_m=config.depth_min_meters,
                max_depth_m=config.depth_max_meters,
                clipping=config.depth_clipping,
            )

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

        # Configure frame listener
        frame_types = FrameType.Color
        if self.use_depth:
            frame_types |= FrameType.Depth
        if self.use_ir:
            frame_types |= FrameType.Ir

        self.listener = SyncMultiFrameListener(frame_types)
        self.device.setColorFrameListener(self.listener)
        if self.use_depth or self.use_ir:
            self.device.setIrAndDepthFrameListener(self.listener)

        # Start device
        self.device.start()

        # Setup registration for depth alignment if needed
        if self.use_depth:
            self.registration = Registration(
                self.device.getIrCameraParams(), self.device.getColorCameraParams()
            )

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
        except Exception:
            pass
        return None

    def _try_opencl_pipeline(self):
        """Attempts to create OpenCL pipeline if available."""
        try:
            if hasattr(pylibfreenect2, "OpenCLPacketPipeline"):
                return pylibfreenect2.OpenCLPacketPipeline()
        except Exception:
            pass
        return None

    def _try_opengl_pipeline(self):
        """Attempts to create OpenGL pipeline if available."""
        try:
            if hasattr(pylibfreenect2, "OpenGLPacketPipeline"):
                return pylibfreenect2.OpenGLPacketPipeline()
        except Exception:
            pass
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

        # Pre-allocate buffer for optimized reading
        depth_buffer = np.empty((424, 512), dtype=np.float32)

        # Wait for new frame
        if not self.listener.waitForNewFrame(frames, timeout_ms):
            self.listener.release(frames)
            raise RuntimeError(f"{self} read_depth timeout after {timeout_ms}ms")

        try:
            depth_frame = frames[FrameType.Depth]

            # Use optimized method if available
            if hasattr(depth_frame, "asarray_optimized"):
                depth_data = depth_frame.asarray_optimized(depth_buffer)
            else:
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

    def read_depth_rgb(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a colorized depth frame as RGB.

        This method reads a depth frame and converts it to an RGB image using
        the configured colormap. The depth values are normalized and mapped to
        colors for visualization.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: Colorized depth as RGB image with shape (height, width, 3)
                        in the configured color mode.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If depth stream is not enabled or reading fails.

        Example:
            ```python
            config = KinectCameraConfig(device_index=0, use_depth=True, depth_colormap="jet")
            camera = KinectCamera(config)
            camera.connect()

            # Read colorized depth
            depth_rgb = camera.read_depth_rgb()
            print(depth_rgb.shape)  # (424, 512, 3)

            camera.disconnect()
            ```
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.use_depth:
            raise RuntimeError(f"Failed to capture depth frame. Depth stream is not enabled for {self}.")

        if self.depth_colorizer is None:
            raise RuntimeError(f"Depth colorizer not initialized for {self}.")

        # Get raw depth frame
        depth_data = self.read_depth(timeout_ms)

        # Colorize the depth data
        depth_rgb = self.depth_colorizer.colorize(depth_data)

        # Convert BGR to RGB if needed (OpenCV colormaps return BGR)
        if self.color_mode == ColorMode.RGB:
            depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)

        return depth_rgb

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
        # Pre-allocate buffers for zero-copy operation
        color_buffer = np.empty((1080, 1920, 3), dtype=np.uint8)
        depth_buffer = np.empty((424, 512), dtype=np.float32)

        while not self.stop_event.is_set():
            try:
                frames = FrameMap()
                # Use shorter timeout for more responsive shutdown
                if self.listener.waitForNewFrame(frames, 100):
                    try:
                        # Read color frame using optimized method
                        color_frame = frames[FrameType.Color]

                        # Check if asarray_optimized is available
                        if hasattr(color_frame, "asarray_optimized"):
                            # Use optimized method that does BGRX->BGR conversion internally
                            color_data = color_frame.asarray_optimized(color_buffer)
                        else:
                            # Fallback to old method
                            color_raw = color_frame.asarray()
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
                        depth_rgb = None
                        if self.use_depth:
                            depth_frame = frames[FrameType.Depth]

                            # Use optimized method if available
                            if hasattr(depth_frame, "asarray_optimized"):
                                depth_data = depth_frame.asarray_optimized(depth_buffer)
                            else:
                                depth_data = depth_frame.asarray().copy()

                            if self.enable_bilateral_filter:
                                depth_data = cv2.bilateralFilter(depth_data.astype(np.float32), 5, 50, 50)

                            if self.enable_edge_filter:
                                # Edge-aware filtering on normalized depth
                                depth_normalized = np.clip(depth_data / 8000.0 * 255, 0, 255).astype(np.uint8)
                                depth_filtered = cv2.medianBlur(depth_normalized, 5)
                                depth_data = depth_filtered.astype(np.float32) * 8000.0 / 255.0

                            # Apply depth range limits before colorization
                            depth_data = np.clip(depth_data, self.min_depth * 1000, self.max_depth * 1000)

                            # Colorize the depth data
                            if self.depth_colorizer is not None:
                                depth_rgb = self.depth_colorizer.colorize(depth_data)

                                # Convert BGR to RGB if needed (OpenCV colormaps return BGR)
                                if self.color_mode == ColorMode.RGB:
                                    depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)

                            if self.rotation is not None:
                                depth_data = cv2.rotate(depth_data, self.rotation)
                                if depth_rgb is not None:
                                    depth_rgb = cv2.rotate(depth_rgb, self.rotation)

                        # Read IR if enabled
                        ir_data = None
                        if self.use_ir:
                            ir_frame = frames[FrameType.Ir]

                            # Pre-allocate IR buffer if needed
                            if not hasattr(self, "_ir_buffer"):
                                self._ir_buffer = np.empty((424, 512), dtype=np.float32)

                            # Use optimized method if available
                            if hasattr(ir_frame, "asarray_optimized"):
                                ir_data = ir_frame.asarray_optimized(self._ir_buffer)
                            else:
                                ir_data = ir_frame.asarray().copy()

                            if self.rotation is not None:
                                ir_data = cv2.rotate(ir_data, self.rotation)

                        # Store frames thread-safely
                        with self.frame_lock:
                            self.latest_frame = color_data
                            self.latest_depth = depth_data
                            self.latest_depth_rgb = depth_rgb
                            self.latest_ir = ir_data
                        self.new_frame_event.set()

                        # Frame stored successfully

                    finally:
                        self.listener.release(frames)

            except DeviceNotConnectedError:
                break
            except Exception:
                pass  # Silently ignore frame read errors

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
        Reads the latest available frame asynchronously.

        When use_depth=True, returns colorized depth. Otherwise returns color frame.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The latest captured frame. Returns colorized depth if use_depth=True,
                       otherwise returns color frame.

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
            # Always return color frame, depth is accessed via async_read_all()
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def async_read_all(self, timeout_ms: float = 200) -> dict[str, np.ndarray]:
        """
        Reads all enabled streams asynchronously.

        This method retrieves the most recent frames for all enabled streams (color,
        colorized depth) captured by the background thread. It's more efficient than
        calling individual read methods when multiple streams are needed.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for frames.

        Returns:
            dict: Dictionary containing available streams:
                - "color": RGB/BGR color frame (always present)
                - "depth_rgb": Colorized depth as RGB (if use_depth=True)

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frames become available within timeout.
            RuntimeError: If the background thread died unexpectedly.

        Example:
            ```python
            config = KinectCameraConfig(device_index=0, use_depth=True)
            camera = KinectCamera(config)
            camera.connect()

            # Read all streams at once
            frames = camera.async_read_all()
            color = frames["color"]  # (1080, 1920, 3)
            if "depth_rgb" in frames:
                depth_colored = frames["depth_rgb"]  # (424, 512, 3)

            camera.disconnect()
            ```
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frames from {self} after {timeout_ms}ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            # Always include color frame
            frames = {"color": self.latest_frame}

            # Add colorized depth if available
            if self.use_depth and self.latest_depth_rgb is not None:
                frames["depth_rgb"] = self.latest_depth_rgb

            self.new_frame_event.clear()

        # Validate we have at least the color frame
        if frames["color"] is None:
            raise RuntimeError(f"Internal error: Event set but no color frame available for {self}.")

        return frames

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
