

from dataclasses import dataclass
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.common.robot_devices.cameras.opencv import find_camera_indices
from lerobot.common.robot_devices.cameras.utils import save_color_image, save_depth_image, write_shape

SERIAL_NUMBER_INDEX = 1

def find_camera_indices(raise_when_empty):
    camera_ids = []
    for device in rs.context().query_devices():
        serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
        camera_ids.append(serial_number)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError("Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware.")

    return camera_ids

def benchmark_cameras(cameras, out_dir, save_images=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        now = time.time()
        for camera in cameras:
            if camera.use_depth:
                color_image, depth_image = camera.capture_image()
            else:
                color_image = camera.capture_image()

            if save_images:
                image_path = out_dir / f"camera_{camera.camera_index:02}.png"
                print(f"Write to {image_path}")
                save_color_image(color_image, image_path, write_shape=True)

                if camera.use_depth:
                    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                    depth_image_path = out_dir / f"camera_{camera.camera_index:02}_depth.png"
                    print(f"Write to {depth_image_path}")
                    save_depth_image(depth_image_path, depth_image, write_shape=True)

        dt_s = (time.time() - now)
        dt_ms = dt_s * 1000
        freq = 1 / dt_s
        print(f"Latency (ms): {dt_ms:.2f}\tFrequency: {freq:.2f}")

        if save_images:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Pre-defined configs that worked

@dataclass
class IntelRealSenseCameraConfig:
    """
    Example of tested options for Intel Real Sense D405:
    
    ```python
    IntelRealSenseCameraConfig(30, 640, 480)
    IntelRealSenseCameraConfig(60, 640, 480)
    IntelRealSenseCameraConfig(90, 640, 480)
    IntelRealSenseCameraConfig(30, 1280, 720)

    IntelRealSenseCameraConfig(30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(60, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(90, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(30, 1280, 720, use_depth=True)
    ```
    """
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color: str = "rgb"
    use_depth: bool = False
    force_hardware_reset: bool = True



class IntelRealSenseCamera():
    # TODO(rcadene): improve dosctring
    """
    Using this class requires:
    - [installing `librealsense` and its python wrapper `pyrealsense2`](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
    - [updating the camera(s) firmware](https://dev.intelrealsense.com/docs/firmware-releases-d400)

    Example of getting the `camera_index` for your camera(s):
    ```bash
    rs-fw-update -l

    > Connected devices:
    > 1) [USB] Intel RealSense D405 s/n 128422270109, update serial number: 133323070634, firmware version: 5.16.0.1
    > 2) [USB] Intel RealSense D405 s/n 128422271609, update serial number: 130523070758, firmware version: 5.16.0.1
    > 3) [USB] Intel RealSense D405 s/n 128422271614, update serial number: 133323070576, firmware version: 5.16.0.1
    > 4) [USB] Intel RealSense D405 s/n 128422271393, update serial number: 133323070271, firmware version: 5.16.0.1
    ```

    Example of uage:

    ```python
    camera = IntelRealSenseCamera(128422270109)  # serial number (s/n)
    color_image = camera.capture_image()
    ```

    Example of capturing additional depth image:

    ```python
    config = IntelRealSenseCameraConfig(use_depth=True)
    camera = IntelRealSenseCamera(128422270109, config)
    color_image, depth_image = camera.capture_image()
    ```
    """

    def __init__(self,
            camera_index: int | None = None,
            config: IntelRealSenseCameraConfig | None = None,
        ):
        if config is None:
            config = IntelRealSenseCameraConfig()
        self.camera_index = camera_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color = config.color
        self.use_depth = config.use_depth
        self.force_hardware_reset = config.force_hardware_reset

        # TODO(rcadene): move these two check in config dataclass
        if self.color not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {self.color} is provided.")

        if (self.fps or self.width or self.height) and not (self.fps and self.width and self.height):
            raise ValueError(f"Expected all fps, width and height to be set, when one of them is set, but {self.fps=}, {self.width=}, {self.height=}.")

        if camera_index is None:
            available_camera_indices = find_camera_indices(raise_when_empty=True)
            raise ValueError(f"`camera_index` is expected to be a serial number of one of these available cameras ({available_camera_indices}), but {camera_index} is provided instead.")

        config = rs.config()
        config.enable_device(str(self.camera_index))

        if self.fps and self.width and self.height:
            # TODO(rcadene): can we set rgb8 directly?
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        else:
            config.enable_stream(rs.stream.color)

        if self.use_depth:
            if self.fps and self.width and self.height:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            else:
                config.enable_stream(rs.stream.depth)

        self.camera = rs.pipeline()
        try:
            self.camera.start(config)
        except RuntimeError:
            # Verify that the provided `camera_index` is valid before printing the traceback
            available_camera_indices = find_camera_indices(raise_when_empty=True)
            if camera_index not in available_camera_indices:
                raise ValueError(f"`camera_index` is expected to be a serial number of one of these available cameras {available_camera_indices}, but {camera_index} is provided instead.")
            traceback.print_exc()


    def capture_image(self, temporary_color: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        frame = self.camera.wait_for_frames()

        color_frame = frame.get_color_frame()
        if not color_frame:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")
        color_image = np.asanyarray(color_frame.get_data())

        if temporary_color is None:
            requested_color = self.color
        else:
            requested_color = temporary_color
        
        if requested_color not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {requested_color} is provided.")
        
        # OpenCV uses BGR format as default (blue, green red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if self.use_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(f"Can't capture depth image from camera {self.camera_index}.")
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image
        else:
            return color_image

    def disconnect(self):
        if getattr(self, "camera", None):
            try:
                self.camera.stop()
            except RuntimeError as e:
                if "stop() cannot be called before start()" in str(e):
                    # skip this runtime error
                    return
                traceback.print_exc()


    def __del__(self):
        self.disconnect()


if __name__ == "__main__":
    # Works well!
    # use_depth = False
    # fps = 90
    # width = 640
    # height = 480

    # # Works well!
    # use_depth = True
    # fps = 90
    # width = 640
    # height = 480

    # # Doesn't work well, latency varies too much
    # use_depth = True
    # fps = 30
    # width = 1280
    # height = 720

    # Works well
    use_depth = False
    fps = 30
    width = 1280
    height = 720

    config = IntelRealSenseCameraConfig640x480Fps30()
    # config = IntelRealSenseCameraConfig(fps, width, height, use_depth=use_depth)
    cameras = [
        IntelRealSenseCamera(0, config),
        # IntelRealSenseCamera(128422270109, config),
        # IntelRealSenseCamera(128422271609, config),
        # IntelRealSenseCamera(128422271614, config),
        # IntelRealSenseCamera(128422271393, config),
    ]

    out_dir = "outputs/benchmark_cameras/intelrealsense/"
    out_dir += f"{config.width}x{config.height}_{config.fps}_depth_{config.use_depth}"
    benchmark_cameras(cameras, out_dir, save_images=False)
