

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread
import time
import traceback
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


# from lerobot.common.robot_devices.cameras.opencv import find_camera_indices
from lerobot.common.robot_devices.cameras.utils import save_color_image, save_depth_image

SERIAL_NUMBER_INDEX = 1


def find_camera_indices(raise_when_empty=True):
    camera_ids = []
    for device in rs.context().query_devices():
        serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
        camera_ids.append(serial_number)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError("Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware.")

    return camera_ids


def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path, camera_ids: list[int] | None = None, fps=None, width=None, height=None, record_time_s=2
):
    if camera_ids is None:
        camera_ids = find_camera_indices()

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        camera = OpenCVCamera(cam_idx, fps=fps, width=width, height=height)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.width}, height={camera.height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    camera.index,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            if time.perf_counter() - start_time > record_time_s:
                break

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            frame_index += 1

    print(f"Images have been saved to {images_dir}")

def benchmark_cameras(cameras, out_dir=None, save_images=False):
    
    if save_images:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        now = time.time()
        for camera in cameras:
            print(f"Camera is detected with serial number {camera.camera_index}")
            camera.connect()
            if camera.use_depth:
                color_image, depth_image = camera.capture_image("bgr" if save_images else "rgb")
            else:
                color_image = camera.capture_image("bgr" if save_images else "rgb")

            if save_images:
                image_path = out_dir / f"camera_{camera.camera_index:02}.png"
                print(f"Write to {image_path}")
                save_color_image(color_image, image_path, write_shape=True)

                if camera.use_depth:
                    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                    depth_image_path = out_dir / f"camera_{camera.camera_index:02}_depth.png"
                    print(f"Write to {depth_image_path}")
                    save_depth_image(depth_image_path, depth_image, write_shape=True)
            
            # camera.disconnect()

        dt_s = (time.time() - now)
        dt_ms = dt_s * 1000
        freq = 1 / dt_s
        print(f"Latency (ms): {dt_ms:.2f}\tFrequency: {freq:.2f}")

        if save_images:
            print("Images Saved")
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
    AVAILABLE_CAMERA_INDICES = find_camera_indices()

    def __init__(self,
            camera_index: int | None = None,
            config: IntelRealSenseCameraConfig | None = None,
            **kwargs,
        ):
        if config is None:
            config = IntelRealSenseCameraConfig()
        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)

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

        if self.camera_index is None:
            raise ValueError(f"`camera_index` is expected to be a serial number of one of these available cameras ({IntelRealSenseCamera.AVAILABLE_CAMERA_INDICES}), but {camera_index} is provided instead.")

        self.camera = None
        self.is_connected = False

        self.t = Thread(target=self.capture_image_loop, args=())
        self.t.daemon = True
        self._color_image = None

    def connect(self):
        if self.is_connected:
            raise ValueError(f"Camera {self.camera_index} is already connected.")

        config = rs.config()
        config.enable_device(str(self.camera_index))

        if self.fps and self.width and self.height:
            # TODO(rcadene): can we set rgb8 directly?
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
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
            if self.camera_index not in IntelRealSenseCamera.AVAILABLE_CAMERA_INDICES:
                raise ValueError(f"`camera_index` is expected to be a serial number of one of these available cameras {IntelRealSenseCamera.AVAILABLE_CAMERA_INDICES}, but {self.camera_index} is provided instead.")
            traceback.print_exc()

        self.is_connected = True
        try:
            self.t.start()
        except Exception as e:
            print(f"Error starting thread: {e}")
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

    def capture_image_loop(self):
        print("Capturing Image")
        while True:
            try:
                self._color_image = self.capture_image()
            except Exception as e:
                print(f"Error in capture_image_loop: {e}")
                traceback.print_exc()
                break                                                                                                       

    def read(self):
        while self._color_image is None:
            time.sleep(0.1)
        return self._color_image

    def disconnect(self):

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )
        
        if getattr(self, "camera", None):
            try:
                self.camera.stop()
            except RuntimeError as e:
                if "stop() cannot be called before start()" in str(e):
                    # skip this runtime error
                    return
                traceback.print_exc()

    def __del__(self):
        print("Delete")
        self.disconnect()


def save_images_config(config, out_dir: Path):
    camera_ids = IntelRealSenseCamera.AVAILABLE_CAMERA_INDICES
    cameras = []
    print(f"Available camera indices: {camera_ids}")
    for camera_idx in camera_ids:
        camera = IntelRealSenseCamera(camera_idx, config)
        cameras.append(camera)

    out_dir = out_dir.parent / f"{out_dir.name}_{config.width}x{config.height}_{config.fps}_depth_{config.use_depth}"
    benchmark_cameras(cameras, out_dir, save_images=True)
    print("Save Image Done")

def benchmark_config(config, camera_ids: list[int]):
    cameras = [IntelRealSenseCamera(idx, config) for idx in camera_ids]
    benchmark_cameras(cameras)


if __name__ == "__main__":          
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["save_images", 'benchmark'], default="save_images")
    parser.add_argument("--camera-ids", type=int, nargs="*", default=[218622272670, 128422271347])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=str, default=640)
    parser.add_argument("--height", type=str, default=480)
    parser.add_argument("--use-depth", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default="outputs/benchmark_cameras/intelrealsense/2024_06_22_1738")
    args = parser.parse_args()

    config = IntelRealSenseCameraConfig(args.fps, args.width, args.height, use_depth=bool(args.use_depth))
    # config = IntelRealSenseCameraConfig()
    # config = IntelRealSenseCameraConfig(60, 640, 480)
    # config = IntelRealSenseCameraConfig(90, 640, 480)
    # config = IntelRealSenseCameraConfig(30, 1280, 720)

    if args.mode == "save_images":
        save_images_config(config, args.out_dir)
    elif args.mode == "benchmark":
        benchmark_config(config, args.camera_ids)
    else:
        raise ValueError(args.mode)
    
    print("Program Ended")
    

# if __name__ == "__main__":
#     # Works well!
#     # use_depth = False
#     # fps = 90
#     # width = 640
#     # height = 480

#     # # Works well!
#     # use_depth = True
#     # fps = 90
#     # width = 640
#     # height = 480

#     # # Doesn't work well, latency varies too much
#     # use_depth = True
#     # fps = 30
#     # width = 1280
#     # height = 720

#     # Works well
#     use_depth = False
#     fps = 30
#     width = 1280
#     height = 720

#     config = IntelRealSenseCameraConfig()
#     # config = IntelRealSenseCameraConfig(fps, width, height, use_depth=use_depth)
#     cameras = [
#         # IntelRealSenseCamera(0, config),
#         # IntelRealSenseCamera(128422270109, config),
#         IntelRealSenseCamera(128422271609, config),
#         IntelRealSenseCamera(128422271614, config),
#         IntelRealSenseCamera(128422271393, config),
#     ]

#     out_dir = "outputs/benchmark_cameras/intelrealsense/2024_06_22_1729"
#     out_dir += f"{config.width}x{config.height}_{config.fps}_depth_{config.use_depth}"
#     benchmark_cameras(cameras, out_dir, save_images=False)
