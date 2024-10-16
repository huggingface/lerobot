import argparse
import concurrent.futures
import importlib
import shutil
import time
from pathlib import Path

import cv2
from PIL import Image

from lerobot.scripts.control_robot import busy_wait


def save_image(img_array, camera_index, frame_index, images_dir):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        print(f"Image saved to: {path}")
    except Exception as e:
        print(f"Failed to save image: {e}")


def save_images_from_cameras(
    driver: str,
    images_dir: Path,
    camera_ids: list[int] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given camera index.
    """
    # Dynamically import the appropriate camera class based on the brand
    if driver == "intelrealsense":
        camera_module = importlib.import_module("lerobot.common.robot_devices.cameras.intelrealsense")
        camera_class = camera_module.IntelRealSenseCamera
        find_camera_indices = camera_module.find_camera_indices
    elif driver == "opencv":
        camera_module = importlib.import_module("lerobot.common.robot_devices.cameras.opencv")
        camera_class = camera_module.OpenCVCamera
        find_camera_indices = camera_module.find_camera_indices
    else:
        raise ValueError(
            f"Unsupported camera driver: {driver}. Note: the drivers we currently support are opencv and intelrealsense."
        )

    if camera_ids is None:
        camera_ids = find_camera_indices()

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        camera = camera_class(cam_idx, fps=fps, width=width, height=height)
        camera.connect()
        print(
            f"{camera.__class__.__name__}({camera.camera_index}, fps={camera.fps}, width={camera.width}, height={camera.height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()

    # Use ThreadPoolExecutor for saving images asynchronously
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        try:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # Capture image
                    image = camera.read() if fps is None else camera.async_read()
                    if image is None:
                        print("No Frame")
                    else:
                        bgr_converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # Submit the save_image function to be executed in the background
                        executor.submit(
                            save_image,
                            bgr_converted_image,
                            camera.camera_index,
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
        finally:
            print(f"Images have been saved to {images_dir}")
            for camera in cameras:
                camera.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--driver", type=str, required=True, help="Camera driver (e.g., intelrealsense, opencv)"
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `IntelRealSenseCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per second for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
