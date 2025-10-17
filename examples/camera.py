from datetime import datetime
from pathlib import Path

from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
camera = OpenCVCamera(config)
camera.connect()

output_dir = Path("./outputs/captured_images")
output_dir.mkdir(parents=True, exist_ok=True)
session_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

# Read frames asynchronously in a loop via `async_read(timeout_ms)`
try:
    for i in range(10):
        frame = camera.async_read(timeout_ms=200)

        if frame is None:
            print(f"Async frame {i} could not be read; skipping")
            continue

        image_path = output_dir / f"{session_prefix}_frame_{i:03d}.png"
        Image.fromarray(frame).save(image_path)
        print(f"Async frame {i} shape: {frame.shape}, saved to {image_path}")
finally:
    camera.disconnect()
    