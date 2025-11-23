import cv2
from pathlib import Path

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Working camera pair: Camera 0 (Microsoft Lifecam - Front) + Camera 1 (Lenovo - Top)
camera_config_front = OpenCVCameraConfig(
    index_or_path='/dev/video0',
    fps=30.0,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG",
)
camera_config_top = OpenCVCameraConfig(
    index_or_path='/dev/video2',
    fps=30.0,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="MJPG",
)

# Create output directory
output_dir = Path("outputs/captured_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Instantiate and connect cameras
camera_front = OpenCVCamera(camera_config_front)
camera_top = OpenCVCamera(camera_config_top)
camera_front.connect()
camera_top.connect()

# Read and save frames
print("Streaming cameras... Press 'q' to quit or Ctrl+C")
print(f"Saving frames to {output_dir}")
try:
    frame_count = 0
    while True:
        # Read frames from both cameras
        frame_front = camera_front.async_read(timeout_ms=500)
        frame_top = camera_top.async_read(timeout_ms=500)

        # Convert RGB to BGR for OpenCV saving and display
        frame_front_bgr = cv2.cvtColor(frame_front, cv2.COLOR_RGB2BGR)
        frame_top_bgr = cv2.cvtColor(frame_top, cv2.COLOR_RGB2BGR)

        # Display frames in real-time windows
        cv2.imshow('Front Camera', frame_front_bgr)
        cv2.imshow('Top Camera', frame_top_bgr)

        # Save frames periodically (every 30 frames)
        if frame_count % 30 == 0:
            front_path = output_dir / f"front_frame_{frame_count:06d}.png"
            top_path = output_dir / f"top_frame_{frame_count:06d}.png"
            cv2.imwrite(str(front_path), frame_front_bgr)
            cv2.imwrite(str(top_path), frame_top_bgr)
            print(f"Frame {frame_count} - Front: {frame_front.shape}, Top: {frame_top.shape} - Saved")
        
        frame_count += 1

        # Wait 1ms and check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    camera_front.disconnect()
    camera_top.disconnect()
    cv2.destroyAllWindows()
    print(f"Captured {frame_count} frames total")
