import cv2

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Working camera pair: Camera 0 (Microsoft Lifecam - Front) + Camera 1 (Lenovo - Top)
camera_config_front = OpenCVCameraConfig(
    index_or_path=0,
    fps=30.0,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
)
camera_config_top = OpenCVCameraConfig(
    index_or_path=1,
    fps=30.0,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
)

# Instantiate and connect cameras
camera_front = OpenCVCamera(camera_config_front)
camera_top = OpenCVCamera(camera_config_top)
camera_front.connect()
camera_top.connect()

# Create named windows for display
cv2.namedWindow("Front Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Top Camera", cv2.WINDOW_NORMAL)

# Read and display frames asynchronously
print("Streaming cameras... Press 'q' to quit")
try:
    frame_count = 0
    while True:
        # Read frames from both cameras
        frame_front = camera_front.async_read(timeout_ms=500)
        frame_top = camera_top.async_read(timeout_ms=500)

        # Convert RGB to BGR for OpenCV display
        frame_front_bgr = cv2.cvtColor(frame_front, cv2.COLOR_RGB2BGR)
        frame_top_bgr = cv2.cvtColor(frame_top, cv2.COLOR_RGB2BGR)

        # Display frames in windows
        cv2.imshow("Front Camera", frame_front_bgr)
        cv2.imshow("Top Camera", frame_top_bgr)

        # Print frame info periodically
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} - Front: {frame_front.shape}, Top: {frame_top.shape}")
        frame_count += 1

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

finally:
    camera_front.disconnect()
    camera_top.disconnect()
    cv2.destroyAllWindows()
