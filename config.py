from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

config = OpenCVCameraConfig(
    index_or_path=2,      
    fps=30,
    width=1280,
    height=720,
    color_mode=ColorMode.BGR,
    fourcc="MJPG",          
    rotation=Cv2Rotation.NO_ROTATION
)

camera = OpenCVCamera(config)
camera.connect()

try:
    frame = camera.async_read(timeout_ms=1000)
    print(frame.shape)
finally:
    camera.disconnect()
