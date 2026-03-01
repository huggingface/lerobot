import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# First camera
config1 = OpenCVCameraConfig(
    index_or_path=0,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,            # good for ML
    rotation=Cv2Rotation.NO_ROTATION
)
camera1 = OpenCVCamera(config1)
camera1.connect()

# Second camera
config2 = OpenCVCameraConfig(
    index_or_path=3,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,            # also RGB
    rotation=Cv2Rotation.ROTATE_180
)
camera2 = OpenCVCamera(config2)
camera2.connect()

try:
    while True:
        # Grab latest available frames (non-blocking)
        frame1 = camera1.async_read()
        frame2 = camera2.async_read()

        if frame1 is not None:
            # convert for OpenCV display (expects BGR)
            cv2.imshow("Camera 0", cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))

        if frame2 is not None:
            cv2.imshow("Camera 3", cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))

        # Needed for display refresh + quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera1.disconnect()
    camera2.disconnect()
    cv2.destroyAllWindows()
