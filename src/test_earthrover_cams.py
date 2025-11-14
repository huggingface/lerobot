import sys
import cv2
from lerobot.cameras.earthrover_mini_camera import EarthRoverMiniCamera
from lerobot.cameras.earthrover_mini_camera.configuration_earthrover_mini import EarthRoverMiniCameraConfig, ColorMode

# EXAMPLE TESTING FILE FOR EARTHROVER CAMERAS

front_main_config = EarthRoverMiniCameraConfig(
    index_or_path=EarthRoverMiniCameraConfig.FRONT_CAM_MAIN,  # front main stream
    color_mode=ColorMode.RGB
)

front_sub_config = EarthRoverMiniCameraConfig(
    index_or_path=EarthRoverMiniCameraConfig.FRONT_CAM_SUB,  # front sub stream
    color_mode=ColorMode.RGB
)

rear_main_config = EarthRoverMiniCameraConfig(
    index_or_path=EarthRoverMiniCameraConfig.REAR_CAM_MAIN,  # rear main stream
    color_mode=ColorMode.RGB
)

rear_sub_config = EarthRoverMiniCameraConfig(
    index_or_path=EarthRoverMiniCameraConfig.REAR_CAM_SUB,  # rear sub stream
    color_mode=ColorMode.RGB
)

config_list = [front_main_config, front_sub_config, rear_main_config, rear_sub_config]
print(config_list)
# Create all cameras
cameras = [EarthRoverMiniCamera(cfg) for cfg in config_list]
print(f"cameras item = {cameras[0]}")

# Connect to all cameras
for cam in cameras:
    print(f"Connecting to camera {cam.config.index_or_path}...")
    cam.connect()
    if cam.is_connected:
        print(f"{cam.config.index_or_path} connected successfully!")
    else:
        print(f"Failed to connect to {cam.config.index_or_path}. Exiting...")
        sys.exit(1)

# Read frames from cameras
try:
    while True:
        for idx, cam in enumerate(cameras):
            frame = cam.read()
            cv2.imshow(f"RTSP Stream {idx}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    for cam in cameras:
        cam.disconnect()
    cv2.destroyAllWindows()
