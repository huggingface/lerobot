from lerobot.cameras.reachy2_camera import Reachy2Camera, Reachy2CameraConfig
import time


camera_config = Reachy2CameraConfig(name="teleop", image_type="left")
camera = Reachy2Camera(camera_config)

camera.connect()

frame = camera.read()
print(frame)

frame_async = camera.async_read()
print(frame_async)
