import sys
import cv2
from dataclasses import dataclass, field
import time

from lerobot.robots.robot import Robot
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.robots.earthrover_mini_plus import (EarthRoverMiniPlusConfig, EarthRover_Mini)
from lerobot.cameras.earthrover_mini_camera import EarthRoverMiniCamera
from lerobot.cameras.earthrover_mini_camera.configuration_earthrover_mini import EarthRoverMiniCameraConfig, ColorMode
# EXAMPLE TESTING FILE FOR EARTHROVER CAMERAS

"""
So far, config functions seem to work properly,
but now need to add in the client connection part from the EarthRoverMiniPlus class

Ideally using the blocking api version to start with to check if everything works properly,
although it'll look slow but we'll have a guaranteed answer if it works or not.

Then we'd incorporate the threaded version of the api

config is being declared here. next we need to use the EarthRoverMiniPlusConfig class
to declare config and test again.

If the above works, introduce the EarthRoverMiniPlus class and test again.

"""

client_config = EarthRoverMiniPlusConfig(remote_ip="192.168.11.1", port=8888)  # change IP to your robot
#print("client config:" + str(client_config))
client = EarthRover_Mini(client_config)
client.connect()
client.start_camera_stream()

start_time = time.perf_counter()
while time.perf_counter() - start_time < 120:  # Run for 10 seconds
    action_dict = { "linear_velocity": 50.0, "angular_velocity": 80.0 }

    # Step 5: Send action to robot
    client.send_action(action_dict)
    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||| GOING THROUGH LOOP!!! |||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    
client.close_camera_stream()