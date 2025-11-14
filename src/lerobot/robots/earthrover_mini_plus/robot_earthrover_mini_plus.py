import logging
from socket import socket
from typing import Any
import cv2
import threading

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig, EarthRoverMiniCamera

# The import from our low-level API, so we can call actual functions on the robot
from earth_rover_mini_sdk import EarthRoverMini_API



#logger = logging.get_logger(__name__)


class EarthRover_Mini(Robot):

    config_class = EarthRoverMiniPlusConfig
    name = "earthrover_mini_plus"

    def __init__(self, config: EarthRoverMiniPlusConfig):

        super().__init__(config)
        self.config = config
        self.earth_rover: None

        #No motors
        #self.base_motors = [] # todo
        self.is_connected = False

        self.cameras = make_cameras_from_configs(config.cameras)
        self.thread_stop_event = None
        self.camera_thread = None

        print("Cameras made from config:" + str(self.cameras))
   
    def is_connected(self) -> bool:
        # Connected iff all the cameras are connected
        return self.is_connected
    
    # Connects to all robot devices, currently just the cameras
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        self.earth_rover = EarthRoverMini_API(ip="192.168.11.1", port=8888)
        #EarthRoverMiniPlus(self.config)
        self.earth_rover.connect()
        #asyncio.run(self.earth_rover.connect())
        for cam in self.cameras.values():
            print(f"Connecting to camera {cam.config.index_or_path}...")
            cam.connect()
            if cam.is_connected:
                print(f"{cam.config.index_or_path} connected successfully!")
            else:
                print(f"Failed to connect to {cam.config.index_or_path}. Exiting...")
                raise DeviceNotConnectedError
        
        # Currently doesn't do anything, no configuration needed? Only need to connect.
        self.configure()

        # Change the is_connected class value
        self.is_connected = True

    def start_camera_stream(self):
        if self.camera_thread and self.camera_thread.is_alive():
            print("Camera stream already running.")
            return
        self.thread_stop_event = threading.Event()
        self.camera_thread = threading.Thread(target=self.update_stream, args=(self.thread_stop_event,), daemon=True)
        self.camera_thread.start()
        
    
    def update_stream(self, stop_event):
        while not stop_event.is_set():
            for idx, cam in enumerate(self.cameras.values()):
                frame = cam.read()
                if frame is None:
                    continue
                
                cv2.imshow(f"RTSP Stream {idx}", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

        print("Stopping camera stream thread...")

        # for cam in self.cameras.keys():
        #         print(str(cam))
        #         print((str(self.cameras[cam])))
        #         frame = self.cameras[cam].read()
        #         cv2.imshow(f"RTSP Stream {idx}", frame)
        #         if cv2.waitKey(1) & 0xFF == ord("q"):
        #             break

    def close_camera_stream(self):
        if self.thread_stop_event:
            self.thread_stop_event.set()
        
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)

        for cam in self.cameras.values():
            cam.disconnect()

        cv2.destroyAllWindows()

    def is_calibrated(self) -> bool:
        return self.is_calibrated
    
    def calibrate(self) -> None:
        """
        Make this a calibration state machine for imu, mag, accelerometer data
        """
        # Calibrate the IMU, motors, etc?
        if self.calibration:
            pass
            #logger.info(f"\nRunning calibration of {self}")

        # todo
    
    # Not necessary for right now, no configuration needed for the EarthRover
    # Just need to connect using the socket in connect, so possibly configure is unnecessary
    def configure(self):
        # todo
        pass

    @property
    def _motor_rpms_ft(self) -> dict[str, type]:
        return {
            "motor_Fl": float,
            "motor_Fr": float,
            "motor_Br": float,
            "motor_Bl": float,
        }
    
    @property
    def _speed_and_heading_ft(self) -> dict[str, type]:
        return {
                "speed": float,
                "heading": float,
            }
    

    @property
    def _imu_ft(self) -> dict[str, type]:
        return {
            "accel_x": float, "accel_y": float, "accel_z": float,
            "gyro_x": float, "gyro_y": float, "gyro_z": float,
            "mag_x": float, "mag_y": float, "mag_z": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motor_rpms_ft, **self._imu_ft, **self._speed_and_heading_ft, **self._cameras_ft}
    

    @property
    def action_features(self) -> dict:
        return self._speed_and_heading_ft

    


    
    def get_observation(self) -> dict[str, Any]:
        #calls function in earthrover object to get observation data:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dct: dict[str: Any]={}
        obs_dct.update(self.earth_rover.get_telemetry())
        for cam_key, cam in self.cameras.items():
            obs_dct[cam_key] = cam.async_read()

        return obs_dct


        

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send control commands to Earthrover Mini Plus"""
       
        # send_ctl_cmd(self.socket, self.speed, self.angular)

        """
        Example of possible action dictionary:

        action = {
            "linear_velocity": 0.2,
            "angular_velocity": 5
        }
        """
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # The action is the movement command, with a linear velocity and angular velocity
        if "linear_velocity" in action or "angular_velocity" in action:
            print("ennterrrrrrrrrrrrrrrrrrrrrrr")
            v = action["linear_velocity"]
            w = action["angular_velocity"]
            # Call the api call for move, should be higher level not send_ctl_cmd
        else:
            return None
        self.earth_rover.move_continuous_loop( speed=int(v),angular= int(w))
        return
        # return await self.earth_rover.move( speed=int(v),angular= int(w),duration=int(10))
            




    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()
        self.earth_rover.disconnect()
        #logger.info(f"{self} disconnected.")

