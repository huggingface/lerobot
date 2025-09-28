from matplotlib import pyplot as plt
import numpy as np
from lerobot.robots.meca.mecaconfig import MecaConfig
from mecademicpy.robot import Robot as MecademicRobot
from lerobot.robots import Robot
from lerobot.cameras import make_cameras_from_configs
from typing import Any
import cv2


class Meca(Robot):
    config_class = MecaConfig
    name = "meca"

    def __init__(self, config: MecaConfig):
        super().__init__(config)
        self.robot = MecademicRobot()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connected = False
        self.resetting = False
        self.gripper_state = 0  # Assume gripper starts open
        self.ip = config.ip


    @property
    def _ee_pose(self) -> dict[str, type]:
        return {
            "x" : float,
            "y" : float,
            "z" : float,
            "a" : float,
            "b" : float,
            "g" : float,
            "gripper" : float
        }
    
    @property
    def _ee_pose_delta(self) -> dict[str, type]:
        return {
            "dx" : float,
            "dy" : float,
            "dz" : float,
            "da" : float,
            "db" : float,
            "dg" : float,
            "gripper" : float
        }
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (224, 224, 3) for cam in self.cameras
        }
    
    @property
    def observation_features(self) -> dict:
        return {**self._ee_pose, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict:
        return self._ee_pose_delta

    def connect(self, calibrate: bool = True) -> None:
        self.resetting = True
        self.robot.Connect(self.ip)
        self.robot.ResetError()
        
        if self.robot.GetStatusRobot(True).homing_state == 0:
            print("⚠️ Robot not homed, homing now...")
            self.robot.ActivateAndHome()
            self.robot.WaitHomed()
        else:
            self.robot.ResumeMotion()
        self.robot.SetJointVel(10)
        self.robot.SetJointAcc(50)
        self.robot.SetCartAcc(100)
        
        self.robot.GripperOpen()
        self.robot.MoveJoints(1, 44, 17, 1, -30, 0)

        print("🤖 Robot ready.")

        # 🔑 Connect all cameras here
        for cam in self.cameras.values():
            try:
                cam.connect()
                print(f"📷 Connected {cam}")
            except Exception as e:
                print(f"⚠️ Failed to connect {cam}: {e}")

        self.connected = True
        self.resetting = False



    def disconnect(self) -> None:
        try:
            self.robot.Disconnect()
        except Exception as e:
            print(f"⚠️ Robot disconnect error: {e}")

        for cam in self.cameras.values():
            if getattr(cam, "is_connected", False):
                try:
                    cam.disconnect()
                except Exception as e:
                    print(f"⚠️ Camera disconnect error: {e}")
        self.connected = False
    
    def reset(self) -> None:
        if not self.connected:
            raise RuntimeError("Robot not connected. Call connect() before resetting.")
        self.robot.ResetError()
        self.robot.ActivateAndHome()
        self.robot.WaitHomed()
        self.robot.ResumeMotion()
        self.robot.MoveJoints(1, 44, 17, 1, -30, 0)
        self.gripper_state = 0  # Assume gripper starts open
        print("🤖 Robot reset and ready.")

    def move_rest_position(self) -> dict:
        self.resetting = True
        self.robot.MoveJoints(1, 44, 17, 1, -30, 0)
        self.robot.GripperOpen()
        self.gripper_state = 0
        self.resetting = False

    @property
    def is_connected(self) -> bool:
        return self.connected
    
    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed for Meca500
    
    def calibrate(self) -> None:
        pass  # No calibration needed for Meca500

    def configure(self) -> None:
        """No-op configuration for Meca500 (already configured on connect)."""
        pass

    def get_observation(self) -> dict:
        obs = {}
        pose = np.array(self.robot.GetPose(), dtype=np.float32)
        obs.update({
            "x": pose[0],
            "y": pose[1],
            "z": pose[2],
            "a": pose[3],
            "b": pose[4],
            "g": pose[5],
            "gripper": self.gripper_state
        })
        for cam_name, cam in self.cameras.items():
            frame = cam.async_read(timeout_ms=100)
            if frame is not None:
                obs[cam_name] = frame
        return obs
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.connected:
            raise RuntimeError("Robot not connected. Call connect() before sending actions.")
        if self.resetting:
            return self.get_observation()  # Ignore actions while resetting

        dx = action.get("dx", 0.0)
        dy = action.get("dy", 0.0)
        dz = action.get("dz", 0.0)
        da = action.get("da", 0.0)
        db = action.get("db", 0.0)
        dg = action.get("dg", 0.0)
        gripper = action.get("gripper", 0)

        # ⚡️ Non-blocking relative motion
        self.robot.MoveLinRelWRF(-dz, -dx, dy, -dg, da, -db)

        # Gripper toggle (edge detection recommended)
        if gripper is not None:
            if gripper and self.gripper_state == 0:
                self.robot.GripperClose()
                self.gripper_state = 1
            elif not gripper and self.gripper_state == 1:
                self.robot.GripperOpen()
                self.gripper_state = 0

        return self.get_observation()

    def center_crop(self, img, output_size=(224, 224)):
        """
        Center crop + resize an image.
        Args:
            img: numpy array (H, W, C)
            output_size: tuple (h, w) for final size
        Returns:
            Cropped and resized numpy array
        """
        h, w = img.shape[:2]
        new_h, new_w = min(h, w), min(h, w)  # square crop

        # Top-left corner of the crop
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop and resize
        cropped = img[top:top + new_h, left:left + new_w]
        return cv2.resize(cropped, output_size)

    
    def teleop_action_processor(self, action: tuple) -> dict:
        # Already in the right shape from OmniTeleoperator
        return action[0]

    def robot_observation_processor(self, obs: tuple) -> dict:
        processed = dict(obs)  # shallow copy
        for cam in ["top", "bottom"]:
            if cam in processed:
                img = processed[cam]


                # Crop
                processed[cam] = self.center_crop(img, (224, 224))

                # Flip bottom
                if cam == "bottom":
                    processed[cam] = cv2.flip(processed[cam], -1)
                else:  
                    cv2.waitKey(1)



        return processed

    
    def robot_action_processor(self, action: tuple) -> dict:
        # No processing needed for Meca500
        return action[0]
    


