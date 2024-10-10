import logging
import time
from datetime import datetime
import os
import numpy as np
from collections import deque
from typing import Protocol
from aubo_robot_controller import Auboi5Robot, RobotError, RobotErrorType, RobotControllerDataReader
from camera_handlers import SpinnakerCamera, USBCamera
import h5py
from PIL import Image
import asyncio
import torch

CONTROL_DT = 1 / 5
TIME_FORMAT = "%Y%m%d_%H%M%S_%f"


# class Robot(Protocol):
#     robot_type: str

#     def connect(self): ...
#     def run_calibration(self): ...
#     def teleop_step(self, record_data=False): ...
#     def capture_observation(self): ...
#     def send_action(self, action): ...
#     def disconnect(self): ...


class Auboi5RobotController:
    def __init__(self, session_name: str, joint_start):
        self.robot_type = "Auboi5"
        self.logger = self.initialize_logger()
        self.session_name = session_name
        self.joint_start = joint_start
        self.output_dirs = self.create_output_directories()
        self.robot = Auboi5Robot()
        self.handle = None
        self.row_count = 0
        self.hdf5_file, self.hdf5_path = self.setup_hdf5_writer()
        self.timestamp_buffer = deque(maxlen=20)
        self.cameras = self.initialize_cameras()
        self.is_connected = False
        self.robot_controller = RobotControllerDataReader(ip='172.31.0.199', port=8891)

    def initialize_logger(self):
        logger = logging.getLogger("Auboi5RobotController")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def create_output_directories(self):
        base_dir = os.path.join("/home/ubuntu/apps/ricliu/data_collection/data", self.session_name)
        os.makedirs(base_dir, exist_ok=True)
        camera_dir = os.path.join(base_dir, "camera")
        os.makedirs(camera_dir, exist_ok=True)
        return {"base": base_dir, "camera": camera_dir}

    def setup_hdf5_writer(self):
        hdf5_file_path = os.path.join(self.output_dirs["base"], f"{self.session_name}.hdf5")
        hdf5_file = h5py.File(hdf5_file_path, mode="w")
        observations_group = hdf5_file.create_group("observations")
        images_group = observations_group.create_group("images")
        hdf5_file.create_dataset("action", (0, 6), maxshape=(None, 6), dtype='f')
        hdf5_file.create_dataset("observations/qpos", (0, 6), maxshape=(None, 6), dtype='f')
        images_group.create_dataset("camera_1", (0, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype='u1')
        images_group.create_dataset("camera_2", (0, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype='u1')
        return hdf5_file, hdf5_file_path

    def initialize_cameras(self):
        spinnaker_camera = SpinnakerCamera(os.path.join(self.output_dirs["camera"], "flir"))
        arducam_1 = USBCamera("/dev/video0", os.path.join(self.output_dirs["camera"], "arducam1"))
        return [spinnaker_camera, arducam_1]

    def connect(self):
        self.logger.info("Connecting to the robot...")
        Auboi5Robot.initialize()
        self.handle = self.robot.create_context()
        self.logger.info(f"robot.rshd={self.handle}")
        ip = '172.31.0.199'
        port = 8899
        result = self.robot.connect(ip, port)
        if result != RobotErrorType.RobotError_SUCC:
            self.logger.error(f"Connect server {ip}:{port} failed.")
            raise ConnectionError("Failed to connect to the robot.")

        self.robot.project_startup()
        self.robot.enable_robot_event()
        self.robot.init_profile()
        self.is_connected = True

    def run_calibration(self):
        if self.is_connected:
            self.logger.info("Calibrating the robot...")
            self.robot.set_joint_maxacc(joint_maxacc=(0.05,) * 6)
            self.robot.set_joint_maxvelc(joint_maxvelc=(0.1,) * 6)
            self.move_to_starting_position(self.joint_start)
        else:
            self.logger.error("Robot is not connected. Cannot calibrate.")

    def move_to_starting_position(self, joint_start):
        self.logger.info("Moving to starting position...")
        self.robot.move_joint(joint_start, True)

    def teleop_step(self, record_data=False):
        if not self.is_connected:
            raise ConnectionError("Robot is not connected.")
        joint = self.robot_controller.get_joint_position()
        robot_pos = self.get_robot_pose(joint)
        if record_data:
            image_data = self.save_camera_frames()
            timestamp = time.time()
            self.collect_data(timestamp, image_data, robot_pos, joint)
            asyncio.run(self.ensure_control_frequency(timestamp))
        return robot_pos, joint

    def capture_observation(self):
        """Capture the observation data and return it as a dictionary of torch.tensors."""
        if not self.is_connected:
            raise ConnectionError("Robot is not connected.")
        # Get the robot's joint positions
        joint = self.robot_controller.get_joint_position()
        robot_pos = self.get_robot_pose(joint)
        # Capture images from cameras
        image_data = self.save_camera_frames()
        # Convert robot position and joint data to torch.tensor
        joint_tensor = torch.tensor(joint, dtype=torch.float32)
        robot_pos_tensor = torch.tensor(robot_pos, dtype=torch.float32)
        # Convert images to torch.tensor
        images = {}
        for i, (image_path, image) in enumerate(image_data):
            if image.shape != (480, 640, 3):
                # If the image is grayscale, convert it to RGB
                image_rgb = np.stack([image] * 3, axis=-1)
                image_resized = Image.fromarray(image_rgb).resize((640, 480))
                image = np.array(image_resized)
            # Convert the image to a tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to (C, H, W) format
            images[f"camera_{i + 1}"] = image_tensor
        # Create the observation dictionary
        observation = {
            "observation.state": torch.cat([robot_pos_tensor, joint_tensor]),
            "observation.joint_positions": joint_tensor,
            "observation.robot_pos": robot_pos_tensor,
        }
        # Add image tensors to the observation dictionary
        for key, image_tensor in images.items():
            observation[f"observation.images.{key}"] = image_tensor
        return observation

    def send_action(self, action):
        if not self.is_connected:
            raise ConnectionError("Robot is not connected.")
        self.logger.info(f"Sending action: {action}")
        # result = self.robot.move_joint(action, True)
        # RSHD = libpyauboi5.create_context()
        # move_to_target_in_cartesian((0.253410,-0.647988,0.102317),(169.232040,7.532587,54.388241))
        result = self.robot.move_to_target_in_cartesian(action[:3], action[3:])
        if result != RobotErrorType.RobotError_SUCC:
            self.logger.error("Failed to execute action.")
        return action

    def disconnect(self):
        self.logger.info("Disconnecting the robot...")
        self.robot.disconnect()
        self.is_connected = False
        # Close HDF5 file
        self.hdf5_file.close()
        self.logger.info("Robot and cameras disconnected successfully.")

    def get_robot_pose(self, joint_positions):
        if joint_positions is not None:
            pos1 = self.robot.forward_kin(joint_positions)
            rpy = self.robot.quaternion_to_rpy(pos1['ori'])
            robot_pos = np.concatenate((np.array(pos1['pos']), np.array(rpy)))
            return robot_pos
        return None

    def save_camera_frames(self):
        camera_data = []
        for camera in self.cameras:
            save_result = camera.save_latest_frame()
            if len(save_result) == 3:
                image_path, timestamp, image = save_result
                camera_data.append((image_path, image))
            else:
                self.logger.warning(f"Unexpected return from {camera.__class__.__name__}: {save_result}")
        return camera_data

    def collect_data(self, timestamp, image_data, pos, joint):
        formatted_timestamp = datetime.fromtimestamp(timestamp).strftime(TIME_FORMAT)
        self.hdf5_file["action"].resize((self.row_count + 1, 6))
        self.hdf5_file["action"][self.row_count, :] = pos
        self.hdf5_file["observations/qpos"].resize((self.row_count + 1, 6))
        self.hdf5_file["observations/qpos"][self.row_count, :] = joint

        for i, (image_path, image) in enumerate(image_data):
            if image.shape == (480, 640, 3):
                self.hdf5_file[f"observations/images/camera_{i + 1}"].resize((self.row_count + 1, 480, 640, 3))
                self.hdf5_file[f"observations/images/camera_{i + 1}"][self.row_count] = image
            else:
                image_rgb = np.stack([image] * 3, axis=-1)
                image_resized = Image.fromarray(image_rgb).resize((640, 480))
                image = np.array(image_resized)
                self.hdf5_file[f"observations/images/camera_{i + 1}"].resize((self.row_count + 1, 480, 640, 3))
                self.hdf5_file[f"observations/images/camera_{i + 1}"][self.row_count] = image

        self.hdf5_file.flush()
        self.row_count += 1

    async def ensure_control_frequency(self, last_command_timestamp):
        current_time = time.time()
        elapsed_time = current_time - last_command_timestamp
        if elapsed_time < CONTROL_DT:
            await asyncio.sleep(CONTROL_DT - elapsed_time)


def main():
    # 定义初始关节位置
    joint_start = (-0.9060847470754707, -0.17845974473485482, 1.609424421163638, 0.3646234885603236, 1.5166347159241875,
                   -0.49550400039325937)
    session_name = "act30"
    # 实例化Auboi5RobotController
    robot_controller = Auboi5RobotController(session_name=session_name, joint_start=joint_start)
    # 连接机器人
    try:
        robot_controller.connect()
        robot_controller.run_calibration()
        # 远程操作步骤
        for i in range(10):  # 假设执行10次远程操作
            # observation, joint_positions = robot_controller.teleop_step(record_data=True)
            # print(f"Step {i+1}: Observation - {observation}, Joint Positions - {joint_positions}")
            # 模拟动作发送
            # action =(-103.413872 / 180.0 * np.pi, -32.761689 / 180.0 * np.pi,
            #                 94.513234 / 180.0 * np.pi, 43.838026 / 180.0 * np.pi,
            #                 87.304222 / 180.0 * np.pi, -25.619585 / 180.0 * np.pi)
            action = (0.253410, -0.647988, 0.102317, 169.232040, 7.532587, 54.388241)
            robot_controller.send_action(action)

        # 获取一次观察数据
        observation = robot_controller.capture_observation()
        print("Captured Observation:", observation)

    except Exception as e:
        print("An error occurred:", e)
    finally:
        # 断开连接
        robot_controller.disconnect()


if __name__ == "__main__":
    main()
