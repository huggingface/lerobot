# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper utilities to visualize robot data with Rerun.

This module provides classes for logging URDF data, video streams, and teleoperation data to Rerun.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import av
import av.video.stream
import cv2
import defusedxml.ElementTree
import numpy as np
import rerun as rr

from lerobot.common.constants import URDFS
from lerobot.common.motors.motors_bus import Motor, MotorNormMode
from lerobot.common.robots.robot import Robot
from lerobot.common.teleoperators.teleoperator import Teleoperator


def _init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


@runtime_checkable
class HasMotors(Protocol):
    motors: Dict[str, Motor]


@runtime_checkable
class HasBusWithMotors(Protocol):
    bus: HasMotors


@dataclass
class JointInfo:
    path: str
    xyz: str
    lower: float
    upper: float


class URDFLogger:
    def __init__(
        self,
        robot: Robot,
        urdf_path: Optional[Union[str, Path]] = None,
        entity_path_prefix: Optional[str] = None,
    ):
        """
        Initialize the URDFLogger with the path to the URDF file and optional motor names.

        :param robot: The robot instance containing the URDF path and motor information.
        :param urdf_path: Path to the URDF file. If not provided, it will be constructed based on the robot type.
        :param entity_path_prefix: Optional prefix for the entity path in Rerun. Defaults to robot type.
        :raises FileNotFoundError: If the URDF file does not exist at the specified path.
        """
        self.robot = robot
        self.urdf_path = Path(urdf_path) if urdf_path else get_urdf_path_for_robot(robot)
        self.entity_path_prefix = entity_path_prefix or robot.robot_type
        self._joint_paths: Optional[Dict[str, JointInfo]] = None

    @property
    def joint_paths(self):
        if self._joint_paths is None:
            self._joint_paths = get_revolute_joint_child_paths(self.urdf_path)
        return self._joint_paths

    def log_urdf(self, recording_stream: Optional[rr.RecordingStream] = None):
        """
        This function logs the URDF file as a static asset in Rerun.

        :param recording_stream: Optional Rerun recording stream to log the URDF file. If not provided,
                                it will use the global recording stream.
        :raises FileNotFoundError: If the URDF file does not exist at the specified path.
        :raises ValueError: If no recording stream is provided and no global recording stream is found.
        """
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        if recording_stream is None:
            recording_stream = rr.get_global_data_recording()
            if recording_stream is None:
                raise ValueError("No global recording stream found. Please provide a recording stream.")
        recording_stream.log_file_from_path(
            self.urdf_path.absolute(), entity_path_prefix=self.entity_path_prefix, static=True
        )

    def log_joint_angles(self, joint_positions: Dict[str, float]):
        """
        Log the joint paths to Rerun.

        :param joint_positions: Dictionary mapping joint names to their positions.
        """
        for joint_name, position in joint_positions.items():
            joint_name = joint_name.replace(".pos", "")
            if joint_name in self.joint_paths:
                path = self.joint_paths[joint_name].path
                if self.entity_path_prefix:
                    path = f"{self.entity_path_prefix}/{path}"
                fixed_axis = list(map(float, self.joint_paths[joint_name].xyz.split(" ")))
                angle_rad = self._get_angle_rad(joint_name, position)
                rr.log(
                    path,
                    rr.Transform3D(rotation=rr.datatypes.RotationAxisAngle(axis=fixed_axis, angle=angle_rad)),
                )

    def _get_motor_modes(self) -> Dict[str, MotorNormMode]:
        """
        Get the motor modes for the robot.

        :return: A dictionary mapping motor names to their modes.
        """
        if isinstance(self.robot, HasBusWithMotors):
            return {name: motor.norm_mode for name, motor in self.robot.bus.motors.items()}
        return {}

    def _get_angle_rad(self, joint_name: str, position: float) -> float:
        """
        Get the calibrated angle in radians for a joint.

        :param joint_name: Name of the joint.
        :param position: Raw position of the joint.
        :return: Joint angle in radians.
        """
        # position is -100 to 100 range
        if joint_name not in self.joint_paths:
            raise ValueError(f"Joint name '{joint_name}' not found in URDF paths.")

        def normalize(position, motor_mode):
            """
            Based on the motor mode, normalize to [0, 1]
            """
            if motor_mode == MotorNormMode.RANGE_M100_100:
                return (position + 100) / 200
            elif motor_mode == MotorNormMode.RANGE_0_100:
                return position / 100
            elif motor_mode == MotorNormMode.DEGREES:
                return (position + 180) / 360
            else:
                raise ValueError(f"Unsupported motor mode: {motor_mode}")

        joint_info = self.joint_paths[joint_name]
        lower_limit_rad = joint_info.lower
        upper_limit_rad = joint_info.upper
        if lower_limit_rad is None or upper_limit_rad is None:
            raise ValueError(f"Joint '{joint_name}' does not have defined limits in the URDF.")

        motor_mode = self._get_motor_modes().get(joint_name, MotorNormMode.RANGE_M100_100)
        radians = normalize(position, motor_mode) * (upper_limit_rad - lower_limit_rad) + lower_limit_rad

        return radians

    def __repr__(self):
        return (
            f"<URDFLogger robot={self.robot.robot_type} urdf_path={self.urdf_path} "
            f"entity_path_prefix={self.entity_path_prefix}>"
        )


def get_urdf_path_for_robot(robot: Robot) -> Path:
    """
    Get the URDF path for a given robot instance.

    :param robot: The robot instance.
    :return: The path to the URDF file.
    :raises FileNotFoundError: If the URDF file does not exist at the specified path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    urdf_path = repo_root / URDFS / robot.robot_type / (robot.robot_type + ".urdf")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}. Please ensure the URDF file exists.")
    return urdf_path


def get_revolute_joint_child_paths(urdf_path: Union[str, Path]) -> Dict[str, JointInfo]:
    """
    Parse a URDF file to extract information about revolute joints and their child links.

    :param urdf_path: Path to the URDF file.
    :return: A dictionary mapping joint names to JointInfo dataclasses containing the path, xyz coordinates, and limits.
    :raises FileNotFoundError: If the URDF file does not exist at the specified path.
    :raises defusedxml.ElementTree.ParseError: If the URDF file is not a valid XML file or cannot be parsed.
    :raises KeyError: If a required attribute is missing in the URDF file (e.g., 'name', 'link', 'xyz', 'lower', 'upper').
    """
    tree = defusedxml.ElementTree.parse(urdf_path)
    root = tree.getroot()

    # Map child link to (parent link, joint name)
    child_to_parent_joint = {}
    joint_to_child = {}
    joint_to_xyz = {}
    joint_to_limits = {}

    for joint in root.findall(".//joint[@type='revolute']"):
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        if parent is not None and child is not None:
            parent_link = parent.attrib["link"]
            child_link = child.attrib["link"]
            joint_name = joint.attrib["name"]
            child_to_parent_joint[child_link] = (parent_link, joint_name)
            joint_to_child[joint_name] = child_link
            xyz = axis.attrib["xyz"] if axis is not None and "xyz" in axis.attrib else None
            lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None
            upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None
            joint_to_xyz[joint_name] = xyz
            joint_to_limits[joint_name] = (lower, upper)

    # Build full path for a link, inserting joint names
    path_cache = {}

    def build_path(link):
        if link in path_cache:
            return path_cache[link]
        if link not in child_to_parent_joint:
            path_cache[link] = link
        else:
            parent_link, joint_name = child_to_parent_joint[link]
            path_cache[link] = f"{build_path(parent_link)}/{joint_name}/{link}"
        return path_cache[link]

    # Map joint name to JointInfo dataclass
    return {
        joint: JointInfo(
            path=build_path(child),
            xyz=joint_to_xyz[joint],
            lower=joint_to_limits[joint][0],
            upper=joint_to_limits[joint][1],
        )
        for joint, child in joint_to_child.items()
    }


class VideoLogger:
    def __init__(
        self,
        stream_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30,
        codec: str = "libx264",
        preset: str = "ultrafast",
        tune: str = "zerolatency",
    ):
        """
        Initializes the VideoLogger to log video frames to a Rerun stream.

        Be sure to call `close()` when done to finalize the video stream, or use it within a `with` statement to ensure proper cleanup.

        :param stream_name: The name of the Rerun stream to log video frames to.
        :param width: The width of the video frames. If None, it will be determined from the first frame logged.
        :param height: The height of the video frames. If None, it will be determined from the first frame logged.
        :param fps: Frames per second for the video stream.
        :param codec: The codec to use for encoding the video. Default is "libx264".
        :param preset: The encoding preset to use. Default is "ultrafast".
        :param tune: The tuning option for the codec. Default is "zerolatency".
        """
        self.stream_name = stream_name
        self._width = width
        self._height = height
        self.fps = fps
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: Optional[float] = None
        self._frame_interval = 1.0 / fps
        self._lock = threading.Lock()
        self._encoder_initialized = False
        self._codec = codec
        self._preset = preset
        self._tune = tune
        rr.log(stream_name, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

    def _open_encoder(self, width: int, height: int):
        self.container = av.open("/dev/null", "w", format="h264")
        stream = self.container.add_stream(self._codec, rate=self.fps)
        if not isinstance(stream, av.video.stream.VideoStream):
            raise RuntimeError("Failed to create a video stream for encoding.")
        stream.width = width
        stream.height = height
        stream.rate = Fraction(self.fps, 1)
        stream.max_b_frames = 0
        stream.options = {  # type: ignore[attr-defined]
            "preset": self._preset,
            "tune": self._tune,
            "vbv_bufsize": "1",
            "vbv_maxrate": str(self.fps * width * height * 3),
        }
        self.stream = stream
        self._encoder_initialized = True

    def log_frame(self, img: np.ndarray):
        """
        Encodes and logs a single frame as video to the Rerun stream.

        :param img: The image frame to log, expected to be in HWC format (height, width, channels).
        :raises RuntimeError: If the video stream is not initialized or if the image shape does not match the expected dimensions.
        :return: None
        """
        now = time.perf_counter()
        with self._lock:
            if self._last_frame_time is not None and (now - self._last_frame_time) < self._frame_interval:
                return  # Skip frame: too soon for target FPS
            self._last_frame_time = now

            if self._width is None or self._height is None:
                self._height, self._width = img.shape[:2]

            if not self._encoder_initialized:
                self._open_encoder(self._width, self._height)

            if img.shape != (self._height, self._width, 3):
                img = cv2.resize(img, (self._width, self._height))

            if self._last_frame is not None and np.array_equal(img, self._last_frame):
                return  # Skip identical frame

            self._last_frame = img.copy()
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in self.stream.encode(frame):
                if packet.pts is not None and packet.time_base is not None:
                    rr.set_time(self.stream_name, duration=float(packet.pts * packet.time_base))
                rr.log(self.stream_name, rr.VideoStream.from_fields(sample=bytes(packet)))

    def close(self):
        """
        Finalizes the video stream by flushing any remaining frames and closing the container.
        """
        with self._lock:
            if self._encoder_initialized:
                for packet in self.stream.encode():
                    if packet.pts is not None and packet.time_base is not None:
                        rr.set_time(self.stream_name, duration=float(packet.pts * packet.time_base))
                    rr.log(self.stream_name, rr.VideoStream.from_fields(sample=bytes(packet)))
                self.container.close()
                self._encoder_initialized = False

    def __repr__(self):
        return f"VideoLogger(stream_name={self.stream_name}, width={self._width}, height={self._height}, fps={self.fps})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RerunRobotLogger:
    def __init__(
        self,
        teleop: Optional[Teleoperator] = None,
        robot: Optional[Robot] = None,
        image_stream_type: str = "video",
        fps: int = 60,
        image_size: Optional[tuple[int, int]] = None,
        log_urdf: bool = True,
    ):
        """
        Initializes the RerunRobotLogger to log robot and teleoperation data to Rerun.

        :param teleop: Optional teleoperator instance to log actions.
        :param robot: Optional robot instance to log observations and joint angles.
        :param image_stream_type: Type of image stream to log. Options are "video", "jpeg", "raw", or "none".
        :param fps: Frames per second for video logging.
        :param image_size: Optional tuple (width, height) for video frame size.
            If None, it will use the size from the robot's observation features.
        :param log_urdf: Whether to log the URDF of the robot. Defaults to True.
        """
        self.robot = robot
        self.teleop = teleop

        self.robot_urdf_logger: Optional[URDFLogger] = None
        self.image_stream_type = image_stream_type
        self.video_loggers: Dict[str, VideoLogger] = {}

        self.fps = fps
        self.width, self.height = (image_size[0], image_size[1]) if image_size else (None, None)

        self.log_urdf = log_urdf and robot is not None

    def init(self, session_name: str = "lerobot_control_loop"):
        _init_rerun(session_name=session_name)

        if self.robot is not None and self.image_stream_type == "video":
            self.video_loggers: Dict[str, VideoLogger] = {
                cam_name: self._create_video_logger(cam_name, shape)
                for cam_name, shape in self.robot.observation_features.items()
                if isinstance(shape, tuple)
            }
        if self.log_urdf and self.robot is not None:
            try:
                self.robot_urdf_logger = URDFLogger(self.robot)
                self.robot_urdf_logger.log_urdf()
            except FileNotFoundError as e:
                logging.warning(
                    f"URDF file not found for robot {self.robot.robot_type}. Skipping URDF logging: {e}"
                )
                self.log_urdf = False

    def cleanup(self):
        """
        Cleanup the logger, closing any video loggers and shutting down Rerun.
        """
        for video_logger in self.video_loggers.values():
            video_logger.close()
        rr.rerun_shutdown()

    def log_all(self, observation=None, action=None, sync_time: bool = True):
        """
        Log all observations, actions, and joint angles to Rerun.
        If `observations` and `actions` are not provided, it will fetch them from the robot and teleoperator instances.
        If `sync_time` is True, it will also set the current time in Rerun.
        This is useful for synchronizing the logs with the robot's time.
        """
        if sync_time:
            rr.set_time("time", duration=np.timedelta64(time.time_ns(), "ns"))

        if observation is None:
            observation = observation or self.robot.get_observation() if self.robot else {}
        if action is None:
            action = self.teleop.get_action() if self.teleop else {}

        self.log_observations(observation)
        self.log_joint_angles(observation)
        self.log_actions(action)

    def log_observations(self, observations: Dict[str, Any]):
        if self.robot is None:
            logging.warning("No robot instance available for logging observations.")
            return

        for obs, val in observations.items():
            if isinstance(val, float):
                rr.log(["observation", obs], rr.Scalars(val))
            elif isinstance(val, np.ndarray) and obs in self.video_loggers:
                self.log_frame(val, obs)

    def log_actions(self, actions: Dict[str, Any]):
        if self.teleop is None:
            logging.warning("No teleoperator instance available for logging actions.")
            return
        for act, val in actions.items():
            if isinstance(val, float):
                rr.log(["action", act], rr.Scalars(val))

    def log_joint_angles(self, angles: Dict[str, Any]):
        if self.robot is None:
            logging.warning("No robot instance available for logging joint angles.")
            return
        if self.log_urdf and self.robot_urdf_logger is not None:
            self.robot_urdf_logger.log_joint_angles(angles)

    def log_frame(self, frame: np.ndarray, cam_name: str):
        """
        Log a single video frame to the specified stream name.

        :param frame: The image frame to log, expected to be in HWC format (height, width, channels).
        :param stream_name: The name of the Rerun stream to log the frame to.
        """
        if self.image_stream_type == "video":
            logger = self.video_loggers.get(cam_name, self._create_video_logger(cam_name, frame.shape[:2]))
            logger.log_frame(frame)
        elif self.image_stream_type == "jpeg":
            rr.log(cam_name, rr.Image(frame).compress(jpeg_quality=60), static=True)
        elif self.image_stream_type == "raw":
            rr.log(cam_name, rr.Image(frame), static=True)

    def _create_video_logger(self, cam_name: str, shape: tuple[int, int]) -> VideoLogger:
        return VideoLogger(
            stream_name=f"observation/{cam_name}",
            height=self.height or shape[0],
            width=self.width or shape[1],
            fps=self.fps,
        )

    def __repr__(self):
        return (
            f"RerunRobotLogger(teleop={self.teleop}, robot={self.robot}, "
            f"image_stream_type={self.image_stream_type}, fps={self.fps}, "
            f"image_size=({self.width}, {self.height}), log_urdf={self.log_urdf})"
        )
