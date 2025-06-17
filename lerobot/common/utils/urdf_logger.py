import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Union, runtime_checkable

import rerun as rr

from lerobot.common.constants import URDFS
from lerobot.common.motors.motors_bus import Motor, MotorNormMode
from lerobot.common.robots.robot import Robot


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
                angle_rad = self._get_calibrated_angle(joint_name, position)
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

    def _get_calibrated_angle(self, joint_name: str, position: float) -> float:
        """
        Get the calibrated angle in radians for a joint.

        :param joint_name: Name of the joint.
        :param position: Raw position of the joint.
        :return: Calibrated angle.
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
    :raises ET.ParseError: If the URDF file is not a valid XML file or cannot be parsed.
    :raises KeyError: If a required attribute is missing in the URDF file (e.g., 'name', 'link', 'xyz', 'lower', 'upper').
    """
    tree = ET.parse(urdf_path)
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
