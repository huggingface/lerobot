import time
from typing import Dict, Any
from collections.abc import Sequence 
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

class Pose:

    def __init__(self, robot: Any, name: str, **joint_targets: float):
        self.robot = robot
        self.name = name

        # Discover valid feature keys once
        self.features = list(robot.action_features.keys())
        # Initialize all joints to 0°
        self.angles: Dict[str, float] = {feat: 0.0 for feat in self.features}

        # Override specified joint_targets by matching exact feature keys
        for joint, value in joint_targets.items():
            if joint not in self.features:
                raise ValueError(f"No joint feature named '{joint}'")
            self.angles[joint] = value

        """
    Friendly Pose class for defining and applying named robot poses.

    Usage:
        robot = SO101Follower(config)
        flat_pose = Pose.flat(robot)
        flat_pose.apply(hold_time=2.0)

        right45 = Pose.flat_45_right(robot)
        right45.apply()

        # Custom pose: specify exact feature keys, e.g.: elbow_flex.pos=30.0, wrist_roll.pos=-15.0
        custom = Pose(robot, "custom", elbow_flex.pos=30.0, wrist_roll.pos=-15.0)
        custom.apply(hold_time=1.5)

    Joint feature usage for SO101 follower:
    - shoulder_pan.pos   : rotate base left/right (pan)
    - shoulder_lift.pos  : lift lower arm up/down
    - elbow_flex.pos     : bend elbow joint
    - wrist_roll.pos     : rotate wrist around axis
    - gripper.pos        : open/close gripper
    """

    @staticmethod
    def read_pose(robot: SO101Follower) -> Dict[str, float]:
        """
        Reads and returns the current joint angles (or other observations)
        from the robot as a dict mapping feature names to values.
        """
        obs = robot.get_observation()
        # If in degree mode, these are degrees; otherwise [-1,1] normalized.
        print("Current joint angles:", obs)
        return obs
    
    def apply(self, hold_time: float = 2.0):
        """Sends the pose to the robot and holds for the specified time."""
        print(f"Applying pose '{self.name}': {self.angles}")
        self.robot.send_action(self.angles)
        time.sleep(hold_time)

    @classmethod
    def flat(cls, robot: Any) -> 'Pose':
        """Creates a flat pose (all joints at 0°)."""
        return cls(robot, "flat")

    @classmethod
    def flat_45_right(cls, robot: Any) -> 'Pose':
        """Flat pose with 45° right pan rotation."""
        # Specify the full feature key for pan rotation
        return cls(robot, "flat_45_right", **{"shoulder_pan.pos": 45.0, "shoulder_lift.pos": 90.0, "elbow_flex.pos": -90.0})

    @classmethod
    def rest(cls, robot: Any) -> 'Pose':
        """
        Creates the default rest pose using predefined joint angles.
        """
        return cls(
            robot,
            "rest_default",
            **{
                'shoulder_pan.pos': 0.0,
                'shoulder_lift.pos': -105.0,
                'elbow_flex.pos': 100.0,
                'wrist_flex.pos': 65.0,
                'wrist_roll.pos': 0.0,
                'gripper.pos': 0
            }
        )
  
    @classmethod
    def move_to_xyz(
        self,
        kinematics: RobotKinematics,
        current_joint_pos: Sequence[float],
        target_xyz: Sequence[float],
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> Sequence[float]:
        """
        Compute the joint angles that bring the end‑effector to `target_xyz`.

        Args
        ----
        kinematics        : A pre‑configured ``RobotKinematics`` solver
        current_joint_pos : Current joint positions (deg or rad, as expected by IK)
        target_xyz        : Desired Cartesian position [x, y, z] in metres
        position_weight   : Weight on translational error in IK optimisation
        orientation_weight: Weight on rotational error in IK optimisation

        Returns
        -------
        Sequence[float]   : Joint angles (same units as input) that achieve the pose
        """
        # 4×4 homogeneous transform with desired translation only
        desired_ee_pose = np.eye(4)
        desired_ee_pose[:3, 3] = target_xyz

        target_joint_positions = kinematics.inverse_kinematics(
            current_joint_pos,
            desired_ee_pose,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
        return target_joint_positions
    # ------------------------------------------------------------------


def main():
    # Configure and connect the SO101 follower
    cfg = SO101FollowerConfig(port="COM6", id="follower", use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect()

    # Initialise a kinematics solver once
    kin = RobotKinematics(
        urdf_path="lerobot\robot_config\so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
    )

    try:
        # Apply several named poses
        # Pose.flat(robot).apply(hold_time=2.0)
        # Pose.flat_45_right(robot).apply(hold_time=5.0)
        # Pose.read_pose(robot)
        # Pose.rest(robot).apply(hold_time=1.0)
        # Pose.move_to_uv(robot, u=0.3, v=0.5)

        # Start from a rest pose
        rest_pose = Pose.rest(robot)
        rest_pose.apply(hold_time=1.0)

        # Compute IK for a Cartesian target and send the command
        current_joints = list(Pose.read_pose(robot).values())
        target_xyz = [0.2417, 0.2012, 0.1027]   # metres
        new_joints = rest_pose.move_to_xyz(kin, current_joints, target_xyz)

        print("Moving to XYZ:", target_xyz, "→ joints:", new_joints)
        robot.send_action(dict(zip(rest_pose.features, new_joints)))

    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
