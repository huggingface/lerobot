import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kinematics import *




class TestKinematics(unittest.TestCase):
    def setUp(self):
        self.robot = Robot(robot_type="so100")
        self.kin = RobotKinematics()
        self.q_init = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
        self.q_init_dh = self.robot.from_mech_to_dh(self.q_init)
        self.T_start = self.kin.forward_kinematics(self.robot, self.q_init_dh)

    def test_basic_usage(self):
        "test basic usage"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([0.0, 0.0, -0.1])
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=True)
        q_final_mech = self.robot.from_dh_to_mech(q_final_dh)
        self.robot.check_joint_limits(q_final_mech)

    def test_unreachable_pose(self):
        "test assert error when final pose is far from T_goal"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([1.0, 0.1, 0.1])  # set an UNREACHABLE goal pose to trigger error
        # check method
        with self.assertRaises(AssertionError) as context:
            self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        # Check the correct ERROR is triggered
        self.assertIn("Large position error", str(context.exception))

    def test_joint_limits(self):
        "test assert error when joint limits are out of bound"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([0.3, -0.2, -0.3])
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        q_final_mech = self.robot.from_dh_to_mech(q_final_dh)
        # check method
        with self.assertRaises(AssertionError) as context:
            self.robot.check_joint_limits(q_final_mech)
        # Check the correct ERROR is triggered
        self.assertIn("Joint limits out of bound", str(context.exception))

    def test_interpolator_delta(self):
        "test assert error when interpolation delta parameter is zero, triggering division by zero"
        with self.assertRaises(AssertionError) as context:
            self.kin._interp_init(self.T_start, np.eye(4), delta=0.0)
        # Check the correct ERROR is triggered
        self.assertIn("Delta must be strictly greater than zero", str(context.exception))

    def test_dh2mechanical(self):
        "validate dh2mechanical conversion against known values"
        q_dh = np.array([-1.57079633, 1.31859625, -1.31859625, -3.14159265, 0.0])
        expected_mech = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2])
        q_mech = self.robot.from_dh_to_mech(q_dh)
        assert np.allclose(q_mech, expected_mech, atol=1e-5), f"Expected {expected_mech}, got {q_mech}"

    def test_mechanical2dh(self):
        "validate mechanical2dh conversion against known values"
        q_mech = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
        expected_dh = np.array([-1.57079633, 1.31859625, -1.31859625, -3.14159265, 0.0])
        q_dh = self.robot.from_mech_to_dh(q_mech)
        assert np.allclose(q_dh, expected_dh, atol=1e-5), f"Expected {expected_dh}, got {q_dh}"


if __name__ == "__main__":
    unittest.main()
