import unittest

import numpy as np
from kinematics import *


class TestKinematics(unittest.TestCase):
    def setUp(self):
        self.robot = Robot(robot_type="so100")
        self.kin = RobotKinematics()
        self.q_init = np.array([0.0, 0, 0.0, -np.pi / 2, 0.0, 0.0])
        # since everyone may set different out of bound values, I do unit testing with these fixed values
        self.mech_joint_limits_low = np.deg2rad([-90.0, -90.0, -90.0, -90.0, -90.0, -90.0])
        self.mech_joint_limits_up = np.deg2rad([90.0, 90.0, 90.0, 90.0, 90.0, 90.0])
        # same here for dh2mech and mech2dh methods
        self.q_init_dh = self.q_init[:-1]  # self.robot.from_mech_to_dh(self.q_init)
        self.T_start = self.kin.forward_kinematics(self.robot, self.q_init_dh)

    def test_basic_usage(self):
        "test bsic usage"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([-0.2, 0.1, 0.1])
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        q_final_mech = q_final_dh[:]  # self.robot.from_dh_to_mech(q_final)
        self.robot.check_joint_limits(q_final_mech)

    def test_unreachable_pose(self):
        "test assert error when final pose is far from T_goal"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([1.0, 0.1, 0.1])  # set an UNREACHABLE goal pose to trigger error
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        q_final_mech = q_final_dh[:]  # self.robot.from_dh_to_mech(q_final)
        self.robot.check_joint_limits(q_final_mech)

    def test_joint_limits(self):
        "test assert error when joint limits are out of bound"
        T_goal = self.T_start.copy()
        T_goal[:3, 3] += np.array([-0.5, -0.15, 0.3])
        q_final_dh = self.kin.inverse_kinematics(self.robot, self.q_init_dh, T_goal, use_orientation=False)
        q_final_mech = q_final_dh[:]  # self.robot.from_dh_to_mech(q_final)
        self.robot.check_joint_limits(q_final_mech)

    def test_interpolator_delta(self):
        "test assert error when interpolation delta parameter is zero, triggering division by zero"
        n_steps = self.kin._interp_init(self.T_start, np.eye(4), delta=0.0)

    def test_dh2mechanical(self):
        "test dh2mechanical conversion"
        pass

    def test_mechanical2dh(self):
        "test mechanical2dh conversion"
        pass


if __name__ == "__main__":
    unittest.main()
