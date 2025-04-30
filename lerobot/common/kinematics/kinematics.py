import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Robot:
    
    # Follow this convention: theta , d, a, alpha
    ROBOT_DH_TABLES = {
        "so100": [
            [0, 0.0542, 0.0304, np.pi / 2],
            [0, 0.0, 0.116, 0.0],
            [0, 0.0, 0.1347, 0.0],
            [0, 0.0, 0.0, -np.pi / 2],
            [0, 0.15, 0.0, 0.0],
        ]
    }
    
    # to be filled with correct numbers based on your motors assembly 
    FROM_DH_TO_MECH = {"so100": np.deg2rad([90.0, 90.0, 90.0, 90.0, 90.0])}
    FROM_MECH_TO_DH = {"so100": np.deg2rad([90.0, 90.0, 90.0, 90.0, 90.0])}
    
    # to be filled with correct numbers based on your motors assembly (here gripper is included)
    MECH_JOINT_LIMITS_LOW = {"so100": np.deg2rad([-90.0, -90.0, -90.0, -90.0, -90.0, -90.0])}
    MECH_JOINT_LIMITS_UP = {"so100": np.deg2rad([90.0, 90.0, 90.0, 90.0, 90.0, 90.0])}

    def __init__(self, robot_type="so100"):
        if robot_type not in Robot.ROBOT_DH_TABLES:
            raise ValueError(
                f"Unknown robot type: {robot_type}. Available: {list(Robot.ROBOT_DH_TABLES.keys())}"
            )

        # set robot model
        self.robot_type = robot_type
        self.dh_table = Robot.ROBOT_DH_TABLES[robot_type]
        self.dh2mech = Robot.FROM_DH_TO_MECH[robot_type]
        self.mech2dh = Robot.FROM_MECH_TO_DH[robot_type]
        self.mech_joint_limits_low = Robot.MECH_JOINT_LIMITS_LOW[robot_type]
        self.mech_joint_limits_up = Robot.MECH_JOINT_LIMITS_UP[robot_type]

        # set worldTbase and nTtool frames
        # NB: worldTbase and nTtool can be used ONLY with:
        # use_orientation = True (see inverse_kinematics() method)
        self.worldTbase = np.eye(4)
        self.nTtool = np.eye(4)

    def from_dh_to_mech(self, q_dh): 
        """convert joint positions from DH to mechanical coordinates"""
        
        # to be implemented based on motors assembly
        
        return q_dh

    def from_mech_to_dh(self, q_mech): 
        """convert joint positions from mechanical to DH coordinates"""
        
        # to be implemented based on motors assembly
        
        return q_mech[:-1] # skip last DOF because it is the gripper
    
    def check_joint_limits(self, q_vec):
        """raise an error in case mechanical joint limits are exceeded"""
        
        for i,q in enumerate(q_vec):
                assert (
                        self.mech_joint_limits_low[i] <= q <= self.mech_joint_limits_up[i]
                    ), f"[ERROR] Joint limits out of bound. J{i+1} = {q}, but limits are ({self.mech_joint_limits_low[i]}, {self.mech_joint_limits_up[i]})"


class RobotUtils:
    @staticmethod
    def calc_distance(p1, p2):
        """compute distance between two 3D vectors"""

        return np.linalg.norm(p2 - p1)

    @staticmethod
    def inv_homog_mat(T):
        """invert homogenous transformation matrix"""

        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    @staticmethod
    def calc_lin_err(T_current, T_desired):
        """compute linear error between 2 homogenous transformations"""

        return T_desired[:3, 3] - T_current[:3, 3]

    @staticmethod
    def calc_ang_err(T_current, T_desired):
        """compute angular error between two homogenous transformations"""

        R_current = T_current[:3, :3]
        R_desired = T_desired[:3, :3]
        return 0.5 * (
            np.cross(R_current[:, 0], R_desired[:, 0])
            + np.cross(R_current[:, 1], R_desired[:, 1])
            + np.cross(R_current[:, 2], R_desired[:, 2])
        )

    @staticmethod
    def calc_dh_matrix(dh, theta):
        """compute dh matrix"""

        _, d, a, alpha = dh
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array(
            [[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]]
        )

    @staticmethod
    def calc_an_jac_n(robot, q):
        """compute analytical jacobian wrt n-frame"""

        DOF = len(q)
        J = np.zeros((6, DOF))
        U = np.eye(4)

        for j in reversed(range(DOF)):
            T_j = RobotUtils.calc_dh_matrix(robot.dh_table[j], q[j])
            U = T_j @ U
            px, py = U[0, 3], U[1, 3]
            d = np.array(
                [-U[0, 0] * py + U[1, 0] * px, -U[0, 1] * py + U[1, 1] * px, -U[0, 2] * py + U[1, 2] * px]
            )
            delta = U[2, :3]
            J[:3, j] = d
            J[3:, j] = delta
        return J

    @staticmethod
    def calc_an_jac_0(robot, q, T_current):
        """compute analytical jacobian wrt base-frame"""

        J_end = RobotUtils.calc_an_jac_n(robot, q)
        R_mat = T_current[:3, :3]
        J_transform = np.zeros((6, 6))
        J_transform[:3, :3] = R_mat
        J_transform[3:, 3:] = R_mat
        J_world = J_transform @ J_end
        return J_world

    @staticmethod
    def dls_pseudoinv(J_an, lambda_val=0.001):
        """compute Damped Least Squares Right-Pseudo-Inverse"""

        JT = J_an.T
        JJT = J_an @ JT
        J_pinv = JT @ np.linalg.inv(JJT + lambda_val * np.eye(JJT.shape[0]))
        return J_pinv


class RobotKinematics:
    def __init__(self):
        pass

    def forward_kinematics(self, robot, q):
        """compute forward kinematics (worldTtool)"""

        baseTn = self._forward_kinematics_baseTn(robot, q)

        return robot.worldTbase @ baseTn @ robot.nTtool

    def _forward_kinematics_baseTn(self, robot, q):
        """compute forward kinematics (baseTn)"""

        T = np.eye(4)
        DOF = len(q)

        for i in range(DOF):
            dh = robot.dh_table[i]
            T_link = RobotUtils.calc_dh_matrix(dh, q[i])
            T = T @ T_link

        return T

    def _inverse_kinematics_step_baseTn(
        self, robot, q_start, T_desired, use_orientation=True, k=0.8, n_iter=50
    ):
        """compute inverse kinematics (T_desired must be expressed in baseTn)"""

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        for _ in range(n_iter):
            # compute current pose baseTn
            T_current = self._forward_kinematics_baseTn(robot, q)

            # compute linear error
            err_lin = RobotUtils.calc_lin_err(T_current, T_desired)

            # decide whether to use full jacobian or not
            if use_orientation:
                err_ang = RobotUtils.calc_ang_err(T_current, T_desired)  # compute angular error
                error = np.concatenate((err_lin, err_ang))  # total error
                J_an = RobotUtils.calc_an_jac_0(robot, q, T_current)  # full jacobian
            else:
                error = err_lin  # total error
                J_an = RobotUtils.calc_an_jac_0(robot, q, T_current)[:3, :]  # take only the position part

            # stop if error is minimum
            if np.linalg.norm(error) < 1e-5:
                break

            # Damped Least Squares Right-Pseudo-Inverse
            J_pinv = RobotUtils.dls_pseudoinv(J_an)

            # keep integrating resulting joint positions
            q += k * (J_pinv @ error)

        return q

    def inverse_kinematics(self, robot, q_start, desired_worldTtool, use_orientation=True, k=0.8, n_iter=50):
        """compute inverse kinematics (T_desired must be expressed in worldTtool)
        It is performed an interpolation both for linear and angular components"""

        # I compute ikine with baseTn
        desired_baseTn = (
            RobotUtils.inv_homog_mat(robot.worldTbase)
            @ desired_worldTtool
            @ RobotUtils.inv_homog_mat(robot.nTtool)
        )

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        # init interpolator
        n_steps = self._interp_init(self._forward_kinematics_baseTn(robot, q), desired_baseTn)

        for i in range(0, n_steps + 1):
            # current setpoint as baseTn
            T_desired_interp = self._interp_execute(i)

            # get updated joint positions
            q = self._inverse_kinematics_step_baseTn(robot, q, T_desired_interp, use_orientation, k, n_iter)

        # check final error
        current_worldTtool = self.forward_kinematics(robot, q)
        err_lin = RobotUtils.calc_lin_err(current_worldTtool, desired_worldTtool)
        lin_error_norm = np.linalg.norm(err_lin)
        assert lin_error_norm < 1e-2, f"[ERROR] Large position error ({lin_error_norm:.4f}). Check target reachability (position/orientation)"

        return q

    def _interp_init(self, T_start, T_final, delta=0.01):
        """Initialize interpolator parameters"""

        # init
        self.t_start = T_start[:3, 3]
        self.t_final = T_final[:3, 3]
        R_start = T_start[:3, :3]
        R_final = T_final[:3, :3]

        # Create SLERP object
        times = [0, 1]
        rotations = R.from_matrix([R_start, R_final])
        self.slerp = Slerp(times, rotations)

        # divide trajectory in steps
        dist = RobotUtils.calc_distance(self.t_final, self.t_start)
        self.n_steps = int(np.ceil(dist / delta))

        return self.n_steps

    def _interp_execute(self, i):
        """Compute Cartesian pose setpoint for the current step"""

        # compute current step
        s = i / self.n_steps
        t_interp = (1 - s) * self.t_start + s * self.t_final
        R_interp = self.slerp(s).as_matrix()

        # compute current setpoint
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp

        return T_interp


if __name__ == "__main__":
    
    ## basic usage demo ##
    
    # init
    robot = Robot(robot_type="so100")
    kin = RobotKinematics()

    # get current joint positions
    q_init = np.array([0.0, 0, 0.0, -np.pi / 2, 0.0, 0.0])  # full extended arm (6th DOF is gripper)

    # convert from mechanical angle to dh angle
    q_init_dh = robot.from_mech_to_dh(q_init)

    # compute start pose
    T_start = kin.forward_kinematics(robot, q_init_dh)
    print("T_start = \n", T_start)

    # Define goal pose
    T_goal = T_start.copy()
    T_goal[:3, 3] += np.array([-0.2, 0.1, 0.1])
    print("T_goal = \n", T_goal)

    # IK with internal interpolation
    q_final_dh = kin.inverse_kinematics(robot, q_init_dh, T_goal, use_orientation=False, k=0.8, n_iter=50)
    T_final = kin.forward_kinematics(robot, q_final_dh)

    print("Final joint angles = ", q_final_dh)
    print("Final pose direct kinematics = \n", T_final)

    print("err_lin = ", RobotUtils.calc_lin_err(T_goal, T_final))
    print("err_ang = ", RobotUtils.calc_ang_err(T_goal, T_final))

    # convert from dh angle to mechanical angle
    q_final_mech = robot.from_dh_to_mech(q_final_dh)
    
    # add gripper position
    gripper_pose = np.deg2rad(0.0)
    q_final_mech = np.append(q_final_mech, gripper_pose)
    
    # raise an error in case joint limits are exceeded
    robot.check_joint_limits(q_final_mech)
    

    
    
    
    