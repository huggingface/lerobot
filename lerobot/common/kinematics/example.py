from kinematics import *




# init
robot = Robot(robot_type="so100")
kin = RobotKinematics()

# get current joint positions
q_init = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
print("q_init_mechanical: ", np.rad2deg(q_init))

# convert from mechanical angle to dh angle
q_init_dh = robot.from_mech_to_dh(q_init)
print("q_init_dh: ", np.rad2deg(q_init_dh))
print("q_init_dh: ", q_init_dh)

# compute start pose
T_start = kin.forward_kinematics(robot, q_init_dh)
print("T_start = \n", T_start)

# Define goal pose
T_goal = T_start.copy()
T_goal[:3, 3] += np.array([0.0, 0.0, -0.1])
print("T_goal = \n", T_goal)

# IK with internal interpolation
q_final_dh = kin.inverse_kinematics(robot, q_init_dh, T_goal, use_orientation=True, k=0.8, n_iter=50)
T_final = kin.forward_kinematics(robot, q_final_dh)

print("Final joint angles = ", q_final_dh)
print("Final pose direct kinematics = \n", T_final)

print("err_lin = ", RobotUtils.calc_lin_err(T_goal, T_final))
print("err_ang = ", RobotUtils.calc_ang_err(T_goal, T_final))

# convert from dh angle to mechanical angle
q_final_mech = robot.from_dh_to_mech(q_final_dh)
print("q_final_mech: ", np.rad2deg(q_final_mech))

# add gripper position
gripper_pose = np.deg2rad(0.0)
q_final_mech = np.append(q_final_mech, gripper_pose)

# raise an error in case joint limits are exceeded
robot.check_joint_limits(q_final_mech)
