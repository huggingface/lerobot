# Just teleop
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{}' --control.type=teleoperate

python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=teleoperate --control.display_data=true


# Record one episode locally
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/aloha_test --control.num_episodes=1 --robot.cameras='{}' --control.push_to_hub=false --control.fps=200
# with camera
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/aloha_test --control.num_episodes=1 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.fps=200

# Visualize 
python lerobot/scripts/visualize_dataset.py  --repo-id ${USER}/aloha_test   --episode-index 0
# Replay
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=replay --control.fps=60 --control.repo_id=${USER}/aloha_test --control.episode=0 

# Record dataset
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/aloha_mug --control.num_episodes=100 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3

# Train
python lerobot/scripts/train.py --dataset.repo_id=${USER}/aloha_test --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha_test --job_name=diifPo_aloha_test --policy.device=cuda --wandb.enable=true

# Rollout
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=60 --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/eval_aloha_mug --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_aloha_mug/checkpoints/last/pretrained_model/


# Misc mapping stuff


default teleop
state tensor([ 91.7578, 198.8965, 199.5996, 174.1992, 174.3750,   5.9766,  22.2363, -10.6348,  68.1441,  92.1094, 196.2598, 196.1719, 172.9688, 173.1445, 4.3066,  20.0391,  -5.7129,  59.3301])

default teleop in radian
array([ 1.60147572,  3.4713988 ,  3.48367021,  3.04034959,  3.04341788, 0.10431135,  0.38809665, -0.18561228,  1.18933891,  1.6076123 , 3.4253797 ,  3.42384555,  3.01887506,  3.02194161,  0.07516435, 0.34974827, -0.09970891,  1.03550559])

after mapping to qpos
map = [1.60147572, 3.4713988, 3.04034959, 0.10431135,  0.38809665, -0.18561228,  1.18933891, 1.18933891, 1.6076123 , 3.4253797, 3.01887506, 0.07516435, 0.34974827, -0.09970891,  1.03550559, 1.03550559]


in configuration.q
['left/waist',
'left/shoulder', 
'left/elbow', 
'left/forearm_roll', 
'left/wrist_angle', 
'left/wrist_rotate', 
'left/left_finger', 
'left/right_finger', 
'right/waist', 
'right/shoulder', 
'right/elbow', 
'right/forearm_roll', 
'right/wrist_angle', 
'right/wrist_rotate', 
'right/left_finger', 
'right/right_finger']

in state
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),

idx mapping
0 - 0
1 - 1
2 - 3
3 - 5
4 - 6
5 - 7

6 - 8
7 - 8

8 - 9
9 - 10
10 - 12
11 - 14
12 - 15
13 - 16

14 - 17
15 - 17


mapping real to sim (for IK, for the right hand) (*sign + offset)
signs = [-1, -1, 1, 1, 1, 1, ]
offsets = [pi/2, pi/2, -pi/2, 0, 0, 0]


### viewer

import mujoco.viewer
import mujoco
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("aloha_mj_description")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)


### fk/ik

def add_eef_pose(dataset):
    model = load_robot_description("aloha_mj_description")
    configuration = mink.Configuration(model)
    episode_outputs = []
    for i in tqdm(range(len(dataset))):
        states = dataset[i]['observation.state']
        action = dataset[i]['action']
        
        right_eef_pose_state = []
        right_gripper_articulation_state = []
        for vec in states:
            eef_pose, eef_pose_se3 = forward_kinematics(configuration, vec)
            right_eef_pose_state.append(eef_pose)
            right_gripper_articulation_state.append([vec[-1]])
        right_eef_pose_state = np.concatenate([np.array(right_eef_pose_state), np.array(right_gripper_articulation_state)], axis=1)

        # right_eef_pose_action = []
        # right_gripper_articulation_action = []
        # for vec in action:
        #     eef_pose, eef_pose_se3 = forward_kinematics(configuration, vec)
        #     right_eef_pose_action.append(eef_pose)
        #     right_gripper_articulation_action.append([vec[-1]])
        # right_eef_pose_action = np.concatenate([np.array(right_eef_pose_action), np.array(right_gripper_articulation_action)], axis=1)

        # output = inverse_kinematics(configuration, right_eef_pose_action[-25]) # compare with action[-25]
        
        episode_outputs.append(right_eef_pose_state)
        if dataset[i]['episode_index'].item() != 0:
            break

    return dataset


## ik code

import mink
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import pytorch3d.transforms as transforms
from tqdm import tqdm

def map_sim2real(vec):
    """
    inverse of map_real2sim
    sim = real*sign + offset
    real = (sim - offset)*sign
    """
    # Set gripper fingers to 0, we don't care about them for IK
    sign = np.array([-1, -1, -1, 1, 1, 1, 1, 1, 0,    
                      -1, -1, -1, 1, 1, 1, 1, 1, 0])
    offset = np.array([np.pi/2, 0, 0, -np.pi/2, -np.pi/2, 0, 0, 0, 0,
                       np.pi/2, 0, 0, -np.pi/2, -np.pi/2, 0, 0, 0, 0])
    vec = (vec - offset)*sign

    # Inverted from real2sim
    real_min, real_max = 0.23, 3.59 
    sim_min, sim_max = -1.26, 1.85 

    vec[1] = (vec[1] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    vec[2] = (vec[2] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    vec[10] = (vec[10] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    vec[11] = (vec[11] - sim_min)*((real_max-real_min)/(sim_max-sim_min)) + real_min
    return vec


def inverse_kinematics(configuration, ee_pose):
    rot_6d, trans, articulation = ee_pose[:6], ee_pose[6:9], ee_pose[9]
    pose_matrix = np.eye(4)
    pose_matrix[:3,3] = trans
    pose_matrix[:3,:3] = transforms.rotation_6d_to_matrix(torch.tensor(rot_6d)[None]).squeeze().numpy()
    ee_pose_se3 = mink.lie.se3.SE3.from_matrix(pose_matrix)
    
    ee_task = mink.FrameTask(frame_name="right/gripper", frame_type="site", position_cost=1., orientation_cost=1.)
    ee_task.set_target(ee_pose_se3)
    n_iter = 200
    dt = 0.01
    thresh = 1e-3
    for i in range(n_iter):
        vel = mink.solve_ik(configuration, [ee_task], dt=dt, solver='daqp')
        configuration.integrate_inplace(vel, dt)

        err = ee_task.compute_error(configuration)
        print(i, np.linalg.norm(err))
        if np.linalg.norm(err) < thresh: break
    
    Q = configuration.q
    vec = np.array(
        [
            Q[0],
            Q[1],
            Q[1],
            Q[2],
            Q[2],
            Q[3],
            Q[4],
            Q[5],
            0,

            Q[8],
            Q[9],
            Q[9],
            Q[10],
            Q[10],
            Q[11],
            Q[12],
            Q[13],
            0,
        ]
    )
    vec = np.rad2deg(map_sim2real(vec))
    vec[-1] = articulation
    return vec

ALOHA_MODEL = load_robot_description("aloha_mj_description")
ALOHA_CONFIGURATION = mink.Configuration(ALOHA_MODEL)
