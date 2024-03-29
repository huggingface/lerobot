import numpy as np
from dm_control import mujoco
from dm_control.rl import control

from aloha.constants import (
    ASSETS_DIR,
    DT,
)
from aloha.tasks.sim import InsertionTask, TransferCubeTask
from aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)


def make_env_task(task_name):
    # time limit is controlled by StepCounter in env factory
    time_limit = float("inf")

    if "sim_transfer_cube" in task_name:
        xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = TransferCubeTask(random=False)
    elif "sim_insertion" in task_name:
        xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = InsertionTask(random=False)
    elif "sim_end_effector_transfer_cube" in task_name:
        raise NotImplementedError()
        xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = TransferCubeEndEffectorTask(random=False)
    elif "sim_end_effector_insertion" in task_name:
        raise NotImplementedError()
        xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = InsertionEndEffectorTask(random=False)
    else:
        raise NotImplementedError(task_name)

    env = control.Environment(
        physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
    )
    return env


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose
