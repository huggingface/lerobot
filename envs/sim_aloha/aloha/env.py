from dm_control import mujoco
from dm_control.rl import control

from aloha.constants import ASSETS_DIR, DT
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
