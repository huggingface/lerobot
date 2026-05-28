import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.robots.so_follower.config_so100_follower import SO100FollowerConfig
# from lerobot.robots.so_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.robots.assembling_sim import AssemblingSim, AssemblingSimCut, AssemblingSimConfig

episode_idx = 0

sim_config = AssemblingSimConfig(
    xml_path="scene.xml",
    sim_timestep=0.001,
    control_hz=20,
    mode="fast",   # "realtime" | "fast"
    max_episode_steps=1000,
    use_task_space=True,
    render_mode="all",   # None | "human" | "rgb_array" | "all"
    camera_names=["cam_front", "cam_side", "cam_gripper"],
    resolution=(224, 224),

    action_pos_scale=1000,
    action_angle_scale=100
)

# Initialize the robot and teleoperator
robot = AssemblingSimCut(sim_config)
robot.connect()

dataset = LeRobotDataset("local/ACT_assembling_sim_s2", episodes=[episode_idx])
actions = dataset.hf_dataset.select_columns("action")

log_say(f"Replaying episode {episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()