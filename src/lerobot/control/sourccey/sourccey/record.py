from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import BiSourcceyLeader
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.config_bi_sourccey_leader import BiSourcceyLeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.configs import parser
from dataclasses import dataclass, field

@dataclass
class DatasetRecordConfig:
    repo_id: str = "sourccey-003/sourccey-003__tyler_eaxmpel"
    num_episodes: int = 5
    episode_time_s: int = 60
    reset_time_s: int = 5
    task: str = "Follow me"
    fps: int = 30
    push_to_hub: bool = False
    private: bool = False

@dataclass
class SourcceyRecordConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.243"
    left_arm_port: str = "COM4"
    right_arm_port: str = "COM3"
    keyboard: str = "keyboard"
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)

@parser.wrap()
def record(cfg: SourcceyRecordConfig):
    # Create the robot and teleoperator configurations
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    teleop_arm_config = BiSourcceyLeaderConfig(left_arm_port=cfg.left_arm_port, right_arm_port=cfg.right_arm_port, id=cfg.id)
    keyboard_config = KeyboardTeleopConfig(id=cfg.keyboard)

    robot = SourcceyClient(robot_config)
    leader_arm = BiSourcceyLeader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.dataset.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    init_rerun(session_name="sourccey_record")

    listener, events = init_keyboard_listener()

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm of keyboard is not connected!")

    recorded_episodes = 0
    while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes}")

        # Run the record loop
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.task,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Logic for reset env
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.task,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # Upload to hub and clean up
    # dataset.push_to_hub()
    log_say("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    listener.stop()

def main():
    record()

if __name__ == "__main__":
    main()


