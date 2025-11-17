import time
from dataclasses import dataclass

from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import BiSourcceyLeader
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.config_bi_sourccey_leader import BiSourcceyLeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.configs import parser


@dataclass
class SourcceyTeleoperateConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.243"
    left_arm_port: str = "COM4"
    right_arm_port: str = "COM3"
    keyboard_port: str = "keyboard"
    fps: int = 30

@parser.wrap()
def teleoperate(cfg: SourcceyTeleoperateConfig):
    # Create the robot and teleoperator configurations
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    teleop_arm_config = BiSourcceyLeaderConfig(left_arm_port=cfg.left_arm_port, right_arm_port=cfg.right_arm_port, id=cfg.id)
    keyboard_config = KeyboardTeleopConfig(id=cfg.keyboard_port)

    robot = SourcceyClient(robot_config)
    leader_arm = BiSourcceyLeader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    init_rerun(session_name="sourccey_teleop")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm of keyboard is not connected!")

    print("Teleoperating Sourccey")
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()

        arm_action = leader_arm.get_action()

        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        log_rerun_data(observation, {**arm_action, **base_action})

        action = {**arm_action, **base_action}

        robot.send_action(action)

        busy_wait(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))

def main():
    teleoperate()

if __name__ == "__main__":
    main()
