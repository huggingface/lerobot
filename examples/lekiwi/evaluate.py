import time
from dataclasses import dataclass
from enum import Enum

import draccus
import rerun as rr

from examples.lekiwi.utils import display_data
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, log_say
from lerobot.common.utils.visualization_utils import _init_rerun


class PolicyType(Enum):
    ACT = "act"
    SMOLVLA = "smolvla"
    PI0 = "pi0"


@dataclass
class EvaluateConfig:
    nb_cycles: int = 9000
    fps: int = 30
    robot_ip: str = "192.168.1.204"
    robot_id: str = "lekiwi"
    policy_type: PolicyType = PolicyType.ACT
    policy_name: str = "outputs/train/act_lekiwi_001_tape_c/checkpoints/020000/pretrained_model"
    task_description: str = "Grab the tape and put it in the cup"
    display_data: bool = False
    rerun_session_name: str = "lekiwi_evaluation"
    play_sounds: bool = True


def load_policy(policy_type: PolicyType, policy_name: str) -> PreTrainedPolicy:
    """Load the specified policy type."""
    if policy_type == PolicyType.ACT:
        policy = ACTPolicy.from_pretrained(policy_name)
        return policy
    # elif policy_type == PolicyType.SMOLVLA:
    #     return SmolvlaPolicy.from_pretrained(policy_name)
    # elif policy_type == PolicyType.PI0:
    #     return Pi0Policy.from_pretrained(policy_name)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")


def evaluate_loop(
    robot,
    policy: PreTrainedPolicy,
    fps: int,
    control_time_s: int | None = None,
    task_description: str | None = None,
    should_display_data: bool = False,
):
    """Evaluation loop that handles policy inference and robot control."""
    policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()

    obs_features = hw_to_dataset_features(robot.observation_features, "observation", False)
    device = get_safe_torch_device(policy.config.device)

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation = robot.get_observation()
        observation_frame = build_dataset_frame(obs_features, observation, prefix="observation")
        action_values = predict_action(
            observation_frame,
            policy,
            device,
            False,
            task=task_description,
            robot_type=robot.robot_type,
        )
        action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        robot.send_action(action)

        if should_display_data:
            display_data(observation, action)

        # Maintain timing
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


@draccus.wrap()
def evaluate(cfg: EvaluateConfig):
    if cfg.display_data:
        _init_rerun(session_name=cfg.rerun_session_name)

    # Initialize robot
    robot_config = LeKiwiClientConfig(remote_ip=cfg.robot_ip, id=cfg.robot_id)
    robot = LeKiwiClient(robot_config)

    # Load policy based on type
    print(f"Loading {cfg.policy_type.value} policy: {cfg.policy_name}")
    policy = load_policy(cfg.policy_type, cfg.policy_name)

    # Connect to robot
    robot.connect()
    if not robot.is_connected:
        print("Failed to connect to robot")
        return

    try:
        control_time_s = cfg.nb_cycles / cfg.fps

        log_say(f"Starting evaluation for {control_time_s:.1f} seconds", cfg.play_sounds)

        evaluate_loop(
            robot=robot,
            policy=policy,
            fps=cfg.fps,
            control_time_s=control_time_s,
            task_description=cfg.task_description,
            should_display_data=cfg.display_data,
        )

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user")
    finally:
        print("Cleaning up...")
        rr.rerun_shutdown()
        robot.disconnect()
        log_say("Evaluation ended", cfg.play_sounds)
        print("Evaluation ended")


if __name__ == "__main__":
    evaluate()
