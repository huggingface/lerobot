#!/usr/bin/env python
"""Run a progress-conditioned ACT policy on the SO101 space-bar task."""

import logging
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.rollout import BaseStrategyConfig, RolloutConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from _spacebar_progress import (
    PROGRESS_FEATURE_NAMES,
    progress_vector,
    read_dataset_repo_id_from_train_config,
)

logger = logging.getLogger(__name__)


@dataclass
class ProgressACTRolloutConfig(RolloutConfig):
    """Rollout config with progress-feature-specific controls."""

    training_dataset_repo_id: str | None = None
    training_dataset_root: str | None = None
    progress_duration: float | None = None
    seed_move_duration_s: float = 3.0
    seed_move_fps: int = 50

    def __post_init__(self):
        super().__post_init__()


def resolve_training_dataset(cfg: ProgressACTRolloutConfig) -> LeRobotDataset:
    repo_id = cfg.training_dataset_repo_id
    if repo_id is None:
        repo_id = read_dataset_repo_id_from_train_config(cfg.policy.pretrained_path)
    if repo_id is None:
        raise ValueError(
            "Could not infer the training dataset repo id from train_config.json. "
            "Pass --training_dataset_repo_id=<user>/so101_blind_task2_progress_v1."
        )
    root = Path(cfg.training_dataset_root) if cfg.training_dataset_root else None
    dataset = LeRobotDataset(repo_id, root=root)
    if OBS_ENV_STATE not in dataset.meta.features:
        raise ValueError(f"{repo_id} does not contain {OBS_ENV_STATE}; use the progress-augmented dataset.")
    if dataset.meta.features[OBS_ENV_STATE].get("names") != PROGRESS_FEATURE_NAMES:
        logger.warning(
            "%s names are %s, expected %s",
            OBS_ENV_STATE,
            dataset.meta.features[OBS_ENV_STATE].get("names"),
            PROGRESS_FEATURE_NAMES,
        )
    return dataset


def load_policy_and_processors(cfg: ProgressACTRolloutConfig):
    policy_cfg = cfg.policy
    policy_cfg.device = cfg.device
    if OBS_ENV_STATE not in policy_cfg.input_features:
        raise ValueError(f"Policy input features do not include {OBS_ENV_STATE}. Was it trained on the derived dataset?")
    if policy_cfg.type != "act":
        logger.warning("This rollout script is tuned for ACT; loaded policy type is %s", policy_cfg.type)

    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(policy_cfg.pretrained_path, config=policy_cfg)
    policy.to(cfg.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    return policy, preprocessor, postprocessor


def reset_inference(policy, preprocessor, postprocessor) -> None:
    for obj in (policy, preprocessor, postprocessor):
        reset = getattr(obj, "reset", None)
        if callable(reset):
            reset()


def move_to_position(robot, target: dict[str, float], duration_s: float, fps: int) -> None:
    try:
        current_obs = robot.get_observation()
        current_pos = {key: current_obs[key] for key in target if key in current_obs}
        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            alpha = step / steps
            action = {key: current_pos[key] * (1.0 - alpha) + target[key] * alpha for key in current_pos}
            robot.send_action(action)
            precise_sleep(1.0 / fps)
    except Exception as exc:
        logger.warning("Could not move robot to target position: %s", exc)


def seed_start_position(
    dataset: LeRobotDataset,
    robot,
    episode: int,
    duration_s: float,
    fps: int,
) -> None:
    state_feature = dataset.meta.features.get(OBS_STATE)
    if state_feature is None or "names" not in state_feature:
        raise ValueError(f"{OBS_STATE} names are required to seed the robot start position.")
    episode_meta = dataset.meta.episodes[episode]
    start_idx = int(episode_meta["dataset_from_index"])
    state = np.asarray(dataset[start_idx][OBS_STATE], dtype=np.float32)
    motor_names = state_feature["names"]
    if len(motor_names) != len(state):
        raise ValueError(f"Motor names length {len(motor_names)} does not match state length {len(state)}.")
    target = {name: float(state[i]) for i, name in enumerate(motor_names)}
    logger.info("Seeding robot from episode %d start pose: %s", episode, {k: f"{v:.3f}" for k, v in target.items()})
    move_to_position(robot, target, duration_s=duration_s, fps=fps)


def add_progress_to_observation(
    raw_observation: dict,
    elapsed_s: float,
    progress_duration_s: float,
    fps: float,
) -> dict:
    frame_count = max(int(round(progress_duration_s * fps)), 2)
    virtual_frame = int(round(np.clip(elapsed_s / progress_duration_s, 0.0, 1.0) * (frame_count - 1)))
    progress = progress_vector(virtual_frame, frame_count)
    observation = dict(raw_observation)
    for name, value in zip(PROGRESS_FEATURE_NAMES, progress, strict=True):
        observation[name] = float(value)
    return observation


def infer_action(
    policy,
    preprocessor,
    postprocessor,
    observation_frame: dict,
    cfg: ProgressACTRolloutConfig,
) -> torch.Tensor:
    device = torch.device(cfg.device or "cpu")
    autocast_ctx = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and getattr(policy.config, "use_amp", False)
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        observation = {key: value.copy() for key, value in observation_frame.items()}
        observation = prepare_observation_for_inference(
            observation,
            device,
            task=cfg.task,
            robot_type=cfg.robot.type if cfg.robot else None,
        )
        observation = preprocessor(observation)
        action = policy.select_action(observation)
        action = postprocessor(action)
    return action.squeeze(0).cpu()


@parser.wrap()
def rollout(cfg: ProgressACTRolloutConfig) -> None:
    init_logging()
    if cfg.fps <= 0:
        cfg.fps = 30.0
    if cfg.duration <= 0:
        cfg.duration = 10.0
    progress_duration_s = cfg.progress_duration if cfg.progress_duration is not None else cfg.duration
    if progress_duration_s <= 0:
        raise ValueError("--progress_duration must be positive.")
    if not isinstance(cfg.strategy, BaseStrategyConfig):
        raise ValueError("Use --strategy.type=base for this no-recording progress rollout.")

    if cfg.display_data:
        init_rerun(session_name="progress_act_rollout", ip=cfg.display_ip, port=cfg.display_port)

    dataset = resolve_training_dataset(cfg)
    dataset_features = dataset.meta.features
    policy, preprocessor, postprocessor = load_policy_and_processors(cfg)
    reset_inference(policy, preprocessor, postprocessor)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    robot = make_robot_from_config(cfg.robot)
    logger.info("Connecting robot (%s)...", cfg.robot.type if cfg.robot else "?")
    robot.connect()
    initial_observation = robot.get_observation()
    initial_position = {key: value for key, value in initial_observation.items() if key.endswith(".pos")}

    try:
        seed_start_position(
            dataset,
            robot,
            episode=cfg.strategy.seed_episode,
            duration_s=cfg.seed_move_duration_s,
            fps=cfg.seed_move_fps,
        )
        reset_inference(policy, preprocessor, postprocessor)
        logger.info(
            "Starting progress ACT rollout: duration=%.2fs fps=%.1f progress_duration=%.2fs",
            cfg.duration,
            cfg.fps,
            progress_duration_s,
        )
        start_time = time.perf_counter()
        control_interval = 1.0 / cfg.fps
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            elapsed_s = loop_start - start_time
            if elapsed_s >= cfg.duration:
                break

            raw_observation = robot.get_observation()
            observation_values = add_progress_to_observation(raw_observation, elapsed_s, progress_duration_s, cfg.fps)
            observation_frame = build_dataset_frame(dataset_features, observation_values, prefix=OBS_STR)
            action_tensor = infer_action(policy, preprocessor, postprocessor, observation_frame, cfg)
            action_dict = make_robot_action(action_tensor, dataset_features)
            sent_action = robot.send_action(action_dict)

            if cfg.display_data:
                log_rerun_data(observation=observation_frame, action=sent_action)

            dt = time.perf_counter() - loop_start
            if (sleep_s := control_interval - dt) > 0:
                precise_sleep(sleep_s)
            else:
                logger.warning("Control loop slower than target FPS: %.1f Hz vs %.1f Hz", 1.0 / dt, cfg.fps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if robot.is_connected:
            if cfg.return_to_initial_position and initial_position:
                logger.info("Returning robot to initial position before disconnect")
                move_to_position(robot, initial_position, duration_s=3.0, fps=50)
            robot.disconnect()
        logger.info("Progress ACT rollout finished")


def main() -> None:
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
