#!/usr/bin/env python
"""
OpenArms Policy Evaluation with Relative Actions

Two modes supported (based on training config):
  Mode 1: Relative actions only (use_relative_state=False)
    - Policy outputs relative action deltas
    - State input is absolute
  Mode 2: Relative actions + state (use_relative_state=True)
    - Policy outputs relative action deltas  
    - State input is also converted to relative

Example usage:
    python examples/openarms/evaluate_relative.py
"""

import time
from pathlib import Path

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.core import RobotAction
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, precise_sleep, predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.relative_actions import (
    convert_from_relative_actions_dict,
    convert_state_to_relative,
    PerTimestepNormalizer,
)
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# Configuration
HF_MODEL_ID = "your-org/your-relative-policy"
HF_EVAL_DATASET_ID = "your-org/your-eval-dataset"
TASK_DESCRIPTION = "your task description"

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 1000

FOLLOWER_LEFT_PORT = "can0"
FOLLOWER_RIGHT_PORT = "can1"

CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=FPS),
}


def make_robot_action(action_values: dict, features: dict) -> RobotAction:
    robot_action = {}
    for key in features:
        if key.startswith(ACTION + "."):
            action_key = key.removeprefix(ACTION + ".")
            if action_key in action_values:
                robot_action[action_key] = action_values[action_key]
    return robot_action


def load_relative_config(model_path: Path | str) -> tuple[PerTimestepNormalizer | None, bool]:
    """Load normalizer and relative_state setting from checkpoint."""
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    normalizer = None
    use_relative_state = False
    
    # Try local path first
    if model_path.exists():
        stats_path = model_path / "relative_stats.pt"
        if stats_path.exists():
            normalizer = PerTimestepNormalizer.load(stats_path)
            print(f"Loaded per-timestep stats from: {stats_path}")
        
        config_path = model_path / "train_config.json"
        if config_path.exists():
            cfg = TrainPipelineConfig.from_pretrained(model_path)
            use_relative_state = getattr(cfg, "use_relative_state", False)
    else:
        # Try hub
        try:
            from huggingface_hub import hf_hub_download
            stats_file = hf_hub_download(repo_id=str(model_path), filename="relative_stats.pt")
            normalizer = PerTimestepNormalizer.load(stats_file)
            print("Loaded per-timestep stats from hub")
            
            config_file = hf_hub_download(repo_id=str(model_path), filename="train_config.json")
            cfg = TrainPipelineConfig.from_pretrained(Path(config_file).parent)
            use_relative_state = getattr(cfg, "use_relative_state", False)
        except Exception as e:
            print(f"Warning: Could not load relative config: {e}")
    
    return normalizer, use_relative_state


def inference_loop_relative(
    robot,
    policy,
    preprocessor,
    postprocessor,
    dataset,
    events,
    fps: int,
    control_time_s: float,
    single_task: str,
    display_data: bool = True,
    state_key: str = "observation.state",
    relative_normalizer: PerTimestepNormalizer | None = None,
    use_relative_state: bool = False,
):
    """
    Inference loop for relative action policies.
    
    If use_relative_state=True, also converts observation state to relative.
    """
    device = get_safe_torch_device(policy.config.device)
    timestamp = 0
    start_t = time.perf_counter()
    
    while timestamp < control_time_s:
        loop_start = time.perf_counter()
        
        if events["exit_early"] or events["stop_recording"]:
            break
        
        obs = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
        
        # Convert state to relative if using full UMI mode
        if use_relative_state and state_key in observation_frame:
            state_tensor = observation_frame[state_key]
            if isinstance(state_tensor, torch.Tensor):
                observation_frame[state_key] = convert_state_to_relative(state_tensor)
        
        # Policy inference (outputs normalized relative actions)
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        
        # Unnormalize actions
        if relative_normalizer is not None:
            action_keys = [k for k in action_values.keys() if not k.startswith("task")]
            action_tensor = torch.tensor([[action_values[k] for k in action_keys]])
            action_tensor = action_tensor.unsqueeze(1)
            action_unnorm = relative_normalizer.unnormalize(action_tensor)
            for i, k in enumerate(action_keys):
                action_values[k] = action_unnorm[0, 0, i].item()
        
        # Convert to absolute
        relative_action = make_robot_action(action_values, dataset.features)
        absolute_action = convert_from_relative_actions_dict(relative_action, current_pos)
        
        robot.send_action(absolute_action)
        
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, absolute_action, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        
        if display_data:
            log_rerun_data(observation=obs, action=absolute_action)
        
        dt = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt)
        timestamp = time.perf_counter() - start_t


def main():
    print("=" * 60)
    print("  OpenArms Evaluation - Relative Actions")
    print("=" * 60)
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Dataset: {HF_EVAL_DATASET_ID}")
    print(f"Episodes: {NUM_EPISODES}, Duration: {EPISODE_TIME_SEC}s")
    
    # Load relative action config
    relative_normalizer, use_relative_state = load_relative_config(HF_MODEL_ID)
    mode = "actions + state" if use_relative_state else "actions only"
    print(f"Mode: relative {mode}")
    
    # Setup robot
    follower_config = OpenArmsFollowerConfig(
        port_left=FOLLOWER_LEFT_PORT,
        port_right=FOLLOWER_RIGHT_PORT,
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=10.0,
        cameras=CAMERA_CONFIG,
    )
    
    follower = OpenArmsFollower(follower_config)
    follower.connect(calibrate=False)
    
    if not follower.is_connected:
        raise RuntimeError("Robot failed to connect!")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    action_features_hw = {k: v for k, v in follower.action_features.items() if k.endswith(".pos")}
    
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
    )
    
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / HF_EVAL_DATASET_ID
    if dataset_path.exists():
        print(f"\nDataset exists at: {dataset_path}")
        if input("Continue? (y/n): ").strip().lower() != 'y':
            follower.disconnect()
            return
    
    dataset = LeRobotDataset.create(
        repo_id=HF_EVAL_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=12, 
    )
    
    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID
    policy = make_policy(policy_config, ds_meta=dataset.meta)
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarms_eval_relative")
    episode_idx = 0
    
    print("\nControls: ESC=stop, →=next episode, ←=rerecord")
    
    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Episode {episode_idx + 1} of {NUM_EPISODES}")
            
            inference_loop_relative(
                robot=follower,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                events=events,
                fps=FPS,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                relative_normalizer=relative_normalizer,
                use_relative_state=use_relative_state,
            )
            
            if events.get("rerecord_episode", False):
                log_say("Re-recording")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                print(f"Saving episode {episode_idx + 1}...")
                dataset.save_episode()
                episode_idx += 1
            
            events["exit_early"] = False
            
            if not events["stop_recording"] and episode_idx < NUM_EPISODES:
                input("Press ENTER for next episode...")
        
        print(f"\nDone! {episode_idx} episodes recorded")
        log_say("Complete", blocking=True)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        follower.disconnect()
        if listener is not None:
            listener.stop()
        dataset.finalize()
        print("Uploading to Hub...")
        dataset.push_to_hub(private=True)


if __name__ == "__main__":
    main()
