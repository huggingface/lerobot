#!/usr/bin/env python
"""
RaC (Recovery and Correction) Data Collection with RTC for OpenArms Robot.

Combines RaC paradigm with Real-Time Chunking (RTC) for smooth policy execution.
RTC enables large flow-matching policies (Pi0, Pi0.5, SmolVLA) to produce reactive
motion despite high inference latency by asynchronously generating action chunks.

The workflow:
1. Policy runs via RTC (async action generation) - teleop is idle
2. Press SPACE/right pedal to pause - teleop moves to match robot position
3. Press 'c'/left pedal to take control - human provides RECOVERY + CORRECTION
4. Press →/right pedal to end episode (save and continue to next)

Controls:
    SPACE/Right pedal  - Pause policy
    c/Left pedal       - Take control (start correction)
    →/Right pedal      - End episode (when in correction mode)
    ESC                - Stop recording and push to hub

Usage:
    python examples/rac/rac_data_collection_openarms_rtc.py \
        --robot.type=openarms_follower \
        --robot.port_right=can0 \
        --robot.port_left=can1 \
        --robot.cameras="{ left_wrist: {type: opencv, index_or_path: 0}, right_wrist: {type: opencv, index_or_path: 2}}" \
        --teleop.type=openarms_mini \
        --teleop.port_right=/dev/ttyUSB0 \
        --teleop.port_left=/dev/ttyUSB1 \
        --policy.path=outputs/train/my_policy/checkpoints/last/pretrained_model \
        --dataset.repo_id=my_user/rac_rtc_dataset \
        --dataset.single_task="Pick up the cube"
"""

import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import make_default_processors
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.openarms_mini.config_openarms_mini import OpenArmsMiniConfig  # noqa: F401
from lerobot.utils.control_utils import is_headless
from lerobot.utils.hub import HubMixin
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotWrapper:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: dict) -> None:
        with self.lock:
            self.robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        return self.robot.observation_features

    @property
    def action_features(self) -> dict:
        return self.robot.action_features

    @property
    def name(self) -> str:
        return self.robot.name

@dataclass
class RaCRTCDatasetConfig:
    """Dataset configuration for RaC + RTC."""
    repo_id: str = "lerobot/rac_rtc_openarms"
    single_task: str = "task"
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: float = 120
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = True
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4


@dataclass
class RaCRTCConfig(HubMixin):
    """Configuration for RaC data collection with RTC."""

    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: RaCRTCDatasetConfig
    policy: PreTrainedConfig | None = None

    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=20,
            max_guidance_weight=5.0,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        )
    )

    device: str = "cuda"
    action_queue_size_to_get_new_actions: int = 30
    interpolation: bool = True
    play_sounds: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("policy.path is required")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def init_keyboard_listener(events: dict):
    """Initialize keyboard listener with RaC controls."""
    if is_headless():
        logger.warning("Headless environment - keyboard controls unavailable")
        return None

    from pynput import keyboard

    def on_press(key):
        try:
            if events["in_reset"]:
                if key == keyboard.Key.space or key == keyboard.Key.right:
                    events["start_next_episode"] = True
                elif hasattr(key, "char") and key.char == "c":
                    events["start_next_episode"] = True
                elif key == keyboard.Key.esc:
                    events["stop_recording"] = True
                    events["start_next_episode"] = True
            else:
                if key == keyboard.Key.space:
                    if not events["policy_paused"] and not events["correction_active"]:
                        print("\n[RaC] ⏸ PAUSED - Press 'c' to take control")
                        events["policy_paused"] = True
                elif hasattr(key, "char") and key.char == "c":
                    if events["policy_paused"] and not events["correction_active"]:
                        print("\n[RaC] ▶ Taking control...")
                        events["start_correction"] = True
                elif key == keyboard.Key.right:
                    print("\n[RaC] → End episode")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("\n[RaC] ← Re-record episode")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    events["stop_recording"] = True
                    events["exit_early"] = True
        except Exception as e:
            print(f"Key error: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    start_pedal_listener(events)
    return listener


def start_pedal_listener(events: dict):
    """Start foot pedal listener if available."""
    import threading

    try:
        from evdev import InputDevice, ecodes
    except ImportError:
        return

    PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    KEY_LEFT = "KEY_A"
    KEY_RIGHT = "KEY_C"

    def pedal_reader():
        try:
            dev = InputDevice(PEDAL_DEVICE)
            print(f"[Pedal] Connected: {dev.name}")

            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue

                from evdev import categorize

                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]

                if key.keystate != 1:
                    continue

                if events["in_reset"]:
                    if code in [KEY_LEFT, KEY_RIGHT]:
                        events["start_next_episode"] = True
                else:
                    if code == KEY_RIGHT:
                        if events["correction_active"]:
                            events["exit_early"] = True
                        elif not events["policy_paused"]:
                            print("\n[Pedal] ⏸ PAUSED")
                            events["policy_paused"] = True
                    elif code == KEY_LEFT:
                        if events["policy_paused"] and not events["correction_active"]:
                            print("\n[Pedal] ▶ Taking control...")
                            events["start_correction"] = True

        except FileNotFoundError:
            pass
        except PermissionError:
            logger.warning(f"[Pedal] Permission denied for {PEDAL_DEVICE}")
        except Exception:
            pass

    thread = threading.Thread(target=pedal_reader, daemon=True)
    thread.start()


def get_actions_thread(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RaCRTCConfig,
    policy_active: Event,
    fps: int,
):
    """Thread for async action generation via RTC."""
    try:
        logger.info("[GET_ACTIONS] Starting RTC action generation thread")

        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / fps

        hw_features = hw_to_dataset_features(robot.observation_features, "observation")
        policy_device = policy.config.device

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": cfg.device}},
        )

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions
        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if not policy_active.is_set():
                time.sleep(0.01)
                continue

            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk) if inference_latency else 0

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    hw_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.dataset.single_task]
                obs_with_policy_features["robot_type"] = robot.name

                preprocessed_obs = preprocessor(obs_with_policy_features)

                actions = policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                original_actions = actions.squeeze(0).clone()
                postprocessed_actions = postprocessor(actions).squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
            else:
                time.sleep(0.01)

        logger.info("[GET_ACTIONS] Shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


def move_robot_to_zero(robot, duration_s: float = 2.0, fps: int = 50):
    """Smoothly move robot to zero position."""
    obs = robot.get_observation()
    current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
    target_pos = {k: 0.0 for k in current_pos}

    print(f"[RaC] Moving robot to zero ({duration_s}s)...")
    steps = int(duration_s * fps)
    for step in range(steps + 1):
        t = step / steps
        interp_pos = {k: current_pos[k] * (1 - t) + target_pos[k] * t for k in current_pos}
        robot.send_action(interp_pos)
        time.sleep(1 / fps)


@parser.wrap()
def main(cfg: RaCRTCConfig):
    """Main RaC + RTC data collection."""
    init_logging()

    fps = cfg.dataset.fps

    print("=" * 65)
    print("  RaC Data Collection with RTC - OpenArms")
    print("=" * 65)
    print(f"  Policy: {cfg.policy.pretrained_path}")
    print(f"  Dataset: {cfg.dataset.repo_id}")
    print(f"  Task: {cfg.dataset.single_task}")
    print(f"  Policy Hz: {fps}")
    print(f"  Robot Hz: {fps * 2 if cfg.interpolation else fps}")
    print(f"  Interpolation: {cfg.interpolation}")
    print(f"  RTC Enabled: {cfg.rtc.enabled}")
    print("=" * 65)

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "policy_paused": False,
        "correction_active": False,
        "start_correction": False,
        "in_reset": False,
        "start_next_episode": False,
    }

    shutdown_event = Event()
    policy_active = Event()

    follower = make_robot_from_config(cfg.robot)
    follower.connect()
    robot = RobotWrapper(follower)
    logger.info("Robot connected")

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()
    teleop.disable_torque()
    logger.info("Teleop connected")

    teleop_proc, robot_proc, obs_proc = make_default_processors()
    action_features_hw = {k: v for k, v in follower.action_features.items() if k.endswith(".pos")}

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=fps,
        root=cfg.dataset.root,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
        * len(follower.cameras if hasattr(follower, "cameras") else []),
    )
    dataset_lock = Lock()

    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    policy = policy.to(cfg.device)
    policy.eval()
    logger.info(f"Policy loaded: {policy.name}")

    action_queue = ActionQueue(cfg.rtc)

    get_actions_t = Thread(
        target=get_actions_thread,
        args=(policy, robot, obs_proc, action_queue, shutdown_event, cfg, policy_active, fps),
        daemon=True,
        name="GetActions",
    )
    get_actions_t.start()

    listener = init_keyboard_listener(events)

    print("\n  Controls:")
    print("    SPACE/Right pedal  - Pause policy")
    print("    c/Left pedal       - Take control")
    print("    →/Right pedal      - End episode (in correction mode)")
    print("    ESC                - Stop and push to hub")
    print("=" * 65 + "\n")

    action_keys = [k for k in robot.action_features.keys() if k.endswith(".pos")]

    if cfg.interpolation:
        interp_factor = 2
        robot_interval = 1.0 / (fps * interp_factor)
    else:
        interp_factor = 1
        robot_interval = 1.0 / fps

    try:
        recorded = 0
        while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"RaC episode {recorded + 1}", play_sounds=cfg.play_sounds)

            move_robot_to_zero(follower, duration_s=2.0)

            action_queue = ActionQueue(cfg.rtc)
            events["policy_paused"] = False
            events["correction_active"] = False
            events["start_correction"] = False
            events["exit_early"] = False

            frame_buffer = []
            prev_action: Tensor | None = None
            interpolated_actions: list[Tensor] = []
            interp_idx = 0

            robot_send_count = 0
            policy_consume_count = 0
            last_hz_print = time.perf_counter()

            policy_active.set()
            episode_start = time.perf_counter()

            while (time.perf_counter() - episode_start) < cfg.dataset.episode_time_s:
                loop_start = time.perf_counter()

                if events["exit_early"]:
                    break

                if events["start_correction"] and not events["correction_active"]:
                    policy_active.clear()
                    print("[RaC] Moving teleop to robot position...")
                    obs = robot.get_observation()
                    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
                    teleop.smooth_move_to(robot_pos, duration_s=2.0, fps=50)
                    teleop.disable_torque()
                    events["correction_active"] = True
                    events["start_correction"] = False
                    print("[RaC] Correction mode - you have control")

                obs = robot.get_observation()
                obs_filtered = {k: v for k, v in obs.items() if k in robot.observation_features}
                obs_frame = build_dataset_frame(dataset_features, obs_filtered, prefix="observation")

                if events["correction_active"]:
                    robot_action = teleop.get_action()
                    for key in robot_action:
                        if "gripper" in key:
                            robot_action[key] = -0.65 * robot_action[key]
                    robot.send_action(robot_action)

                    action_frame = build_dataset_frame(dataset_features, robot_action, prefix="action")
                    frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                    frame_buffer.append(frame)

                elif events["policy_paused"]:
                    pass

                else:
                    if interp_idx >= len(interpolated_actions):
                        new_action = action_queue.get()
                        if new_action is not None:
                            current_action = new_action.cpu()
                            policy_consume_count += 1

                            if cfg.interpolation and prev_action is not None:
                                mid = prev_action + 0.5 * (current_action - prev_action)
                                interpolated_actions = [mid, current_action]
                            else:
                                interpolated_actions = [current_action]

                            prev_action = current_action
                            interp_idx = 0
                        else:
                            # No action yet - hold current position while waiting for first inference
                            hold_pos = {k: v for k, v in obs_filtered.items() if k.endswith(".pos")}
                            robot.send_action(hold_pos)
                            robot_send_count += 1

                    if interp_idx < len(interpolated_actions):
                        action_to_send = interpolated_actions[interp_idx]
                        interp_idx += 1

                        action_dict = {}
                        for i, key in enumerate(action_keys):
                            if i < len(action_to_send):
                                action_dict[key] = action_to_send[i].item()

                        action_processed = robot_proc((action_dict, None))
                        robot.send_action(action_processed)
                        robot_send_count += 1

                        action_frame = build_dataset_frame(dataset_features, action_dict, prefix="action")
                        frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                        frame_buffer.append(frame)

                now = time.perf_counter()
                if now - last_hz_print >= 5.0:
                    elapsed = now - last_hz_print
                    actual_robot_hz = robot_send_count / elapsed if elapsed > 0 else 0
                    actual_policy_hz = policy_consume_count / elapsed if elapsed > 0 else 0
                    logger.info(f"[ACTOR] Actual Hz - Robot: {actual_robot_hz:.1f}, Policy: {actual_policy_hz:.1f}")
                    robot_send_count = 0
                    policy_consume_count = 0
                    last_hz_print = now

                dt = time.perf_counter() - loop_start
                sleep_time = max(0, robot_interval - dt - 0.001)
                if sleep_time > 0:
                    precise_sleep(sleep_time)

            policy_active.clear()

            if events["rerecord_episode"]:
                log_say("Re-recording", play_sounds=cfg.play_sounds)
                events["rerecord_episode"] = False
                continue

            for frame in frame_buffer:
                dataset.add_frame(frame)

            with dataset_lock:
                if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                    logger.info(f"Saving episode ({dataset.episode_buffer['size']} frames)")
                    dataset.save_episode()

            recorded += 1

            if recorded < cfg.num_episodes and not events["stop_recording"]:
                events["in_reset"] = True
                events["start_next_episode"] = False
                print("\n[RaC] RESET - Press any key/pedal to enable teleop")

                obs = robot.get_observation()
                robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
                teleop.smooth_move_to(robot_pos, duration_s=2.0, fps=50)

                while not events["start_next_episode"] and not events["stop_recording"]:
                    precise_sleep(0.05)

                if events["stop_recording"]:
                    break

                events["start_next_episode"] = False
                teleop.disable_torque()
                print("[RaC] Teleop enabled - move to start, then press key/pedal")

                while not events["start_next_episode"] and not events["stop_recording"]:
                    action = teleop.get_action()
                    for key in action:
                        if "gripper" in key:
                            action[key] = -0.65 * action[key]
                    robot.send_action(action)
                    precise_sleep(1 / fps)

                events["in_reset"] = False
                events["start_next_episode"] = False

        log_say("Recording complete", play_sounds=cfg.play_sounds, blocking=True)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        shutdown_event.set()
        policy_active.clear()

        if get_actions_t.is_alive():
            get_actions_t.join(timeout=5.0)

        follower.disconnect()
        teleop.disconnect()

        if listener:
            listener.stop()

        dataset.finalize()
        if cfg.dataset.push_to_hub:
            logger.info("Pushing to hub...")
            dataset.push_to_hub(private=cfg.dataset.private)

        logger.info("Done")


if __name__ == "__main__":
    main()

