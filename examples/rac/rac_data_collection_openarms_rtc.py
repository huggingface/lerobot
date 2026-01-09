#!/usr/bin/env python
"""
RaC (Recovery and Correction) Data Collection for OpenArms Robot with RTC.

This combines RaC data collection with Real-Time Chunking (RTC) for smooth policy execution.
RTC enables large flow-matching policies (Pi0, Pi0.5, SmolVLA) to produce reactive motion
despite high inference latency by asynchronously generating action chunks.

The workflow:
1. Policy runs autonomously with RTC (teleop is idle/free)
2. Press SPACE to pause - teleop moves to match robot position
3. Press 'c' to take control - teleop is free, human provides RECOVERY + CORRECTION
4. Press → to end episode (save and continue to next)
5. Reset, then do next rollout

Keyboard Controls:
    SPACE  - Pause policy (teleop mirrors robot, no recording)
    c      - Take control (teleop free, recording correction)
    →      - End episode (save and continue to next)
    ←      - Re-record episode
    ESC    - Stop recording and push dataset to hub

Usage:
    python examples/rac/rac_data_collection_openarms_rtc.py \
        --robot.port_right=can0 \
        --robot.port_left=can1 \
        --teleop.port_right=/dev/ttyUSB0 \
        --teleop.port_left=/dev/ttyUSB1 \
        --policy.path=outputs/train/my_policy/checkpoints/last/pretrained_model \
        --dataset.repo_id=my_user/rac_openarms_dataset \
        --dataset.single_task="Pick up the cube"
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from threading import Event, Thread
from typing import Any

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import (
    IdentityProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.openarms_mini.config_openarms_mini import OpenArmsMiniConfig  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RaCRTCDatasetConfig:
    repo_id: str = "lerobot/rac_openarms_rtc"
    single_task: str = "default task"
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: float = 500
    reset_time_s: float = 30
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class RaCRTCConfig:
    robot: RobotConfig = field(default_factory=lambda: OpenArmsFollowerConfig(
        port_left="can0",
        port_right="can1",
    ))
    teleop: TeleoperatorConfig = field(default_factory=lambda: OpenArmsMiniConfig(
        port_left="/dev/ttyUSB1",
        port_right="/dev/ttyUSB0",
    ))
    dataset: RaCRTCDatasetConfig = field(default_factory=RaCRTCDatasetConfig)
    policy: PreTrainedConfig | None = None
    
    rtc: RTCConfig = field(default_factory=lambda: RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    ))
    
    interpolation: bool = True
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = False
    device: str = "cuda"
    action_queue_size_to_get_new_actions: int = 30

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


def init_rac_keyboard_listener():
    """Initialize keyboard listener with RaC-specific controls."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "policy_paused": False,
        "correction_active": False,
        "in_reset": False,
        "start_next_episode": False,
    }

    if is_headless():
        logging.warning("Headless environment - keyboard controls unavailable")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if events["in_reset"]:
                if key == keyboard.Key.space or key == keyboard.Key.right:
                    print("\n[RaC] Starting next episode...")
                    events["start_next_episode"] = True
                elif hasattr(key, 'char') and key.char == 'c':
                    print("\n[RaC] Starting next episode...")
                    events["start_next_episode"] = True
                elif key == keyboard.Key.esc:
                    print("[RaC] ESC - Stop recording, pushing to hub...")
                    events["stop_recording"] = True
                    events["start_next_episode"] = True
            else:
                if key == keyboard.Key.space:
                    if not events["policy_paused"] and not events["correction_active"]:
                        print("\n[RaC] ⏸ PAUSED - Policy stopped, teleop moving to robot position")
                        print("      Press 'c' or START to take control")
                        events["policy_paused"] = True
                elif hasattr(key, 'char') and key.char == 'c':
                    if events["policy_paused"] and not events["correction_active"]:
                        print("\n[RaC] ▶ START pressed - taking control")
                        events["start_next_episode"] = True
                elif key == keyboard.Key.right:
                    print("[RaC] → End episode")
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("[RaC] ← Re-record episode")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("[RaC] ESC - Stop recording, pushing to hub...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
        except Exception as e:
            print(f"Key error: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    start_pedal_listener(events)
    
    return listener, events


def start_pedal_listener(events: dict):
    """Start foot pedal listener thread if evdev is available."""
    import threading
    
    try:
        from evdev import InputDevice, ecodes  # noqa: F401
    except ImportError:
        logging.info("[Pedal] evdev not installed - pedal support disabled")
        return
    
    PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    KEY_LEFT = "KEY_A"
    KEY_RIGHT = "KEY_C"
    
    def pedal_reader():
        try:
            dev = InputDevice(PEDAL_DEVICE)
            print(f"[Pedal] Connected: {dev.name}")
            print(f"[Pedal] Right=pause/next, Left=take control/start")
            
            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue
                
                from evdev import categorize  # noqa: F401
                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]
                
                if key.keystate != 1:
                    continue
                
                if events["in_reset"]:
                    if code in [KEY_LEFT, KEY_RIGHT]:
                        print("\n[Pedal] Starting next episode...")
                        events["start_next_episode"] = True
                else:
                    if code == KEY_RIGHT:
                        if events["correction_active"]:
                            print("\n[Pedal] → End episode")
                            events["exit_early"] = True
                        elif not events["policy_paused"]:
                            print("\n[Pedal] ⏸ PAUSED - Policy stopped")
                            events["policy_paused"] = True
                    elif code == KEY_LEFT:
                        if events["policy_paused"] and not events["correction_active"]:
                            print("\n[Pedal] ▶ START pressed - taking control")
                            events["start_next_episode"] = True
                        
        except FileNotFoundError:
            logging.info(f"[Pedal] Device not found: {PEDAL_DEVICE}")
        except PermissionError:
            logging.warning(f"[Pedal] Permission denied. Run: sudo setfacl -m u:$USER:rw {PEDAL_DEVICE}")
        except Exception as e:
            logging.debug(f"[Pedal] Error: {e}")
    
    thread = threading.Thread(target=pedal_reader, daemon=True)
    thread.start()


def make_identity_processors():
    """Create identity processors for RaC recording."""
    teleop_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    obs_proc = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return teleop_proc, robot_proc, obs_proc


class SharedState:
    """Thread-safe shared state for RTC inference thread."""
    def __init__(self):
        self.obs: dict | None = None
        self.action_queue: ActionQueue | None = None


def rtc_inference_thread(
    policy,
    shared_state: SharedState,
    shutdown_event: Event,
    policy_active: Event,
    cfg: RaCRTCConfig,
    hw_features: dict,
    preprocessor,
    postprocessor,
):
    """Background thread that generates action chunks using RTC.
    
    This thread:
    - Waits for policy_active to be set
    - Uses observation from shared_state.obs (set by main loop)
    - Generates action chunks and puts them in shared_state.action_queue
    """
    logger.info("[RTC] Inference thread started (waiting for policy_active signal)")
    logger.info("[RTC] Thread is IDLE - will not do anything until main loop activates policy")
    
    latency_tracker = LatencyTracker()
    time_per_chunk = 1.0 / cfg.dataset.fps
    policy_device = policy.config.device
    
    get_actions_threshold = cfg.action_queue_size_to_get_new_actions
    if not cfg.rtc.enabled:
        get_actions_threshold = 0
    
    inference_count = 0
    was_active = False
    
    while not shutdown_event.is_set():
        if not policy_active.is_set():
            if was_active:
                logger.info("[RTC] Policy deactivated, thread going idle")
                was_active = False
            time.sleep(0.01)
            continue
        
        if not was_active:
            logger.info("[RTC] Policy activated! Starting inference loop")
            was_active = True
        
        action_queue = shared_state.action_queue
        if action_queue is None:
            time.sleep(0.01)
            continue
            
        if action_queue.qsize() <= get_actions_threshold:
            obs = shared_state.obs
            if obs is None:
                time.sleep(0.01)
                continue
            
            current_time = time.perf_counter()
            action_index_before_inference = action_queue.get_action_index()
            prev_actions = action_queue.get_left_over()
            
            inference_latency = latency_tracker.max()
            inference_delay = math.ceil(inference_latency / time_per_chunk) if inference_latency else 0
            
            obs_with_policy_features = build_dataset_frame(hw_features, obs, prefix="observation")
            
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
            obs_with_policy_features["robot_type"] = "openarms_follower"
            
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
            
            inference_count += 1
            if inference_count % 10 == 0:
                logger.debug(f"[RTC] Inference #{inference_count}, latency={new_latency:.2f}s, queue={action_queue.qsize()}")
        else:
            time.sleep(0.005)
    
    logger.info("[RTC] Inference thread shutting down")


@safe_stop_image_writer
def rac_rtc_rollout_loop(
    robot,
    teleop,
    shared_state: SharedState,
    policy_active: Event,
    dataset: LeRobotDataset,
    events: dict,
    cfg: RaCRTCConfig,
    action_keys: list[str],
) -> dict:
    """RaC rollout loop with RTC for smooth policy execution."""
    logger.info("[ROLLOUT] Starting rollout loop...")
    
    fps = cfg.dataset.fps
    single_task = cfg.dataset.single_task
    control_time_s = cfg.dataset.episode_time_s
    
    frame_buffer = []
    stats = {
        "total_frames": 0,
        "autonomous_frames": 0,
        "paused_frames": 0,
        "correction_frames": 0,
    }

    teleop.disable_torque()
    was_paused = False
    waiting_for_takeover = False
    
    # Interpolation state
    prev_action: Tensor | None = None
    interpolated_actions: list[Tensor] = []
    interp_idx = 0
    
    if cfg.interpolation:
        interp_factor = 2
        control_interval = 1.0 / (fps * interp_factor)
        logger.info(f"[ROLLOUT] Interpolation ON: {fps}Hz -> {fps * interp_factor}Hz")
    else:
        interp_factor = 1
        control_interval = 1.0 / fps
        logger.info(f"[ROLLOUT] Interpolation OFF: {fps}Hz")
    
    # Hz tracking
    robot_send_count = 0
    policy_consume_count = 0
    last_hz_time = time.perf_counter()
    last_record_time = 0.0
    
    timestamp = 0
    start_t = time.perf_counter()
    first_iteration = True

    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            break

        # Get observation (always from main thread - only place robot is read)
        if first_iteration:
            logger.info("[ROLLOUT] First iteration - reading observation from robot...")
        obs = robot.get_observation()
        if first_iteration:
            logger.info("[ROLLOUT] First observation read OK")
            first_iteration = False
        obs_filtered = {k: v for k, v in obs.items() if k in robot.observation_features}
        
        # Update shared observation for RTC thread
        shared_state.obs = obs_filtered

        # State transition: entering paused state
        if events["policy_paused"] and not was_paused:
            policy_active.clear()  # Stop RTC inference
            robot_pos = {k: v for k, v in obs_filtered.items() if k.endswith(".pos")}
            print("[RaC] Moving teleop to robot position (2s smooth transition)...")
            teleop.smooth_move_to(robot_pos, duration_s=2.0, fps=50)
            print("[RaC] Teleop aligned. Press START to take control.")
            events["start_next_episode"] = False
            waiting_for_takeover = True
            was_paused = True
            # Reset interpolation state
            prev_action = None
            interpolated_actions = []
            interp_idx = 0

        # Wait for start button before enabling correction mode
        if waiting_for_takeover and events["start_next_episode"]:
            print("[RaC] Start pressed - enabling teleop control...")
            teleop.disable_torque()
            events["start_next_episode"] = False
            events["correction_active"] = True
            waiting_for_takeover = False

        obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)

        if events["correction_active"]:
            # Human controlling - record correction data
            robot_action = teleop.get_action()
            for key in robot_action:
                if "gripper" in key:
                    robot_action[key] = -0.65 * robot_action[key]
            robot.send_action(robot_action)
            robot_send_count += 1
            stats["correction_frames"] += 1
            
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": single_task}
            frame_buffer.append(frame)
            stats["total_frames"] += 1
            
        elif waiting_for_takeover:
            # Waiting for START - policy stopped, no recording
            stats["paused_frames"] += 1
            
        elif events["policy_paused"]:
            # Paused - teleop tracks robot position
            robot_pos = {k: v for k, v in obs_filtered.items() if k.endswith(".pos")}
            teleop.send_feedback(robot_pos)
            stats["paused_frames"] += 1
            
        else:
            # Policy execution with RTC
            policy_active.set()
            action_queue = shared_state.action_queue
            
            # Get next action from queue (with interpolation)
            if interp_idx >= len(interpolated_actions):
                new_action = action_queue.get() if action_queue else None
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
            
            if interp_idx < len(interpolated_actions):
                action_to_send = interpolated_actions[interp_idx]
                interp_idx += 1
                
                action_dict = {}
                for i, key in enumerate(action_keys):
                    if i < len(action_to_send):
                        action_dict[key] = action_to_send[i].item()
                
                robot.send_action(action_dict)
                robot_send_count += 1
                stats["autonomous_frames"] += 1
                
                # Record at dataset fps (not interpolated rate)
                now = time.perf_counter()
                if now - last_record_time >= (1.0 / fps):
                    last_record_time = now
                    action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                    frame = {**obs_frame, **action_frame, "task": single_task}
                    frame_buffer.append(frame)
                    stats["total_frames"] += 1

        # Print Hz stats every 5 seconds
        now = time.perf_counter()
        if now - last_hz_time >= 5.0:
            elapsed = now - last_hz_time
            actual_robot_hz = robot_send_count / elapsed if elapsed > 0 else 0
            actual_policy_hz = policy_consume_count / elapsed if elapsed > 0 else 0
            mode = "CORRECTION" if events["correction_active"] else ("PAUSED" if events["policy_paused"] else "POLICY")
            logger.info(f"[Hz] Robot: {actual_robot_hz:.1f}, Policy: {actual_policy_hz:.1f}, Mode: {mode}")
            robot_send_count = 0
            policy_consume_count = 0
            last_hz_time = now

        if cfg.display_data:
            log_rerun_data(observation=obs_filtered, action=action_dict if 'action_dict' in dir() else {})

        dt = time.perf_counter() - loop_start
        sleep_time = control_interval - dt
        if sleep_time > 0:
            precise_sleep(sleep_time)
        timestamp = time.perf_counter() - start_t

    policy_active.clear()
    teleop.disable_torque()

    for frame in frame_buffer:
        dataset.add_frame(frame)

    return stats


def reset_loop(robot, teleop, events: dict, fps: int):
    """Reset period where human repositions environment."""
    print("\n" + "=" * 65)
    print("  [RaC] RESET - Moving teleop to robot position...")
    print("=" * 65)
    
    events["in_reset"] = True
    events["start_next_episode"] = False
    
    obs = robot.get_observation()
    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features}
    teleop.smooth_move_to(robot_pos, duration_s=2.0, fps=50)
    
    print("  Teleop aligned. Press any key/pedal to enable teleoperation")
    while not events["start_next_episode"] and not events["stop_recording"]:
        precise_sleep(0.05)
    
    if events["stop_recording"]:
        return
    
    events["start_next_episode"] = False
    teleop.disable_torque()
    print("  Teleop enabled - move robot to starting position")
    print("  Press any key/pedal to start next episode")

    while not events["start_next_episode"] and not events["stop_recording"]:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        for key in action:
            if "gripper" in key:
                action[key] = -0.65 * action[key]
        robot.send_action(action)
        dt = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt)
    
    events["in_reset"] = False
    events["start_next_episode"] = False
    events["exit_early"] = False
    events["policy_paused"] = False
    events["correction_active"] = False


@parser.wrap()
def rac_rtc_collect(cfg: RaCRTCConfig) -> LeRobotDataset:
    """Main RaC data collection function with RTC."""
    init_logging()
    logging.info(pformat(cfg.__dict__))

    if cfg.display_data:
        init_rerun(session_name="rac_rtc_collection_openarms")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    
    teleop_proc, robot_proc, obs_proc = make_identity_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    shutdown_event = Event()
    policy_active = Event()
    shared_state = SharedState()
    rtc_thread = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
            if hasattr(robot, "cameras") and robot.cameras:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
        else:
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras if hasattr(robot, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

        # Load policy
        logger.info(f"Loading policy from: {cfg.policy.pretrained_path}")
        policy_class = get_policy_class(cfg.policy.type)
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path)
        policy.config.rtc_config = cfg.rtc
        policy.init_rtc_processor()
        policy = policy.to(cfg.device)
        policy.eval()
        logger.info(f"Policy loaded: {policy.name}")

        # Setup preprocessor/postprocessor for RTC thread
        hw_features = hw_to_dataset_features(robot.observation_features, "observation")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot.connect()
        logger.info("Robot connected, waiting for CAN bus to stabilize...")
        time.sleep(1.0)  # Let CAN bus stabilize
        
        # Test read to verify robot communication is working
        logger.info("Testing robot communication...")
        test_obs = robot.get_observation()
        logger.info(f"Robot test read OK, got {len(test_obs)} observation keys")
        time.sleep(0.5)
        
        teleop.connect()
        listener, events = init_rac_keyboard_listener()

        # Get action keys for the robot
        action_keys = [k for k in robot.action_features.keys() if k.endswith(".pos")]
        logger.info(f"Action keys: {action_keys}")

        # Start RTC inference thread (it will be idle until policy_active is set)
        logger.info("Starting RTC inference thread (will be idle until episode starts)...")
        rtc_thread = Thread(
            target=rtc_inference_thread,
            args=(
                policy,
                shared_state,
                shutdown_event,
                policy_active,
                cfg,
                hw_features,
                preprocessor,
                postprocessor,
            ),
            daemon=True,
            name="RTCInference",
        )
        rtc_thread.start()
        logger.info("Started RTC inference thread")

        print("\n" + "=" * 65)
        print("  RaC (Recovery and Correction) Data Collection with RTC")
        print("=" * 65)
        print(f"  Policy: {cfg.policy.pretrained_path}")
        print(f"  Task: {cfg.dataset.single_task}")
        print(f"  RTC Enabled: {cfg.rtc.enabled}")
        print(f"  Interpolation: {cfg.interpolation}")
        print(f"  Policy Hz: {cfg.dataset.fps}, Robot Hz: {cfg.dataset.fps * 2 if cfg.interpolation else cfg.dataset.fps}")
        print()
        print("  Controls:")
        print("    SPACE  - Pause policy (teleop tracks robot, no recording)")
        print("    c      - Take control (start correction, recording)")
        print("    →      - End episode (save)")
        print("    ←      - Re-record episode")
        print("    ESC    - Stop session and push to hub")
        print("=" * 65 + "\n")

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"RaC episode {dataset.num_episodes}", cfg.play_sounds)
                
                # Create fresh action queue for this episode
                shared_state.action_queue = ActionQueue(cfg.rtc)
                shared_state.obs = None
                
                logger.info(f"\n{'='*40}")
                logger.info(f"Episode {recorded + 1} / {cfg.dataset.num_episodes}")
                logger.info(f"{'='*40}")

                stats = rac_rtc_rollout_loop(
                    robot=robot,
                    teleop=teleop,
                    shared_state=shared_state,
                    policy_active=policy_active,
                    dataset=dataset,
                    events=events,
                    cfg=cfg,
                    action_keys=action_keys,
                )

                logging.info(f"Episode stats: {stats}")

                if events["rerecord_episode"]:
                    log_say("Re-recording", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1

                if recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                    reset_loop(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=cfg.dataset.fps,
                    )

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        
        shutdown_event.set()
        policy_active.clear()
        
        if rtc_thread and rtc_thread.is_alive():
            logger.info("Waiting for RTC thread to finish...")
            rtc_thread.join(timeout=2.0)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


def main():
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    rac_rtc_collect()


if __name__ == "__main__":
    main()
