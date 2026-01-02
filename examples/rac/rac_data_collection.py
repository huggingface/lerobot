#!/usr/bin/env python
"""
RaC (Recovery and Correction) Data Collection with Policy Rollout + Human Intervention.

This implements the RaC paradigm from "RaC: Robot Learning for Long-Horizon Tasks
by Scaling Recovery and Correction" (Hu et al., 2025) for LeRobot.

RaC improves upon standard data collection (BC) and prior human-in-the-loop methods
(DAgger, HG-DAgger) by explicitly collecting recovery and correction behaviors:

The workflow:
1. Policy runs autonomously
2. Press SPACE to pause - robot holds position
3. Press 'c' to take control - human provides RECOVERY + CORRECTION
4. Press → to end episode (save and continue to next)
5. Reset, then do next rollout

Key RaC Rules:
- Rule 1 (Recover then Correct): Every intervention = recovery + correction (both human)
- Rule 2 (Terminate after Intervention): Episode ends after correction

The recovery segment (teleoperating back to good state) is recorded as training data -
this teaches the policy how to recover from errors.

Keyboard Controls:
    SPACE  - Pause policy (robot holds position, no recording)
    c      - Take control (start correction, recording resumes)
    →      - End episode (save and continue to next)
    ←      - Re-record episode
    ESC    - Stop recording and push dataset to hub

Usage:
    python examples/rac/rac_data_collection.py \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58760431541 \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --teleop.type=so100_leader \
        --teleop.port=/dev/tty.usbmodem58760431551 \
        --policy.path=outputs/train/my_policy/checkpoints/last/pretrained_model \
        --dataset.repo_id=my_user/rac_dataset \
        --dataset.single_task="Pick up the cube"
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    IdentityProcessor,
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
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class RaCDatasetConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: float = 120
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
class RaCConfig:
    robot: RobotConfig
    dataset: RaCDatasetConfig
    policy: PreTrainedConfig
    teleop: TeleoperatorConfig
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def init_rac_keyboard_listener():
    """Initialize keyboard listener with RaC-specific controls."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "policy_paused": False,      # SPACE pressed - policy paused, teleop tracking robot
        "correction_active": False,  # 'c' pressed - human controlling, recording correction
        "in_reset": False,           # True during reset period
        "start_next_episode": False, # Signal to start next episode
    }

    if is_headless():
        logging.warning("Headless environment - keyboard controls unavailable")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if events["in_reset"]:
                # During reset: any action key starts next episode
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
                # During episode
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
        from evdev import InputDevice, ecodes
    except ImportError:
        logging.info("[Pedal] evdev not installed - pedal support disabled")
        return
    
    PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    KEY_LEFT = "KEY_A"   # Left pedal
    KEY_RIGHT = "KEY_C"  # Right pedal
    
    def pedal_reader():
        try:
            dev = InputDevice(PEDAL_DEVICE)
            print(f"[Pedal] Connected: {dev.name}")
            print(f"[Pedal] Right=pause/next, Left=take control/start")
            
            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue
                
                from evdev import categorize
                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]
                
                # Only trigger on key down
                if key.keystate != 1:
                    continue
                
                if events["in_reset"]:
                    # During reset: either pedal starts next episode
                    if code in [KEY_LEFT, KEY_RIGHT]:
                        print("\n[Pedal] Starting next episode...")
                        events["start_next_episode"] = True
                else:
                    # During episode
                    if code == KEY_RIGHT:
                        # Right pedal: SPACE (pause) when running, → (next) when in correction
                        if events["correction_active"]:
                            print("\n[Pedal] → End episode")
                            events["exit_early"] = True
                        elif not events["policy_paused"]:
                            print("\n[Pedal] ⏸ PAUSED - Policy stopped, teleop moving to robot")
                            print("        Press left pedal to take control")
                            events["policy_paused"] = True
                    
                    elif code == KEY_LEFT:
                        # Left pedal: START (take control) when paused
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
        steps=[IdentityProcessor()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessor()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    obs_proc = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessor()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return teleop_proc, robot_proc, obs_proc


def move_robot_to_zero(robot: Robot, duration_s: float = 2.0, fps: int = 50):
    """Smoothly move all robot joints to zero position."""
    obs = robot.get_observation()
    current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
    target_pos = {k: 0.0 for k in current_pos}
    
    print(f"[RaC] Moving robot to zero position ({duration_s}s)...")
    steps = int(duration_s * fps)
    for step in range(steps + 1):
        t = step / steps
        interp_pos = {k: current_pos[k] * (1 - t) + target_pos[k] * t for k in current_pos}
        robot.send_action(interp_pos)
        time.sleep(1 / fps)
    print("[RaC] Robot at zero position.")

@safe_stop_image_writer
def rac_rollout_loop(
    robot: Robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    dataset: LeRobotDataset,
    events: dict,
    fps: int,
    control_time_s: float,
    single_task: str,
    display_data: bool = True,
) -> dict:
    """
    RaC rollout loop with two-stage intervention:
    
    1. Policy runs autonomously (recording)
    2. SPACE: Policy pauses (NOT recording) - robot holds position
    3. 'c': Human takes control (recording correction)
    4. →: End episode
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = get_safe_torch_device(policy.config.device)
    frame_buffer = []

    stats = {
        "total_frames": 0,
        "autonomous_frames": 0,
        "paused_frames": 0,
        "correction_frames": 0,
    }

    last_robot_action = None
    was_paused = False
    was_correction_active = False
    waiting_for_takeover = False
    timestamp = 0
    start_t = time.perf_counter()

    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            break

        # Detect transition to paused state
        if events["policy_paused"] and not was_paused:
            obs = robot.get_observation()
            robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
            print("[RaC] Moving teleop to robot position (2s smooth transition)...")
            teleop.smooth_move_to(robot_pos, duration_s=2.0, fps=50)
            print("[RaC] Teleop aligned. Press START to take control.")
            events["start_next_episode"] = False
            waiting_for_takeover = True
            was_paused = True

        # Wait for start button before enabling correction mode
        if waiting_for_takeover and events["start_next_episode"]:
            print("[RaC] Start pressed - enabling teleop control...")
            events["start_next_episode"] = False
            events["correction_active"] = True
            waiting_for_takeover = False
            was_correction_active = True

        obs = robot.get_observation()
        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        if events["correction_active"]:
            # Human controlling - record correction data
            robot_action = teleop.get_action()
            robot.send_action(robot_action)
            stats["correction_frames"] += 1
            
            # Record this frame
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": single_task}
            frame_buffer.append(frame)
            stats["total_frames"] += 1
            
        elif waiting_for_takeover:
            # Waiting for START - policy stopped, no recording, robot holds position
            if last_robot_action is not None:
                robot.send_action(last_robot_action)
            stats["paused_frames"] += 1
            
        elif events["policy_paused"]:
            # Paused and user acknowledged - hold last position, don't record
            if last_robot_action is not None:
                robot.send_action(last_robot_action)
            stats["paused_frames"] += 1
            robot_action = last_robot_action
            
        else:
            # Normal policy execution - record
            action_values = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            robot_action: RobotAction = make_robot_action(action_values, dataset.features)
            robot.send_action(robot_action)
            last_robot_action = robot_action
            stats["autonomous_frames"] += 1
            
            # Record this frame
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": single_task}
            frame_buffer.append(frame)
            stats["total_frames"] += 1

        if display_data and robot_action is not None:
            log_rerun_data(observation=obs, action=robot_action)

        dt = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt)
        timestamp = time.perf_counter() - start_t

    for frame in frame_buffer:
        dataset.add_frame(frame)

    return stats


def reset_loop(
    robot: Robot,
    teleop: Teleoperator,
    events: dict,
    fps: int,
):
    """Reset period where human repositions environment. Two-stage: enable teleop, then start episode."""
    print("\n" + "=" * 65)
    print("  [RaC] RESET - Moving teleop to robot position...")
    print("=" * 65)
    
    # Enter reset mode
    events["in_reset"] = True
    events["start_next_episode"] = False
    
    # Move teleop to match robot position to avoid sudden jumps
    obs = robot.get_observation()
    robot_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
    teleop.smooth_move_to(robot_pos, duration_s=2.0, fps=50)
    
    # Stage 1: Wait for user to press start to enable teleoperation
    print("  Teleop aligned. Press any key/pedal to enable teleoperation")
    while not events["start_next_episode"] and not events["stop_recording"]:
        precise_sleep(0.05)
    
    if events["stop_recording"]:
        return
    
    # Stage 2: Enable teleop and let user move robot to starting position
    events["start_next_episode"] = False
    teleop.disable_torque()
    print("  Teleop enabled - move robot to starting position")
    print("  Press any key/pedal to start next episode")

    # Wait for user to signal ready for next episode
    while not events["start_next_episode"] and not events["stop_recording"]:
        loop_start = time.perf_counter()

        action = teleop.get_action()
        robot.send_action(action)

        dt = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt)
    
    # Exit reset mode and clear flags for next episode
    events["in_reset"] = False
    events["start_next_episode"] = False
    events["exit_early"] = False
    events["policy_paused"] = False
    events["correction_active"] = False


@parser.wrap()
def rac_collect(cfg: RaCConfig) -> LeRobotDataset:
    """Main RaC data collection function."""
    init_logging()
    logging.info(pformat(cfg.__dict__))

    if cfg.display_data:
        init_rerun(session_name="rac_collection")

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

        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot.connect()
        teleop.connect()
        listener, events = init_rac_keyboard_listener()

        print("\n" + "=" * 65)
        print("  RaC (Recovery and Correction) Data Collection")
        print("=" * 65)
        print("  Policy runs autonomously until you intervene.")
        print()
        print("  Controls:")
        print("    SPACE  - Pause policy (robot holds position, no recording)")
        print("    c      - Take control (start correction, recording)")
        print("    →      - End episode (save)")
        print("    ←      - Re-record episode")
        print("    ESC    - Stop session and push to hub")
        print("=" * 65 + "\n")

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"RaC episode {dataset.num_episodes}", cfg.play_sounds)
                
                move_robot_to_zero(robot, duration_s=2.0, fps=cfg.dataset.fps)

                stats = rac_rollout_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    events=events,
                    fps=cfg.dataset.fps,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
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

                # Reset between episodes
                if recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                    reset_loop(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=cfg.dataset.fps,
                    )

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

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
    rac_collect()


if __name__ == "__main__":
    main()

