#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenArms Policy Evaluation with RTC and Interpolation

Evaluates a trained policy with:
- RTC (Real-Time Chunking) for async inference - decouples policy from robot loop
- Smooth action interpolation for high-frequency robot control
- Velocity feedforward for smoother tracking
- Adjustable PID gains

Example usage:
    python examples/openarms/evaluate_interpolation.py
"""

import logging
import math
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
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
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================== MODEL & TASK CONFIG ========================
HF_MODEL_ID = "lerobot-data-collection/three-folds-pi0"  # TODO: Replace with your trained model
HF_EVAL_DATASET_ID = "lerobot-data-collection/three-folds-pi0_eval_interp"  # TODO: Replace
TASK_DESCRIPTION = "three-folds-dataset"  # TODO: Replace with your task

# ======================== TIMING CONFIG ========================
CAMERA_FPS = 30           # Camera hardware limit (fixed)
POLICY_FPS = 30           # What the policy was trained with
ROBOT_FPS = 50            # Robot command rate (higher = smoother interpolation)

NUM_EPISODES = 1
EPISODE_TIME_SEC = 300
RESET_TIME_SEC = 60

# ======================== RTC CONFIG ========================
RTC_ENABLED = True
RTC_EXECUTION_HORIZON = 20
RTC_MAX_GUIDANCE_WEIGHT = 5.0
ACTION_QUEUE_SIZE_TO_GET_NEW_ACTIONS = 30  # Should be > inference_delay + execution_horizon

# ======================== PID TUNING ========================
CUSTOM_KP_SCALE = 1.0     # Scale factor for position gain (0.5-1.0, lower = smoother)
CUSTOM_KD_SCALE = 1.0     # Scale factor for damping gain (1.0-2.0, higher = less overshoot)
USE_VELOCITY_FEEDFORWARD = False  # Enable velocity feedforward for smoother tracking

# ======================== ROBOT CONFIG ========================
FOLLOWER_LEFT_PORT = "can0"
FOLLOWER_RIGHT_PORT = "can1"

USE_LEADER_FOR_RESETS = False
LEADER_LEFT_PORT = "can2"
LEADER_RIGHT_PORT = "can3"

DEVICE = "cuda"

CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=1280, height=720, fps=CAMERA_FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=1280, height=720, fps=CAMERA_FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=CAMERA_FPS),
}


class RobotWrapper:
    """Thread-safe wrapper for robot operations."""

    def __init__(self, robot: OpenArmsFollower):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: dict, **kwargs) -> None:
        with self.lock:
            self.robot.send_action(action, **kwargs)

    @property
    def observation_features(self) -> dict:
        with self.lock:
            return self.robot.observation_features

    @property
    def action_features(self) -> dict:
        with self.lock:
            return self.robot.action_features

    @property
    def name(self) -> str:
        return self.robot.name


class ActionInterpolator:
    """Interpolate between consecutive actions for smoother robot control."""

    def __init__(self, policy_fps: int, robot_fps: int):
        self.policy_fps = policy_fps
        self.robot_fps = robot_fps
        self.substeps_per_policy_step = robot_fps / policy_fps
        self.prev_action: Tensor | None = None
        self.curr_action: Tensor | None = None
        self.substep = 0
        self.last_interpolated: Tensor | None = None

    def update(self, new_action: Tensor) -> None:
        self.prev_action = self.curr_action
        self.curr_action = new_action
        self.substep = 0

    def get_interpolated_action(self) -> tuple[Tensor | None, Tensor | None]:
        """Returns (interpolated_action, estimated_velocity)"""
        if self.curr_action is None:
            return None, None
        if self.prev_action is None:
            self.last_interpolated = self.curr_action.clone()
            return self.curr_action, torch.zeros_like(self.curr_action)

        t = min(self.substep / self.substeps_per_policy_step, 1.0)
        self.substep += 1

        interpolated = self.prev_action * (1 - t) + self.curr_action * t

        dt = 1.0 / self.robot_fps
        if self.last_interpolated is not None:
            velocity = (interpolated - self.last_interpolated) / dt
        else:
            velocity = (self.curr_action - self.prev_action) * self.policy_fps

        self.last_interpolated = interpolated.clone()
        return interpolated, velocity

    def reset(self):
        self.prev_action = None
        self.curr_action = None
        self.substep = 0
        self.last_interpolated = None


class HzTracker:
    """Track and display actual loop frequency."""
    
    def __init__(self, name: str = "Robot", window_size: int = 100, print_interval: float = 2.0):
        self.name = name
        self.timestamps = deque(maxlen=window_size)
        self.last_print_time = 0
        self.print_interval = print_interval
        
    def tick(self) -> float | None:
        now = time.perf_counter()
        self.timestamps.append(now)
        
        if len(self.timestamps) < 2:
            return None
            
        hz = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
        
        if now - self.last_print_time >= self.print_interval:
            print(f"{self.name} Hz: {hz:.1f}")
            self.last_print_time = now
            
        return hz
    
    def get_avg_hz(self) -> float | None:
        if len(self.timestamps) < 2:
            return None
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
    
    def reset(self):
        self.timestamps.clear()
        self.last_print_time = 0


def get_actions_thread(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    episode_active: Event,
    rtc_config: RTCConfig,
    policy_fps: int,
    task: str,
    pretrained_path: str,
    device: str,
):
    """Thread function to asynchronously generate action chunks from the policy."""
    try:
        logger.info("[GET_ACTIONS] Starting action generation thread")

        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / policy_fps

        hw_features = hw_to_dataset_features(robot.observation_features, "observation")
        policy_device = device

        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=pretrained_path,
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully")

        get_actions_threshold = ACTION_QUEUE_SIZE_TO_GET_NEW_ACTIONS if rtc_config.enabled else 0

        while not shutdown_event.is_set():
            if not episode_active.is_set():
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

                # Filter out non-feature keys (like _timing_breakdown)
                obs_for_frame = {k: v for k, v in obs_processed.items() if not k.startswith("_")}
                
                # Debug: log keys on first iteration
                if action_queue.qsize() == 0:
                    logger.info(f"[GET_ACTIONS] obs_for_frame keys: {list(obs_for_frame.keys())}")
                    logger.info(f"[GET_ACTIONS] hw_features keys: {list(hw_features.keys())}")
                    # Check expected vs actual image keys
                    expected_img_keys = [k.removeprefix("observation.images.") 
                                         for k in hw_features if "images" in k]
                    logger.info(f"[GET_ACTIONS] Expected image keys: {expected_img_keys}")
                    for k in expected_img_keys:
                        logger.info(f"[GET_ACTIONS] '{k}' in obs_for_frame: {k in obs_for_frame}")

                obs_with_policy_features = build_dataset_frame(
                    hw_features, obs_for_frame, prefix="observation"
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

                obs_with_policy_features["task"] = [task]
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

                if ACTION_QUEUE_SIZE_TO_GET_NEW_ACTIONS < rtc_config.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions too small. "
                        "Should be higher than inference delay + execution horizon."
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )

                logger.debug(
                    f"[GET_ACTIONS] Generated chunk, latency={new_latency:.3f}s, "
                    f"delay={new_delay}, queue_size={action_queue.qsize()}"
                )
            else:
                time.sleep(0.01)

        logger.info("[GET_ACTIONS] Action generation thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


def actor_thread(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    episode_active: Event,
    interpolator: ActionInterpolator,
    robot_hz_tracker: HzTracker,
    robot_fps: int,
    action_keys: list[str],
    custom_kp: dict | None,
    custom_kd: dict | None,
    use_velocity_ff: bool,
):
    """Thread function to execute interpolated actions on the robot at high frequency."""
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / robot_fps

        while not shutdown_event.is_set():
            if not episode_active.is_set():
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()

            action = action_queue.get()
            if action is not None:
                interpolator.update(action.cpu())

            smooth_action, velocity = interpolator.get_interpolated_action()

            if smooth_action is not None:
                action_dict = {}
                for i, key in enumerate(action_keys):
                    if i < len(smooth_action):
                        action_dict[key] = smooth_action[i].item()

                action_processed = robot_action_processor((action_dict, None))

                vel_ff = None
                if use_velocity_ff and velocity is not None:
                    vel_ff = {}
                    for i, key in enumerate(action_keys):
                        if i < len(velocity):
                            motor_name = key.replace(".pos", "")
                            vel_ff[motor_name] = velocity[i].item()

                robot.send_action(action_processed, custom_kp=custom_kp, custom_kd=custom_kd, velocity_feedforward=vel_ff)
                action_count += 1

            robot_hz_tracker.tick()

            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - dt_s - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


def build_custom_gains(robot, kp_scale: float | None, kd_scale: float | None) -> tuple[dict | None, dict | None]:
    """Build custom KP/KD gains for the robot."""
    if kp_scale is None and kd_scale is None:
        return None, None
        
    custom_kp = {}
    custom_kd = {}
    for arm in ["right", "left"]:
        bus = robot.robot.bus_right if arm == "right" else robot.robot.bus_left
        for i, motor_name in enumerate(bus.motors):
            full_name = f"{arm}_{motor_name}"
            default_kp = robot.robot.config.position_kp[i] if isinstance(robot.robot.config.position_kp, list) else robot.robot.config.position_kp
            default_kd = robot.robot.config.position_kd[i] if isinstance(robot.robot.config.position_kd, list) else robot.robot.config.position_kd
            custom_kp[full_name] = default_kp * (kp_scale or 1.0)
            custom_kd[full_name] = default_kd * (kd_scale or 1.0)
    
    return custom_kp, custom_kd


def main():
    """Main evaluation function with RTC and interpolation."""
    print("=" * 60)
    print("OpenArms Policy Evaluation with RTC + Interpolation")
    print("=" * 60)
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Dataset: {HF_EVAL_DATASET_ID}")
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"\n--- Timing ---")
    print(f"Policy FPS: {POLICY_FPS}Hz")
    print(f"Robot FPS: {ROBOT_FPS}Hz (interpolated)")
    print(f"\n--- RTC ---")
    print(f"RTC Enabled: {RTC_ENABLED}")
    print(f"Execution Horizon: {RTC_EXECUTION_HORIZON}")
    print(f"Max Guidance Weight: {RTC_MAX_GUIDANCE_WEIGHT}")
    print(f"\n--- PID ---")
    print(f"KP scale: {CUSTOM_KP_SCALE}, KD scale: {CUSTOM_KD_SCALE}")
    print(f"Velocity FF: {USE_VELOCITY_FEEDFORWARD}")
    print(f"\n--- Episodes ---")
    print(f"Episodes: {NUM_EPISODES}, Duration: {EPISODE_TIME_SEC}s")
    print("=" * 60)

    shutdown_event = Event()
    episode_active = Event()
    
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
        raise RuntimeError("Follower robot failed to connect!")

    robot = RobotWrapper(follower)
    logger.info("Follower robot connected")

    leader = None
    if USE_LEADER_FOR_RESETS:
        leader_config = OpenArmsLeaderConfig(
            port_left=LEADER_LEFT_PORT,
            port_right=LEADER_RIGHT_PORT,
            can_interface="socketcan",
            id="openarms_leader",
            manual_control=False,
        )
        
        leader = OpenArmsLeader(leader_config)
        leader.connect(calibrate=False)
        
        if not leader.is_connected:
            raise RuntimeError("Leader robot failed to connect!")
        
        if leader.pin_robot is not None:
            leader.bus_right.enable_torque()
            leader.bus_left.enable_torque()
            time.sleep(0.1)
            print("Leader connected with gravity compensation")
        else:
            print("Leader connected (no gravity compensation)")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    action_features_hw = {}
    for key, value in follower.action_features.items():
        if key.endswith(".pos"):
            action_features_hw[key] = value
    
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
        choice = input("Continue and append? (y/n): ").strip().lower()
        if choice != 'y':
            print("Aborting.")
            follower.disconnect()
            if leader:
                leader.disconnect()
            return
    
    dataset = LeRobotDataset.create(
        repo_id=HF_EVAL_DATASET_ID,
        fps=POLICY_FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=12, 
    )
    
    # Load policy with RTC support
    logger.info(f"Loading policy from: {HF_MODEL_ID}")
    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID
    
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(HF_MODEL_ID, config=policy_config)
    
    rtc_config = RTCConfig(
        enabled=RTC_ENABLED,
        execution_horizon=RTC_EXECUTION_HORIZON,
        max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()
    
    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 support RTC"
    
    policy = policy.to(DEVICE)
    policy.eval()
    
    logger.info(f"Policy loaded: {policy.name}")

    print(f"\nRunning evaluation...")
    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarms_eval_rtc_interp")
    
    action_keys = [k for k in robot.action_features.keys() if k.endswith(".pos")]
    custom_kp, custom_kd = build_custom_gains(robot, CUSTOM_KP_SCALE, CUSTOM_KD_SCALE)
    
    if custom_kp:
        print(f"Custom gains applied")
    if USE_VELOCITY_FEEDFORWARD:
        print("Velocity feedforward: enabled")
    
    episode_idx = 0
    get_actions_t = None
    actor_t = None
    
    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Evaluating episode {episode_idx + 1} of {NUM_EPISODES}")
            print(f"\n--- Episode {episode_idx + 1}/{NUM_EPISODES} ---")
            
            action_queue = ActionQueue(rtc_config)
            interpolator = ActionInterpolator(policy_fps=POLICY_FPS, robot_fps=ROBOT_FPS)
            robot_hz_tracker = HzTracker(name="Robot", window_size=100, print_interval=2.0)
            
            get_actions_t = Thread(
                target=get_actions_thread,
                args=(
                    policy, robot, robot_observation_processor, action_queue,
                    shutdown_event, episode_active, rtc_config, POLICY_FPS,
                    TASK_DESCRIPTION, HF_MODEL_ID, DEVICE,
                ),
                daemon=True,
                name="GetActions",
            )
            get_actions_t.start()
            
            actor_t = Thread(
                target=actor_thread,
                args=(
                    robot, robot_action_processor, action_queue,
                    shutdown_event, episode_active, interpolator, robot_hz_tracker,
                    ROBOT_FPS, action_keys, custom_kp, custom_kd, USE_VELOCITY_FEEDFORWARD,
                ),
                daemon=True,
                name="Actor",
            )
            actor_t.start()
            
            logger.info("Started inference and actor threads")
            
            episode_active.set()
            episode_start_time = time.time()
            
            while (time.time() - episode_start_time) < EPISODE_TIME_SEC:
                if events["exit_early"] or events["stop_recording"] or shutdown_event.is_set():
                    break
                
                elapsed = time.time() - episode_start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    robot_hz = robot_hz_tracker.get_avg_hz()
                    hz_str = f"{robot_hz:.1f}" if robot_hz else "N/A"
                    logger.info(
                        f"Progress: {elapsed:.0f}/{EPISODE_TIME_SEC}s, "
                        f"queue={action_queue.qsize()}, hz={hz_str}"
                    )
                
                time.sleep(0.5)
            
            episode_active.clear()
            
            robot_hz = robot_hz_tracker.get_avg_hz()
            hz_str = f"{robot_hz:.1f}" if robot_hz else "N/A"
            logger.info(f"Episode {episode_idx + 1} done. Avg Hz: {hz_str}")
            
            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                print(f"Saving episode ({dataset.episode_buffer['size']} frames)...")
                dataset.save_episode()
                episode_idx += 1
            
            if not events["stop_recording"] and episode_idx < NUM_EPISODES:
                if USE_LEADER_FOR_RESETS and leader:
                    log_say("Reset the environment using leader arms")
                    print(f"\nManual reset ({RESET_TIME_SEC}s)...")
                    
                    dt = 1 / CAMERA_FPS
                    reset_start_time = time.perf_counter()
                    
                    while time.perf_counter() - reset_start_time < RESET_TIME_SEC:
                        if events["exit_early"] or events["stop_recording"]:
                            break
                        
                        loop_start = time.perf_counter()
                        leader_action = leader.get_action()
                        
                        leader_positions_deg = {}
                        leader_velocities_deg_per_sec = {}
                        
                        for motor in leader.bus_right.motors:
                            pos_key = f"right_{motor}.pos"
                            vel_key = f"right_{motor}.vel"
                            if pos_key in leader_action:
                                leader_positions_deg[f"right_{motor}"] = leader_action[pos_key]
                            if vel_key in leader_action:
                                leader_velocities_deg_per_sec[f"right_{motor}"] = leader_action[vel_key]
                        
                        for motor in leader.bus_left.motors:
                            pos_key = f"left_{motor}.pos"
                            vel_key = f"left_{motor}.vel"
                            if pos_key in leader_action:
                                leader_positions_deg[f"left_{motor}"] = leader_action[pos_key]
                            if vel_key in leader_action:
                                leader_velocities_deg_per_sec[f"left_{motor}"] = leader_action[vel_key]
                        
                        leader_positions_rad = {k: np.deg2rad(v) for k, v in leader_positions_deg.items()}
                        leader_gravity_torques_nm = leader._gravity_from_q(leader_positions_rad)
                        
                        leader_velocities_rad_per_sec = {k: np.deg2rad(v) for k, v in leader_velocities_deg_per_sec.items()}
                        leader_friction_torques_nm = leader._friction_from_velocity(
                            leader_velocities_rad_per_sec, friction_scale=1.0
                        )
                        
                        leader_total_torques_nm = {}
                        for motor_name in leader_gravity_torques_nm:
                            gravity = leader_gravity_torques_nm.get(motor_name, 0.0)
                            friction = leader_friction_torques_nm.get(motor_name, 0.0)
                            leader_total_torques_nm[motor_name] = gravity + friction
                        
                        for motor in leader.bus_right.motors:
                            full_name = f"right_{motor}"
                            position = leader_positions_deg.get(full_name, 0.0)
                            torque = leader_total_torques_nm.get(full_name, 0.0)
                            kd = leader.get_damping_kd(motor)
                            leader.bus_right._mit_control(
                                motor=motor, kp=0.0, kd=kd,
                                position_degrees=position, velocity_deg_per_sec=0.0, torque=torque,
                            )
                        
                        for motor in leader.bus_left.motors:
                            full_name = f"left_{motor}"
                            position = leader_positions_deg.get(full_name, 0.0)
                            torque = leader_total_torques_nm.get(full_name, 0.0)
                            kd = leader.get_damping_kd(motor)
                            leader.bus_left._mit_control(
                                motor=motor, kp=0.0, kd=kd,
                                position_degrees=position, velocity_deg_per_sec=0.0, torque=torque,
                            )
                        
                        follower_action = {}
                        for joint in leader_positions_deg.keys():
                            pos_key = f"{joint}.pos"
                            if pos_key in leader_action:
                                follower_action[pos_key] = leader_action[pos_key]
                        
                        if follower_action:
                            follower.send_action(follower_action)
                        
                        loop_duration = time.perf_counter() - loop_start
                        sleep_time = dt - loop_duration
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
                    print("Reset complete")
                else:
                    log_say("Waiting for manual reset")
                    input("Press ENTER when ready...")
        
        print(f"\nEvaluation complete! {episode_idx} episodes recorded")
        log_say("Evaluation complete", blocking=True)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        shutdown_event.set()
        episode_active.clear()
        
        if get_actions_t is not None and get_actions_t.is_alive():
            get_actions_t.join(timeout=2.0)
        
        if actor_t is not None and actor_t.is_alive():
            actor_t.join(timeout=2.0)
        
        if leader:
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
            leader.disconnect()

        follower.disconnect()
        logger.info("Follower disconnected")
        
        if listener is not None:
            listener.stop()
        
        dataset.finalize()
        print("\nUploading to Hugging Face Hub...")
        dataset.push_to_hub(private=True)


if __name__ == "__main__":
    main()

