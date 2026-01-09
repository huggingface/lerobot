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
OpenArms Policy Evaluation with Interpolation

Evaluates a trained policy with smooth action interpolation:
- Decoupled camera capture (CAMERA_FPS) from robot control (ROBOT_FPS)
- Speed multiplier to execute actions faster than training
- Velocity feedforward for smoother tracking
- Adjustable PID gains

Example usage:
    python examples/openarms/evaluate_interpolation.py
"""

import time
from collections import deque
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import log_say, get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun


# ======================== MODEL & TASK CONFIG ========================
HF_MODEL_ID = "lerobot-data-collection/three-folds-pi0"  # TODO: Replace with your trained model
HF_EVAL_DATASET_ID = "lerobot-data-collection/three-folds-pi0_eval_interp"  # TODO: Replace
TASK_DESCRIPTION = "three-folds-dataset"  # TODO: Replace with your task

# ======================== TIMING CONFIG ========================
CAMERA_FPS = 30           # Camera hardware limit (fixed)
POLICY_FPS = 30           # What the policy was trained with
SPEED_MULTIPLIER = 1.2    # Execute actions faster (1.0 = normal, 1.2 = 20% faster)
ROBOT_FPS = 50            # Robot command rate (higher = smoother interpolation)

# Derived values
EFFECTIVE_POLICY_FPS = int(POLICY_FPS * SPEED_MULTIPLIER)  # How fast we consume actions (36Hz at 1.2x)

NUM_EPISODES = 1
EPISODE_TIME_SEC = 300
RESET_TIME_SEC = 60

# ======================== PID TUNING ========================
# Set to None to use robot config defaults
CUSTOM_KP_SCALE = 0.7     # Scale factor for position gain (0.5-1.0, lower = smoother)
CUSTOM_KD_SCALE = 1.3     # Scale factor for damping gain (1.0-2.0, higher = less overshoot)
USE_VELOCITY_FEEDFORWARD = True  # Enable velocity feedforward for smoother tracking

# ======================== ROBOT CONFIG ========================
FOLLOWER_LEFT_PORT = "can0"
FOLLOWER_RIGHT_PORT = "can1"

USE_LEADER_FOR_RESETS = True
LEADER_LEFT_PORT = "can2"
LEADER_RIGHT_PORT = "can3"

# Camera config uses CAMERA_FPS (hardware limit)
CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=CAMERA_FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=CAMERA_FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=CAMERA_FPS),
}


class ActionInterpolator:
    """Interpolate between policy actions for smoother robot control with velocity estimation."""
    
    def __init__(self, effective_policy_fps: int, robot_fps: int):
        self.effective_policy_fps = effective_policy_fps
        self.robot_fps = robot_fps
        self.substeps_per_policy_step = robot_fps / effective_policy_fps
        self.prev_action: dict | None = None
        self.curr_action: dict | None = None
        self.substep = 0
        self.last_interpolated: dict | None = None
        
    def update(self, new_action: dict) -> None:
        self.prev_action = self.curr_action
        self.curr_action = new_action
        self.substep = 0
        
    def get_interpolated_action(self) -> tuple[dict | None, dict | None]:
        """Returns (interpolated_position, estimated_velocity_deg_per_sec)"""
        if self.curr_action is None:
            return None, None
        if self.prev_action is None:
            self.last_interpolated = self.curr_action.copy()
            return self.curr_action, {k: 0.0 for k in self.curr_action}
            
        t = min(self.substep / self.substeps_per_policy_step, 1.0)
        self.substep += 1
        
        interpolated = {}
        velocity = {}
        dt = 1.0 / self.robot_fps
        
        for key in self.curr_action:
            prev = self.prev_action.get(key, self.curr_action[key])
            curr = self.curr_action[key]
            interpolated[key] = prev * (1 - t) + curr * t
            
            if self.last_interpolated is not None and key in self.last_interpolated:
                velocity[key] = (interpolated[key] - self.last_interpolated[key]) / dt
            else:
                velocity[key] = (curr - prev) * self.effective_policy_fps
        
        self.last_interpolated = interpolated.copy()
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


def interpolated_eval_loop(
    robot,
    policy,
    preprocessor,
    postprocessor,
    robot_observation_processor,
    robot_action_processor,
    dataset,
    events,
    interpolator: ActionInterpolator,
    robot_hz_tracker: HzTracker,
    camera_fps: int,
    effective_policy_fps: int,
    robot_fps: int,
    control_time_s: float,
    task: str,
    kp_scale: float | None = None,
    kd_scale: float | None = None,
    use_velocity_ff: bool = False,
):
    """
    Run evaluation with decoupled camera and robot control:
    - Camera captures at camera_fps (hardware limit)
    - Policy inference runs when new camera frame is available
    - Actions are consumed at effective_policy_fps (sped up by SPEED_MULTIPLIER)
    - Robot receives interpolated commands at robot_fps (smoothest)
    """
    from lerobot.scripts.lerobot_record import build_dataset_frame, make_robot_action
    from lerobot.utils.visualization_utils import log_rerun_data
    
    camera_dt = 1.0 / camera_fps
    policy_dt = 1.0 / effective_policy_fps
    robot_dt = 1.0 / robot_fps
    
    interpolator.reset()
    robot_hz_tracker.reset()
    policy.reset()
    
    # Build custom gains if scaling is enabled
    custom_kp = None
    custom_kd = None
    if kp_scale is not None or kd_scale is not None:
        custom_kp = {}
        custom_kd = {}
        for arm in ["right", "left"]:
            bus = robot.bus_right if arm == "right" else robot.bus_left
            for i, motor_name in enumerate(bus.motors):
                full_name = f"{arm}_{motor_name}"
                default_kp = robot.config.position_kp[i] if isinstance(robot.config.position_kp, list) else robot.config.position_kp
                default_kd = robot.config.position_kd[i] if isinstance(robot.config.position_kd, list) else robot.config.position_kd
                custom_kp[full_name] = default_kp * (kp_scale or 1.0)
                custom_kd[full_name] = default_kd * (kd_scale or 1.0)
        print(f"Custom gains: kp_scale={kp_scale}, kd_scale={kd_scale}")
    
    if use_velocity_ff:
        print("Velocity feedforward: enabled")
    
    last_camera_time = -camera_dt
    last_policy_action_time = -policy_dt
    cached_observation = None
    cached_robot_action = None
    
    start_time = time.perf_counter()
    
    print(f"\nStarting interpolated eval loop:")
    print(f"  Camera: {camera_fps}Hz | Policy actions consumed: {effective_policy_fps}Hz | Robot: {robot_fps}Hz")
    
    while time.perf_counter() - start_time < control_time_s:
        if events["exit_early"] or events["stop_recording"]:
            break
            
        loop_start = time.perf_counter()
        elapsed = loop_start - start_time
        
        # === CAMERA CAPTURE (at camera_fps, decoupled from robot) ===
        if elapsed - last_camera_time >= camera_dt:
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
            
            # Run policy inference with fresh observation
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            
            act_processed = make_robot_action(action_values, dataset.features)
            cached_robot_action = robot_action_processor((act_processed, obs))
            cached_observation = (obs_processed, observation_frame, act_processed)
            
            last_camera_time = elapsed
        
        # === ACTION UPDATE (at effective_policy_fps, faster than camera if speed > 1) ===
        if elapsed - last_policy_action_time >= policy_dt and cached_robot_action is not None:
            interpolator.update(cached_robot_action)
            last_policy_action_time = elapsed
            
            # Save to dataset at effective policy rate
            if dataset is not None and cached_observation is not None:
                obs_processed, observation_frame, act_processed = cached_observation
                action_frame = build_dataset_frame(dataset.features, act_processed, prefix="action")
                frame = {**observation_frame, **action_frame, "task": task}
                dataset.add_frame(frame)
                log_rerun_data(observation=obs_processed, action=act_processed)
        
        # === ROBOT COMMAND (at robot_fps, highest rate for smoothness) ===
        smooth_action, velocity = interpolator.get_interpolated_action()
        if smooth_action is not None:
            vel_ff = velocity if use_velocity_ff else None
            robot.send_action(smooth_action, custom_kp=custom_kp, custom_kd=custom_kd, velocity_feedforward=vel_ff)
        
        robot_hz_tracker.tick()
        
        # Maintain robot control rate
        sleep_time = robot_dt - (time.perf_counter() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Print final stats
    robot_hz = robot_hz_tracker.get_avg_hz()
    if robot_hz:
        print(f"\nFinal average robot Hz: {robot_hz:.1f}")


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("OpenArms Policy Evaluation with Interpolation")
    print("=" * 60)
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Dataset: {HF_EVAL_DATASET_ID}")
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"\n--- Timing ---")
    print(f"Camera FPS: {CAMERA_FPS} (hardware limit)")
    print(f"Policy trained at: {POLICY_FPS}Hz")
    print(f"Speed multiplier: {SPEED_MULTIPLIER}x")
    print(f"Effective policy FPS: {EFFECTIVE_POLICY_FPS}Hz (actions consumed)")
    print(f"Robot FPS: {ROBOT_FPS}Hz (interpolated commands)")
    print(f"\n--- PID Tuning ---")
    print(f"KP scale: {CUSTOM_KP_SCALE}")
    print(f"KD scale: {CUSTOM_KD_SCALE}")
    print(f"Velocity feedforward: {USE_VELOCITY_FEEDFORWARD}")
    print(f"\n--- Episodes ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Duration: {EPISODE_TIME_SEC}s per episode")
    print(f"Reset time: {RESET_TIME_SEC}s")
    print(f"Leader for resets: {USE_LEADER_FOR_RESETS}")
    print("=" * 60)
    
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
            print(f"Leader connected with gravity compensation")
        else:
            print(f"Leader connected (no gravity compensation)")

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
    
    # Dataset uses effective policy FPS (sped up rate)
    dataset = LeRobotDataset.create(
        repo_id=HF_EVAL_DATASET_ID,
        fps=EFFECTIVE_POLICY_FPS,
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
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )

    print(f"\nRunning evaluation...")
    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarms_evaluation_interp")
    
    interpolator = ActionInterpolator(effective_policy_fps=EFFECTIVE_POLICY_FPS, robot_fps=ROBOT_FPS)
    robot_hz_tracker = HzTracker(name="Robot", window_size=100, print_interval=2.0)
    
    episode_idx = 0
    
    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Evaluating episode {episode_idx + 1} of {NUM_EPISODES}")
            print(f"\n--- Episode {episode_idx + 1}/{NUM_EPISODES} ---")
            
            interpolated_eval_loop(
                robot=follower,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_observation_processor=robot_observation_processor,
                robot_action_processor=robot_action_processor,
                dataset=dataset,
                events=events,
                interpolator=interpolator,
                robot_hz_tracker=robot_hz_tracker,
                camera_fps=CAMERA_FPS,
                effective_policy_fps=EFFECTIVE_POLICY_FPS,
                robot_fps=ROBOT_FPS,
                control_time_s=EPISODE_TIME_SEC,
                task=TASK_DESCRIPTION,
                kp_scale=CUSTOM_KP_SCALE,
                kd_scale=CUSTOM_KD_SCALE,
                use_velocity_ff=USE_VELOCITY_FEEDFORWARD,
            )
            
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
        if leader:
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
            leader.disconnect()

        follower.disconnect()
        
        if listener is not None:
            listener.stop()
        
        dataset.finalize()
        print("\nUploading to Hugging Face Hub...")
        dataset.push_to_hub(private=True)


if __name__ == "__main__":
    main()

