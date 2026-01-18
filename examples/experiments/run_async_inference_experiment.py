#!/usr/bin/env python3
"""
Latency-Adaptive Async Inference Experiment Runner

This script runs experiments on a REAL ROBOT to validate the latency-adaptive
async inference algorithm. It assumes the policy server is already running.

Usage:
    python examples/experiments/run_async_inference_experiment.py --experiment_config spike --output_dir results/
    python examples/experiments/run_async_inference_experiment.py --drop_obs '[{"start_s": 5, "duration_s": 1}]'
"""

import argparse
import json
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lerobot.async_inference.robot_client_improved import RobotClientImproved
from lerobot.async_inference.configs_improved import RobotClientImprovedConfig
from lerobot.async_inference.utils.simulation import DropConfig, DropEvent
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig


DEFAULT_SERVER_ADDRESS = "192.168.4.37:8080"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_2026_01_03"
DEFAULT_MODEL_PATH = "/home/jack/code/self-driving-screwdriver-robot/wandb_downloads/so101_smolvla_pickplaceorangecube_e100_20260108_203916/100000/pretrained_model/"
DEFAULT_TASK = "Pick up the orange cube and place it on the black X marker with the white background"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    estimator: str
    cooldown: bool
    latency_k: float = 1.5
    epsilon: int = 1
    duration_s: float = 60.0
    fps: int = 30
    actions_per_chunk: int = 50
    drop_obs_config: DropConfig | None = None
    drop_action_config: DropConfig | None = None
    spikes: list[dict] = field(default_factory=list)


ESTIMATOR_COMPARISON_CONFIG = [
    ExperimentConfig(name=f"estimator_{est}", estimator=est, cooldown=True, duration_s=15.0)
    for est in ["jk", "max_last_10"]
]

K_PARAMETER_CONFIG = [
    ExperimentConfig(name=f"jk_K{k}_cooldown_on", estimator="jk", cooldown=True, latency_k=k)
    for k in [0.5, 1.0, 1.5, 2.0, 4.0]
]

EPSILON_CONFIG = [
    ExperimentConfig(name=f"jk_eps{eps}_cooldown_on", estimator="jk", cooldown=True, epsilon=eps)
    for eps in [0, 1, 2, 3, 5]
]

QUICK_TEST_CONFIG = [
    ExperimentConfig(name="jk_cooldown_on", estimator="jk", cooldown=True, duration_s=30.0),
    ExperimentConfig(name="jk_cooldown_off", estimator="jk", cooldown=False, duration_s=30.0),
]

OBS_DROP_CONFIG = [
    ExperimentConfig(
        name="jk_drop_at_5s",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(drops=[DropEvent(start_s=5.0, duration_s=1.0)]),
    ),
    ExperimentConfig(
        name="jk_drops_at_5s_and_15s",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(drops=[
            DropEvent(start_s=5.0, duration_s=1.0),
            DropEvent(start_s=15.0, duration_s=1.0),
        ]),
    ),
]

ACTION_DROP_CONFIG = [
    ExperimentConfig(name="jk_no_action_drops", estimator="jk", cooldown=True),
    ExperimentConfig(
        name="jk_action_drop_at_5s",
        estimator="jk",
        cooldown=True,
        drop_action_config=DropConfig(drops=[DropEvent(start_s=5.0, duration_s=1.0)]),
    ),
    ExperimentConfig(
        name="jk_action_drops_at_5s_and_15s",
        estimator="jk",
        cooldown=True,
        drop_action_config=DropConfig(drops=[
            DropEvent(start_s=5.0, duration_s=1.0),
            DropEvent(start_s=15.0, duration_s=1.0),
        ]),
    ),
]

DROP_RECOVERY_COMPARISON_CONFIG = [
    ExperimentConfig(
        name="cooldown_drop_at_10s",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(drops=[DropEvent(start_s=10.0, duration_s=1.0)]),
    ),
    ExperimentConfig(
        name="no_cooldown_drop_at_10s",
        estimator="jk",
        cooldown=False,
        drop_obs_config=DropConfig(drops=[DropEvent(start_s=10.0, duration_s=1.0)]),
    ),
]

SPIKE_CONFIG = [
    ExperimentConfig(
        name="jk_spike_at_5s",
        estimator="jk",
        cooldown=True,
        spikes=[{"start_s": 5.0, "delay_ms": 2000}],
        duration_s=20.0,
    ),
    ExperimentConfig(
        name="max_last_10_spike_at_5s",
        estimator="max_last_10",
        cooldown=True,
        spikes=[{"start_s": 5.0, "delay_ms": 2000}],
        duration_s=20.0,
    ),
]

SPIKE_ESTIMATOR_COMPARISON_CONFIG = [
    ExperimentConfig(
        name="jk_with_spikes",
        estimator="jk",
        cooldown=True,
        spikes=[{"start_s": 5.0, "delay_ms": 2000}, {"start_s": 15.0, "delay_ms": 2000}],
        duration_s=30.0,
    ),
    ExperimentConfig(
        name="max_last_10_with_spikes",
        estimator="max_last_10",
        cooldown=True,
        spikes=[{"start_s": 5.0, "delay_ms": 2000}, {"start_s": 15.0, "delay_ms": 2000}],
        duration_s=30.0,
    ),
]

ALL_EXPERIMENT_CONFIGS = {
    "estimator_comparison": ESTIMATOR_COMPARISON_CONFIG,
    "k_parameter": K_PARAMETER_CONFIG,
    "epsilon": EPSILON_CONFIG,
    "quick_test": QUICK_TEST_CONFIG,
    "obs_drop": OBS_DROP_CONFIG,
    "action_drop": ACTION_DROP_CONFIG,
    "drop_recovery": DROP_RECOVERY_COMPARISON_CONFIG,
    "spike": SPIKE_CONFIG,
    "spike_estimator": SPIKE_ESTIMATOR_COMPARISON_CONFIG
}


def create_robot_config() -> SO101FollowerConfig:
    camera_cfg = {
        "camera2": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0",
            width=800, height=600, fps=30, fourcc="MJPG",
            use_threaded_async_read=True, allow_stale_frames=True,
        ),
        "camera1": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0",
            width=800, height=600, fps=30, fourcc="MJPG",
            use_threaded_async_read=True, allow_stale_frames=True,
        ),
    }
    return SO101FollowerConfig(port=DEFAULT_ROBOT_PORT, id=DEFAULT_ROBOT_ID, cameras=camera_cfg)


def create_client_config(
    config: ExperimentConfig,
    metrics_path: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
) -> RobotClientImprovedConfig:
    """Create a client config for a single experiment."""
    robot_cfg = create_robot_config()
    return RobotClientImprovedConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cuda",
        policy_type="smolvla",
        pretrained_name_or_path=DEFAULT_MODEL_PATH,
        actions_per_chunk=config.actions_per_chunk,
        fps=config.fps,
        latency_estimator_type=config.estimator,
        cooldown_enabled=config.cooldown,
        latency_k=config.latency_k,
        epsilon=config.epsilon,
        latency_alpha=0.125,
        latency_beta=0.25,
        diagnostics_enabled=True,
        diagnostics_interval_s=2.0,
        diagnostics_window_s=10.0,
        control_use_deadline_clock=True,
        obs_fallback_on_failure=True,
        obs_fallback_max_age_s=2.0,
        trajectory_viz_enabled=True,
        drop_obs_config=config.drop_obs_config,
        drop_action_config=config.drop_action_config,
        spikes=config.spikes,
        experiment_metrics_path=str(metrics_path),
    )


def run_single_experiment(
    client: RobotClientImproved,
    config: ExperimentConfig,
    metrics_path: Path,
    task: str = DEFAULT_TASK,
) -> dict:
    """Run a single experiment using an already-connected client.
    
    The client should already be started (robot connected, policy loaded).
    This function runs the control loop for the specified duration, then
    signals stop but does NOT disconnect the robot.
    """
    print(f"\nRunning experiment: {config.name}")
    print(f"  Estimator: {config.estimator}, Cooldown: {config.cooldown}")
    if config.drop_obs_config:
        print(f"  Drop obs: {config.drop_obs_config}")
    if config.drop_action_config:
        print(f"  Drop action: {config.drop_action_config}")
    if config.spikes:
        print(f"  Spikes: {config.spikes}")
    print(f"  Duration: {config.duration_s}s")

    # Use threading event to signal stop (instead of calling client.stop())
    stop_event = threading.Event()
    
    def stop_after_duration():
        time.sleep(config.duration_s)
        stop_event.set()
        # Signal the client to stop the control loop (but not disconnect)
        client.signal_stop()

    timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
    
    # Reset client state for new experiment
    client.reset_for_new_experiment(str(metrics_path))
    
    # Start threads
    obs_thread = threading.Thread(target=client.observation_sender, daemon=True)
    action_thread = threading.Thread(target=client.action_receiver, daemon=True)
    obs_thread.start()
    action_thread.start()
    timer_thread.start()
    
    print(f"  Running for {config.duration_s}s...")
    try:
        client.control_loop(task=task)
    except Exception as e:
        import traceback
        print(f"Control loop error: {e}")
        traceback.print_exc()
    
    # Wait for threads to finish (they should exit when stop_event is set)
    stop_event.wait(timeout=config.duration_s + 5.0)
    
    success = metrics_path.exists()
    print(f"  Experiment finished. Metrics saved: {success}")
    return {"success": success, "metrics_path": str(metrics_path)}


def run_experiment(config: ExperimentConfig, output_dir: Path, server_address: str = DEFAULT_SERVER_ADDRESS, task: str = DEFAULT_TASK) -> dict:
    """Run a single standalone experiment (creates and tears down client)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.name}_{timestamp}"
    metrics_path = output_dir / f"{exp_name}.csv"

    print(f"\nRunning experiment: {config.name}")
    print(f"  Estimator: {config.estimator}, Cooldown: {config.cooldown}")
    if config.drop_obs_config:
        print(f"  Drop obs: {config.drop_obs_config}")
    if config.drop_action_config:
        print(f"  Drop action: {config.drop_action_config}")
    if config.spikes:
        print(f"  Spikes: {config.spikes}")

    client_cfg = create_client_config(config, metrics_path, server_address)
    client = RobotClientImproved(client_cfg)
    stop_requested = threading.Event()

    def stop_after_duration():
        time.sleep(config.duration_s)
        stop_requested.set()
        client.signal_stop()

    def signal_handler(sig, frame):
        stop_requested.set()
        client.signal_stop()

    timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        print(f"  Starting client...")
        if client.start():
            print(f"  Client started successfully")
            obs_thread = threading.Thread(target=client.observation_sender, daemon=True)
            action_thread = threading.Thread(target=client.action_receiver, daemon=True)
            obs_thread.start()
            action_thread.start()
            timer_thread.start()
            print(f"  Running for {config.duration_s}s...")
            try:
                client.control_loop(task=task)
            except Exception as e:
                import traceback
                print(f"Control loop error: {e}")
                traceback.print_exc()
            success = metrics_path.exists()
            print(f"  Experiment finished. Metrics saved: {success}")
            return {"success": success, "metrics_path": str(metrics_path)}
        else:
            print(f"  ERROR: Client failed to start!")
            return {"success": False, "error": "Client failed to start"}
    except Exception as e:
        import traceback
        print(f"  ERROR: Exception during experiment: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        signal.signal(signal.SIGINT, original_handler)
        # Only disconnect at the very end for standalone experiments
        try:
            client.stop()
        except Exception:
            pass


def run_experiment_config(experiment_config_name: str, output_dir: Path, pause_between_s: float = 10.0, server_address: str = DEFAULT_SERVER_ADDRESS) -> None:
    """Run multiple experiments from a config, keeping the robot connected between them."""
    if experiment_config_name not in ALL_EXPERIMENT_CONFIGS:
        print(f"Unknown experiment config: {experiment_config_name}. Available: {list(ALL_EXPERIMENT_CONFIGS.keys())}")
        return

    configs = ALL_EXPERIMENT_CONFIGS[experiment_config_name]
    print(f"\nRunning experiment config '{experiment_config_name}' with {len(configs)} experiments")
    print("Robot will stay connected between experiments.")

    # Create client with first config (robot connects once)
    first_config = configs[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_metrics_path = output_dir / f"{first_config.name}_{timestamp}.csv"
    client_cfg = create_client_config(first_config, first_metrics_path, server_address)
    client = RobotClientImproved(client_cfg)
    
    # Setup signal handler for clean shutdown
    shutdown_requested = threading.Event()
    original_handler = signal.signal(signal.SIGINT, lambda sig, frame: shutdown_requested.set() or client.signal_stop())

    results = []
    try:
        print(f"\nConnecting robot and policy server...")
        if not client.start():
            print("ERROR: Failed to start client!")
            return
        print("Robot connected. Starting experiments...\n")
        
        for i, config in enumerate(configs):
            if shutdown_requested.is_set():
                print("\nShutdown requested, stopping experiments.")
                break
                
            print(f"\n{'='*50}")
            print(f"[{i+1}/{len(configs)}] {config.name}")
            print(f"{'='*50}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = output_dir / f"{config.name}_{timestamp}.csv"
            
            try:
                # Update client config for this experiment
                # Note: Some settings like estimator_type are set at init, so we update what we can
                client.config.latency_k = config.latency_k
                client.config.epsilon = config.epsilon
                client.config.cooldown_enabled = config.cooldown
                client.config.drop_obs_config = config.drop_obs_config
                client.config.drop_action_config = config.drop_action_config
                client.config.spikes = config.spikes
                client.config.experiment_metrics_path = str(metrics_path)
                
                result = run_single_experiment(client, config, metrics_path)
            except Exception as e:
                import traceback
                print(f"Experiment error: {e}")
                traceback.print_exc()
                result = {"success": False, "error": str(e)}
            
            results.append(result)
            
            if i < len(configs) - 1 and not shutdown_requested.is_set():
                print(f"\nPausing {pause_between_s}s before next experiment...")
                time.sleep(pause_between_s)

    finally:
        signal.signal(signal.SIGINT, original_handler)
        print("\nDisconnecting robot...")
        try:
            client.stop()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        print("Robot disconnected.")

    success_count = sum(1 for r in results if r.get("success"))
    print(f"\nExperiment config complete: {success_count}/{len(results)} succeeded")


def main():
    parser = argparse.ArgumentParser(description="Latency-Adaptive Async Inference Experiment Runner")
    parser.add_argument("--experiment_config", type=str, choices=list(ALL_EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--estimator", type=str, choices=["jk", "max_last_10"], default="jk")
    parser.add_argument("--cooldown", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--latency_k", type=float, default=1.5)
    parser.add_argument("--epsilon", type=int, default=1)
    parser.add_argument("--duration_s", type=float, default=60.0)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--pause_between_s", type=float, default=10.0)
    parser.add_argument("--drop_obs", type=str, default="")
    parser.add_argument("--drop_action", type=str, default="")
    parser.add_argument("--spikes", type=str, default="")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment_config:
        run_experiment_config(args.experiment_config, output_dir, args.pause_between_s, args.server_address)
    else:
        drop_obs_config = DropConfig.from_dicts(json.loads(args.drop_obs)) if args.drop_obs else None
        drop_action_config = DropConfig.from_dicts(json.loads(args.drop_action)) if args.drop_action else None
        spikes = json.loads(args.spikes) if args.spikes else []

        config = ExperimentConfig(
            name=f"single_{args.estimator}_cooldown_{args.cooldown}",
            estimator=args.estimator,
            cooldown=(args.cooldown == "on"),
            latency_k=args.latency_k,
            epsilon=args.epsilon,
            duration_s=args.duration_s,
            drop_obs_config=drop_obs_config,
            drop_action_config=drop_action_config,
            spikes=spikes,
        )
        run_experiment(config, output_dir, server_address=args.server_address)


if __name__ == "__main__":
    main()
