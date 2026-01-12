#!/usr/bin/env python3
"""
Latency-Adaptive Async Inference Experiment Runner

This script runs experiments on a REAL ROBOT to validate the latency-adaptive
async inference algorithm. It assumes the policy server is already running.

Usage:
    python examples/experiments/latency_adaptive_sweep.py --sweep spike --output_dir results/
    python examples/experiments/latency_adaptive_sweep.py --drop_obs '[{"start_s": 5, "duration_s": 1}]'
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


ESTIMATOR_COMPARISON_SWEEP = [
    ExperimentConfig(name=f"estimator_{est}", estimator=est, cooldown=True, duration_s=15.0)
    for est in ["jk", "max_last_10"]
]

K_PARAMETER_SWEEP = [
    ExperimentConfig(name=f"jk_K{k}_cooldown_on", estimator="jk", cooldown=True, latency_k=k)
    for k in [0.5, 1.0, 1.5, 2.0, 4.0]
]

EPSILON_SWEEP = [
    ExperimentConfig(name=f"jk_eps{eps}_cooldown_on", estimator="jk", cooldown=True, epsilon=eps)
    for eps in [0, 1, 2, 3, 5]
]

QUICK_TEST_SWEEP = [
    ExperimentConfig(name="jk_cooldown_on", estimator="jk", cooldown=True, duration_s=30.0),
    ExperimentConfig(name="jk_cooldown_off", estimator="jk", cooldown=False, duration_s=30.0),
]

OBS_DROP_SWEEP = [
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

ACTION_DROP_SWEEP = [
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

DROP_RECOVERY_COMPARISON_SWEEP = [
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

SPIKE_SWEEP = [
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

SPIKE_ESTIMATOR_COMPARISON_SWEEP = [
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

ALL_SWEEPS = {
    "estimator_comparison": ESTIMATOR_COMPARISON_SWEEP,
    "k_parameter": K_PARAMETER_SWEEP,
    "epsilon": EPSILON_SWEEP,
    "quick_test": QUICK_TEST_SWEEP,
    "obs_drop": OBS_DROP_SWEEP,
    "action_drop": ACTION_DROP_SWEEP,
    "drop_recovery": DROP_RECOVERY_COMPARISON_SWEEP,
    "spike": SPIKE_SWEEP,
    "spike_estimator": SPIKE_ESTIMATOR_COMPARISON_SWEEP,
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


def run_experiment(config: ExperimentConfig, output_dir: Path, server_address: str = DEFAULT_SERVER_ADDRESS, task: str = DEFAULT_TASK) -> dict:
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

    robot_cfg = create_robot_config()
    client_cfg = RobotClientImprovedConfig(
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

    client = RobotClientImproved(client_cfg)
    shutdown_event = threading.Event()

    def stop_after_duration():
        time.sleep(config.duration_s)
        shutdown_event.set()
        client.stop()

    def signal_handler(sig, frame):
        shutdown_event.set()
        client.stop()

    timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        if client.start():
            obs_thread = threading.Thread(target=client.observation_sender, daemon=True)
            action_thread = threading.Thread(target=client.action_receiver, daemon=True)
            obs_thread.start()
            action_thread.start()
            timer_thread.start()
            try:
                client.control_loop(task=task)
            except Exception as e:
                print(f"Control loop error: {e}")
            finally:
                try:
                    client.stop()
                except Exception:
                    pass
            return {"success": metrics_path.exists(), "metrics_path": str(metrics_path)}
        return {"success": False, "error": "Client failed to start"}
    finally:
        signal.signal(signal.SIGINT, original_handler)


def run_sweep(sweep_name: str, output_dir: Path, pause_between_s: float = 10.0, server_address: str = DEFAULT_SERVER_ADDRESS) -> None:
    if sweep_name not in ALL_SWEEPS:
        print(f"Unknown sweep: {sweep_name}. Available: {list(ALL_SWEEPS.keys())}")
        return

    configs = ALL_SWEEPS[sweep_name]
    print(f"\nRunning sweep '{sweep_name}' with {len(configs)} experiments")

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.name}")
        try:
            result = run_experiment(config, output_dir, server_address)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        results.append(result)
        if i < len(configs) - 1:
            time.sleep(pause_between_s)

    success_count = sum(1 for r in results if r.get("success"))
    print(f"\nSweep complete: {success_count}/{len(results)} succeeded")


def main():
    parser = argparse.ArgumentParser(description="Latency-Adaptive Async Inference Experiment Runner")
    parser.add_argument("--sweep", type=str, choices=list(ALL_SWEEPS.keys()))
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

    if args.sweep:
        run_sweep(args.sweep, output_dir, args.pause_between_s, args.server_address)
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
