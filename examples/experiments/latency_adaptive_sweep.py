#!/usr/bin/env python3
"""
Latency-Adaptive Async Inference Experiment Runner

This script runs experiments on a REAL ROBOT to validate the latency-adaptive
async inference algorithm. It assumes the policy server is already running.

Experiments:
- Compare latency estimators (JK vs max_last_10)
- Compare cooldown on/off
- Parameter sweeps (K, epsilon)
- Drop recovery testing (observation and action drops)

Usage:
    # First, start the policy server (in another terminal):
    python examples/tutorial/async-inf/policy_server_improved.py

    # Single experiment
    python examples/experiments/latency_adaptive_sweep.py \
        --estimator jk \
        --cooldown on \
        --duration_s 15 \
        --output_dir results/

    # Sweep mode (runs predefined parameter grid)
    python examples/experiments/latency_adaptive_sweep.py \
        --sweep estimator_comparison \
        --duration_s 60 \
        --output_dir results/sweep/

    # Test observation drops (random 5% drop rate)
    python examples/experiments/latency_adaptive_sweep.py \
        --drop_obs_random_p 0.05 \
        --duration_s 60 \
        --output_dir results/

    # Test observation drops (burst: 1s drop every 20s)
    python examples/experiments/latency_adaptive_sweep.py \
        --drop_obs_burst_period_s 20 \
        --drop_obs_burst_duration_s 1 \
        --duration_s 60 \
        --output_dir results/

    # Run drop recovery sweep
    python examples/experiments/latency_adaptive_sweep.py \
        --sweep obs_drop \
        --output_dir results/drop_test/

Note: Each experiment run requires manual reset of the robot/environment.
      The script pauses between runs to allow this.
"""

import argparse
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lerobot.async_inference.robot_client_improved import RobotClientImproved
from lerobot.async_inference.configs_improved import RobotClientImprovedConfig
from lerobot.async_inference.utils.simulation import DropConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig


# =============================================================================
# Default Configuration (matching tutorial setup)
# =============================================================================

DEFAULT_SERVER_ADDRESS = "192.168.4.37:8080"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_2026_01_03"
DEFAULT_MODEL_PATH = "/home/jack/code/self-driving-screwdriver-robot/wandb_downloads/so101_smolvla_pickplaceorangecube_e100_20260108_203916/100000/pretrained_model/"
DEFAULT_TASK = "Pick up the orange cube and place it on the black X marker with the white background"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    estimator: str  # "jk" or "max_last_10"
    cooldown: bool
    latency_k: float = 1.5  # JK scaling factor for deviation
    epsilon: int = 1  # Safety margin in action steps
    duration_s: float = 60.0
    fps: int = 30
    actions_per_chunk: int = 50
    # Drop injection for experiments (None = disabled)
    drop_obs_config: DropConfig | None = None
    drop_action_config: DropConfig | None = None
    # Spike injection for latency testing (passed to server)
    spike_base_delay_ms: float = 0.0  # Base delay in milliseconds
    spike_delay_ms: float = 0.0  # Additional delay during spike (ms)
    spike_period_s: float = 0.0  # Time between spikes (0 = disabled)
    spike_duration_s: float = 0.0  # How long each spike lasts (seconds)


# =============================================================================
# Predefined Experiment Sweeps
# =============================================================================

# Compare JK vs max_last_10 estimators with cooldown on/off
ESTIMATOR_COMPARISON_SWEEP = [
    ExperimentConfig(
        name=f"estimator_{est}",
        estimator=est,
        cooldown=True,
        duration_s=15.0,
    )
    for est in ["jk", "max_last_10"]
]

# Sweep K parameter (JK scaling factor for deviation)
K_PARAMETER_SWEEP = [
    ExperimentConfig(
        name=f"jk_K{k}_cooldown_on",
        estimator="jk",
        cooldown=True,
        latency_k=k,
    )
    for k in [0.5, 1.0, 1.5, 2.0, 4.0]
]

# Sweep epsilon parameter (safety margin)
EPSILON_SWEEP = [
    ExperimentConfig(
        name=f"jk_eps{eps}_cooldown_on",
        estimator="jk",
        cooldown=True,
        epsilon=eps,
    )
    for eps in [0, 1, 2, 3, 5]
]

# Quick test sweep (just 2 configs for testing)
QUICK_TEST_SWEEP = [
    ExperimentConfig(name="jk_cooldown_on", estimator="jk", cooldown=True, duration_s=30.0),
    ExperimentConfig(name="jk_cooldown_off", estimator="jk", cooldown=False, duration_s=30.0),
]

# Observation drop recovery test (tests cooldown mechanism under drops)
OBS_DROP_SWEEP = [
    ExperimentConfig(
        name="jk_no_drops",
        estimator="jk",
        cooldown=True,
    ),
    ExperimentConfig(
        name="jk_random_5pct_drops",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(random_drop_p=0.05),
    ),
    ExperimentConfig(
        name="jk_burst_drops_1s_every_20s",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(burst_period_s=20.0, burst_duration_s=1.0),
    ),
    ExperimentConfig(
        name="jk_burst_drops_2s_every_30s",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(burst_period_s=30.0, burst_duration_s=2.0),
    ),
]

# Action chunk drop recovery test
ACTION_DROP_SWEEP = [
    ExperimentConfig(
        name="jk_no_action_drops",
        estimator="jk",
        cooldown=True,
    ),
    ExperimentConfig(
        name="jk_random_5pct_action_drops",
        estimator="jk",
        cooldown=True,
        drop_action_config=DropConfig(random_drop_p=0.05),
    ),
    ExperimentConfig(
        name="jk_burst_action_drops_1s_every_20s",
        estimator="jk",
        cooldown=True,
        drop_action_config=DropConfig(burst_period_s=20.0, burst_duration_s=1.0),
    ),
]

# Compare cooldown vs merge_reset under drops (key paper contribution)
DROP_RECOVERY_COMPARISON_SWEEP = [
    ExperimentConfig(
        name="cooldown_burst_drops",
        estimator="jk",
        cooldown=True,
        drop_obs_config=DropConfig(burst_period_s=20.0, burst_duration_s=1.0),
    ),
    ExperimentConfig(
        name="no_cooldown_burst_drops",
        estimator="jk",
        cooldown=False,
        drop_obs_config=DropConfig(burst_period_s=20.0, burst_duration_s=1.0),
    ),
]

# Latency spike testing (tests JK estimator adaptation to spikes)
SPIKE_SWEEP = [
    ExperimentConfig(
        name="baseline_no_spike",
        estimator="jk",
        cooldown=True,
    ),
    ExperimentConfig(
        name="spike_100ms_base",
        estimator="jk",
        cooldown=True,
        spike_base_delay_ms=100.0,
    ),
    ExperimentConfig(
        name="spike_2s_every_30s",
        estimator="jk",
        cooldown=True,
        spike_base_delay_ms=100.0,
        spike_delay_ms=2000.0,
        spike_period_s=30.0,
        spike_duration_s=1.0,
    ),
    ExperimentConfig(
        name="spike_1s_every_15s",
        estimator="jk",
        cooldown=True,
        spike_base_delay_ms=100.0,
        spike_delay_ms=1000.0,
        spike_period_s=15.0,
        spike_duration_s=0.5,
    ),
]

# Compare JK vs max_last_10 under latency spikes
SPIKE_ESTIMATOR_COMPARISON_SWEEP = [
    ExperimentConfig(
        name="jk_with_spikes",
        estimator="jk",
        cooldown=True,
        spike_base_delay_ms=100.0,
        spike_delay_ms=2000.0,
        spike_period_s=30.0,
        spike_duration_s=1.0,
    ),
    ExperimentConfig(
        name="max_last_10_with_spikes",
        estimator="max_last_10",
        cooldown=True,
        spike_base_delay_ms=100.0,
        spike_delay_ms=2000.0,
        spike_period_s=30.0,
        spike_duration_s=1.0,
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


# =============================================================================
# Experiment Runner
# =============================================================================


def create_robot_config() -> SO101FollowerConfig:
    """Create robot configuration matching tutorial setup."""
    camera_cfg = {
        "camera2": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0",
            width=800,
            height=600,
            fps=30,
            fourcc="MJPG",
            use_threaded_async_read=True,
            allow_stale_frames=True,
        ),
        "camera1": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0",
            width=800,
            height=600,
            fps=30,
            fourcc="MJPG",
            use_threaded_async_read=True,
            allow_stale_frames=True,
        ),
    }

    return SO101FollowerConfig(
        port=DEFAULT_ROBOT_PORT,
        id=DEFAULT_ROBOT_ID,
        cameras=camera_cfg,
    )


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    task: str = DEFAULT_TASK,
) -> dict:
    """Run a single experiment on the real robot.

    Returns dict with success status and metrics path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.name}_{timestamp}"
    metrics_path = output_dir / f"{exp_name}.csv"

    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name}")
    print(f"  Estimator: {config.estimator}")
    print(f"  Cooldown: {config.cooldown}")
    print(f"  K: {config.latency_k}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Duration: {config.duration_s}s")
    if config.drop_obs_config:
        print(f"  Drop obs: {config.drop_obs_config}")
    if config.drop_action_config:
        print(f"  Drop action: {config.drop_action_config}")
    if config.spike_delay_ms > 0 or config.spike_base_delay_ms > 0:
        print(f"  Spike: base={config.spike_base_delay_ms}ms, spike={config.spike_delay_ms}ms, "
              f"period={config.spike_period_s}s, duration={config.spike_duration_s}s")
    print(f"  Output: {metrics_path}")
    print(f"{'='*60}\n")

    # Create robot and client configs
    robot_cfg = create_robot_config()

    client_cfg = RobotClientImprovedConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cuda",
        policy_type="smolvla",
        pretrained_name_or_path=DEFAULT_MODEL_PATH,
        actions_per_chunk=config.actions_per_chunk,
        fps=config.fps,
        # Experiment parameters
        latency_estimator_type=config.estimator,
        cooldown_enabled=config.cooldown,
        latency_k=config.latency_k,
        epsilon=config.epsilon,
        # Standard settings from tutorial
        latency_alpha=0.125,
        latency_beta=0.25,
        diagnostics_enabled=True,
        diagnostics_interval_s=2.0,
        diagnostics_window_s=10.0,
        control_use_deadline_clock=True,
        obs_fallback_on_failure=True,
        obs_fallback_max_age_s=2.0,
        trajectory_viz_enabled=True,
        # Drop injection for experiments
        drop_obs_config=config.drop_obs_config,
        drop_action_config=config.drop_action_config,
        # Spike injection (passed to server)
        spike_base_delay_ms=config.spike_base_delay_ms,
        spike_delay_ms=config.spike_delay_ms,
        spike_period_s=config.spike_period_s,
        spike_duration_s=config.spike_duration_s,
        # Experiment metrics
        experiment_metrics_path=str(metrics_path),
    )

    # Create client
    client = RobotClientImproved(client_cfg)
    shutdown_event = threading.Event()

    def stop_after_duration():
        time.sleep(config.duration_s)
        print(f"\nDuration elapsed ({config.duration_s}s), stopping...")
        shutdown_event.set()
        client.stop()

    def signal_handler(sig, frame):
        print("\nInterrupted, stopping...")
        shutdown_event.set()
        client.stop()

    timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        if client.start():
            print("Client started successfully")

            # Start helper threads
            obs_thread = threading.Thread(
                target=client.observation_sender,
                name="observation_sender",
                daemon=True,
            )
            action_thread = threading.Thread(
                target=client.action_receiver,
                name="action_receiver",
                daemon=True,
            )

            obs_thread.start()
            action_thread.start()
            timer_thread.start()

            # Run control loop
            try:
                client.control_loop(task=task)
            except Exception as e:
                print(f"Control loop error: {e}")
            finally:
                client.stop()

            print(f"\nExperiment completed: {config.name}")

            if metrics_path.exists():
                return {"success": True, "metrics_path": str(metrics_path)}
            else:
                return {"success": False, "error": "Metrics file not created"}
        else:
            print("Client failed to start")
            return {"success": False, "error": "Client failed to start"}

    finally:
        signal.signal(signal.SIGINT, original_handler)


def run_sweep(
    sweep_name: str,
    output_dir: Path,
    pause_between_s: float = 10.0,
    server_address: str = DEFAULT_SERVER_ADDRESS,
) -> None:
    """Run a predefined sweep of experiments."""
    if sweep_name not in ALL_SWEEPS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available sweeps: {list(ALL_SWEEPS.keys())}")
        return

    configs = ALL_SWEEPS[sweep_name]
    print(f"\nRunning sweep '{sweep_name}' with {len(configs)} experiments")
    print(f"Pause between experiments: {pause_between_s}s")
    print(f"Estimated total time: {len(configs) * (configs[0].duration_s + pause_between_s) / 60:.1f} min\n")

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.name}")

        result = run_experiment(config, output_dir, server_address)
        results.append(result)

        # Pause between experiments (except after last one)
        if i < len(configs) - 1:
            print(f"\n--- Pausing {pause_between_s}s before next experiment ---")
            print("    (Reset robot/environment if needed)")
            try:
                time.sleep(pause_between_s)
            except KeyboardInterrupt:
                print("\nSweep interrupted by user")
                break

    # Summary
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"Sweep complete: {success_count}/{len(results)} experiments succeeded")
    print(f"{'='*60}")

    # List successful experiments
    for r in results:
        if r.get("success"):
            print(f"  OK: {r['metrics_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Latency-Adaptive Async Inference Experiment Runner (Real Robot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--sweep",
        type=str,
        choices=list(ALL_SWEEPS.keys()),
        help="Run a predefined sweep of experiments",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["jk", "max_last_10"],
        default="jk",
        help="Latency estimator type (for single experiment)",
    )
    parser.add_argument(
        "--cooldown",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable or disable cooldown mechanism",
    )
    parser.add_argument(
        "--latency_k",
        type=float,
        default=1.5,
        help="JK scaling factor for deviation (K parameter)",
    )
    parser.add_argument(
        "--epsilon",
        type=int,
        default=1,
        help="Safety margin in action steps",
    )
    parser.add_argument(
        "--duration_s",
        type=float,
        default=60.0,
        help="Experiment duration in seconds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"Policy server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--pause_between_s",
        type=float,
        default=10.0,
        help="Pause between experiments in sweep mode (for robot reset)",
    )

    # Drop injection arguments (for single experiments)
    parser.add_argument(
        "--drop_obs_random_p",
        type=float,
        default=0.0,
        help="Random observation drop probability (0.0-1.0)",
    )
    parser.add_argument(
        "--drop_obs_burst_period_s",
        type=float,
        default=0.0,
        help="Time between observation drop bursts (0 = disabled)",
    )
    parser.add_argument(
        "--drop_obs_burst_duration_s",
        type=float,
        default=0.0,
        help="Duration of each observation drop burst",
    )
    parser.add_argument(
        "--drop_action_random_p",
        type=float,
        default=0.0,
        help="Random action chunk drop probability (0.0-1.0)",
    )
    parser.add_argument(
        "--drop_action_burst_period_s",
        type=float,
        default=0.0,
        help="Time between action chunk drop bursts (0 = disabled)",
    )
    parser.add_argument(
        "--drop_action_burst_duration_s",
        type=float,
        default=0.0,
        help="Duration of each action chunk drop burst",
    )

    # Spike injection arguments (passed to server for experiments)
    parser.add_argument(
        "--spike_base_delay_ms",
        type=float,
        default=0.0,
        help="Base delay in milliseconds (applied to all inferences)",
    )
    parser.add_argument(
        "--spike_delay_ms",
        type=float,
        default=0.0,
        help="Additional delay during spike periods (milliseconds)",
    )
    parser.add_argument(
        "--spike_period_s",
        type=float,
        default=0.0,
        help="Time between spikes (0 = disabled)",
    )
    parser.add_argument(
        "--spike_duration_s",
        type=float,
        default=0.0,
        help="How long each spike lasts (seconds)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        run_sweep(
            sweep_name=args.sweep,
            output_dir=output_dir,
            pause_between_s=args.pause_between_s,
            server_address=args.server_address,
        )
    else:
        # Single experiment
        # Build drop configs if any drop parameters are specified
        drop_obs_config = None
        if args.drop_obs_random_p > 0 or args.drop_obs_burst_period_s > 0:
            drop_obs_config = DropConfig(
                random_drop_p=args.drop_obs_random_p,
                burst_period_s=args.drop_obs_burst_period_s,
                burst_duration_s=args.drop_obs_burst_duration_s,
            )

        drop_action_config = None
        if args.drop_action_random_p > 0 or args.drop_action_burst_period_s > 0:
            drop_action_config = DropConfig(
                random_drop_p=args.drop_action_random_p,
                burst_period_s=args.drop_action_burst_period_s,
                burst_duration_s=args.drop_action_burst_duration_s,
            )

        config = ExperimentConfig(
            name=f"single_{args.estimator}_cooldown_{args.cooldown}",
            estimator=args.estimator,
            cooldown=(args.cooldown == "on"),
            latency_k=args.latency_k,
            epsilon=args.epsilon,
            duration_s=args.duration_s,
            drop_obs_config=drop_obs_config,
            drop_action_config=drop_action_config,
            spike_base_delay_ms=args.spike_base_delay_ms,
            spike_delay_ms=args.spike_delay_ms,
            spike_period_s=args.spike_period_s,
            spike_duration_s=args.spike_duration_s,
        )
        run_experiment(config, output_dir, server_address=args.server_address)


if __name__ == "__main__":
    main()
