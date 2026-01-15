#!/usr/bin/env python3
"""
RTC Parameter Sweep for Action Discontinuity Minimization

This script runs experiments to find optimal RTC (Real-Time Control) parameters
that minimize action discontinuity (L2 distance between overlapping chunks).

Sweeps (default mode):
- rtc_sigma_d: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
- rtc_full_trajectory_alignment: [True, False]

Alex Soare sweep mode (--alex_soare_sweep):
- num_flow_matching_steps (n): [5, 10, 20]
- rtc_max_guidance_weight (Beta): [None] (auto = n)
- rtc_sigma_d: [0.2] (fixed at optimal value)

Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html

The sweep runs on a real robot with a real policy server. Make sure the server
is running before starting the sweep.

Usage:
    # Run the full sweep (uses defaults from tutorial example)
    uv run python examples/experiments/rtc_sweep.py \
        --duration_s 60 \
        --output_dir results/rtc_sweep/ \
        --server_address 192.168.4.37:8080

    # Run Alex Soare sweep (denoising steps + Beta)
    uv run python examples/experiments/rtc_sweep.py \
        --alex_soare_sweep \
        --duration_s 60 \
        --output_dir results/rtc_sweep_alex/ \
        --server_address 192.168.4.37:8080

    # Run a single configuration
    uv run python examples/experiments/rtc_sweep.py \
        --sigma_d 0.4 \
        --full_trajectory_alignment false \
        --num_flow_matching_steps 10 \
        --rtc_max_guidance_weight 10.0 \
        --duration_s 60 \
        --output_dir results/rtc_sweep/ \
        --server_address 192.168.4.37:8080

    # Analyze existing results only
    uv run python examples/experiments/rtc_sweep.py \
        --analyze_only \
        --output_dir results/rtc_sweep/
"""

import argparse
import csv
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RTCSweepConfig:
    """Configuration for a single RTC parameter experiment.

    Alex Soare optimizations (https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html):
    - num_flow_matching_steps (n): Number of denoising steps. Higher = smoother but slower.
    - rtc_max_guidance_weight (Beta): Should scale with n. None = auto (Beta = n).
    - rtc_sigma_d: Prior variance. 0.2 = stronger guidance, 1.0 = original RTC.
    """

    name: str
    rtc_sigma_d: float
    rtc_full_trajectory_alignment: bool
    duration_s: float = 60.0
    fps: int = 30
    actions_per_chunk: int = 50
    rtc_prefix_attention_schedule: str = "linear"
    # Alex Soare parameters
    num_flow_matching_steps: int | None = None  # None = use policy default (e.g., 10)
    rtc_max_guidance_weight: float | None = None  # None = auto (Beta = n)


# =============================================================================
# Parameter Grid
# =============================================================================

# Default sweep: sigma_d and full_trajectory_alignment
SIGMA_D_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
FULL_TRAJ_ALIGNMENT_VALUES = [False, True]

# Alex Soare sweep: denoising steps and Beta
# Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
# Key insight: Beta should scale with n (denoising steps)
DENOISING_STEPS_VALUES = [5, 10, 20]
MAX_GUIDANCE_WEIGHT_VALUES: list[float | None] = [None]  # None = Beta = n (auto)


def generate_sweep_configs(duration_s: float = 60.0) -> list[RTCSweepConfig]:
    """Generate all configurations for the default RTC parameter sweep (sigma_d, full_traj)."""
    configs = []
    for sigma_d in SIGMA_D_VALUES:
        for full_traj in FULL_TRAJ_ALIGNMENT_VALUES:
            name = f"rtc_sigma{sigma_d}_fulltraj{full_traj}"
            configs.append(
                RTCSweepConfig(
                    name=name,
                    rtc_sigma_d=sigma_d,
                    rtc_full_trajectory_alignment=full_traj,
                    duration_s=duration_s,
                )
            )
    return configs


def generate_alex_soare_sweep_configs(duration_s: float = 60.0) -> list[RTCSweepConfig]:
    """Generate configurations for the Alex Soare sweep (denoising steps + Beta).

    This sweep follows the recommendations from:
    https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html

    Key findings:
    - sigma_d = 0.2 is optimal (fixed)
    - Beta should scale with n (use None for auto)
    - Sweep n = [5, 10, 20] to trade off smoothness vs speed
    """
    configs = []
    # Fixed optimal values from blog
    sigma_d = 0.2
    full_traj = False  # Use gradient-based guidance

    for n in DENOISING_STEPS_VALUES:
        for beta in MAX_GUIDANCE_WEIGHT_VALUES:
            beta_str = "auto" if beta is None else f"{beta}"
            name = f"rtc_n{n}_beta{beta_str}_sigma{sigma_d}"
            configs.append(
                RTCSweepConfig(
                    name=name,
                    rtc_sigma_d=sigma_d,
                    rtc_full_trajectory_alignment=full_traj,
                    num_flow_matching_steps=n,
                    rtc_max_guidance_weight=beta,
                    duration_s=duration_s,
                )
            )
    return configs


# =============================================================================
# Experiment Runner
# =============================================================================


def run_experiment(
    config: RTCSweepConfig,
    output_dir: Path,
    server_address: str,
    robot_config_path: str | None,
    policy_type: str,
    pretrained_name_or_path: str,
    robot_id: str,
    robot_port: str,
    task: str,
) -> dict[str, Any]:
    """Run a single RTC parameter experiment on a real robot.

    This connects to an already-running policy server and runs the robot client
    with the specified RTC parameters.
    """
    # Import here to avoid import issues when just parsing args
    from lerobot.async_inference.robot_client_improved import (
        RobotClientImproved,
    )
    from lerobot.async_inference.configs_improved import RobotClientImprovedConfig
    from lerobot.robots.so101_follower import SO101FollowerConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.name}_{timestamp}"
    metrics_path = output_dir / f"{exp_name}.csv"
    config_path = output_dir / f"{exp_name}_config.json"

    print(f"\n{'='*60}")
    print(f"Running RTC experiment: {config.name}")
    print(f"  sigma_d: {config.rtc_sigma_d}")
    print(f"  full_trajectory_alignment: {config.rtc_full_trajectory_alignment}")
    print(f"  num_flow_matching_steps (n): {config.num_flow_matching_steps or 'default'}")
    print(f"  rtc_max_guidance_weight (Beta): {config.rtc_max_guidance_weight or 'auto (=n)'}")
    print(f"  Duration: {config.duration_s}s")
    print(f"  Output: {metrics_path}")
    print(f"{'='*60}\n")

    # Save configuration
    config_data = {
        "name": config.name,
        "rtc_sigma_d": config.rtc_sigma_d,
        "rtc_full_trajectory_alignment": config.rtc_full_trajectory_alignment,
        "num_flow_matching_steps": config.num_flow_matching_steps,
        "rtc_max_guidance_weight": config.rtc_max_guidance_weight,
        "duration_s": config.duration_s,
        "fps": config.fps,
        "actions_per_chunk": config.actions_per_chunk,
        "server_address": server_address,
        "timestamp": timestamp,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # Create robot config
    # For real robot experiments, we need a proper robot config
    if robot_config_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("robot_config", robot_config_path)
        robot_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(robot_config_module)
        robot_cfg = robot_config_module.robot_config
    else:
        # Camera configuration matching tutorial example
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
        robot_cfg = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
            cameras=camera_cfg,
        )

    # Create client config with RTC parameters
    client_cfg = RobotClientImprovedConfig(
        robot=robot_cfg,
        server_address=server_address,
        policy_device="cpu",  # Not used - server has the policy
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        actions_per_chunk=config.actions_per_chunk,
        fps=config.fps,
        # RTC parameters being swept
        rtc_enabled=True,
        rtc_sigma_d=config.rtc_sigma_d,
        rtc_full_trajectory_alignment=config.rtc_full_trajectory_alignment,
        rtc_prefix_attention_schedule=config.rtc_prefix_attention_schedule,
        # Alex Soare parameters (denoising steps + Beta)
        num_flow_matching_steps=config.num_flow_matching_steps,
        rtc_max_guidance_weight=config.rtc_max_guidance_weight,
        # Metrics collection
        experiment_metrics_path=str(metrics_path),
        diagnostics_enabled=True,
        trajectory_viz_enabled=False,  # Disable to reduce overhead
    )

    # Run experiment
    client = RobotClientImproved(client_cfg)
    shutdown_event = threading.Event()

    def stop_after_duration():
        time.sleep(config.duration_s)
        print(f"Duration elapsed ({config.duration_s}s), stopping client...")
        shutdown_event.set()
        client.stop()

    timer_thread = threading.Thread(target=stop_after_duration, daemon=True)

    def signal_handler(sig, frame):
        print("\nInterrupted, stopping...")
        shutdown_event.set()
        client.stop()

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

            # Run the control loop
            try:
                client.control_loop(task=task)
            except Exception as e:
                print(f"Control loop error: {e}")
            finally:
                client.stop()

            # Get summary from metrics
            if client._experiment_metrics:
                summary = client._experiment_metrics.get_summary()
            else:
                summary = {}

            print(f"\nExperiment completed: {config.name}")
            print(f"  Summary: {summary}")

            return {
                "success": True,
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "config": config.name,
                "summary": summary,
            }
        else:
            print("Client failed to start")
            return {"success": False, "error": "Client failed to start"}

    finally:
        signal.signal(signal.SIGINT, original_handler)


def analyze_results(output_dir: Path) -> dict[str, Any]:
    """Analyze all experiment results and find the best configuration."""
    import pandas as pd

    results = []
    config_files = list(output_dir.glob("*_config.json"))

    for config_file in config_files:
        # Load config
        with open(config_file) as f:
            config = json.load(f)

        # Find corresponding CSV
        csv_file = config_file.with_name(config_file.stem.replace("_config", "") + ".csv")
        if not csv_file.exists():
            continue

        # Load and analyze CSV
        df = pd.read_csv(csv_file)

        # Filter to rows where action was received (have L2 data)
        df_chunks = df[df["chunk_mean_l2"].notna() & (df["chunk_mean_l2"] != "")]
        if len(df_chunks) == 0:
            continue

        df_chunks = df_chunks.copy()
        df_chunks["chunk_mean_l2"] = pd.to_numeric(df_chunks["chunk_mean_l2"])
        df_chunks["chunk_max_l2"] = pd.to_numeric(df_chunks["chunk_max_l2"])

        # Compute aggregate stats
        results.append({
            "name": config["name"],
            "sigma_d": config["rtc_sigma_d"],
            "full_traj_alignment": config["rtc_full_trajectory_alignment"],
            "mean_l2_avg": df_chunks["chunk_mean_l2"].mean(),
            "mean_l2_std": df_chunks["chunk_mean_l2"].std(),
            "mean_l2_max": df_chunks["chunk_mean_l2"].max(),
            "max_l2_max": df_chunks["chunk_max_l2"].max(),
            "chunk_count": len(df_chunks),
            "stall_count": df["stall"].sum() if "stall" in df.columns else 0,
            "stall_fraction": df["stall"].mean() if "stall" in df.columns else 0,
        })

    if not results:
        print("No valid results found")
        return {}

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_l2_avg")

    # Print summary table
    print("\n" + "=" * 80)
    print("RTC Parameter Sweep Results (sorted by mean L2)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Find best configuration
    best = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print(f"  sigma_d: {best['sigma_d']}")
    print(f"  full_trajectory_alignment: {best['full_traj_alignment']}")
    print(f"  mean_l2_avg: {best['mean_l2_avg']:.6f}")
    print(f"  stall_fraction: {best['stall_fraction']:.2%}")
    print("=" * 80)

    # Save summary
    summary_path = output_dir / "sweep_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    return {
        "best_config": best.to_dict(),
        "all_results": results,
    }


def run_sweep(
    output_dir: Path,
    server_address: str,
    duration_s: float,
    robot_config_path: str | None,
    policy_type: str,
    pretrained_name_or_path: str,
    robot_id: str,
    robot_port: str,
    task: str,
    pause_between_s: float = 5.0,
) -> None:
    """Run the full RTC parameter sweep."""
    configs = generate_sweep_configs(duration_s=duration_s)

    print(f"\nRunning RTC parameter sweep with {len(configs)} configurations")
    print(f"  sigma_d values: {SIGMA_D_VALUES}")
    print(f"  full_trajectory_alignment values: {FULL_TRAJ_ALIGNMENT_VALUES}")
    print(f"  Duration per config: {duration_s}s")
    print(f"  Total estimated time: {len(configs) * (duration_s + pause_between_s) / 60:.1f} minutes\n")

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.name}")

        result = run_experiment(
            config=config,
            output_dir=output_dir,
            server_address=server_address,
            robot_config_path=robot_config_path,
            policy_type=policy_type,
            pretrained_name_or_path=pretrained_name_or_path,
            robot_id=robot_id,
            robot_port=robot_port,
            task=task,
        )
        results.append(result)

        # Pause between experiments to let the robot settle
        if i < len(configs) - 1:
            print(f"\nPausing {pause_between_s}s before next experiment...")
            time.sleep(pause_between_s)

    # Analyze results
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"Sweep complete: {success_count}/{len(results)} experiments succeeded")
    print(f"{'='*60}")

    if success_count > 0:
        analyze_results(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="RTC Parameter Sweep for Action Discontinuity Minimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sweep vs single experiment
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the full parameter sweep (default if no sigma_d specified)",
    )
    parser.add_argument(
        "--alex_soare_sweep",
        action="store_true",
        help="Run Alex Soare sweep (denoising steps + Beta) instead of default sweep. "
        "Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html",
    )

    # Single experiment parameters
    parser.add_argument(
        "--sigma_d",
        type=float,
        help="RTC sigma_d value (run single experiment instead of sweep)",
    )
    parser.add_argument(
        "--full_trajectory_alignment",
        type=str,
        choices=["true", "false"],
        default="false",
        help="RTC full_trajectory_alignment setting",
    )
    # Alex Soare parameters
    parser.add_argument(
        "--num_flow_matching_steps",
        type=int,
        default=None,
        help="Number of flow matching denoising steps (n). Higher = smoother but slower. "
        "None = use policy default (e.g., 10 for PI0/SmolVLA)",
    )
    parser.add_argument(
        "--rtc_max_guidance_weight",
        type=float,
        default=None,
        help="RTC max guidance weight (Beta). None = auto (Beta = n). "
        "Alex Soare recommends Beta = n.",
    )

    # Common parameters
    parser.add_argument(
        "--duration_s",
        type=float,
        default=60.0,
        help="Duration of each experiment in seconds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rtc_sweep/",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="192.168.4.37:8080",
        help="Policy server address (must already be running)",
    )
    parser.add_argument(
        "--robot_config",
        type=str,
        default=None,
        help="Path to Python file containing robot_config variable",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="smolvla",
        help="Policy type (e.g., 'smolvla', 'act', 'pi0')",
    )
    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        default="/home/jack/code/self-driving-screwdriver-robot/wandb_downloads/so101_smolvla_pickplaceorangecube_e100_20260108_203916/100000/pretrained_model/",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="so101_follower_2026_01_03",
        help="Robot ID for calibration files",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Robot serial port",
    )
    parser.add_argument(
        "--pause_between_s",
        type=float,
        default=5.0,
        help="Pause between experiments in seconds",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the orange cube and place it on the black X marker with the white background",
        help="Task description for VLA policies",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze existing results (don't run experiments)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analyze_results(output_dir)
        return

    if args.sigma_d is not None:
        # Single experiment with all specified parameters
        n_str = f"_n{args.num_flow_matching_steps}" if args.num_flow_matching_steps else ""
        beta_str = f"_beta{args.rtc_max_guidance_weight}" if args.rtc_max_guidance_weight else ""
        config = RTCSweepConfig(
            name=f"rtc_sigma{args.sigma_d}_fulltraj{args.full_trajectory_alignment}{n_str}{beta_str}",
            rtc_sigma_d=args.sigma_d,
            rtc_full_trajectory_alignment=(args.full_trajectory_alignment == "true"),
            num_flow_matching_steps=args.num_flow_matching_steps,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            duration_s=args.duration_s,
        )
        run_experiment(
            config=config,
            output_dir=output_dir,
            server_address=args.server_address,
            robot_config_path=args.robot_config,
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_name_or_path,
            robot_id=args.robot_id,
            robot_port=args.robot_port,
            task=args.task,
        )
    elif args.alex_soare_sweep:
        # Alex Soare sweep (denoising steps + Beta)
        configs = generate_alex_soare_sweep_configs(duration_s=args.duration_s)
        print(f"\nRunning Alex Soare RTC sweep with {len(configs)} configurations")
        print(f"  num_flow_matching_steps (n): {DENOISING_STEPS_VALUES}")
        print(f"  rtc_max_guidance_weight (Beta): auto (=n)")
        print(f"  rtc_sigma_d: 0.2 (fixed)")
        print(f"  Duration per config: {args.duration_s}s")
        print(f"  Total estimated time: {len(configs) * (args.duration_s + args.pause_between_s) / 60:.1f} minutes\n")

        results = []
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config.name}")
            result = run_experiment(
                config=config,
                output_dir=output_dir,
                server_address=args.server_address,
                robot_config_path=args.robot_config,
                policy_type=args.policy_type,
                pretrained_name_or_path=args.pretrained_name_or_path,
                robot_id=args.robot_id,
                robot_port=args.robot_port,
                task=args.task,
            )
            results.append(result)
            if i < len(configs) - 1:
                print(f"\nPausing {args.pause_between_s}s before next experiment...")
                time.sleep(args.pause_between_s)

        success_count = sum(1 for r in results if r.get("success"))
        print(f"\n{'='*60}")
        print(f"Alex Soare sweep complete: {success_count}/{len(results)} experiments succeeded")
        print(f"{'='*60}")
        if success_count > 0:
            analyze_results(output_dir)
    else:
        # Default sweep (sigma_d + full_trajectory_alignment)
        run_sweep(
            output_dir=output_dir,
            server_address=args.server_address,
            duration_s=args.duration_s,
            robot_config_path=args.robot_config,
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_name_or_path,
            robot_id=args.robot_id,
            robot_port=args.robot_port,
            task=args.task,
            pause_between_s=args.pause_between_s,
        )


if __name__ == "__main__":
    main()
