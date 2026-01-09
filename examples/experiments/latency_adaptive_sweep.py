#!/usr/bin/env python3
"""
Latency-Adaptive Async Inference Experiment Runner

This script runs experiments to validate the latency-adaptive async inference algorithm:
- Compare cooldown on/off
- Compare latency estimators (JK vs max_last_10)
- Test drop recovery scenarios
- Sweep fixed latency configurations

Usage:
    # Single experiment
    python examples/experiments/latency_adaptive_sweep.py \
        --experiment fixed_latency \
        --latency_ms 100 \
        --estimator jk \
        --cooldown on \
        --duration_s 60 \
        --output_dir results/

    # Sweep mode (runs predefined parameter grid)
    python examples/experiments/latency_adaptive_sweep.py \
        --sweep \
        --output_dir results/sweep/
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    latency_ms: float
    estimator: str  # "jk" or "max_last_10"
    cooldown: bool
    drop_obs_burst: str | None = None
    drop_action_burst: str | None = None
    drop_obs_p: float = 0.0
    drop_action_p: float = 0.0
    duration_s: float = 60.0
    fps: int = 30
    actions_per_chunk: int = 50


# =============================================================================
# Predefined Experiment Sweeps
# =============================================================================

FIXED_LATENCY_SWEEP = [
    ExperimentConfig(
        name=f"fixed_L{L}_cooldown_{c}",
        latency_ms=float(L),
        estimator="jk",
        cooldown=c,
    )
    for L in [50, 100, 200, 400, 800]
    for c in [True, False]
]

ESTIMATOR_COMPARISON_SWEEP = [
    ExperimentConfig(
        name=f"estimator_{est}_cooldown_{c}",
        latency_ms=100.0,
        estimator=est,
        cooldown=c,
    )
    for est in ["jk", "max_last_10"]
    for c in [True, False]
]

DROP_RECOVERY_SWEEP = [
    # Action drop bursts
    ExperimentConfig(
        name=f"drop_action_cooldown_{c}",
        latency_ms=100.0,
        estimator="jk",
        cooldown=c,
        drop_action_burst="1s@20s",
    )
    for c in [True, False]
] + [
    # Observation drop bursts
    ExperimentConfig(
        name=f"drop_obs_cooldown_{c}",
        latency_ms=100.0,
        estimator="jk",
        cooldown=c,
        drop_obs_burst="1s@20s",
    )
    for c in [True, False]
] + [
    # Random drops
    ExperimentConfig(
        name=f"random_drops_p{int(p*100)}_cooldown_{c}",
        latency_ms=100.0,
        estimator="jk",
        cooldown=c,
        drop_obs_p=p,
        drop_action_p=p,
    )
    for p in [0.01, 0.05, 0.1]
    for c in [True, False]
]

ALL_SWEEPS = {
    "fixed_latency": FIXED_LATENCY_SWEEP,
    "estimator_comparison": ESTIMATOR_COMPARISON_SWEEP,
    "drop_recovery": DROP_RECOVERY_SWEEP,
}


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    server_host: str = "localhost",
    server_port: int = 8080,
) -> dict[str, Any]:
    """Run a single experiment and return results.

    This spawns both server and client processes in simulation mode,
    runs for the specified duration, then collects metrics.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.name}_{timestamp}"
    metrics_path = output_dir / f"{exp_name}.csv"

    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name}")
    print(f"  Latency: {config.latency_ms}ms")
    print(f"  Estimator: {config.estimator}")
    print(f"  Cooldown: {config.cooldown}")
    print(f"  Duration: {config.duration_s}s")
    print(f"  Output: {metrics_path}")
    print(f"{'='*60}\n")

    # Build server command (using draccus CLI format)
    server_cmd = [
        sys.executable, "-m", "lerobot.async_inference.policy_server_improved",
        f"--host={server_host}",
        f"--port={server_port}",
        f"--fps={config.fps}",
        "--mock_policy=true",
        f"--mock_inference_delay_ms={config.latency_ms}",
    ]

    # Build client command
    client_cmd = [
        sys.executable, "-c",
        _generate_client_script(config, server_host, server_port, metrics_path),
    ]

    # Start server
    print(f"Starting mock server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
    )
    time.sleep(3.0)  # Wait for server to start

    # Check if server started
    if server_proc.poll() is not None:
        stdout, _ = server_proc.communicate()
        print(f"Server failed to start:\n{stdout.decode()}")
        return {"success": False, "error": "Server failed to start"}

    try:
        print("Starting mock client...")
        client_proc = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for client to complete
        stdout, stderr = client_proc.communicate(timeout=config.duration_s + 30)
        client_returncode = client_proc.returncode

        if stderr:
            print(f"Client stderr:\n{stderr.decode()}")

        if client_returncode != 0:
            print(f"Client failed with return code {client_returncode}")
            return {"success": False, "error": f"Client failed with code {client_returncode}"}

        print(f"Client stdout:\n{stdout.decode()}")

    finally:
        # Stop server
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    # Check if metrics file was created
    if metrics_path.exists():
        return {
            "success": True,
            "metrics_path": str(metrics_path),
            "config": config.name,
        }
    else:
        return {"success": False, "error": "Metrics file not created"}


def _generate_client_script(
    config: ExperimentConfig,
    server_host: str,
    server_port: int,
    metrics_path: Path,
) -> str:
    """Generate a Python script to run the client."""
    return f'''
import sys
import time
import signal
import threading
from lerobot.async_inference.robot_client_improved import (
    RobotClientImproved,
    RobotClientImprovedConfig,
)
from lerobot.robots.so101_follower import SO101FollowerConfig

# Duration for this experiment
DURATION_S = {config.duration_s}

# Create a minimal robot config (won't be used in simulation mode)
robot_cfg = SO101FollowerConfig(
    port="/dev/null",
    id="simulation",
    cameras={{}},
)

client_cfg = RobotClientImprovedConfig(
    robot=robot_cfg,
    server_address="{server_host}:{server_port}",
    policy_device="cpu",
    policy_type="smolvla",
    pretrained_name_or_path="dummy/model",
    actions_per_chunk={config.actions_per_chunk},
    fps={config.fps},
    latency_estimator_type="{config.estimator}",
    cooldown_enabled={config.cooldown},
    simulation_mode=True,
    drop_obs_p={config.drop_obs_p},
    drop_obs_burst_pattern={repr(config.drop_obs_burst)},
    drop_action_p={config.drop_action_p},
    drop_action_burst_pattern={repr(config.drop_action_burst)},
    experiment_metrics_path="{metrics_path}",
    diagnostics_enabled=False,
)

print(f"Creating client with metrics path: {repr(str(metrics_path))}", file=sys.stderr)

client = RobotClientImproved(client_cfg)

# Set up a timer to stop the client after duration
def stop_after_duration():
    time.sleep(DURATION_S)
    print(f"Duration elapsed ({{DURATION_S}}s), stopping client...", file=sys.stderr)
    client.stop()

timer_thread = threading.Thread(target=stop_after_duration, daemon=True)

if client.start():
    print("Client started successfully", file=sys.stderr)
    
    # Start helper threads
    obs_thread = threading.Thread(target=client.observation_sender, daemon=True)
    action_thread = threading.Thread(target=client.action_receiver, daemon=True)
    obs_thread.start()
    action_thread.start()
    timer_thread.start()

    # Run the actual control loop (this records metrics)
    try:
        client.control_loop()
    except Exception as e:
        print(f"Control loop error: {{e}}", file=sys.stderr)
    finally:
        client.stop()
        print("Experiment completed successfully")
else:
    print("Client failed to start", file=sys.stderr)
    sys.exit(1)
'''


def run_sweep(sweep_name: str, output_dir: Path) -> None:
    """Run a predefined sweep of experiments."""
    if sweep_name not in ALL_SWEEPS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available sweeps: {list(ALL_SWEEPS.keys())}")
        return

    configs = ALL_SWEEPS[sweep_name]
    print(f"\nRunning sweep '{sweep_name}' with {len(configs)} experiments\n")

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config.name}")
        result = run_experiment(config, output_dir)
        results.append(result)

    # Summary
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*60}")
    print(f"Sweep complete: {success_count}/{len(results)} experiments succeeded")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Latency-Adaptive Async Inference Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["fixed_latency", "latency_spikes", "drop_recovery", "custom"],
        default="custom",
        help="Experiment type to run",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=list(ALL_SWEEPS.keys()),
        help="Run a predefined sweep of experiments",
    )
    parser.add_argument(
        "--latency_ms",
        type=float,
        default=100.0,
        help="Fixed latency in milliseconds (for custom experiments)",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["jk", "max_last_10"],
        default="jk",
        help="Latency estimator type",
    )
    parser.add_argument(
        "--cooldown",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable or disable cooldown mechanism",
    )
    parser.add_argument(
        "--drop_obs_burst",
        type=str,
        default=None,
        help="Observation drop burst pattern, e.g., '1s@20s'",
    )
    parser.add_argument(
        "--drop_action_burst",
        type=str,
        default=None,
        help="Action drop burst pattern, e.g., '1s@20s'",
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
        "--server_port",
        type=int,
        default=8080,
        help="Server port for experiment",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        run_sweep(args.sweep, output_dir)
    else:
        # Single experiment
        config = ExperimentConfig(
            name=f"custom_{args.experiment}",
            latency_ms=args.latency_ms,
            estimator=args.estimator,
            cooldown=(args.cooldown == "on"),
            drop_obs_burst=args.drop_obs_burst,
            drop_action_burst=args.drop_action_burst,
            duration_s=args.duration_s,
        )
        run_experiment(config, output_dir, server_port=args.server_port)


if __name__ == "__main__":
    main()
