#!/usr/bin/env python3
"""
DRTC Experiment Runner

This script runs experiments on a REAL ROBOT to validate the DRTC algorithm. It assumes the policy server is already running.

Experiment parameters are defined in YAML config files that live in
examples/experiments/configs/.

Usage:
    python examples/experiments/run_async_inference_experiment.py --config mixture_of_faults
    python examples/experiments/run_async_inference_experiment.py --config spike --output_dir results/experiments
    python examples/experiments/run_async_inference_experiment.py --config path/to/custom.yaml
"""

import argparse
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from lerobot.async_inference.robot_client_drtc import RobotClientDrtc
from lerobot.async_inference.configs_drtc import RobotClientDrtcConfig
from lerobot.async_inference.utils.simulation import (
    DisconnectConfig, DisconnectEvent,
    DropConfig, DropEvent, DuplicateConfig, DuplicateEvent, ReorderConfig, ReorderEvent,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig


DEFAULT_SERVER_ADDRESS = "192.168.4.37:8080"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_2026_01_03"
DEFAULT_MODEL_PATH = "jackvial/so101_smolvla_pickplaceorangecube_e100"
DEFAULT_TASK = "Pick up the orange cube and place it on the black X marker with the white background"

CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    estimator: str
    cooldown: bool
    # Hardware
    robot_type: str = "so101"
    gpu: str = ""
    client_host: str = ""
    server_host: str = ""
    # Policy
    policy_type: str = "smolvla"
    pretrained_name_or_path: str = DEFAULT_MODEL_PATH
    # DRTC parameters
    latency_k: float = 2.0
    epsilon: int = 2
    s_min: int = 15
    latency_alpha: float = 0.125
    latency_beta: float = 0.25
    # Timing
    duration_s: float = 60.0
    fps: int = 60
    actions_per_chunk: int = 50
    # Butterworth filter
    action_filter_mode: str = "butterworth"
    action_filter_butterworth_cutoff: float = 3.0
    action_filter_butterworth_order: int = 2
    action_filter_gain: float = 1.4
    action_filter_past_buffer_size: int = 10
    # Drop/spike/duplicate/reorder/disconnect injection
    drop_obs_config: DropConfig | None = None
    drop_action_config: DropConfig | None = None
    dup_obs_config: DuplicateConfig | None = None
    dup_action_config: DuplicateConfig | None = None
    reorder_obs_config: ReorderConfig | None = None
    reorder_action_config: ReorderConfig | None = None
    disconnect_config: DisconnectConfig | None = None
    spikes: list[dict] = field(default_factory=list)


# ---- YAML config loading ----

# Scalar fields that map 1:1 from YAML keys to ExperimentConfig constructor args.
_SCALAR_FIELDS = frozenset({
    "name", "estimator", "cooldown",
    "robot_type", "gpu", "client_host", "server_host",
    "policy_type", "pretrained_name_or_path",
    "latency_k", "epsilon", "s_min", "latency_alpha", "latency_beta",
    "duration_s", "fps", "actions_per_chunk",
    "action_filter_mode", "action_filter_butterworth_cutoff",
    "action_filter_butterworth_order", "action_filter_gain",
    "action_filter_past_buffer_size",
})


def _parse_experiment_dict(d: dict) -> ExperimentConfig:
    """Convert a raw YAML dict into an ExperimentConfig."""
    kwargs: dict = {k: v for k, v in d.items() if k in _SCALAR_FIELDS}

    # Fault-injection lists -> typed config objects
    if "drop_obs" in d:
        kwargs["drop_obs_config"] = DropConfig(drops=[DropEvent(**e) for e in d["drop_obs"]])
    if "drop_action" in d:
        kwargs["drop_action_config"] = DropConfig(drops=[DropEvent(**e) for e in d["drop_action"]])
    if "dup_obs" in d:
        kwargs["dup_obs_config"] = DuplicateConfig(duplicates=[DuplicateEvent(**e) for e in d["dup_obs"]])
    if "dup_action" in d:
        kwargs["dup_action_config"] = DuplicateConfig(duplicates=[DuplicateEvent(**e) for e in d["dup_action"]])
    if "reorder_obs" in d:
        kwargs["reorder_obs_config"] = ReorderConfig(reorders=[ReorderEvent(**e) for e in d["reorder_obs"]])
    if "reorder_action" in d:
        kwargs["reorder_action_config"] = ReorderConfig(reorders=[ReorderEvent(**e) for e in d["reorder_action"]])
    if "disconnect" in d:
        kwargs["disconnect_config"] = DisconnectConfig(disconnects=[DisconnectEvent(**e) for e in d["disconnect"]])
    if "spikes" in d:
        kwargs["spikes"] = d["spikes"]

    return ExperimentConfig(**kwargs)


def load_experiments_from_yaml(path: Path) -> list[ExperimentConfig]:
    """Load one or more ExperimentConfig from a YAML file.

    Supports two formats:

    **Single experiment** -- top-level dict IS the experiment::

        name: my_experiment
        estimator: jk
        cooldown: true

    **Multi-experiment** -- has an ``experiments`` key (and optional ``defaults``)::

        defaults:
          estimator: jk
          cooldown: true
        experiments:
          - name: run_1
          - name: run_2
            estimator: max_last_10
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    if "experiments" in raw:
        defaults = raw.get("defaults", {})
        configs = []
        for exp_dict in raw["experiments"]:
            merged = {**defaults, **exp_dict}
            configs.append(_parse_experiment_dict(merged))
        return configs

    return [_parse_experiment_dict(raw)]


def resolve_config_path(config_arg: str) -> Path:
    """Resolve a ``--config`` argument to a YAML file path.

    Accepts:
      - A relative or absolute path to a ``.yaml`` file.
      - A bare name (e.g. ``spike``), which resolves to
        ``examples/experiments/configs/<name>.yaml``.
    """
    path = Path(config_arg)
    if path.exists():
        return path

    # Try appending .yaml
    if not config_arg.endswith(".yaml"):
        with_ext = Path(config_arg + ".yaml")
        if with_ext.exists():
            return with_ext

    # Try the bundled configs directory
    in_configs = CONFIGS_DIR / config_arg
    if in_configs.exists():
        return in_configs
    if not config_arg.endswith(".yaml"):
        in_configs_yaml = CONFIGS_DIR / (config_arg + ".yaml")
        if in_configs_yaml.exists():
            return in_configs_yaml

    raise FileNotFoundError(
        f"Config not found: {config_arg} (also tried {CONFIGS_DIR / config_arg})"
    )


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
) -> RobotClientDrtcConfig:
    """Create a client config for a single experiment."""
    robot_cfg = create_robot_config()
    return RobotClientDrtcConfig(
        robot=robot_cfg,
        server_address=server_address,
        robot_type=config.robot_type,
        gpu=config.gpu,
        client_host=config.client_host,
        server_host=config.server_host,
        policy_device="cuda",
        policy_type=config.policy_type,
        pretrained_name_or_path=config.pretrained_name_or_path,
        actions_per_chunk=config.actions_per_chunk,
        fps=config.fps,
        s_min=config.s_min,
        latency_estimator_type=config.estimator,
        cooldown_enabled=config.cooldown,
        latency_k=config.latency_k,
        epsilon=config.epsilon,
        latency_alpha=config.latency_alpha,
        latency_beta=config.latency_beta,
        # Butterworth filter
        action_filter_mode=config.action_filter_mode,
        action_filter_butterworth_cutoff=config.action_filter_butterworth_cutoff,
        action_filter_butterworth_order=config.action_filter_butterworth_order,
        action_filter_gain=config.action_filter_gain,
        action_filter_past_buffer_size=config.action_filter_past_buffer_size,
        # Diagnostics and robustness
        metrics_diagnostic_enabled=True,
        metrics_diagnostic_interval_s=2.0,
        metrics_diagnostic_window_s=10.0,
        control_use_deadline_clock=True,
        obs_fallback_on_failure=True,
        obs_fallback_max_age_s=2.0,
        trajectory_viz_enabled=True,
        # Drop/spike/duplicate/reorder/disconnect injection
        drop_obs_config=config.drop_obs_config,
        drop_action_config=config.drop_action_config,
        dup_obs_config=config.dup_obs_config,
        dup_action_config=config.dup_action_config,
        reorder_obs_config=config.reorder_obs_config,
        reorder_action_config=config.reorder_action_config,
        disconnect_config=config.disconnect_config,
        spikes=config.spikes,
        metrics_path=str(metrics_path),
    )


def run_single_experiment(
    client: RobotClientDrtc,
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
    if config.dup_obs_config:
        print(f"  Dup obs: {config.dup_obs_config}")
    if config.dup_action_config:
        print(f"  Dup action: {config.dup_action_config}")
    if config.reorder_obs_config:
        print(f"  Reorder obs: {config.reorder_obs_config}")
    if config.reorder_action_config:
        print(f"  Reorder action: {config.reorder_action_config}")
    if config.disconnect_config:
        print(f"  Disconnect: {config.disconnect_config}")
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
    
    # Wait for timer to fire (in case control_loop exited early due to error)
    stop_event.wait(timeout=config.duration_s + 5.0)

    # CRITICAL: Join worker threads before returning so they don't race
    # with the next experiment's threads on the robot's USB serial bus.
    # signal_stop() has already been called (sets shutdown_event and cancels
    # the active gRPC stream), so threads should exit promptly.
    obs_thread.join(timeout=5.0)
    action_thread.join(timeout=5.0)
    if obs_thread.is_alive():
        print("  WARNING: obs_sender thread did not exit in time")
    if action_thread.is_alive():
        print("  WARNING: action_receiver thread did not exit in time")
    
    success = metrics_path.exists()
    print(f"  Experiment finished. Metrics saved: {success}")
    if success:
        exp_dir = metrics_path.parent
        print(f"  Metrics file: {metrics_path}")
        print("")
        print("  To plot:")
        print(f"    uv run python examples/experiments/plot_results.py --input {exp_dir}")
    return {"success": success, "metrics_path": str(metrics_path)}


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    task: str = DEFAULT_TASK,
    experiment_name: str | None = None,
) -> dict:
    """Run a single standalone experiment (creates and tears down client)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        # Use the provided name verbatim; append timestamp if folder already exists.
        exp_dir = output_dir / experiment_name
        if exp_dir.exists():
            exp_dir = output_dir / f"{experiment_name}_{timestamp}"
        exp_name = exp_dir.name
    else:
        exp_name = f"{config.name}_{timestamp}"
        exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / f"{exp_name}.csv"

    print(f"\nRunning experiment: {config.name}")
    print(f"  Estimator: {config.estimator}, Cooldown: {config.cooldown}")
    if config.drop_obs_config:
        print(f"  Drop obs: {config.drop_obs_config}")
    if config.drop_action_config:
        print(f"  Drop action: {config.drop_action_config}")
    if config.dup_obs_config:
        print(f"  Dup obs: {config.dup_obs_config}")
    if config.dup_action_config:
        print(f"  Dup action: {config.dup_action_config}")
    if config.reorder_obs_config:
        print(f"  Reorder obs: {config.reorder_obs_config}")
    if config.reorder_action_config:
        print(f"  Reorder action: {config.reorder_action_config}")
    if config.disconnect_config:
        print(f"  Disconnect: {config.disconnect_config}")
    if config.spikes:
        print(f"  Spikes: {config.spikes}")

    client_cfg = create_client_config(config, metrics_path, server_address)
    client = RobotClientDrtc(client_cfg)

    def stop_after_duration():
        time.sleep(config.duration_s)
        client.signal_stop()

    def signal_handler(sig, frame):
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
            # Wait for the timer thread to finish (it calls signal_stop which flushes)
            timer_thread.join(timeout=5.0)
            # Ensure metrics are flushed from the main thread in case signal_stop
            # hasn't finished or was never called (e.g. control loop exited early).
            if client._metrics.experiment is not None and client.config.metrics_path:
                try:
                    client._metrics.experiment.flush(client.config.metrics_path)
                except Exception:
                    pass
            success = metrics_path.exists()
            print(f"  Experiment finished. Metrics saved: {success}")
            if success:
                exp_dir = metrics_path.parent
                print(f"  Metrics file: {metrics_path}")
                print("")
                print("  To plot:")
                print(f"    uv run python examples/experiments/plot_results.py --input {exp_dir}")
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


def run_experiment_config(configs: list[ExperimentConfig], output_dir: Path, pause_between_s: float = 10.0, server_address: str = DEFAULT_SERVER_ADDRESS) -> None:
    """Run multiple experiments from a config, keeping the robot connected between them."""
    print(f"\nRunning {len(configs)} experiments")
    print("Robot will stay connected between experiments.")

    # Create client with first config (robot connects once)
    first_config = configs[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_exp_dir = output_dir / f"{first_config.name}_{timestamp}"
    first_exp_dir.mkdir(parents=True, exist_ok=True)
    first_metrics_path = first_exp_dir / f"{first_config.name}_{timestamp}.csv"
    client_cfg = create_client_config(first_config, first_metrics_path, server_address)
    client = RobotClientDrtc(client_cfg)
    
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
            exp_dir = output_dir / f"{config.name}_{timestamp}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = exp_dir / f"{config.name}_{timestamp}.csv"
            
            try:
                # Update client config for this experiment
                # Note: Some settings like estimator_type are set at init, so we update what we can
                client.config.latency_k = config.latency_k
                client.config.epsilon = config.epsilon
                client.config.cooldown_enabled = config.cooldown
                client.config.drop_obs_config = config.drop_obs_config
                client.config.drop_action_config = config.drop_action_config
                client.config.dup_obs_config = config.dup_obs_config
                client.config.dup_action_config = config.dup_action_config
                client.config.reorder_obs_config = config.reorder_obs_config
                client.config.reorder_action_config = config.reorder_action_config
                client.config.disconnect_config = config.disconnect_config
                client.config.spikes = config.spikes
                client.config.metrics_path = str(metrics_path)
                
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
    parser = argparse.ArgumentParser(
        description="DRTC Experiment Runner",
        epilog=(
            "Config files live in examples/experiments/configs/. "
            "Pass a bare name (e.g. spike) or a path to a .yaml file."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file, or a bare config name from examples/experiments/configs/",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help=(
            "Optional custom run name (single-experiment configs only). "
            "Overrides the name from the YAML file."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="results/experiments")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--pause_between_s", type=float, default=10.0)

    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    configs = load_experiments_from_yaml(config_path)
    print(f"Loaded {len(configs)} experiment(s) from {config_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(configs) == 1:
        experiment_name = (args.experiment_name or "").strip() or None
        run_experiment(
            configs[0],
            output_dir,
            server_address=args.server_address,
            task=DEFAULT_TASK,
            experiment_name=experiment_name,
        )
    else:
        run_experiment_config(
            configs,
            output_dir,
            pause_between_s=args.pause_between_s,
            server_address=args.server_address,
        )


if __name__ == "__main__":
    main()
