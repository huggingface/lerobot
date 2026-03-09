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
import logging
import signal
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from lerobot.async_inference.configs_drtc import RobotClientDrtcConfig
from lerobot.async_inference.robot_client_drtc import RobotClientDrtc
from lerobot.async_inference.utils.simulation import (
    DisconnectConfig,
    DisconnectEvent,
    DropConfig,
    DropEvent,
    DuplicateConfig,
    DuplicateEvent,
    ReorderConfig,
    ReorderEvent,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig, SO101FollowerConfig

logger = logging.getLogger(__name__)


DEFAULT_SERVER_ADDRESS = "192.168.4.37:8080"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_2026_01_03"
DEFAULT_CAMERA1_PATH = "/dev/video2"
DEFAULT_CAMERA2_PATH = "/dev/video6"
DEFAULT_CAMERA_WIDTH = 800
DEFAULT_CAMERA_HEIGHT = 600
DEFAULT_CAMERA_FPS = 30
DEFAULT_CAMERA_FOURCC = "MJPG"
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
    robot_port: str = DEFAULT_ROBOT_PORT
    robot_id: str = DEFAULT_ROBOT_ID
    camera1_path: str = DEFAULT_CAMERA1_PATH
    camera2_path: str = DEFAULT_CAMERA2_PATH
    camera_width: int = DEFAULT_CAMERA_WIDTH
    camera_height: int = DEFAULT_CAMERA_HEIGHT
    camera_fps: int = DEFAULT_CAMERA_FPS
    camera_fourcc: str | None = DEFAULT_CAMERA_FOURCC
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
    # Flow matching / RTC
    num_flow_matching_steps: int | None = 8
    rtc_enabled: bool = True
    rtc_max_guidance_weight: float | None = None
    rtc_prefix_attention_schedule: str = "linear"
    rtc_sigma_d: float = 0.2
    rtc_full_trajectory_alignment: bool = False
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
    # Diagnostics
    full_diagnostics: bool = False
    trajectory_viz_enabled: bool = False


# ---- YAML config loading ----

# Scalar fields that map 1:1 from YAML keys to ExperimentConfig constructor args.
_SCALAR_FIELDS = frozenset(
    {
        "name",
        "estimator",
        "cooldown",
        "robot_type",
        "gpu",
        "client_host",
        "server_host",
        "robot_port",
        "robot_id",
        "camera1_path",
        "camera2_path",
        "camera_width",
        "camera_height",
        "camera_fps",
        "camera_fourcc",
        "policy_type",
        "pretrained_name_or_path",
        "latency_k",
        "epsilon",
        "s_min",
        "latency_alpha",
        "latency_beta",
        "duration_s",
        "fps",
        "actions_per_chunk",
        "num_flow_matching_steps",
        "rtc_enabled",
        "rtc_max_guidance_weight",
        "rtc_prefix_attention_schedule",
        "rtc_sigma_d",
        "rtc_full_trajectory_alignment",
        "action_filter_mode",
        "action_filter_butterworth_cutoff",
        "action_filter_butterworth_order",
        "action_filter_gain",
        "action_filter_past_buffer_size",
        "full_diagnostics",
        "trajectory_viz_enabled",
    }
)


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
        kwargs["dup_action_config"] = DuplicateConfig(
            duplicates=[DuplicateEvent(**e) for e in d["dup_action"]]
        )
    if "reorder_obs" in d:
        kwargs["reorder_obs_config"] = ReorderConfig(reorders=[ReorderEvent(**e) for e in d["reorder_obs"]])
    if "reorder_action" in d:
        kwargs["reorder_action_config"] = ReorderConfig(
            reorders=[ReorderEvent(**e) for e in d["reorder_action"]]
        )
    if "disconnect" in d:
        kwargs["disconnect_config"] = DisconnectConfig(
            disconnects=[DisconnectEvent(**e) for e in d["disconnect"]]
        )
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

    raise FileNotFoundError(f"Config not found: {config_arg} (also tried {CONFIGS_DIR / config_arg})")


def create_robot_config(config: ExperimentConfig) -> SO100FollowerConfig | SO101FollowerConfig:
    camera_fourcc = (
        config.camera_fourcc.strip() if isinstance(config.camera_fourcc, str) else config.camera_fourcc
    )
    if camera_fourcc == "":
        camera_fourcc = None

    camera_cfg = {
        "camera2": OpenCVCameraConfig(
            index_or_path=config.camera2_path,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
            fourcc=camera_fourcc,
        ),
        "camera1": OpenCVCameraConfig(
            index_or_path=config.camera1_path,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
            fourcc=camera_fourcc,
        ),
    }
    robot_type_normalized = config.robot_type.strip().lower()
    if robot_type_normalized in {"so101", "so101_follower"}:
        return SO101FollowerConfig(port=config.robot_port, id=config.robot_id, cameras=camera_cfg)
    if robot_type_normalized in {"so100", "so100_follower"}:
        return SO100FollowerConfig(port=config.robot_port, id=config.robot_id, cameras=camera_cfg)

    raise ValueError(
        f"Unsupported robot_type '{config.robot_type}'. "
        "Supported values: so101, so101_follower, so100, so100_follower."
    )


def create_client_config(
    config: ExperimentConfig,
    metrics_path: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    trajectory_viz_ws_url: str | None = None,
) -> RobotClientDrtcConfig:
    """Create a client config for a single experiment."""
    robot_cfg = create_robot_config(config)
    client_kwargs = {
        "robot": robot_cfg,
        "server_address": server_address,
        "robot_type": config.robot_type,
        "gpu": config.gpu,
        "client_host": config.client_host,
        "server_host": config.server_host,
        "policy_device": "cuda",
        "policy_type": config.policy_type,
        "pretrained_name_or_path": config.pretrained_name_or_path,
        "actions_per_chunk": config.actions_per_chunk,
        "fps": config.fps,
        "s_min": config.s_min,
        "latency_estimator_type": config.estimator,
        "cooldown_enabled": config.cooldown,
        "latency_k": config.latency_k,
        "epsilon": config.epsilon,
        "latency_alpha": config.latency_alpha,
        "latency_beta": config.latency_beta,
        # Flow matching / RTC
        "num_flow_matching_steps": config.num_flow_matching_steps,
        "rtc_enabled": config.rtc_enabled,
        "rtc_max_guidance_weight": config.rtc_max_guidance_weight,
        "rtc_prefix_attention_schedule": config.rtc_prefix_attention_schedule,
        "rtc_sigma_d": config.rtc_sigma_d,
        "rtc_full_trajectory_alignment": config.rtc_full_trajectory_alignment,
        # Butterworth filter
        "action_filter_mode": config.action_filter_mode,
        "action_filter_butterworth_cutoff": config.action_filter_butterworth_cutoff,
        "action_filter_butterworth_order": config.action_filter_butterworth_order,
        "action_filter_gain": config.action_filter_gain,
        "action_filter_past_buffer_size": config.action_filter_past_buffer_size,
        # Diagnostics and robustness
        "metrics_diagnostic_enabled": True,
        "metrics_diagnostic_interval_s": 2.0,
        "metrics_diagnostic_window_s": 10.0,
        "metrics_diagnostic_verbose": config.full_diagnostics,
        "control_use_deadline_clock": True,
        "obs_fallback_on_failure": True,
        "obs_fallback_max_age_s": 2.0,
        "trajectory_viz_enabled": config.trajectory_viz_enabled,
        # Drop/spike/duplicate/reorder/disconnect injection
        "drop_obs_config": config.drop_obs_config,
        "drop_action_config": config.drop_action_config,
        "dup_obs_config": config.dup_obs_config,
        "dup_action_config": config.dup_action_config,
        "reorder_obs_config": config.reorder_obs_config,
        "reorder_action_config": config.reorder_action_config,
        "disconnect_config": config.disconnect_config,
        "spikes": config.spikes,
        "metrics_path": str(metrics_path),
    }
    if trajectory_viz_ws_url:
        client_kwargs["trajectory_viz_ws_url"] = trajectory_viz_ws_url
    return RobotClientDrtcConfig(**client_kwargs)


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    trajectory_viz_ws_url: str | None = None,
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

    logger.info(f"Running experiment: {config.name}")
    logger.info(
        f"  Estimator: {config.estimator}, Cooldown: {config.cooldown}, Full diagnostics: {config.full_diagnostics}"
    )
    if config.drop_obs_config:
        logger.info(f"  Drop obs: {config.drop_obs_config}")
    if config.drop_action_config:
        logger.info(f"  Drop action: {config.drop_action_config}")
    if config.dup_obs_config:
        logger.info(f"  Dup obs: {config.dup_obs_config}")
    if config.dup_action_config:
        logger.info(f"  Dup action: {config.dup_action_config}")
    if config.reorder_obs_config:
        logger.info(f"  Reorder obs: {config.reorder_obs_config}")
    if config.reorder_action_config:
        logger.info(f"  Reorder action: {config.reorder_action_config}")
    if config.disconnect_config:
        logger.info(f"  Disconnect: {config.disconnect_config}")
    if config.spikes:
        logger.info(f"  Spikes: {config.spikes}")

    client_cfg = create_client_config(
        config,
        metrics_path,
        server_address=server_address,
        trajectory_viz_ws_url=trajectory_viz_ws_url,
    )
    client = RobotClientDrtc(client_cfg)

    def stop_after_duration():
        time.sleep(config.duration_s)
        client.signal_stop()

    def signal_handler(sig, frame):
        client.signal_stop()

    timer_thread = threading.Thread(target=stop_after_duration, daemon=True)
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info("Starting client...")
        if client.start():
            logger.info("Client started successfully")
            obs_thread = threading.Thread(target=client.observation_sender, daemon=True)
            action_thread = threading.Thread(target=client.action_receiver, daemon=True)
            obs_thread.start()
            action_thread.start()
            timer_thread.start()
            logger.info(f"Running for {config.duration_s}s...")
            try:
                client.control_loop(task=task)
            except Exception as e:
                logger.exception(f"Control loop error: {e}")
            # Wait for the timer thread to finish (it calls signal_stop which flushes)
            timer_thread.join(timeout=5.0)
            # Ensure metrics are flushed from the main thread in case signal_stop
            # hasn't finished or was never called (e.g. control loop exited early).
            if client._metrics.experiment is not None and client.config.metrics_path:
                with suppress(Exception):
                    client._metrics.experiment.flush(client.config.metrics_path)
            success = metrics_path.exists()
            logger.info(f"Experiment finished. Metrics saved: {success}")
            if success:
                exp_dir = metrics_path.parent
                logger.info(f"Metrics file: {metrics_path}")
                logger.info("To plot:")
                logger.info(f"  uv run python examples/experiments/plot_results.py --input {exp_dir}")
            return {"success": success, "metrics_path": str(metrics_path)}
        else:
            logger.error("Client failed to start!")
            return {"success": False, "error": "Client failed to start"}
    except Exception as e:
        logger.exception(f"Exception during experiment: {e}")
        return {"success": False, "error": str(e)}
    finally:
        signal.signal(signal.SIGINT, original_handler)
        # Only disconnect at the very end for standalone experiments
        with suppress(Exception):
            client.stop()


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
    parser.add_argument(
        "--trajectory_viz_ws_url",
        type=str,
        default=None,
        help=(
            "Optional WebSocket URL for trajectory visualization. "
            "Used when trajectory visualization is enabled in the experiment config."
        ),
    )
    parser.add_argument("--pause_between_s", type=float, default=10.0)

    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    configs = load_experiments_from_yaml(config_path)
    logger.info(f"Loaded {len(configs)} experiment(s) from {config_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, config in enumerate(configs):
        if len(configs) > 1:
            logger.info(f"{'=' * 50}")
            logger.info(f"[{i + 1}/{len(configs)}] {config.name}")
            logger.info(f"{'=' * 50}")

        experiment_name = (args.experiment_name or "").strip() or None
        result = run_experiment(
            config,
            output_dir,
            server_address=args.server_address,
            trajectory_viz_ws_url=args.trajectory_viz_ws_url,
            task=DEFAULT_TASK,
            experiment_name=experiment_name if len(configs) == 1 else None,
        )
        results.append(result)

        if i < len(configs) - 1:
            logger.info(f"Pausing {args.pause_between_s}s before next experiment...")
            time.sleep(args.pause_between_s)

    if len(configs) > 1:
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"All experiments complete: {success_count}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
