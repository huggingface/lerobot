#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc.
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
Utility script to dry-run or deploy an ACT policy on the Piper robot.

Example usage – dry run against a dataset (no hardware required):

    python -m lerobot_robot_piper.run_act_piper \
        --policy-path outputs/act_piper_demo_merged/last \
        --dataset-repo-id piper-demo-251110-merged \
        --dataset-root /home/hls/datasets/piper-demo-251110-merged \
        --dry-run \
        --max-dry-run-samples 3

Example usage – stream actions to the real robot:

    python -m lerobot_robot_piper.run_act_piper \
        --policy-path outputs/act_piper_demo_merged/last \
        --dataset-repo-id piper-demo-251110-merged \
        --dataset-root /home/hls/codes/lerobot/datasets/piper-demo-251110-merged/checkpoints/010000/pretrained_model \
        --can-interface can0 \
        --bitrate 1000000 \
        --include-gripper \
        --use-degrees \
        --loop-frequency 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline
from lerobot.processor.pipeline import TransitionKey
from lerobot.utils.import_utils import register_third_party_devices

from lerobot_robot_piper.lerobot_robot_piper.config_piper import PiperConfig
from lerobot_robot_piper.lerobot_robot_piper.piper import Piper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACT policy on Piper robot, with optional dry run.")
    parser.add_argument("--policy-path", type=str, required=True, help="Directory containing the trained policy.")
    parser.add_argument(
        "--policy-device",
        type=str,
        default="cuda",
        help="Device to place the policy on (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--postprocessor-device",
        type=str,
        default="cpu",
        help="Device for postprocessed action chunks (default: cpu).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repo id used during training (e.g. piper-demo-251110-merged).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing the dataset (meta/, data/, ...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a forward pass on dataset samples without touching the robot.",
    )
    parser.add_argument(
        "--max-dry-run-samples",
        type=int,
        default=5,
        help="Maximum number of dataset samples to evaluate during a dry run.",
    )
    parser.add_argument(
        "--can-interface",
        type=str,
        default=None,
        help="CAN interface used to communicate with the Piper base (e.g. can0). Required for live runs.",
    )
    parser.add_argument("--bitrate", type=int, default=1_000_000, help="CAN bitrate for the Piper SDK.")
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Whether the gripper DoF is present.",
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Do not normalise joint angles; operate directly in degrees.",
    )
    parser.add_argument(
        "--loop-frequency",
        type=float,
        default=10.0,
        help="Control frequency for live runs (Hz). Determines how often actions are sent.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Optional JSON describing cameras for the Piper config.",
    )
    parser.add_argument(
        "--rename-map",
        type=str,
        default=None,
        help="Optional JSON mapping to rename observation keys before preprocessing.",
    )
    parser.add_argument(
        "--map-state-to-environment",
        action="store_true",
        help="Treat 'observation.state' features as environment state inputs for ACT.",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=None,
        help="Maximum duration in seconds for the live run. Omit to run indefinitely.",
    )
    return parser.parse_args()


def load_policy_and_processors(
    policy_path: Path,
    dataset: LeRobotDataset,
    device: str,
    rename_map: dict[str, str] | None = None,
) -> tuple[Any, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = device
    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=rename_map or {})
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def run_dry_run(
    dataset: LeRobotDataset,
    policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    max_samples: int,
    add_env_copy: bool,
    postprocessor_device: str | None,
) -> None:
    limit = min(max_samples, len(dataset))
    print(f"[DryRun] Running on {limit} sample(s) from '{dataset.repo_id}'.")
    with torch.no_grad():
        for idx in range(limit):
            sample = prepare_sample(dict(dataset[idx]), add_env_copy=add_env_copy)
            processed = preprocessor(sample)
            chunk = policy.predict_action_chunk(processed)
            unnorm = postprocessor(chunk)
            chunk_tensor = extract_action_tensor(unnorm)
            if postprocessor_device:
                chunk_tensor = chunk_tensor.to(postprocessor_device)
            print(
                f"[DryRun] Sample {idx}: "
                f"action chunk shape = {tuple(chunk_tensor.shape)}, "
                f"device = {chunk_tensor.device}"
            )
    print("[DryRun] Completed without errors.")


def parse_cameras(cameras_json: str | None) -> dict[str, CameraConfig]:
    if not cameras_json:
        return {}
    try:
        raw_cfgs = json.loads(cameras_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid camera JSON: {exc}") from exc

    cameras: dict[str, CameraConfig] = {}
    for name, cfg in raw_cfgs.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Camera '{name}' must be defined by a JSON object.")
        cam_type = cfg.get("type", "opencv")
        cfg = dict(cfg)
        cfg.pop("type", None)
        if cam_type == "opencv":
            cameras[name] = OpenCVCameraConfig(**cfg)
        else:
            raise ValueError(f"Unsupported camera type '{cam_type}' for camera '{name}'.")
    return cameras


def build_state_vector(raw_obs: dict[str, Any], state_names: list[str]) -> torch.Tensor:
    return torch.tensor([float(raw_obs.get(name, 0.0)) for name in state_names], dtype=torch.float32)


def build_action_dict(action_vec: list[float], action_names: list[str]) -> dict[str, float]:
    return {name: float(val) for name, val in zip(action_names, action_vec, strict=True)}


def prepare_sample(
    sample: dict[str, Any],
    add_env_copy: bool,
    images: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    prepared = dict(sample)
    if images:
        prepared.update(images)
    state_tensor = prepared.get("observation.state")
    if add_env_copy and state_tensor is not None:
        prepared["observation.environment_state"] = state_tensor.clone()
    return prepared


def convert_image_frame(frame: Any) -> torch.Tensor:
    if frame is None:
        raise ValueError("Received empty frame from camera.")
    if isinstance(frame, torch.Tensor):
        tensor = frame
    else:
        tensor = torch.from_numpy(frame)
    if tensor.dim() == 3 and tensor.shape[-1] in (3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def extract_action_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        tensor = output[TransitionKey.ACTION]
    elif isinstance(output, torch.Tensor):
        tensor = output
    else:
        raise TypeError(
            f"Unexpected output type from postprocessor: {type(output)}. "
            "Expected torch.Tensor or dict keyed by TransitionKey.ACTION."
        )

    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    return tensor


def run_live(
    policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    add_env_copy: bool,
    postprocessor_device: str | None,
) -> None:
    if not args.can_interface:
        raise ValueError("--can-interface is required for live robot execution.")

    register_third_party_devices()
    cameras_cfg = parse_cameras(args.cameras)
    robot_cfg = PiperConfig(
        can_interface=args.can_interface,
        bitrate=args.bitrate,
        include_gripper=args.include_gripper,
        use_degrees=args.use_degrees,
        cameras=cameras_cfg,
    )
    robot = Piper(robot_cfg)
    print("[Live] Connecting to Piper robot...")
    robot.connect()
    print("[Live] Connected. Starting control loop.")

    state_names = dataset.meta.features["observation.state"]["names"]
    action_names = dataset.meta.features["action"]["names"]
    action_queue: deque[list[float]] = deque()
    loop_dt = 1.0 / args.loop_frequency if args.loop_frequency > 0 else 0.0
    elapsed = 0.0
    start_time = time.time()

    try:
        with torch.no_grad():
            while args.max_runtime is None or elapsed < args.max_runtime:
                obs = robot.get_observation()
                state_tensor = build_state_vector(obs, state_names)
                sample_inputs: dict[str, Any] = {"observation.state": state_tensor}
                image_tensors: dict[str, torch.Tensor] = {}
                for cam_key in robot.cameras.keys():
                    frame = obs.get(cam_key)
                    if frame is not None:
                        try:
                            image_tensors[f"observation.images.{cam_key}"] = convert_image_frame(frame)
                        except Exception as exc:
                            print(f"[Live] Warning: failed to convert frame from camera '{cam_key}': {exc}")
                sample = prepare_sample(sample_inputs, add_env_copy=add_env_copy, images=image_tensors or None)
                processed = preprocessor(sample)

                if not action_queue:
                    chunk = policy.predict_action_chunk(processed)
                    chunk = postprocessor(chunk)
                    chunk_tensor = extract_action_tensor(chunk)
                    if postprocessor_device:
                        chunk_tensor = chunk_tensor.to(postprocessor_device)
                    chunk_tensor = chunk_tensor.cpu()
                    action_queue.extend(chunk_tensor.tolist())

                if action_queue:
                    action_vec = action_queue.popleft()
                    action_dict = build_action_dict(action_vec, action_names)
                else:
                    action_dict = {name: 0.0 for name in action_names}
                robot.send_action(action_dict)

                if loop_dt > 0:
                    time.sleep(loop_dt)
                elapsed = time.time() - start_time

    except KeyboardInterrupt:
        print("[Live] Stopping control loop (CTRL+C received).")
    finally:
        robot.disconnect()
        print("[Live] Robot disconnected.")


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(args.dataset_repo_id, root=Path(args.dataset_root))
    if args.map_state_to_environment:
        if "observation.state" in dataset.meta.features and "observation.environment_state" not in dataset.meta.features:
            dataset.meta.features["observation.environment_state"] = dataset.meta.features["observation.state"].copy()
        if "observation.state" in dataset.meta.stats and "observation.environment_state" not in dataset.meta.stats:
            dataset.meta.stats["observation.environment_state"] = dataset.meta.stats["observation.state"].copy()

    rename_map = json.loads(args.rename_map) if args.rename_map else None

    policy_path = Path(args.policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy directory '{policy_path}' does not exist.")

    policy, preproc, postproc = load_policy_and_processors(
        policy_path=policy_path,
        dataset=dataset,
        device=args.policy_device,
        rename_map=rename_map,
    )
    if args.postprocessor_device:
        for step in postproc.steps:
            if isinstance(step, DeviceProcessorStep):
                step.device = args.postprocessor_device
                step.__post_init__()

    if args.dry_run:
        run_dry_run(
            dataset=dataset,
            policy=policy,
            preprocessor=preproc,
            postprocessor=postproc,
            max_samples=args.max_dry_run_samples,
            add_env_copy=args.map_state_to_environment,
            postprocessor_device=args.postprocessor_device,
        )

    if not args.dry_run:
        run_live(
            policy,
            preproc,
            postproc,
            dataset,
            args,
            add_env_copy=args.map_state_to_environment,
            postprocessor_device=args.postprocessor_device,
        )


if __name__ == "__main__":
    main()

