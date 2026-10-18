# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired LIBERO evaluation: native policy versus a reference checkout.

Uses LeRobot's inline engine contract with Long-WAM's sync or async scheduler.
No videos, compilation, uploads, or training. One process owns one GPU.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import Future
from functools import partial
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}


class InlineWorker:
    """Synchronous reference transport; scheduling remains in the reference code."""

    def __init__(self, handler):
        self.handler = handler

    def call(self, operation, payload):
        return self.handler(operation, payload)

    def notify(self, operation, payload):
        self.call(operation, payload)

    def submit(self, operation, payload):
        future = Future()
        try:
            future.set_result(self.call(operation, payload))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def close(self, force=False):
        self.handler = None


class ReferencePolicy:
    def __init__(self, args):
        from hydra.utils import instantiate
        from longwam.inference import TextConditioning, load_model, load_run_config
        from longwam.runtime import libero as adapter
        from longwam.runtime.execution import AsyncPolicy, Schedule, SyncPolicy, SyncSchedule
        from longwam.runtime.policy import initialize_policy
        from omegaconf import OmegaConf

        run = load_run_config(args.bundle / "config.yaml")
        run.model.model_id = str(args.backbone)
        run.model.tokenizer_model_id = str(args.tokenizer_model)
        run.data.train.processor.train_transforms = None
        run.seed = args.seed
        run.EVALUATION = OmegaConf.create(
            {
                "num_inference_steps": args.num_inference_steps,
                "binarize_gripper": True,
                "text_cfg_scale": 1.0,
                "rand_device": "cpu",
                "tiled": False,
            }
        )
        self.model = load_model(run, args.bundle / "model.pt", device="cuda", strict=True)
        processor = instantiate(run.data.train.processor).eval()
        processor.set_normalizer_from_stats(
            adapter.load_dataset_stats_from_json(args.bundle / "dataset_stats.json")
        )
        text = TextConditioning(self.model)

        def image(obs):
            return adapter._image_to_model_input(
                obs, run, processor, 448, 224, "cuda", self.model.torch_dtype
            )[0]

        def infer(obs, instruction, frames, clean):
            return adapter._predict_action_chunk(
                obs,
                instruction,
                self.model,
                processor,
                run,
                action_horizon=80,
                input_w=448,
                input_h=224,
                model_device="cuda",
                obs_frame_buffer=frames,
                clean_latents=clean,
                image_tensor=frames[-1].unsqueeze(0),
                text_conditioning=text,
            )

        settings = {
            "execution": args.execution,
            "streaming_vae": args.streaming_vae,
            "hardware": "reference",
            "optimization": {},
            "execute_steps": 10,
            "action_horizon": 80,
            "trigger_stride": 5,
        }
        handler, _ = initialize_policy(settings, components=(run, self.model, image, infer))
        self.runtime = (
            AsyncPolicy(InlineWorker(handler), Schedule(80, 10, 5))
            if args.execution == "async"
            else SyncPolicy(InlineWorker(handler), SyncSchedule(80, 10))
        )

    def reset(self, instruction):
        self.runtime.reset(instruction)

    def action(self, batch):
        images = {}
        for target, key in (("image", "image"), ("wrist_image", "image2")):
            value = batch["observation.images." + key][0]
            images[target] = value.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        return self.runtime.act({"images": images, "state": batch["observation.state"][0].cpu().numpy()})

    def close(self):
        self.runtime.close()


def batched(value):
    return {key: batched(item) for key, item in value.items()} if isinstance(value, dict) else value[None]


def json_write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def reset_fresh_environment(factory, seed, episode):
    """Avoid fixture state leaking through LIBERO's reused-object hard resets."""
    random.seed(seed)
    np.random.seed(seed)
    env = factory()
    try:
        env.init_state_id = episode
        observation, _ = env.reset(seed=seed)
    except BaseException:
        env.close()
        raise
    return env, observation


def observation_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode() + b"\0")
                visit(item[key])
        elif item is None:
            digest.update(b"None\0")
        else:
            array = np.ascontiguousarray(item)
            if array.dtype.hasobject:
                raise ValueError("Cannot fingerprint object arrays")
            digest.update(f"{array.dtype}:{array.shape}\0".encode())
            digest.update(array.tobytes())

    visit(value)
    return digest.hexdigest()


def implementation_digest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/lerobot/policies/longwam").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument(
        "--all-suites", action="store_true", help="Shard all suites across SLURM_PROCID/SLURM_NTASKS"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task-limit", type=int, default=10)
    parser.add_argument("--trace-steps", type=int, default=61)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-atol", type=float, default=1e-5)
    parser.add_argument("--execution", choices=("sync", "async"), default="sync")
    parser.add_argument("--streaming-vae", action="store_true")
    parser.add_argument(
        "--num-inference-steps", type=int, help="Explicit override; otherwise use checkpoint config"
    )
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50 or not 1 <= args.task_limit <= 10 or args.trace_steps < 1:
        parser.error("Require 1..50 episodes, 1..10 tasks, and positive trace steps")
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    workers = int(os.environ.get("SLURM_NTASKS", "1"))
    if args.all_suites:
        if args.suite:
            parser.error("Choose --suite or --all-suites, not both")
        work = [(name, task) for name in SUITES for task in range(args.task_limit)][rank::workers]
        if not work:
            parser.error("More workers than useful task shards")
        output_name = f"worker-{rank}"
    else:
        suite_name = args.suite or SUITES[rank]
        work = [(suite_name, task) for task in range(args.task_limit)]
        output_name = suite_name
    suite_name = work[0][0]
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to record the reference revision")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"{output_name}.json"
    sys.path.insert(0, str(args.reference_root / "src"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Let robosuite follow CUDA_VISIBLE_DEVICES unless explicitly overridden.
    # A Slurm local rank is not a device index inside its GPU cgroup.
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no silent CPU inference fallback")
    print(
        f"START shard={output_name} assignments={work} job={os.environ.get('SLURM_JOB_ID')} "
        f"gpu={torch.cuda.get_device_name()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"egl={os.environ.get('MUJOCO_EGL_DEVICE_ID', 'auto')}",
        flush=True,
    )
    from libero.libero import benchmark

    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

    suite = benchmark.get_benchmark_dict()[suite_name]()
    probe = LiberoEnv(
        suite,
        0,
        suite_name,
        observation_height=256,
        observation_width=256,
        num_steps_wait=30,
        obs_type="pixels_agent_pos",
    )
    try:
        observation, _ = probe.reset(seed=args.seed)
        print(f"EGL_READY camera_shape={observation['pixels']['image'].shape}", flush=True)
    finally:
        probe.close()
    environment_processor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    config = PreTrainedConfig.from_pretrained(args.native_checkpoint)
    if config.prediction_steps != 80 or config.n_action_steps != 10:
        raise ValueError("Native checkpoint does not match the evaluation horizon/replan protocol")
    config.execution = args.execution
    config.trigger_stride = 5 if args.execution == "async" else None
    config.streaming_vae = args.streaming_vae
    if args.num_inference_steps is not None:
        if args.num_inference_steps <= 0:
            parser.error("--num-inference-steps must be positive")
        config.num_inference_steps = args.num_inference_steps
    args.num_inference_steps = config.num_inference_steps
    config.device, config.inference_seed = "cuda", args.seed
    native = LongWAMPolicy.from_pretrained(args.native_checkpoint, config=config).eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=args.native_checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    print("NATIVE_READY", flush=True)
    reference = ReferencePolicy(args)
    print("REFERENCE_READY", flush=True)
    protocol = {
        "assignments": [list(item) for item in work],
        "seed": args.seed,
        "episodes": args.episodes,
        "tasks": args.task_limit,
        "execution": args.execution,
        "denoising": config.denoising,
        "streaming_vae": args.streaming_vae,
        "trigger_stride": config.trigger_stride,
        "resolution": 256,
        "settle_steps": 30,
        "max_steps": MAX_STEPS,
        "horizon": 80,
        "replan": 10,
        "denoising_steps": config.num_inference_steps,
        "trace_steps": args.trace_steps,
        "trace_atol": args.trace_atol,
        "reference_transport": "inline",
        "environment_reset": "fresh_instance_per_episode",
        "require_identical_initial_conditions": True,
        "native_code_sha256": implementation_digest(),
        "reference_commit": subprocess.check_output(
            [git, "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_config_sha256": hashlib.sha256(
            (args.native_checkpoint / "config.json").read_bytes()
        ).hexdigest(),
    }
    result = {
        "protocol": protocol,
        "complete": False,
        "episodes": [],
        "traces": [],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "transformers", "mujoco", "robosuite", "Pillow")
        },
    }
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["protocol"] != protocol:
            raise ValueError("Refusing to mix different evaluation protocols in one result")
        result = previous
    done = {(x["suite"], x["task_id"], x["episode"], x["backend"]) for x in result["episodes"]}

    def features(observation, task):
        batch = environment_processor(preprocess_observation(batched(observation)))
        batch["task"] = [task]
        return batch

    def action(backend, batch):
        if backend == "native":
            # Keep the reference inputs independent of processor implementation.
            native_batch = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            return post(native.select_action(pre(native_batch)))[0].cpu().numpy()
        return reference.action(batch)

    try:
        for suite_name, task_id in work:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            task = suite.get_task(task_id).language
            make_env = partial(
                LiberoEnv,
                suite,
                task_id,
                suite_name,
                episode_length=MAX_STEPS[suite_name],
                observation_height=256,
                observation_width=256,
                num_steps_wait=30,
                obs_type="pixels_agent_pos",
                hard_reset=True,
            )
            env = None
            try:
                # Both schedulers receive identical observations at every tick.
                trace_key = [suite_name, task_id]
                if trace_key not in result.get("trace_tasks", []):
                    result["traces"] = [
                        row for row in result["traces"] if [row["suite"], row["task_id"]] != trace_key
                    ]
                    env, observation = reset_fresh_environment(make_env, args.seed, 0)
                    native.reset()
                    reference.reset(task)
                    for step in range(args.trace_steps):
                        batch = features(observation, task)
                        left, right = action("native", batch), action("reference", batch)
                        error = float(np.max(np.abs(left - right)))
                        result["traces"].append(
                            {
                                "suite": suite_name,
                                "task_id": task_id,
                                "step": step,
                                "max_abs_action_error": error,
                            }
                        )
                        json_write(output, result)
                        if not np.allclose(left, right, atol=args.trace_atol, rtol=0):
                            raise AssertionError(f"Action mismatch at step {step}: max_abs_error={error}")
                        observation, _, terminated, _, _ = env.step(left)
                        if terminated:
                            break
                    result.setdefault("trace_tasks", []).append(trace_key)
                    json_write(output, result)
                    print(
                        f"TRACE_PASS suite={suite_name} task={task_id} steps={step + 1} "
                        f"max_error={max(t['max_abs_action_error'] for t in result['traces'])}",
                        flush=True,
                    )
                for episode in range(args.episodes):
                    for backend in ("native", "reference"):
                        if (suite_name, task_id, episode, backend) in done:
                            continue
                        if env is not None:
                            env.close()
                            env = None
                        env, observation = reset_fresh_environment(make_env, args.seed, episode)
                        initial = {
                            "observation": observation_digest(observation),
                            "simulator_state": observation_digest(env._env.sim.get_state().flatten()),
                        }
                        peers = [
                            row
                            for row in result["episodes"]
                            if (row["suite"], row["task_id"], row["episode"])
                            == (suite_name, task_id, episode)
                        ]
                        if peers and any(row["initial_conditions"] != initial for row in peers):
                            result["initial_condition_error"] = {
                                "suite": suite_name,
                                "task_id": task_id,
                                "episode": episode,
                                "observed": initial,
                                "expected": peers[0]["initial_conditions"],
                            }
                            json_write(output, result)
                            raise AssertionError("Paired episodes have different initial conditions")
                        native.reset() if backend == "native" else reference.reset(task)
                        start = time.monotonic()
                        success = False
                        for step in range(MAX_STEPS[suite_name]):
                            selected = action(backend, features(observation, task))
                            observation, _, terminated, truncated, info = env.step(selected)
                            if (step + 1) % 50 == 0:
                                print(
                                    f"HEARTBEAT {backend} {suite_name} task={task_id} ep={episode} step={step + 1}",
                                    flush=True,
                                )
                            if terminated or truncated:
                                success = bool(info["is_success"])
                                break
                        row = {
                            "suite": suite_name,
                            "backend": backend,
                            "task_id": task_id,
                            "task": task,
                            "episode": episode,
                            "success": success,
                            "steps": step + 1,
                            "seconds": time.monotonic() - start,
                            "initial_conditions": initial,
                        }
                        result["episodes"].append(row)
                        json_write(output, result)
                        print("EPISODE " + json.dumps(row), flush=True)
            finally:
                if env is not None:
                    env.close()
        result["complete"] = len(result["episodes"]) == len(work) * args.episodes * 2
        result["trace_passed"] = len(result.get("trace_tasks", [])) == len(work)
        result["summary"] = {
            backend: {
                "successes": sum(x["success"] for x in result["episodes"] if x["backend"] == backend),
                "episodes": sum(x["backend"] == backend for x in result["episodes"]),
            }
            for backend in ("native", "reference")
        }
        json_write(output, result)
        print("SUMMARY " + json.dumps(result["summary"]), flush=True)
    finally:
        native.close()
        reference.close()


if __name__ == "__main__":
    main()
