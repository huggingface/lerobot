#!/usr/bin/env python
"""
Benchmark script to measure end-to-end inference latency for LeRobot policies.

This script measures:
1. Preprocessing time
2. Policy inference time (select_action / predict_action_chunk)
3. Postprocessing time
4. Total end-to-end latency

Usage:
    python benchmark_inference_latency.py --policy.path=YieumYoon/groot-bimanual-so100-cbasket-diffusion-003

For your specific setup:
    python benchmark_inference_latency.py \
        --policy.path=YieumYoon/groot-bimanual-so100-cbasket-diffusion-003 \
        --num_iterations=100 \
        --num_warmup=10
"""

import argparse
import gc
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.utils.utils import TimerManager


@dataclass
class BenchmarkConfig:
    """Configuration for latency benchmark."""

    # Policy path (HuggingFace repo or local path)
    policy_path: str = field(default="", metadata={
                             "help": "Path to pretrained policy"})

    # Benchmark settings
    num_warmup: int = field(default=10, metadata={
                            "help": "Number of warmup iterations"})
    num_iterations: int = field(default=100, metadata={
                                "help": "Number of benchmark iterations"})

    # Device settings
    device: str = field(default="cuda", metadata={
                        "help": "Device to run on (cuda/cpu)"})

    # Image settings (matching your camera setup)
    image_height: int = field(default=480, metadata={"help": "Image height"})
    image_width: int = field(default=640, metadata={"help": "Image width"})
    num_cameras: int = field(default=3, metadata={"help": "Number of cameras"})

    # State dimensions (for bimanual SO100: 2 arms * 6 joints = 12)
    state_dim: int = field(default=12, metadata={"help": "State dimension"})

    # Task description
    task: str = field(default="Grab the red cube and put it in a red basket", metadata={
                      "help": "Task description"})


def create_dummy_observation(config: BenchmarkConfig, policy_config) -> dict:
    """Create a dummy observation matching your bimanual setup with 3 cameras."""
    device = config.device

    obs = {}

    # Add images for each camera (matching your setup: left_gripper, top, right_gripper)
    camera_names = ["left_gripper", "top", "right_gripper"]
    for cam_name in camera_names[:config.num_cameras]:
        # Images: (batch, channels, height, width), float32 in [0, 1]
        obs[f"observation.images.{cam_name}"] = torch.rand(
            1, 3, config.image_height, config.image_width,
            dtype=torch.float32, device=device
        )

    # State: (batch, state_dim) - bimanual SO100 has 12 DOF (6 per arm)
    obs["observation.state"] = torch.rand(
        1, config.state_dim, dtype=torch.float32, device=device
    )

    # Add task/language if needed
    obs["task"] = config.task

    return obs


def benchmark_inference(
    policy,
    preprocessor,
    postprocessor,
    observation: dict,
    config: BenchmarkConfig,
) -> dict:
    """Run the full benchmark and return timing statistics."""

    device = torch.device(config.device)
    is_cuda = device.type == "cuda"

    # Timers
    preprocess_timer = TimerManager("Preprocessing", log=False)
    inference_timer = TimerManager("Inference", log=False)
    postprocess_timer = TimerManager("Postprocessing", log=False)
    total_timer = TimerManager("Total E2E", log=False)

    policy.eval()

    # ========== WARMUP ==========
    print(f"\n🔥 Warming up ({config.num_warmup} iterations)...")
    for i in range(config.num_warmup):
        policy.reset()
        if preprocessor:
            preprocessor.reset()
        if postprocessor:
            postprocessor.reset()

        with torch.no_grad():
            if preprocessor:
                processed_obs = preprocessor(observation.copy())
            else:
                processed_obs = observation

            action = policy.select_action(processed_obs)

            if postprocessor:
                action = postprocessor(action)

        if is_cuda:
            torch.cuda.synchronize()

    # Clear memory after warmup
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    # ========== BENCHMARK ==========
    print(f"\n📊 Running benchmark ({config.num_iterations} iterations)...")

    for i in range(config.num_iterations):
        # Reset policy state for fresh inference each time
        policy.reset()
        if preprocessor:
            preprocessor.reset()
        if postprocessor:
            postprocessor.reset()

        if is_cuda:
            torch.cuda.synchronize()

        with total_timer:
            # 1. Preprocess
            with preprocess_timer:
                if preprocessor:
                    processed_obs = preprocessor(observation.copy())
                else:
                    processed_obs = observation
                if is_cuda:
                    torch.cuda.synchronize()

            # 2. Inference
            with inference_timer:
                with torch.no_grad():
                    action = policy.select_action(processed_obs)
                if is_cuda:
                    torch.cuda.synchronize()

            # 3. Postprocess
            with postprocess_timer:
                if postprocessor:
                    action = postprocessor(action)
                if is_cuda:
                    torch.cuda.synchronize()

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{config.num_iterations} - "
                  f"Last: {total_timer.last*1000:.1f}ms")

    return {
        "preprocess": preprocess_timer,
        "inference": inference_timer,
        "postprocess": postprocess_timer,
        "total": total_timer,
    }


def print_results(results: dict, config: BenchmarkConfig):
    """Print formatted benchmark results."""

    print("\n" + "=" * 60)
    print("📈 INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Policy: {config.policy_path}")
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Image size: {config.image_width}x{config.image_height}")
    print(f"Cameras: {config.num_cameras}")
    print("=" * 60)

    for name, timer in results.items():
        print(f"\n📌 {name.upper()}")
        print(f"   Mean:  {timer.avg * 1000:>8.2f} ms")
        print(f"   Min:   {min(timer.history) * 1000:>8.2f} ms")
        print(f"   Max:   {max(timer.history) * 1000:>8.2f} ms")
        print(f"   P50:   {timer.percentile(50) * 1000:>8.2f} ms")
        print(f"   P90:   {timer.percentile(90) * 1000:>8.2f} ms")
        print(f"   P99:   {timer.percentile(99) * 1000:>8.2f} ms")
        print(f"   FPS:   {timer.fps_avg:>8.1f}")

    # Summary
    total = results["total"]
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(
        f"Average end-to-end latency: {total.avg * 1000:.2f} ms ({total.avg:.3f} s)")
    print(f"P90 latency:                {total.percentile(90) * 1000:.2f} ms")
    print(f"P99 latency:                {total.percentile(99) * 1000:.2f} ms")
    print(f"Average FPS:                {total.fps_avg:.1f}")

    # Check requirement
    print("\n" + "-" * 60)
    if total.avg < 1.0:
        print(
            f"✅ PASS: Mean latency ({total.avg*1000:.0f}ms) < 1000ms requirement")
    else:
        print(
            f"❌ FAIL: Mean latency ({total.avg*1000:.0f}ms) >= 1000ms requirement")

    if total.percentile(99) < 1.0:
        print(
            f"✅ PASS: P99 latency ({total.percentile(99)*1000:.0f}ms) < 1000ms requirement")
    else:
        print(
            f"⚠️  WARN: P99 latency ({total.percentile(99)*1000:.0f}ms) >= 1000ms requirement")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency for LeRobot policies")
    parser.add_argument("--policy.path", dest="policy_path", type=str, required=True,
                        help="Path to pretrained policy (HuggingFace repo or local)")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Image height")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Image width")
    parser.add_argument("--num_cameras", type=int, default=3,
                        help="Number of cameras")
    parser.add_argument("--state_dim", type=int, default=12,
                        help="State dimension (bimanual SO100 = 12)")
    parser.add_argument("--task", type=str,
                        default="Grab the red cube and put it in a red basket",
                        help="Task description")

    args = parser.parse_args()

    config = BenchmarkConfig(
        policy_path=args.policy_path,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        num_cameras=args.num_cameras,
        state_dim=args.state_dim,
        task=args.task,
    )

    print("\n" + "=" * 60)
    print("🚀 LeRobot Inference Latency Benchmark")
    print("=" * 60)

    # Check CUDA availability
    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        config.device = "cpu"

    if config.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load policy
    print(f"\n📦 Loading policy from: {config.policy_path}")

    # Load Groot policy directly using from_pretrained
    policy = GrootPolicy.from_pretrained(
        pretrained_name_or_path=config.policy_path,
        strict=False,  # allow missing keys if any extra params were saved
    )
    policy.to(config.device)
    policy.config.device = config.device
    policy.eval()

    print(f"   Policy type: {policy.name}")
    print(f"   Device: {config.device}")

    # Create preprocessor/postprocessor
    print("\n🔧 Creating preprocessor and postprocessor...")
    try:
        preprocessor, postprocessor = make_groot_pre_post_processors(
            config=policy.config,
            # Pass None for benchmarking (no normalization)
            dataset_stats=None,
        )
        print("   ✓ Preprocessor and postprocessor created")
    except Exception as e:
        print(f"   ⚠️  Could not create processors: {e}")
        print("   Running without pre/post processing")
        preprocessor, postprocessor = None, None

    # Create dummy observation
    print("\n🎯 Creating dummy observation...")
    observation = create_dummy_observation(config, policy.config)
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {value}")

    # Run benchmark
    results = benchmark_inference(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        observation=observation,
        config=config,
    )

    # Print results
    print_results(results, config)

    # Memory info
    if config.device == "cuda":
        print(
            f"\n💾 GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")


if __name__ == "__main__":
    main()
