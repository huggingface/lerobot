#!/usr/bin/env python

"""
Script to compare performance with and without RTC enabled.

This script helps identify whether RTC is actually improving or degrading performance
by running multiple inference passes and collecting detailed timing statistics.

Usage:
    # Profile with mock data (no robot needed)
    uv run examples/rtc/profile_rtc_comparison.py \
        --policy_path=helper2424/pi05_check_rtc \
        --device=mps \
        --num_iterations=50

    # Profile with specific RTC config
    uv run examples/rtc/profile_rtc_comparison.py \
        --policy_path=helper2424/pi05_check_rtc \
        --device=mps \
        --num_iterations=50 \
        --execution_horizon=20
"""

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.profiling import (
    clear_profiling_stats,
    enable_profiling,
    get_profiling_stats,
    print_profiling_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProfileResults:
    """Results from profiling run."""

    mode: str  # "with_rtc" or "without_rtc"
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    times: list[float]
    throughput: float  # iterations per second


def create_mock_observation(policy, device: str) -> dict:
    """Create a mock observation for testing.

    Args:
        policy: Policy instance
        device: Device to create tensors on

    Returns:
        Mock observation dictionary
    """
    # Get expected input shapes from policy config
    # This is a simplified version - adjust based on actual policy requirements
    obs = {}

    # Mock image observations (if needed)
    if hasattr(policy.config, "input_shapes"):
        for key, shape in policy.config.input_shapes.items():
            if "image" in key:
                # Typical image shape: (batch, channels, height, width)
                obs[key] = torch.randn(1, *shape, device=device)
            else:
                obs[key] = torch.randn(1, *shape, device=device)

    # Add task if needed
    if "task" in policy.config.__dict__ or hasattr(policy, "accepts_task"):
        obs["task"] = ["Pick up the object"]

    # Mock state observation
    obs["observation.state"] = torch.randn(1, 10, device=device)  # Adjust size as needed

    return obs


def profile_inference(
    policy, observation: dict, num_iterations: int, use_rtc: bool, execution_horizon: int = 10
) -> ProfileResults:
    """Profile policy inference with or without RTC.

    Args:
        policy: Policy instance
        observation: Observation dictionary
        num_iterations: Number of inference iterations to run
        use_rtc: Whether to enable RTC
        execution_horizon: Execution horizon for RTC

    Returns:
        ProfileResults with timing statistics
    """
    mode = "with_rtc" if use_rtc else "without_rtc"
    logger.info(f"\n{'='*80}")
    logger.info(f"Profiling: {mode.upper()}")
    logger.info(f"{'='*80}")

    # Configure RTC
    if use_rtc:
        policy.config.rtc_config.enabled = True
        policy.config.rtc_config.execution_horizon = execution_horizon
        policy.init_rtc_processor()
    else:
        policy.config.rtc_config.enabled = False

    times = []
    prev_actions = None

    # Warmup
    logger.info("Warming up (5 iterations)...")
    for _ in range(5):
        with torch.no_grad():
            if use_rtc:
                _ = policy.predict_action_chunk(
                    observation, inference_delay=0, prev_chunk_left_over=prev_actions
                )
            else:
                _ = policy.predict_action_chunk(observation)

    # Actual profiling
    logger.info(f"Running {num_iterations} profiled iterations...")
    for i in range(num_iterations):
        start = time.perf_counter()

        with torch.no_grad():
            if use_rtc:
                actions = policy.predict_action_chunk(
                    observation, inference_delay=0, prev_chunk_left_over=prev_actions
                )
                # Simulate consuming some actions for next iteration
                if actions.shape[1] > execution_horizon:
                    prev_actions = actions[:, execution_horizon:].clone()
                else:
                    prev_actions = None
            else:
                actions = policy.predict_action_chunk(observation)

        # Synchronize if using CUDA
        if observation["observation.state"].device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            logger.info(f"  Completed {i+1}/{num_iterations} iterations")

    # Calculate statistics
    times_arr = np.array(times)
    results = ProfileResults(
        mode=mode,
        mean_time=float(np.mean(times_arr)),
        std_time=float(np.std(times_arr)),
        min_time=float(np.min(times_arr)),
        max_time=float(np.max(times_arr)),
        times=times,
        throughput=num_iterations / sum(times),
    )

    logger.info(f"\nResults for {mode}:")
    logger.info(f"  Mean time: {results.mean_time*1000:.2f} ms")
    logger.info(f"  Std dev:   {results.std_time*1000:.2f} ms")
    logger.info(f"  Min time:  {results.min_time*1000:.2f} ms")
    logger.info(f"  Max time:  {results.max_time*1000:.2f} ms")
    logger.info(f"  Throughput: {results.throughput:.2f} iter/s")

    return results


def compare_results(results_without_rtc: ProfileResults, results_with_rtc: ProfileResults):
    """Compare and print results from both runs.

    Args:
        results_without_rtc: Results from run without RTC
        results_with_rtc: Results from run with RTC
    """
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}")

    mean_diff = results_with_rtc.mean_time - results_without_rtc.mean_time
    mean_diff_pct = (mean_diff / results_without_rtc.mean_time) * 100

    throughput_diff = results_with_rtc.throughput - results_without_rtc.throughput
    throughput_diff_pct = (throughput_diff / results_without_rtc.throughput) * 100

    logger.info(f"\n{'Metric':<30} {'Without RTC':>15} {'With RTC':>15} {'Difference':>15}")
    logger.info("-" * 80)
    logger.info(
        f"{'Mean time (ms)':<30} "
        f"{results_without_rtc.mean_time*1000:>15.2f} "
        f"{results_with_rtc.mean_time*1000:>15.2f} "
        f"{mean_diff*1000:>+15.2f}"
    )
    logger.info(
        f"{'Std dev (ms)':<30} "
        f"{results_without_rtc.std_time*1000:>15.2f} "
        f"{results_with_rtc.std_time*1000:>15.2f} "
        f"{(results_with_rtc.std_time - results_without_rtc.std_time)*1000:>+15.2f}"
    )
    logger.info(
        f"{'Min time (ms)':<30} "
        f"{results_without_rtc.min_time*1000:>15.2f} "
        f"{results_with_rtc.min_time*1000:>15.2f} "
        f"{(results_with_rtc.min_time - results_without_rtc.min_time)*1000:>+15.2f}"
    )
    logger.info(
        f"{'Max time (ms)':<30} "
        f"{results_without_rtc.max_time*1000:>15.2f} "
        f"{results_with_rtc.max_time*1000:>15.2f} "
        f"{(results_with_rtc.max_time - results_without_rtc.max_time)*1000:>+15.2f}"
    )
    logger.info(
        f"{'Throughput (iter/s)':<30} "
        f"{results_without_rtc.throughput:>15.2f} "
        f"{results_with_rtc.throughput:>15.2f} "
        f"{throughput_diff:>+15.2f}"
    )

    logger.info(f"\n{'='*80}")
    logger.info("VERDICT")
    logger.info(f"{'='*80}")

    if mean_diff_pct < -5:
        logger.info(f"✓ RTC is FASTER by {abs(mean_diff_pct):.1f}%")
        logger.info(f"  Mean time reduced by {abs(mean_diff)*1000:.2f} ms")
    elif mean_diff_pct > 5:
        logger.info(f"✗ RTC is SLOWER by {mean_diff_pct:.1f}%")
        logger.info(f"  Mean time increased by {mean_diff*1000:.2f} ms")
        logger.info("\n  Possible reasons:")
        logger.info("  - RTC overhead exceeds benefits at current execution horizon")
        logger.info("  - Inference delay calculation not accounting for RTC processing")
        logger.info("  - Additional tensor operations in RTC guidance")
    else:
        logger.info(f"≈ Performance is SIMILAR (difference: {mean_diff_pct:+.1f}%)")

    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Profile RTC performance")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to pretrained policy"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=50, help="Number of inference iterations"
    )
    parser.add_argument(
        "--execution_horizon", type=int, default=10, help="RTC execution horizon"
    )
    parser.add_argument(
        "--enable_detailed_profiling",
        action="store_true",
        help="Enable detailed method-level profiling",
    )
    parser.add_argument(
        "--use_torch_compile", action="store_true", help="Use torch.compile for faster inference"
    )

    args = parser.parse_args()

    # Load policy
    logger.info(f"Loading policy from {args.policy_path}")
    config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_class = get_policy_class(config.type)

    # Set compile flag if needed
    if hasattr(config, "compile_model"):
        config.compile_model = args.use_torch_compile

    policy = policy_class.from_pretrained(args.policy_path, config=config)

    # Initialize RTC config
    policy.config.rtc_config = RTCConfig(
        execution_horizon=args.execution_horizon,
        max_guidance_weight=1.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )

    policy = policy.to(args.device)
    policy.eval()

    logger.info(f"Policy loaded: {config.type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Execution horizon: {args.execution_horizon}")

    # Create mock observation
    logger.info("Creating mock observation...")
    observation = create_mock_observation(policy, args.device)

    # Enable detailed profiling if requested
    if args.enable_detailed_profiling:
        enable_profiling()
        logger.info("Detailed profiling enabled")

    # Profile without RTC
    results_without_rtc = profile_inference(
        policy=policy,
        observation=observation,
        num_iterations=args.num_iterations,
        use_rtc=False,
        execution_horizon=args.execution_horizon,
    )

    if args.enable_detailed_profiling:
        logger.info("\nDetailed profiling stats (WITHOUT RTC):")
        print_profiling_summary()
        clear_profiling_stats()

    # Profile with RTC
    results_with_rtc = profile_inference(
        policy=policy,
        observation=observation,
        num_iterations=args.num_iterations,
        use_rtc=True,
        execution_horizon=args.execution_horizon,
    )

    if args.enable_detailed_profiling:
        logger.info("\nDetailed profiling stats (WITH RTC):")
        print_profiling_summary()

    # Compare results
    compare_results(results_without_rtc, results_with_rtc)


if __name__ == "__main__":
    main()

