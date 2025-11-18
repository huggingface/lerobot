#!/usr/bin/env python

"""
Comprehensive profiling script for Pi0 with RTC.

This script demonstrates how to use all the profiling tools to identify
bottlenecks in Pi0 policy inference with RTC enabled.

It profiles:
1. Overall inference time
2. RTC-specific operations (guidance, weights, etc.)
3. Preprocessing/postprocessing
4. Individual method timings

Usage:
    uv run examples/rtc/profile_pi0_rtc_detailed.py \
        --policy_path=helper2424/pi05_check_rtc \
        --device=mps \
        --num_iterations=20 \
        --execution_horizon=20 \
        --enable_rtc_profiling
"""

import argparse
import logging
import sys
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.profiling import (
    ProfileContext,
    clear_profiling_stats,
    enable_profiling,
    get_profiling_stats,
    print_profiling_summary,
)

# Import monkey patching for RTC profiling
try:
    from examples.rtc.add_rtc_profiling import monkey_patch_rtc_profiling
except ImportError:
    logging.warning("Could not import add_rtc_profiling, detailed RTC profiling disabled")
    monkey_patch_rtc_profiling = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_observation(policy_config, device: str) -> dict:
    """Create a mock observation matching policy requirements.
    
    Args:
        policy_config: Policy configuration
        device: Device to create tensors on
        
    Returns:
        Mock observation dictionary
    """
    obs = {}
    
    # Create mock state observation
    state_dim = 10  # Typical robot state dimension
    obs["observation.state"] = torch.randn(1, state_dim, device=device)
    
    # Create mock images if needed
    # For Pi0, we typically need at least one image
    image_height = 224
    image_width = 224
    
    # Common image keys for Pi0
    image_keys = ["observation.images.gripper", "observation.images.front"]
    
    for key in image_keys:
        # Images should be [B, C, H, W] and normalized to [0, 1]
        obs[key] = torch.rand(1, 3, image_height, image_width, device=device)
    
    # Add task
    obs["task"] = ["Pick up the object"]
    
    # Add language tokens and attention mask (required for Pi0)
    # These are mock values - in real usage they come from tokenizer
    max_seq_len = 32
    obs["observation.language_tokens"] = torch.randint(0, 1000, (1, max_seq_len), device=device)
    obs["observation.language_attention_mask"] = torch.ones(1, max_seq_len, device=device)
    
    return obs


def profile_single_iteration(
    policy,
    preprocessor,
    postprocessor,
    observation: dict,
    prev_actions: torch.Tensor | None,
    use_rtc: bool,
    inference_delay: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """Profile a single inference iteration.
    
    Args:
        policy: Policy instance
        preprocessor: Observation preprocessor
        postprocessor: Action postprocessor
        observation: Input observation
        prev_actions: Previous action chunk (for RTC)
        use_rtc: Whether RTC is enabled
        inference_delay: Inference delay in timesteps
        
    Returns:
        Tuple of (actions, new_prev_actions, timings)
    """
    timings = {}
    
    with ProfileContext("iteration.total"):
        # Preprocessing
        with ProfileContext("iteration.preprocessing"):
            preprocessed_obs = preprocessor(observation)
        
        # Policy inference
        with ProfileContext("iteration.policy_inference"):
            if use_rtc:
                actions = policy.predict_action_chunk(
                    preprocessed_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )
            else:
                actions = policy.predict_action_chunk(preprocessed_obs)
        
        # Clone for next iteration (if RTC)
        new_prev_actions = None
        if use_rtc:
            with ProfileContext("iteration.prepare_prev_actions"):
                execution_horizon = policy.config.rtc_config.execution_horizon
                if actions.shape[1] > execution_horizon:
                    new_prev_actions = actions[:, execution_horizon:].clone()
        
        # Postprocessing
        with ProfileContext("iteration.postprocessing"):
            processed_actions = postprocessor(actions)
    
    return processed_actions, new_prev_actions, timings


def main():
    parser = argparse.ArgumentParser(description="Detailed profiling for Pi0 with RTC")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to pretrained policy")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--num_iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--execution_horizon", type=int, default=10, help="RTC execution horizon")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--enable_rtc_profiling", action="store_true", help="Enable detailed RTC profiling")
    parser.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DETAILED PI0 RTC PROFILING")
    logger.info("="*80)
    logger.info(f"Policy: {args.policy_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Iterations: {args.num_iterations}")
    logger.info(f"Execution Horizon: {args.execution_horizon}")
    logger.info(f"RTC Profiling: {args.enable_rtc_profiling}")
    logger.info("="*80 + "\n")
    
    # Enable profiling
    enable_profiling()
    
    # Apply RTC profiling if requested
    if args.enable_rtc_profiling:
        if monkey_patch_rtc_profiling is not None:
            monkey_patch_rtc_profiling()
            logger.info("✓ Detailed RTC profiling enabled\n")
        else:
            logger.warning("⚠ Could not enable detailed RTC profiling\n")
    
    # Load policy
    logger.info("Loading policy...")
    config = PreTrainedConfig.from_pretrained(args.policy_path)
    
    if hasattr(config, "compile_model"):
        config.compile_model = args.use_torch_compile
    
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(args.policy_path, config=config)
    
    # Configure RTC
    policy.config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=1.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy.init_rtc_processor()
    
    policy = policy.to(args.device)
    policy.eval()
    
    logger.info(f"✓ Policy loaded: {config.type}\n")
    
    # Create preprocessor and postprocessor
    logger.info("Loading preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
        },
    )
    logger.info("✓ Preprocessor/postprocessor loaded\n")
    
    # Create mock observation
    logger.info("Creating mock observation...")
    observation = create_mock_observation(config, args.device)
    logger.info("✓ Mock observation created\n")
    
    # Warmup
    logger.info(f"Warming up ({args.warmup_iterations} iterations)...")
    prev_actions = None
    for i in range(args.warmup_iterations):
        with torch.no_grad():
            _, prev_actions, _ = profile_single_iteration(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                observation=observation,
                prev_actions=prev_actions,
                use_rtc=True,
                inference_delay=0,
            )
    
    # Clear warmup stats
    clear_profiling_stats()
    logger.info("✓ Warmup complete\n")
    
    # Profiled run WITH RTC
    logger.info(f"Running profiled iterations WITH RTC ({args.num_iterations} iterations)...")
    prev_actions = None
    iteration_times = []
    
    for i in range(args.num_iterations):
        start = time.perf_counter()
        
        with torch.no_grad():
            _, prev_actions, _ = profile_single_iteration(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                observation=observation,
                prev_actions=prev_actions,
                use_rtc=True,
                inference_delay=0,
            )
        
        # Sync CUDA if needed
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        iteration_times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Completed {i+1}/{args.num_iterations}")
    
    logger.info("✓ Profiling complete\n")
    
    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("ITERATION TIMING SUMMARY")
    logger.info("="*80)
    
    times_arr = np.array(iteration_times)
    logger.info(f"Mean time:       {np.mean(times_arr)*1000:.2f} ms")
    logger.info(f"Median time:     {np.median(times_arr)*1000:.2f} ms")
    logger.info(f"Std dev:         {np.std(times_arr)*1000:.2f} ms")
    logger.info(f"Min time:        {np.min(times_arr)*1000:.2f} ms")
    logger.info(f"Max time:        {np.max(times_arr)*1000:.2f} ms")
    logger.info(f"Total time:      {np.sum(times_arr):.2f} s")
    logger.info(f"Throughput:      {len(times_arr)/np.sum(times_arr):.2f} iter/s")
    logger.info("="*80 + "\n")
    
    # Print detailed profiling breakdown
    print_profiling_summary(sort_by="total")
    
    # Print key insights
    stats = get_profiling_stats()
    
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)
    
    # Find bottlenecks
    if stats:
        policy_inference_time = stats.get("iteration.policy_inference", {}).get("mean", 0)
        preprocessing_time = stats.get("iteration.preprocessing", {}).get("mean", 0)
        postprocessing_time = stats.get("iteration.postprocessing", {}).get("mean", 0)
        
        total_time = policy_inference_time + preprocessing_time + postprocessing_time
        
        if total_time > 0:
            logger.info(f"\nTime breakdown:")
            logger.info(f"  Policy inference:  {policy_inference_time*1000:.2f} ms ({policy_inference_time/total_time*100:.1f}%)")
            logger.info(f"  Preprocessing:     {preprocessing_time*1000:.2f} ms ({preprocessing_time/total_time*100:.1f}%)")
            logger.info(f"  Postprocessing:    {postprocessing_time*1000:.2f} ms ({postprocessing_time/total_time*100:.1f}%)")
        
        # RTC-specific insights
        if args.enable_rtc_profiling:
            rtc_guidance = stats.get("rtc.denoise_step.guidance_computation", {}).get("mean", 0)
            rtc_autograd = stats.get("rtc.denoise_step.autograd_correction", {}).get("mean", 0)
            rtc_base = stats.get("rtc.denoise_step.base_denoising", {}).get("mean", 0)
            
            if rtc_guidance > 0:
                logger.info(f"\nRTC breakdown:")
                logger.info(f"  Base denoising:    {rtc_base*1000:.2f} ms")
                logger.info(f"  Guidance compute:  {rtc_guidance*1000:.2f} ms")
                logger.info(f"  Autograd correct:  {rtc_autograd*1000:.2f} ms")
                logger.info(f"  RTC overhead:      {(rtc_guidance - rtc_base)*1000:.2f} ms")
        
        # Recommendations
        logger.info("\nRecommendations:")
        
        if preprocessing_time > policy_inference_time * 0.3:
            logger.info("  ⚠ Preprocessing is taking >30% of time")
            logger.info("    → Consider reducing image resolution")
            logger.info("    → Consider using fewer cameras")
        
        if args.enable_rtc_profiling and rtc_autograd > rtc_base * 0.5:
            logger.info("  ⚠ RTC autograd overhead is significant")
            logger.info("    → This is expected, but consider increasing execution_horizon")
            logger.info("    → Try torch.compile if not already enabled")
        
        if not args.use_torch_compile:
            logger.info("  💡 torch.compile not enabled")
            logger.info("    → Try --use_torch_compile for potential speedup")
    
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProfiling interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

