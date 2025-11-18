#!/usr/bin/env python

"""
Script to add profiling instrumentation to RTCProcessor.

This script shows which methods to profile in the RTC code to identify bottlenecks.
You can either:
1. Apply these changes directly to modeling_rtc.py
2. Use monkey patching to add profiling without modifying source
3. Use as reference for manual instrumentation

Usage:
    # Option 1: Monkey patch (no source changes)
    python examples/rtc/add_rtc_profiling.py

    # Option 2: Apply changes to source
    # Copy the profiled methods below into src/lerobot/policies/rtc/modeling_rtc.py
"""

import logging

import torch
from torch import Tensor

from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.profiling import ProfileContext, enable_profiling, is_profiling_enabled

logger = logging.getLogger(__name__)


def profile_denoise_step(self, x_t, prev_chunk_left_over, inference_delay, time, original_denoise_step_partial, execution_horizon=None) -> Tensor:
    """Profiled version of denoise_step."""
    
    if not is_profiling_enabled():
        # Call original implementation if profiling disabled
        return self._original_denoise_step(x_t, prev_chunk_left_over, inference_delay, time, original_denoise_step_partial, execution_horizon)
    
    with ProfileContext("rtc.denoise_step.total"):
        # In the original implementation, the time goes from 0 to 1 and
        # In our implementation, the time goes from 1 to 0
        # So we need to invert the time
        tau = 1 - time

        if prev_chunk_left_over is None:
            # First step, no guidance - return v_t
            with ProfileContext("rtc.denoise_step.base_denoising"):
                v_t = original_denoise_step_partial(x_t)
            return v_t

        with ProfileContext("rtc.denoise_step.setup"):
            x_t = x_t.clone().detach()

            squeezed = False
            if len(x_t.shape) < 3:
                x_t = x_t.unsqueeze(0)
                squeezed = True

            if len(prev_chunk_left_over.shape) < 3:
                prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

            if execution_horizon is None:
                execution_horizon = self.rtc_config.execution_horizon

            if execution_horizon > prev_chunk_left_over.shape[1]:
                execution_horizon = prev_chunk_left_over.shape[1]

            batch_size = x_t.shape[0]
            action_chunk_size = x_t.shape[1]
            action_dim = x_t.shape[2]

        # Padding
        with ProfileContext("rtc.denoise_step.padding"):
            if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
                padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
                padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
                prev_chunk_left_over = padded

        # Get prefix weights
        with ProfileContext("rtc.denoise_step.get_prefix_weights"):
            weights = (
                self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
                .to(x_t.device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )

        # Main RTC guidance computation
        with ProfileContext("rtc.denoise_step.guidance_computation"):
            with torch.enable_grad():
                # Base denoising
                with ProfileContext("rtc.denoise_step.base_denoising"):
                    v_t = original_denoise_step_partial(x_t)
                
                x_t.requires_grad_(True)

                # Compute x1_t
                with ProfileContext("rtc.denoise_step.compute_x1_t"):
                    x1_t = x_t - time * v_t

                # Compute error
                with ProfileContext("rtc.denoise_step.compute_error"):
                    err = (prev_chunk_left_over - x1_t) * weights
                    grad_outputs = err.clone().detach()

                # Compute correction via autograd
                with ProfileContext("rtc.denoise_step.autograd_correction"):
                    correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        # Compute guidance weight
        with ProfileContext("rtc.denoise_step.compute_guidance_weight"):
            max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
            tau_tensor = torch.as_tensor(tau)
            squared_one_minus_tau = (1 - tau_tensor) ** 2
            inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
            c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
            guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
            guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        # Apply guidance
        with ProfileContext("rtc.denoise_step.apply_guidance"):
            result = v_t - guidance_weight * correction

        # Cleanup
        with ProfileContext("rtc.denoise_step.cleanup"):
            if squeezed:
                result = result.squeeze(0)
                correction = correction.squeeze(0)
                x1_t = x1_t.squeeze(0)
                err = err.squeeze(0)

            self.track(
                time=time,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
            )

        return result


def monkey_patch_rtc_profiling():
    """Apply profiling to RTCProcessor via monkey patching.
    
    This modifies the RTCProcessor class at runtime to add profiling
    without changing source files.
    """
    logger.info("Applying RTC profiling monkey patch...")
    
    # Save original method
    RTCProcessor._original_denoise_step = RTCProcessor.denoise_step
    
    # Replace with profiled version
    RTCProcessor.denoise_step = profile_denoise_step
    
    logger.info("✓ RTC profiling enabled")


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("RTC PROFILING INSTRUMENTATION")
    print("="*80)
    print("\nThis script provides profiling for RTCProcessor methods.")
    print("\nOption 1: Monkey Patch (Recommended)")
    print("-" * 40)
    print("Add to your script:")
    print("""
    from lerobot.utils.profiling import enable_profiling, print_profiling_summary
    from examples.rtc.add_rtc_profiling import monkey_patch_rtc_profiling

    # Enable profiling
    enable_profiling()
    monkey_patch_rtc_profiling()

    # ... run your code ...

    # Print results
    print_profiling_summary()
    """)
    
    print("\nOption 2: Manual Source Modification")
    print("-" * 40)
    print("1. Copy profile_denoise_step() from this file")
    print("2. Replace denoise_step() in src/lerobot/policies/rtc/modeling_rtc.py")
    print("3. Add profiling imports at top of file")
    
    print("\nKey Metrics to Watch:")
    print("-" * 40)
    print("- rtc.denoise_step.base_denoising     - Time for base policy inference")
    print("- rtc.denoise_step.autograd_correction - Time computing gradients")
    print("- rtc.denoise_step.guidance_computation - Total guidance overhead")
    print("- rtc.denoise_step.get_prefix_weights  - Time computing weights")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_usage()

