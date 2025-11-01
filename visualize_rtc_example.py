#!/usr/bin/env python

"""
Example script demonstrating how to use the gradient visualization feature
in RTCProcessor to visualize the correction term computation.

Usage:
    1. Install torchviz: uv pip install torchviz graphviz
    2. Install graphviz system package: brew install graphviz (macOS)
    3. Run this script: python visualize_rtc_example.py
"""

import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def example_denoiser(x_t):
    """Simple mock denoiser that returns random velocity."""
    return torch.randn_like(x_t)


def main():
    # Create RTC configuration
    rtc_config = RTCConfig(
        execution_horizon=5,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=10.0,
    )

    # Create RTCProcessor with gradient visualization enabled
    processor = RTCProcessor(
        rtc_config=rtc_config,
        verbose=True,
        visualize_gradients=True,  # Enable gradient visualization
        viz_output_dir="./rtc_viz_output",  # Output directory for graphs
    )

    # Setup dummy data
    batch_size = 2
    action_chunk_size = 10
    action_dim = 7

    x_t = torch.randn(batch_size, action_chunk_size, action_dim)
    prev_chunk_left_over = torch.randn(batch_size, action_chunk_size, action_dim)
    inference_delay = 2
    time = torch.tensor(0.5)

    # Run denoise step - this will generate the visualization
    print("Running denoise step with gradient visualization...")
    result, correction, x1_t, err = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=inference_delay,
        time=time,
        original_denoise_step_partial=example_denoiser,
    )

    print(f"\nResult shape: {result.shape}")
    print(f"Correction shape: {correction.shape}")
    print("\nCheck './rtc_viz_output/' for generated PNG visualizations!")
    print("- rtc_correction_forward_graph_0.png: Shows forward computation graph")
    print("- rtc_correction_gradient_graph_0.png: Shows gradient computation graph")


if __name__ == "__main__":
    main()
