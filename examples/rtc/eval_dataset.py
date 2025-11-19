#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Evaluate Real-Time Chunking (RTC) performance on dataset samples.

This script takes two random samples from a dataset:
- Uses actions from the first sample as previous chunk
- Generates new actions for the second sample with and without RTC

It compares action predictions with and without RTC on dataset samples,
measuring consistency and ground truth alignment.

Usage:
    # Basic usage with smolvla policy
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=mps \
        --rtc.max_guidance_weight=10.0 \
        --rtc.prefix_attention_schedule=EXP \
        --seed=10

    # Basic usage with pi0.5 policy
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=lerobot/pi05_libero_finetuned \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --rtc.execution_horizon=10 \
        --device=mps
        --seed=10

    # Basic usage with pi0.5 policy with cuda device
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=lerobot/pi05_libero_finetuned \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --rtc.execution_horizon=8 \
        --device=cuda

    # Basic usage with pi0 policy with cuda device
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=lerobot/pi0_libero_finetuned \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --rtc.execution_horizon=8 \
        --device=cuda

    uv run python examples/rtc/eval_dataset.py \
        --policy.path=lipsop/reuben_pi0 \
        --dataset.repo_id=ReubenLim/so101_cube_in_cup \
        --rtc.execution_horizon=8 \
        --device=cuda

    # With torch.compile for faster inference (PyTorch 2.0+)
    # Note: CUDA graphs disabled by default due to in-place ops in denoising loop
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=mps \
        --use_torch_compile=true \
        --torch_compile_mode=max-autotune

    # With torch.compile on CUDA (CUDA graphs disabled by default)
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=cuda \
        --use_torch_compile=true \
        --torch_compile_mode=reduce-overhead

    # Enable CUDA graphs (advanced - may cause tensor aliasing errors)
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --use_torch_compile=true \
        --torch_compile_backend=inductor \
        --torch_compile_mode=max-autotune \
        --torch_compile_disable_cudagraphs=false
"""

import gc
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_matplotlib_available():
    """Check if matplotlib is available, raise helpful error if not."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for RTC debug visualizations. "
            "Please install it by running:\n"
            "  uv pip install matplotlib"
        )


@dataclass
class RTCEvalConfig(HubMixin):
    """Configuration for RTC evaluation."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=20,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            debug=True,
            debug_maxlen=1000,
        )
    )

    # Device configuration
    device: str | None = field(
        default=None,
        metadata={"help": "Device to run on (cuda, cpu, mps, auto)"},
    )

    # Output configuration
    output_dir: str = field(
        default="rtc_debug_output",
        metadata={"help": "Directory to save debug visualizations"},
    )

    # Seed configuration
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )

    inference_delay: int = field(
        default=4,
        metadata={"help": "Inference delay for RTC"},
    )

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    def __post_init__(self):
        # Parse policy path
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (--policy.path)")

        # Auto-detect device if not specified
        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            logging.info(f"Auto-detected device: {self.device}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class RTCEvaluator:
    """Evaluator for RTC on dataset samples."""

    def __init__(self, cfg: RTCEvalConfig):
        self.cfg = cfg
        self.device = cfg.device

        # Load dataset with proper delta_timestamps based on policy configuration
        # Calculate delta_timestamps using the same logic as make_dataset factory
        logging.info(f"Loading dataset: {cfg.dataset.repo_id}")

        # Get dataset metadata to extract FPS
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id)

        # Calculate delta_timestamps from policy's delta_indices
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        # Create dataset with calculated delta_timestamps
        self.dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            delta_timestamps=delta_timestamps,
        )
        logging.info(f"Dataset loaded: {len(self.dataset)} samples, {self.dataset.num_episodes} episodes")

        # Create preprocessor/postprocessor
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

        logging.info("=" * 80)
        logging.info("Ready to run evaluation with sequential policy loading:")
        logging.info("  1. policy_prev_chunk - Generate reference chunk, then destroy")
        logging.info("  2. policy_no_rtc - Generate without RTC, then destroy")
        logging.info("  3. policy_rtc - Generate with RTC, then destroy")
        logging.info("  Note: Only one policy in memory at a time for efficient memory usage")
        logging.info("=" * 80)

    def _init_policy(self, name: str, rtc_enabled: bool, rtc_debug: bool):
        """Initialize a single policy instance with specified RTC configuration.

        Args:
            name: Name identifier for logging purposes
            rtc_enabled: Whether to enable RTC for this policy
            rtc_debug: Whether to enable debug tracking for this policy

        Returns:
            Configured policy instance with optional torch.compile applied
        """
        logging.info(f"Initializing {name}...")

        # Load policy from pretrained
        policy_class = get_policy_class(self.cfg.policy.type)

        config = PreTrainedConfig.from_pretrained(self.cfg.policy.pretrained_path)

        if self.cfg.policy.type == "pi05" or self.cfg.policy.type == "pi0":
            config.compile_model = self.cfg.use_torch_compile

        policy = policy_class.from_pretrained(self.cfg.policy.pretrained_path, config=config)
        policy = policy.to(self.device)
        policy.eval()

        # Configure RTC
        rtc_config = RTCConfig(
            enabled=rtc_enabled,
            execution_horizon=self.cfg.rtc.execution_horizon,
            max_guidance_weight=self.cfg.rtc.max_guidance_weight,
            prefix_attention_schedule=self.cfg.rtc.prefix_attention_schedule,
            debug=rtc_debug,
            debug_maxlen=self.cfg.rtc.debug_maxlen,
        )
        policy.config.rtc_config = rtc_config
        policy.init_rtc_processor()

        logging.info(f"  RTC enabled: {rtc_enabled}")
        logging.info(f"  RTC debug: {rtc_debug}")
        logging.info(f"  Policy config: {config}")

        # Apply torch.compile to predict_action_chunk method if enabled
        if self.cfg.use_torch_compile:
            policy = self._apply_torch_compile(policy, name)

        logging.info(f"✓ {name} initialized successfully")
        return policy

    def _apply_torch_compile(self, policy, policy_name: str):
        """Apply torch.compile to the policy's predict_action_chunk method.

        Args:
            policy: Policy instance to compile
            policy_name: Name for logging purposes

        Returns:
            Policy with compiled predict_action_chunk method
        """

        # PI models handle their own compilation
        if policy.type == "pi05" or policy.type == "pi0":
            return policy

        try:
            # Check if torch.compile is available (PyTorch 2.0+)
            if not hasattr(torch, "compile"):
                logging.warning(
                    f"  [{policy_name}] torch.compile is not available. Requires PyTorch 2.0+. "
                    f"Current version: {torch.__version__}. Skipping compilation."
                )
                return policy

            logging.info(f"  [{policy_name}] Applying torch.compile to predict_action_chunk...")
            logging.info(f"    Backend: {self.cfg.torch_compile_backend}")
            logging.info(f"    Mode: {self.cfg.torch_compile_mode}")
            logging.info(f"    Disable CUDA graphs: {self.cfg.torch_compile_disable_cudagraphs}")
            logging.info("    Note: Debug tracker excluded from compilation via @torch._dynamo.disable")

            # Compile the predict_action_chunk method
            # - Debug tracker is excluded from compilation via @torch._dynamo.disable
            # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
            compile_kwargs = {
                "backend": self.cfg.torch_compile_backend,
                "mode": self.cfg.torch_compile_mode,
            }

            # Disable CUDA graphs if requested (prevents tensor aliasing issues)
            if self.cfg.torch_compile_disable_cudagraphs:
                compile_kwargs["options"] = {"triton.cudagraphs": False}

            original_method = policy.predict_action_chunk
            compiled_method = torch.compile(original_method, **compile_kwargs)
            policy.predict_action_chunk = compiled_method
            logging.info(f"  ✓ [{policy_name}] Successfully compiled predict_action_chunk")

        except Exception as e:
            logging.error(f"  [{policy_name}] Failed to apply torch.compile: {e}")
            logging.warning(f"  [{policy_name}] Continuing without torch.compile")

        return policy

    def _destroy_policy(self, policy, policy_name: str):
        """Explicitly destroy a policy and free all associated memory.

        This method performs aggressive cleanup to ensure maximum memory is freed,
        which is critical for large models (e.g., VLAs with billions of parameters).

        Args:
            policy: Policy instance to destroy
            policy_name: Name for logging purposes
        """
        logging.info(f"  Destroying {policy_name} and freeing memory...")

        try:
            # Step 1: Move policy to CPU to free GPU/MPS memory
            policy.cpu()

            # Step 2: Delete the policy object
            del policy

            # Step 3: Force garbage collection to reclaim memory immediately
            gc.collect()

            # Step 4: Clear device-specific caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            logging.info(f"  ✓ {policy_name} destroyed and memory freed")

        except Exception as e:
            logging.warning(f"  Warning: Error during {policy_name} cleanup: {e}")

    def run_evaluation(self):
        """Run evaluation on two random dataset samples using three separate policies.

        Note: Policies are deinitalized after each step to free memory. Large models
        (e.g., VLA models with billions of parameters) cannot fit three instances in
        memory simultaneously. By deleting and garbage collecting after each step,
        we ensure only one policy is loaded at a time.
        """
        # Create output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logging.info(f"Output directory: {self.cfg.output_dir}")

        logging.info("=" * 80)
        logging.info("Starting RTC evaluation")
        logging.info(f"Inference delay: {self.cfg.inference_delay}")
        logging.info("=" * 80)

        # Load two random samples from dataset
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        loader_iter = iter(data_loader)
        first_sample = next(loader_iter)
        second_sample = next(loader_iter)

        preprocessed_first_sample = self.preprocessor(first_sample)
        preprocessed_second_sample = self.preprocessor(second_sample)

        # ============================================================================
        # Step 1: Generate previous chunk using policy_prev_chunk
        # ============================================================================
        # This policy is only used to generate the reference chunk and then freed
        logging.info("=" * 80)
        logging.info("Step 1: Generating previous chunk with policy_prev_chunk")
        logging.info("=" * 80)

        # Initialize policy 1
        policy_prev_chunk_policy = self._init_policy(
            name="policy_prev_chunk",
            rtc_enabled=False,
            rtc_debug=False,
        )
        with torch.no_grad():
            prev_chunk_left_over = policy_prev_chunk_policy.predict_action_chunk(
                preprocessed_first_sample,
            )[:, :25, :].squeeze(0)
        logging.info(f"  Generated prev_chunk shape: {prev_chunk_left_over.shape}")

        # Destroy policy_prev_chunk to free memory for large models
        self._destroy_policy(policy_prev_chunk_policy, "policy_prev_chunk")

        # ============================================================================
        # Step 2: Generate actions WITHOUT RTC using policy_no_rtc
        # ============================================================================
        logging.info("=" * 80)
        logging.info("Step 2: Generating actions WITHOUT RTC with policy_no_rtc")
        logging.info("=" * 80)

        set_seed(self.cfg.seed)

        # Initialize policy 2
        policy_no_rtc_policy = self._init_policy(
            name="policy_no_rtc",
            rtc_enabled=False,
            rtc_debug=True,
        )

        # Sample noise (use same noise for both RTC and non-RTC for fair comparison)
        noise_size = (1, policy_no_rtc_policy.config.chunk_size, policy_no_rtc_policy.config.max_action_dim)
        noise = policy_no_rtc_policy.model.sample_noise(noise_size, self.device)
        noise_clone = noise.clone()
        policy_no_rtc_policy.rtc_processor.reset_tracker()
        with torch.no_grad():
            no_rtc_actions = policy_no_rtc_policy.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise,
            )
        no_rtc_tracked_steps = policy_no_rtc_policy.rtc_processor.tracker.get_all_steps()
        logging.info(f"  Tracked {len(no_rtc_tracked_steps)} steps without RTC")
        logging.info(f"  Generated no_rtc_actions shape: {no_rtc_actions.shape}")

        # Destroy policy_no_rtc to free memory before loading policy_rtc
        self._destroy_policy(policy_no_rtc_policy, "policy_no_rtc")

        # ============================================================================
        # Step 3: Generate actions WITH RTC using policy_rtc
        # ============================================================================
        logging.info("=" * 80)
        logging.info("Step 3: Generating actions WITH RTC with policy_rtc")
        logging.info("=" * 80)

        set_seed(self.cfg.seed)

        # Initialize policy 3
        policy_rtc_policy = self._init_policy(
            name="policy_rtc",
            rtc_enabled=True,
            rtc_debug=True,
        )
        policy_rtc_policy.rtc_processor.reset_tracker()
        with torch.no_grad():
            rtc_actions = policy_rtc_policy.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise_clone,
                inference_delay=self.cfg.inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=self.cfg.rtc.execution_horizon,
            )
        rtc_tracked_steps = policy_rtc_policy.rtc_processor.get_all_debug_steps()
        logging.info(f"  Tracked {len(rtc_tracked_steps)} steps with RTC")
        logging.info(f"  Generated rtc_actions shape: {rtc_actions.shape}")

        # Save num_steps before destroying policy (needed for plotting)
        try:
            num_steps = policy_rtc_policy.config.num_steps
        except Exception as e:
            logging.error(f"  Error getting num_steps: {e}")
            num_steps = policy_rtc_policy.config.num_inference_steps
            logging.warning(f"  Using num_inference_steps: {num_steps} instead of num_steps")

        # Destroy policy_rtc after final use
        self._destroy_policy(policy_rtc_policy, "policy_rtc")

        # Plot and save results
        logging.info("=" * 80)
        logging.info("Plotting results...")
        self.plot_tracked_data(rtc_tracked_steps, no_rtc_tracked_steps, prev_chunk_left_over, num_steps)

        # Plot final actions comparison
        logging.info("=" * 80)
        logging.info("Plotting final actions comparison...")
        self.plot_final_actions_comparison(rtc_actions, no_rtc_actions, prev_chunk_left_over)

        logging.info("=" * 80)
        logging.info("Evaluation completed successfully")

    def plot_final_actions_comparison(self, rtc_actions, no_rtc_actions, prev_chunk_left_over):
        """Plot final action predictions comparison on a single chart.

        Args:
            rtc_actions: Final actions from RTC policy
            no_rtc_actions: Final actions from non-RTC policy
            prev_chunk_left_over: Previous chunk used as ground truth
        """
        _check_matplotlib_available()

        # Remove batch dimension if present
        rtc_actions_plot = rtc_actions.squeeze(0).cpu() if len(rtc_actions.shape) == 3 else rtc_actions.cpu()
        no_rtc_actions_plot = (
            no_rtc_actions.squeeze(0).cpu() if len(no_rtc_actions.shape) == 3 else no_rtc_actions.cpu()
        )
        prev_chunk_plot = prev_chunk_left_over.cpu()

        # Create figure with 6 subplots (one per action dimension)
        fig, axes = plt.subplots(6, 1, figsize=(16, 12))
        fig.suptitle("Final Action Predictions Comparison (Raw)", fontsize=16)

        # Plot each action dimension
        for dim_idx, ax in enumerate(axes):
            # Plot previous chunk (ground truth) in red
            RTCDebugVisualizer.plot_waypoints(
                [ax],
                prev_chunk_plot[:, dim_idx : dim_idx + 1],
                start_from=0,
                color="red",
                label="Previous Chunk (Ground Truth)",
                linewidth=2.5,
                alpha=0.8,
            )

            # Plot no-RTC actions in blue
            RTCDebugVisualizer.plot_waypoints(
                [ax],
                no_rtc_actions_plot[:, dim_idx : dim_idx + 1],
                start_from=0,
                color="blue",
                label="No RTC",
                linewidth=2,
                alpha=0.7,
            )

            # Plot RTC actions in green
            RTCDebugVisualizer.plot_waypoints(
                [ax],
                rtc_actions_plot[:, dim_idx : dim_idx + 1],
                start_from=0,
                color="green",
                label="RTC",
                linewidth=2,
                alpha=0.7,
            )

            # Add vertical lines for inference delay and execution horizon
            inference_delay = self.cfg.inference_delay
            execution_horizon = self.cfg.rtc.execution_horizon

            if inference_delay > 0:
                ax.axvline(
                    x=inference_delay - 1,
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Inference Delay ({inference_delay})",
                )

            if execution_horizon > 0:
                ax.axvline(
                    x=execution_horizon,
                    color="purple",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Execution Horizon ({execution_horizon})",
                )

            ax.set_ylabel(f"Dim {dim_idx}", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set x-axis ticks to show all integer values
            max_len = max(rtc_actions_plot.shape[0], no_rtc_actions_plot.shape[0], prev_chunk_plot.shape[0])
            ax.set_xticks(range(0, max_len, max(1, max_len // 20)))  # Show ~20 ticks
            ax.set_xlim(-0.5, max_len - 0.5)

        axes[-1].set_xlabel("Step", fontsize=10)

        # Collect legend handles and labels from first subplot
        handles, labels = axes[0].get_legend_handles_labels()
        # Remove duplicates while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels, strict=True):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)

        # Add legend outside the plot area (to the right)
        fig.legend(
            unique_handles,
            unique_labels,
            loc="center right",
            fontsize=9,
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0.9,
        )

        # Save figure
        output_path = os.path.join(self.cfg.output_dir, "final_actions_comparison.png")
        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on right
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved final actions comparison to {output_path}")
        plt.close(fig)

    def plot_tracked_data(self, rtc_tracked_steps, no_rtc_tracked_steps, prev_chunk_left_over, num_steps):
        _check_matplotlib_available()

        # Create side-by-side figures for denoising visualization
        fig_xt, axs_xt = self._create_figure("x_t Denoising: No RTC (left) vs RTC (right)")
        fig_vt, axs_vt = self._create_figure("v_t Denoising: No RTC (left) vs RTC (right)")
        fig_corr, axs_corr = self._create_figure("Correction: No RTC (left) vs RTC (right)")
        fig_x1t, axs_x1t = self._create_figure(
            "x1_t Predicted State & Error: No RTC (left - empty) vs RTC (right)"
        )
        self._plot_denoising_steps_from_tracker(
            rtc_tracked_steps,
            axs_xt[:, 1],  # Right column for x_t
            axs_vt[:, 1],  # Right column for v_t
            axs_corr[:, 1],  # Right column for correction
            axs_x1t[:, 1],  # Right column for x1_t
            num_steps,
            add_labels=True,  # Add labels for RTC (right column)
        )

        self._plot_denoising_steps_from_tracker(
            no_rtc_tracked_steps,
            axs_xt[:, 0],  # Left column for x_t
            axs_vt[:, 0],  # Left column for v_t
            axs_corr[:, 0],  # Left column for correction
            axs_x1t[:, 0],  # Left column for x1_t
            num_steps,
            add_labels=False,  # No labels for No RTC (left column)
        )

        # Plot no-RTC x_t data on right chart as orange dashed line for comparison
        self._plot_no_rtc_xt_reference(no_rtc_tracked_steps, axs_xt[:, 1], num_steps)

        # Plot ground truth on x_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_xt[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        # Plot ground truth on x1_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_x1t[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        # Plot ground truth on x_t axes (no labels for left column)
        RTCDebugVisualizer.plot_waypoints(
            axs_xt[:, 0], prev_chunk_left_over, start_from=0, color="red", label=None
        )

        RTCDebugVisualizer.plot_waypoints(
            axs_x1t[:, 0], prev_chunk_left_over, start_from=0, color="red", label=None
        )

        # Add legends outside the plot area for each figure
        self._add_figure_legend(fig_xt, axs_xt)
        self._add_figure_legend(fig_vt, axs_vt)
        self._add_figure_legend(fig_corr, axs_corr)
        self._add_figure_legend(fig_x1t, axs_x1t)

        # Save denoising plots
        self._save_figure(fig_xt, os.path.join(self.cfg.output_dir, "denoising_xt_comparison.png"))
        self._save_figure(fig_vt, os.path.join(self.cfg.output_dir, "denoising_vt_comparison.png"))
        self._save_figure(fig_corr, os.path.join(self.cfg.output_dir, "denoising_correction_comparison.png"))
        self._save_figure(fig_x1t, os.path.join(self.cfg.output_dir, "denoising_x1t_comparison.png"))

    def _create_figure(self, title):
        fig, axs = plt.subplots(6, 2, figsize=(24, 12))
        fig.suptitle(title, fontsize=16)

        for ax in axs[:, 0]:
            ax.set_title("No RTC (N/A)" if ax == axs[0, 0] else "", fontsize=12)
        for ax in axs[:, 1]:
            ax.set_title("RTC" if ax == axs[0, 1] else "", fontsize=12)

        return fig, axs

    def _add_figure_legend(self, fig, axs):
        """Add a legend outside the plot area on the right side.

        Args:
            fig: Matplotlib figure to add legend to
            axs: Array of axes to collect legend handles from
        """
        # Collect all handles and labels from the first row of axes (right column)
        handles, labels = axs[0, 1].get_legend_handles_labels()

        # Remove duplicates while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels, strict=True):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)

        # Add legend outside the plot area (to the right, close to charts)
        if unique_handles:
            fig.legend(
                unique_handles,
                unique_labels,
                loc="center left",
                fontsize=8,
                bbox_to_anchor=(0.87, 0.5),
                framealpha=0.9,
                ncol=1,
            )

    def _save_figure(self, fig, path):
        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend/colorbar on right
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved figure to {path}")
        plt.close(fig)

    def _plot_denoising_steps_from_tracker(
        self, tracked_steps, xt_axs, vt_axs, corr_axs, x1t_axs, num_steps, add_labels=True
    ):
        """Plot denoising steps from tracker data.

        Args:
            tracked_steps: List of DebugStep objects containing debug steps
            xt_axs: Matplotlib axes for x_t plots (array of 6 axes)
            vt_axs: Matplotlib axes for v_t plots (array of 6 axes)
            corr_axs: Matplotlib axes for correction plots (array of 6 axes)
            x1t_axs: Matplotlib axes for x1_t plots (array of 6 axes)
            num_steps: Total number of denoising steps for colormap
            add_labels: Whether to add legend labels for the plots
        """

        logging.info("=" * 80)
        logging.info(f"Plotting {len(tracked_steps)} steps")

        debug_steps = tracked_steps
        if not debug_steps:
            return

        # Define colors for different denoise steps (using a colormap)
        colors = plt.cm.viridis(np.linspace(0, 1, num_steps))

        for step_idx, debug_step in enumerate(debug_steps):
            color = colors[step_idx % len(colors)]
            label = f"Step {step_idx}" if add_labels else None

            # Plot x_t
            if debug_step.x_t is not None:
                RTCDebugVisualizer.plot_waypoints(
                    xt_axs, debug_step.x_t, start_from=0, color=color, label=label
                )

            # Plot v_t
            if debug_step.v_t is not None:
                RTCDebugVisualizer.plot_waypoints(
                    vt_axs, debug_step.v_t, start_from=0, color=color, label=label
                )

            # Plot correction on separate axes
            if debug_step.correction is not None:
                RTCDebugVisualizer.plot_waypoints(
                    corr_axs,
                    debug_step.correction,
                    start_from=0,
                    color=color,
                    label=label,
                )

            # Plot x1_t (predicted state)
            if x1t_axs is not None and debug_step.x1_t is not None:
                x1t_label = f"x1_t Step {step_idx}" if add_labels else None
                RTCDebugVisualizer.plot_waypoints(
                    x1t_axs,
                    debug_step.x1_t,
                    start_from=0,
                    color=color,
                    label=x1t_label,
                )

            # Plot error in orange dashed
            if x1t_axs is not None and debug_step.err is not None:
                error_chunk = (
                    debug_step.err[0].cpu().numpy()
                    if len(debug_step.err.shape) == 3
                    else debug_step.err.cpu().numpy()
                )

                num_dims = min(error_chunk.shape[-1], 6)
                error_label = f"error Step {step_idx}" if add_labels else None
                for j in range(num_dims):
                    x1t_axs[j].plot(
                        np.arange(0, error_chunk.shape[0]),
                        error_chunk[:, j],
                        color="orange",
                        linestyle="--",
                        alpha=0.7,
                        label=error_label,
                    )

        # Recalculate axis limits after plotting to ensure proper scaling
        self._rescale_axes(xt_axs)
        self._rescale_axes(vt_axs)
        self._rescale_axes(corr_axs)
        self._rescale_axes(x1t_axs)

    def _plot_no_rtc_xt_reference(self, no_rtc_tracked_steps, xt_axs, num_steps):
        """Plot final no-RTC x_t data as orange dashed line on the RTC chart for comparison.

        Args:
            no_rtc_tracked_steps: List of DebugStep objects containing no-RTC debug steps
            xt_axs: Matplotlib axes for x_t plots (array of 6 axes, right column)
            num_steps: Total number of denoising steps for colormap
        """
        debug_steps = no_rtc_tracked_steps
        if not debug_steps:
            return

        # Plot only the final x_t step as orange dashed line
        final_step = debug_steps[-1]
        logging.info("Plotting final no-RTC x_t step as orange dashed reference")

        if final_step.x_t is not None:
            x_t_chunk = (
                final_step.x_t[0].cpu().numpy()
                if len(final_step.x_t.shape) == 3
                else final_step.x_t.cpu().numpy()
            )

            num_dims = min(x_t_chunk.shape[-1], 6)
            for j in range(num_dims):
                xt_axs[j].plot(
                    np.arange(0, x_t_chunk.shape[0]),
                    x_t_chunk[:, j],
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2,
                    label="No RTC (final)" if j == 0 else "",
                )

    def _rescale_axes(self, axes):
        """Rescale axes to show all data with proper margins.

        Args:
            axes: Array of matplotlib axes to rescale
        """
        for ax in axes:
            ax.relim()
            ax.autoscale_view()

            # Add 10% margin to y-axis for better visualization
            ylim = ax.get_ylim()
            y_range = ylim[1] - ylim[0]
            if y_range > 0:  # Avoid division by zero
                margin = y_range * 0.1
                ax.set_ylim(ylim[0] - margin, ylim[1] + margin)

            # Set x-axis ticks to show all integer values
            xlim = ax.get_xlim()
            max_len = int(xlim[1]) + 1
            if max_len > 0:
                ax.set_xticks(range(0, max_len, max(1, max_len // 20)))  # Show ~20 ticks
                ax.set_xlim(-0.5, max_len - 0.5)


@parser.wrap()
def main(cfg: RTCEvalConfig):
    """Main entry point for RTC evaluation."""
    # Set random seed for reproducibility
    set_seed(cfg.seed)

    init_logging()

    logging.info("=" * 80)
    logging.info("RTC Dataset Evaluation")
    logging.info(f"Config: {cfg}")
    logging.info("=" * 80)

    evaluator = RTCEvaluator(cfg)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
