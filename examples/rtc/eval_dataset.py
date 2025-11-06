#!/usr/bin/env python

"""
Evaluate Real-Time Chunking (RTC) performance on dataset samples.

This script takes two random samples from a dataset:
- Uses actions from the first sample as previous chunk
- Generates new actions for the second sample with and without RTC

It compares action predictions with and without RTC on dataset samples,
measuring consistency and ground truth alignment.

Usage:
    # Basic usage
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=mps

    # With torch.compile for faster inference (PyTorch 2.0+)
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=mps \
        --use_torch_compile=true \
        --torch_compile_mode=max-autotune

    # With torch.compile for faster inference (PyTorch 2.0+)
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=cuda \
        --use_torch_compile=true \
        --torch_compile_mode=reduce-overhead

    # With custom compile settings
    uv run python examples/rtc/eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --use_torch_compile=true \
        --torch_compile_backend=inductor \
        --torch_compile_mode=max-autotune
"""

import gc
import logging
import os
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
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

        # Load dataset first (needed for preprocessor)
        logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
        self.dataset = LeRobotDataset(cfg.dataset.repo_id, delta_timestamps={"action": np.arange(50) / 30})
        logging.info(f"Dataset loaded: {len(self.dataset)} samples, {self.dataset.num_episodes} episodes")

        # Create preprocessor/postprocessor
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

        # Initialize three separate policy instances
        # Note: These policies are initialized here but will be freed sequentially during
        # evaluation to manage memory. Large models (e.g., VLAs with billions of parameters)
        # cannot fit three instances in memory simultaneously. Each policy is deleted and
        # memory is freed (via torch.cuda.empty_cache()) immediately after its use.
        logging.info("=" * 80)
        logging.info("Initializing three policy instances:")
        logging.info("  1. policy_prev_chunk (for generating previous chunk)")
        logging.info("  2. policy_no_rtc (for non-RTC inference)")
        logging.info("  3. policy_rtc (for RTC inference)")
        logging.info("  Note: Policies will be freed sequentially during evaluation to manage memory")
        logging.info("=" * 80)

        # Policy 1: For generating previous chunk (RTC disabled, no debug)
        self.policy_prev_chunk = self._init_policy(
            name="policy_prev_chunk",
            rtc_enabled=False,
            rtc_debug=False,
        )

        # Policy 2: For non-RTC inference (RTC disabled, debug enabled)
        self.policy_no_rtc = self._init_policy(
            name="policy_no_rtc",
            rtc_enabled=False,
            rtc_debug=True,
        )

        # Policy 3: For RTC inference (RTC enabled, debug enabled)
        self.policy_rtc = self._init_policy(
            name="policy_rtc",
            rtc_enabled=True,
            rtc_debug=True,
        )

        logging.info("=" * 80)
        logging.info("All policies initialized successfully")
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
        policy = policy_class.from_pretrained(self.cfg.policy.pretrained_path)
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

            # Compile the predict_action_chunk method
            original_method = policy.predict_action_chunk
            compiled_method = torch.compile(
                original_method,
                backend=self.cfg.torch_compile_backend,
                mode=self.cfg.torch_compile_mode,
            )
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

        # Step 1: Generate previous chunk using policy_prev_chunk
        # This policy is only used to generate the reference chunk and then freed
        logging.info("Step 1: Generating previous chunk with policy_prev_chunk")
        with torch.no_grad():
            prev_chunk_left_over = self.policy_prev_chunk.predict_action_chunk(
                preprocessed_first_sample,
            )[:, :25, :].squeeze(0)
        logging.info(f"  Generated prev_chunk shape: {prev_chunk_left_over.shape}")

        # Destroy policy_prev_chunk to free memory for large models
        self._destroy_policy(self.policy_prev_chunk, "policy_prev_chunk")

        # Sample noise (use same noise for both RTC and non-RTC for fair comparison)
        noise_size = (1, self.policy_no_rtc.config.chunk_size, self.policy_no_rtc.config.max_action_dim)
        noise = self.policy_no_rtc.model.sample_noise(noise_size, self.device)
        noise_clone = noise.clone()

        # Step 2: Generate actions WITHOUT RTC using policy_no_rtc
        logging.info("Step 2: Generating actions WITHOUT RTC with policy_no_rtc")
        self.policy_no_rtc.rtc_processor.reset_tracker()
        with torch.no_grad():
            _ = self.policy_no_rtc.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise,
            )
        no_rtc_tracked_steps = self.policy_no_rtc.rtc_processor.tracker.get_all_steps()
        logging.info(f"  Tracked {len(no_rtc_tracked_steps)} steps without RTC")

        # Destroy policy_no_rtc to free memory before loading policy_rtc
        self._destroy_policy(self.policy_no_rtc, "policy_no_rtc")

        # Step 3: Generate actions WITH RTC using policy_rtc
        logging.info("Step 3: Generating actions WITH RTC with policy_rtc")
        self.policy_rtc.rtc_processor.reset_tracker()
        with torch.no_grad():
            _ = self.policy_rtc.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise_clone,
                inference_delay=self.cfg.inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=self.cfg.rtc.execution_horizon,
            )
        rtc_tracked_steps = self.policy_rtc.rtc_processor.get_all_debug_steps()
        logging.info(f"  Tracked {len(rtc_tracked_steps)} steps with RTC")

        # Save num_steps before destroying policy (needed for plotting)
        num_steps = self.policy_rtc.config.num_steps

        # Destroy policy_rtc after final use
        self._destroy_policy(self.policy_rtc, "policy_rtc")

        # Plot and save results
        logging.info("=" * 80)
        logging.info("Plotting results...")
        self.plot_tracked_data(rtc_tracked_steps, no_rtc_tracked_steps, prev_chunk_left_over, num_steps)
        logging.info("=" * 80)
        logging.info("Evaluation completed successfully")

    def plot_tracked_data(self, rtc_tracked_steps, no_rtc_tracked_steps, prev_chunk_left_over, num_steps):
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
        )

        self._plot_denoising_steps_from_tracker(
            no_rtc_tracked_steps,
            axs_xt[:, 0],  # Left column for x_t
            axs_vt[:, 0],  # Left column for v_t
            axs_corr[:, 0],  # Left column for correction
            axs_x1t[:, 0],  # Left column for x1_t
            num_steps,
        )

        # Plot ground truth on x_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_xt[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        # Plot ground truth on x1_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_x1t[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        # Plot ground truth on x_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_xt[:, 0], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        RTCDebugVisualizer.plot_waypoints(
            axs_x1t[:, 0], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

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

    def _save_figure(self, fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        logging.info(f"Saved figure to {path}")
        plt.close(fig)

    def _plot_denoising_steps_from_tracker(self, tracked_steps, xt_axs, vt_axs, corr_axs, x1t_axs, num_steps):
        """Plot denoising steps from tracker data.

        Args:
            tracked_steps: List of DebugStep objects containing debug steps
            xt_axs: Matplotlib axes for x_t plots (array of 6 axes)
            vt_axs: Matplotlib axes for v_t plots (array of 6 axes)
            corr_axs: Matplotlib axes for correction plots (array of 6 axes)
            x1t_axs: Matplotlib axes for x1_t plots (array of 6 axes)
            num_steps: Total number of denoising steps for colormap
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

            # Plot x_t
            if debug_step.x_t is not None:
                RTCDebugVisualizer.plot_waypoints(
                    xt_axs, debug_step.x_t, start_from=0, color=color, label=f"Step {step_idx}"
                )

            # Plot v_t
            if debug_step.v_t is not None:
                RTCDebugVisualizer.plot_waypoints(
                    vt_axs, debug_step.v_t, start_from=0, color=color, label=f"Step {step_idx}"
                )

            # Plot correction on separate axes
            if debug_step.correction is not None:
                RTCDebugVisualizer.plot_waypoints(
                    corr_axs,
                    debug_step.correction,
                    start_from=0,
                    color=color,
                    label=f"Step {step_idx}",
                )

            # Plot x1_t (predicted state)
            if x1t_axs is not None and debug_step.x1_t is not None:
                RTCDebugVisualizer.plot_waypoints(
                    x1t_axs,
                    debug_step.x1_t,
                    start_from=0,
                    color=color,
                    label=f"x1_t Step {step_idx}",
                )

            # Plot error in orange dashed
            if x1t_axs is not None and debug_step.err is not None:
                error_chunk = (
                    debug_step.err[0].cpu().numpy()
                    if len(debug_step.err.shape) == 3
                    else debug_step.err.cpu().numpy()
                )

                num_dims = min(error_chunk.shape[-1], 6)
                for j in range(num_dims):
                    x1t_axs[j].plot(
                        np.arange(0, error_chunk.shape[0]),
                        error_chunk[:, j],
                        color="orange",
                        linestyle="--",
                        alpha=0.7,
                        label=f"error Step {step_idx}",
                    )

        # Recalculate axis limits after plotting to ensure proper scaling
        self._rescale_axes(xt_axs)
        self._rescale_axes(vt_axs)
        self._rescale_axes(corr_axs)
        self._rescale_axes(x1t_axs)

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
