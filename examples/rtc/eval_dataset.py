#!/usr/bin/env python

"""
Evaluate Real-Time Chunking (RTC) performance on dataset samples.

This script takes two random samples from a dataset:
- Uses actions from the first sample as previous chunk
- Generates new actions for the second sample with and without RTC

It compares action predictions with and without RTC on dataset samples,
measuring consistency and ground truth alignment.

Usage:
    python eval_dataset.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --dataset.repo_id=helper2424/check_rtc \
        --rtc.execution_horizon=8 \
        --device=mps
"""

import logging
import os
import random
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Ensure logs are flushed immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


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
    logger.info(f"Random seed set to: {seed}")


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
            max_guidance_weight=5.0,
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
            logger.info(f"Auto-detected device: {self.device}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class RTCEvaluator:
    """Evaluator for RTC on dataset samples."""

    def __init__(self, cfg: RTCEvalConfig):
        self.cfg = cfg
        self.device = cfg.device

        # Load policy
        logger.info(f"Loading policy from {cfg.policy.pretrained_path}")
        policy_class = get_policy_class(cfg.policy.type)
        self.policy = policy_class.from_pretrained(cfg.policy.pretrained_path)
        self.policy = self.policy.to(self.device)
        self.policy.eval()

        # Configure RTC
        cfg.rtc.enabled = True
        cfg.rtc.debug = True  # Enable debug tracking for visualization
        self.policy.config.rtc_config = cfg.rtc
        self.policy.init_rtc_processor()

        logger.info(f"Policy loaded: {self.policy.name}")
        logger.info(f"RTC enabled: {cfg.rtc.enabled}")
        logger.info(f"Execution horizon: {cfg.rtc.execution_horizon}")

        # Load dataset
        logger.info(f"Loading dataset: {cfg.dataset.repo_id}")
        self.dataset = LeRobotDataset(cfg.dataset.repo_id, delta_timestamps={"action": np.arange(50) / 30})
        logger.info(f"Dataset loaded: {len(self.dataset)} samples, {self.dataset.num_episodes} episodes")

        # Create preprocessor/postprocessor
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

    def run_evaluation(self):
        """Run evaluation on two random dataset samples."""
        # Create output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.cfg.output_dir}")

        logger.info("Starting RTC evaluation")
        logger.info(f"Inference delay: {self.cfg.inference_delay}")

        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True)
        loader_iter = iter(data_loader)
        first_sample = next(loader_iter)
        second_sample = next(loader_iter)

        preprocessed_first_sample = self.preprocessor(first_sample)
        preprocessed_second_sample = self.preprocessor(second_sample)

        # Don't postprocess the previous chunk
        prev_chunk_left_over = self.policy.predict_action_chunk(
            preprocessed_first_sample,
        )[:, :25, :].squeeze(0)

        # Sample noise (use same noise for both RTC and non-RTC for fair comparison)
        noise_size = (1, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        noise = self.policy.model.sample_noise(noise_size, self.device)
        noise_clone = noise.clone()

        # Generate actions WITHOUT RTC
        logger.info("Generating actions WITHOUT RTC")
        self.policy.config.rtc_config.enabled = False
        with torch.no_grad():
            _ = self.policy.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise,
            )

        no_rtc_tracked_steps = self.policy.rtc_processor.tracker.get_all_steps()
        self.policy.rtc_processor.reset_tracker()

        # Generate actions WITH RTC
        logger.info("Generating actions WITH RTC")
        self.policy.config.rtc_config.enabled = True
        with torch.no_grad():
            _ = self.policy.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise_clone,
                inference_delay=self.cfg.inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=self.cfg.rtc.execution_horizon,
            )

        # ================================================================

        rtc_tracked_steps = self.policy.rtc_processor.get_all_debug_steps()

        self.plot_tracked_data(rtc_tracked_steps, no_rtc_tracked_steps, prev_chunk_left_over)
        logger.info("Evaluation completed successfully")

    def plot_tracked_data(self, rtc_tracked_steps, no_rtc_tracked_steps, prev_chunk_left_over):
        # Create side-by-side figures for denoising visualization
        fig_xt, axs_xt = self._create_figure("x_t Denoising: No RTC (left) vs RTC (right)")
        fig_vt, axs_vt = self._create_figure("v_t Denoising: No RTC (left) vs RTC (right)")
        fig_x1t, axs_x1t = self._create_figure(
            "x1_t Predicted State & Error: No RTC (left - empty) vs RTC (right)"
        )

        num_steps = self.policy.config.num_steps
        self._plot_denoising_steps_from_tracker(
            rtc_tracked_steps,
            axs_xt[:, 1],  # Right column for x_t
            axs_vt[:, 1],  # Right column for v_t
            axs_x1t[:, 1],  # Right column for x1_t
            num_steps,
        )

        self._plot_denoising_steps_from_tracker(
            no_rtc_tracked_steps,
            axs_xt[:, 0],  # Left column for x_t
            axs_vt[:, 0],  # Left column for v_t
            axs_x1t[:, 0],  # Left column for x1_t
            num_steps,
        )

        # Plot ground truth on x_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_xt[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        RTCDebugVisualizer.plot_waypoints(
            axs_xt[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        # Plot ground truth on x1_t axes
        RTCDebugVisualizer.plot_waypoints(
            axs_x1t[:, 1], prev_chunk_left_over, start_from=0, color="red", label="Ground truth"
        )

        # Save denoising plots
        self._save_figure(fig_xt, os.path.join(self.cfg.output_dir, "denoising_xt_comparison.png"))
        self._save_figure(fig_vt, os.path.join(self.cfg.output_dir, "denoising_vt_comparison.png"))
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
        logger.info(f"Saved figure to {path}")
        plt.close(fig)

    def _plot_denoising_steps_from_tracker(self, tracked_steps, xt_axs, vt_axs, x1t_axs, num_steps):
        """Plot denoising steps from tracker data.

        Args:
            tracked_steps: List of DebugStep objects containing debug steps
            xt_axs: Matplotlib axes for x_t plots (array of 6 axes)
            vt_axs: Matplotlib axes for v_t plots (array of 6 axes)
            x1t_axs: Matplotlib axes for x1_t plots (array of 6 axes)
            num_steps: Total number of denoising steps for colormap
        """

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

            # Plot correction in red
            if debug_step.correction is not None:
                RTCDebugVisualizer.plot_waypoints(
                    vt_axs,
                    debug_step.correction,
                    start_from=0,
                    color="red",
                    label=f"Step corr {step_idx}",
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


@parser.wrap()
def main(cfg: RTCEvalConfig):
    """Main entry point for RTC evaluation."""
    # Set random seed for reproducibility
    set_seed(cfg.seed)

    logger.info("=" * 80)
    logger.info("RTC Dataset Evaluation")
    logger.info(f"Config: {cfg}")
    logger.info("=" * 80)

    evaluator = RTCEvaluator(cfg)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
