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
)
logger = logging.getLogger(__name__)


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

        # Create side-by-side figures for denoising visualization
        fig_xt, axs_xt = plt.subplots(6, 2, figsize=(24, 12))
        fig_xt.suptitle("x_t Denoising: No RTC (left) vs RTC (right)", fontsize=16)

        fig_vt, axs_vt = plt.subplots(6, 2, figsize=(24, 12))
        fig_vt.suptitle("v_t Denoising: No RTC (left) vs RTC (right)", fontsize=16)

        fig_x1t, axs_x1t = plt.subplots(6, 2, figsize=(24, 12))
        fig_x1t.suptitle("x1_t Predicted State & Error: No RTC (left - empty) vs RTC (right)", fontsize=16)

        # Generate actions WITHOUT RTC (plot on left column)
        logger.info("Generating actions WITHOUT RTC")
        self.policy.config.rtc_config.enabled = False
        with torch.no_grad():
            no_rtc_actions = self.policy.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise,
                viz_xt_axs=axs_xt[:, 0],  # Left column for x_t
                viz_vt_axs=axs_vt[:, 0],  # Left column for v_t
            )

        # Generate actions WITH RTC (plot on right column)
        logger.info("Generating actions WITH RTC")
        self.policy.config.rtc_config.enabled = True
        with torch.no_grad():
            rtc_actions = self.policy.predict_action_chunk(
                preprocessed_second_sample,
                noise=noise_clone,
                inference_delay=self.cfg.inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=self.cfg.rtc.execution_horizon,
                viz_xt_axs=axs_xt[:, 1],  # Right column for x_t
                viz_vt_axs=axs_vt[:, 1],  # Right column for v_t
                viz_x1t_axs=axs_x1t[:, 1],  # Right column for x1_t
            )

        # Set titles for denoising plots
        for ax in axs_xt[:, 0]:
            ax.set_title("No RTC" if ax == axs_xt[0, 0] else "", fontsize=12)
        for ax in axs_xt[:, 1]:
            ax.set_title("RTC" if ax == axs_xt[0, 1] else "", fontsize=12)

        for ax in axs_vt[:, 0]:
            ax.set_title("No RTC" if ax == axs_vt[0, 0] else "", fontsize=12)
        for ax in axs_vt[:, 1]:
            ax.set_title("RTC" if ax == axs_vt[0, 1] else "", fontsize=12)

        for ax in axs_x1t[:, 0]:
            ax.set_title("No RTC (N/A)" if ax == axs_x1t[0, 0] else "", fontsize=12)
        for ax in axs_x1t[:, 1]:
            ax.set_title("RTC" if ax == axs_x1t[0, 1] else "", fontsize=12)

        # Save denoising plots
        fig_xt.tight_layout()
        xt_path = os.path.join(self.cfg.output_dir, "denoising_xt_comparison.png")
        fig_xt.savefig(xt_path, dpi=150)
        logger.info(f"Saved x_t denoising comparison to {xt_path}")
        plt.close(fig_xt)

        fig_vt.tight_layout()
        vt_path = os.path.join(self.cfg.output_dir, "denoising_vt_comparison.png")
        fig_vt.savefig(vt_path, dpi=150)
        logger.info(f"Saved v_t denoising comparison to {vt_path}")
        plt.close(fig_vt)

        fig_x1t.tight_layout()
        x1t_path = os.path.join(self.cfg.output_dir, "denoising_x1t_comparison.png")
        fig_x1t.savefig(x1t_path, dpi=150)
        logger.info(f"Saved x1_t predicted state & error comparison to {x1t_path}")
        plt.close(fig_x1t)

        # Create side-by-side comparison: No RTC (left) vs RTC (right)
        fig, axs = plt.subplots(6, 2, figsize=(24, 12))
        fig.suptitle("Final Action Comparison: No RTC (left) vs RTC (right)", fontsize=16)

        # Plot on left column (No RTC)
        self._plot_actions(
            axs[:, 0],
            prev_chunk_left_over[0].cpu().numpy(),
            no_rtc_actions[0].cpu().numpy(),
            "No RTC",
        )

        # Plot on right column (RTC)
        self._plot_actions(
            axs[:, 1],
            prev_chunk_left_over[0].cpu().numpy(),
            rtc_actions[0].detach().cpu().numpy(),
            "RTC",
        )

        plt.tight_layout()
        final_path = os.path.join(self.cfg.output_dir, "final_actions_comparison.png")
        plt.savefig(final_path, dpi=150)
        logger.info(f"Saved final actions comparison to {final_path}")
        plt.close(fig)

        # Visualize debug information if enabled
        self._visualize_debug_info()

        logger.info("Evaluation completed successfully")

    def _plot_actions(self, axs, prev_chunk, predicted_actions, title):
        """Plot actions comparison on given axes."""
        # Ensure arrays are 2D
        if prev_chunk.ndim == 1:
            prev_chunk = prev_chunk.reshape(1, -1)
        if predicted_actions.ndim == 1:
            predicted_actions = predicted_actions.reshape(1, -1)

        for j in range(min(prev_chunk.shape[-1], 6)):  # Limit to 6 dimensions
            axs[j].plot(
                np.arange(prev_chunk.shape[0]),
                prev_chunk[:, j],
                color="green",
                label="Previous Chunk",
            )
            axs[j].plot(
                np.arange(predicted_actions.shape[0]),
                predicted_actions[:, j],
                color="red" if "RTC" in title else "blue",
                label=title,
            )
            axs[j].set_ylabel("Joint angle", fontsize=14)
            axs[j].grid()
            axs[j].legend(loc="upper right", fontsize=14)
            axs[j].set_title(title if j == 0 else "", fontsize=12)
            if j == 2:
                axs[j].set_xlabel("Step #", fontsize=16)

    def _visualize_debug_info(self):
        """Visualize debug information from the RTC processor."""
        # Use proxy method to check if debug is enabled
        if not self.policy.rtc_processor.is_debug_enabled():
            logger.warning("Debug tracking is disabled. Skipping debug visualization.")
            return

        # Get tracker length using proxy method
        if self.policy.rtc_processor.get_tracker_length() == 0:
            logger.warning("No debug steps recorded. Skipping debug visualization.")
            return

        # Create output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"Saving debug visualizations to {self.cfg.output_dir}")

        # Still need direct access to tracker for visualization functions
        # This is acceptable since RTCDebugVisualizer is part of the RTC package
        tracker = self.policy.rtc_processor.tracker

        # Print statistics
        RTCDebugVisualizer.print_debug_statistics(tracker)

        # Plot debug summary
        summary_path = os.path.join(self.cfg.output_dir, "debug_summary.png")
        RTCDebugVisualizer.plot_debug_summary(
            tracker,
            save_path=summary_path,
            show=False,
        )

        # Plot correction heatmap
        heatmap_path = os.path.join(self.cfg.output_dir, "correction_heatmap.png")
        RTCDebugVisualizer.plot_correction_heatmap(
            tracker,
            save_path=heatmap_path,
            show=False,
        )

        # Plot step-by-step comparison (last step)
        step_path = os.path.join(self.cfg.output_dir, "step_comparison_last.png")
        RTCDebugVisualizer.plot_step_by_step_comparison(
            tracker,
            step_idx=-1,
            save_path=step_path,
            show=False,
        )

        # Plot step-by-step comparison (first step)
        step_path_first = os.path.join(self.cfg.output_dir, "step_comparison_first.png")
        if self.policy.rtc_processor.get_tracker_length() > 0:
            RTCDebugVisualizer.plot_step_by_step_comparison(
                tracker,
                step_idx=0,
                save_path=step_path_first,
                show=False,
            )

        logger.info(f"Debug visualizations saved to {self.cfg.output_dir}")


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
