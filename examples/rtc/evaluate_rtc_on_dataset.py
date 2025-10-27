#!/usr/bin/env python

"""
Evaluate Real-Time Chunking (RTC) performance on a dataset.

This script compares action predictions with and without RTC on dataset samples,
measuring consistency and ground truth alignment.

Usage:
    # Basic evaluation
    python evaluate_rtc_on_dataset.py \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=lerobot/pusht \
        --num_iterations=100

    # With custom RTC parameters
    python evaluate_rtc_on_dataset.py \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=lerobot/pusht \
        --num_iterations=100 \
        --skip_steps=3 \
        --rtc.execution_horizon=8

    # Save results to file
    python evaluate_rtc_on_dataset.py \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=lerobot/pusht \
        --num_iterations=100 \
        --output_path=rtc_eval_results.json

    # With custom seed for reproducibility
    python evaluate_rtc_on_dataset.py \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=lerobot/pusht \
        --num_iterations=100 \
        --seed=123
"""

import logging
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
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
    # Set deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


def tensor_stats_str(tensor: Tensor | None, name: str = "tensor") -> str:
    """Generate readable statistics string for a tensor."""
    if tensor is None:
        return f"{name}: None"

    stats = (
        f"{name}:\n"
        f"  shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}\n"
        f"  min={tensor.min().item():.6f}, max={tensor.max().item():.6f}\n"
        f"  mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
    )
    return stats


def compute_mse(pred: Tensor, target: Tensor) -> float:
    """Compute mean squared error between prediction and target."""
    return torch.nn.functional.mse_loss(pred, target).item()


def compute_consistency_metrics(
    rtc_actions: Tensor,
    prev_actions: Tensor | None,
    execution_horizon: int,
) -> dict[str, float]:
    """Compute consistency metrics for RTC actions.

    Measures how well RTC maintains consistency with previous actions
    in the overlap region (prefix).

    Args:
        rtc_actions: Actions generated with RTC (batch, time, action_dim)
        prev_actions: Previous action chunk (time, action_dim) or (batch, time, action_dim)
        execution_horizon: Number of steps where consistency is expected

    Returns:
        Dictionary with consistency metrics
    """
    metrics = {}

    if prev_actions is None:
        metrics["prefix_mse"] = 0.0
        metrics["prefix_max_error"] = 0.0
        return metrics

    # Ensure batch dimension
    if len(prev_actions.shape) == 2:
        prev_actions = prev_actions.unsqueeze(0)

    # Get overlap region
    overlap_size = min(execution_horizon, prev_actions.shape[1], rtc_actions.shape[1])

    rtc_prefix = rtc_actions[:, :overlap_size, :]
    prev_prefix = prev_actions[:, :overlap_size, :]

    # Pad if needed
    if prev_prefix.shape != rtc_prefix.shape:
        if prev_prefix.shape[2] < rtc_prefix.shape[2]:
            # Pad action dimension
            pad_size = rtc_prefix.shape[2] - prev_prefix.shape[2]
            prev_prefix = torch.nn.functional.pad(prev_prefix, (0, pad_size))
        elif prev_prefix.shape[2] > rtc_prefix.shape[2]:
            # Truncate
            prev_prefix = prev_prefix[:, :, : rtc_prefix.shape[2]]

    diff = rtc_prefix - prev_prefix
    metrics["prefix_mse"] = torch.nn.functional.mse_loss(rtc_prefix, prev_prefix).item()
    metrics["prefix_max_error"] = diff.abs().max().item()
    metrics["prefix_mean_error"] = diff.abs().mean().item()

    return metrics


@dataclass
class RTCDatasetEvalConfig(HubMixin):
    """Configuration for RTC dataset evaluation."""

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
        )
    )

    # Evaluation parameters
    num_iterations: int = field(
        default=100,
        metadata={"help": "Number of dataset samples to evaluate"},
    )
    skip_steps: int = field(
        default=1,
        metadata={
            "help": "Number of steps to skip between inferences (simulates inference delay). "
            "skip_steps=1 means evaluate every step, skip_steps=5 means evaluate every 5th step."
        },
    )
    start_episode: int = field(
        default=0,
        metadata={"help": "Episode index to start evaluation from"},
    )

    # Device configuration
    device: str | None = field(
        default=None,
        metadata={"help": "Device to run on (cuda, cpu, mps, auto)"},
    )

    # Output configuration
    output_path: str | None = field(
        default=None,
        metadata={"help": "Path to save evaluation results (JSON format)"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose logging for each sample"},
    )

    # Seed configuration
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )

    inference_delay: int = field(
        default=4,
        metadata={
            "help": "Number of steps to skip between inferences (simulates inference delay). "
            "inference_delay=1 means evaluate every step, inference_delay=5 means evaluate every 5th step."
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
            logger.info(f"Auto-detected device: {self.device}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


class RTCDatasetEvaluator:
    """Evaluator for RTC on dataset samples."""

    def __init__(self, cfg: RTCDatasetEvalConfig):
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
        self.policy.init_rtc_processor(verbose=cfg.verbose)

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

    def run_evaluation(self) -> dict:
        """Run full evaluation on dataset.

        Returns:
            Dictionary with aggregated metrics and detailed results
        """
        logger.info(f"Starting evaluation on {self.cfg.num_iterations} samples")
        logger.info(f"Skip steps (inference delay simulation): {self.cfg.skip_steps}")
        logger.info(f"Inference delay: {self.cfg.inference_delay}")

        batch_size = 1
        prev_actions_chunk = None

        # Determine actual inference delay based on skip_steps
        inference_delay = self.cfg.inference_delay

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        inference_times = 0
        last_inference_i = 0

        prev_chunk_left_over = None

        for i, batch in enumerate(dataloader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(self.device)
                else:
                    batch[key] = value

            # Preprocess observations
            preprocessed_batch = self.preprocessor(batch)

            # Sample noise (use same noise for both RTC and non-RTC for fair comparison)
            noise_size = (1, self.policy.config.chunk_size, self.policy.config.max_action_dim)
            noise = self.policy.model.sample_noise(noise_size, self.device)
            noise_clone = noise.clone()

            if i % 2 == 0:
                prev_chunk_left_over = preprocessed_batch["action"][0, :, :25]
                continue

            # Create side-by-side figures for denoising visualization
            # x_t denoising: No RTC (left) vs RTC (right)
            fig_xt, axs_xt = plt.subplots(6, 2, figsize=(24, 12))
            fig_xt.suptitle("x_t Denoising: No RTC (left) vs RTC (right)", fontsize=16)

            # v_t denoising: No RTC (left) vs RTC (right)
            fig_vt, axs_vt = plt.subplots(6, 2, figsize=(24, 12))
            fig_vt.suptitle("v_t Denoising: No RTC (left) vs RTC (right)", fontsize=16)

            # x1_t denoising: Only RTC has this (left empty, right shows x1_t + error)
            fig_x1t, axs_x1t = plt.subplots(6, 2, figsize=(24, 12))
            fig_x1t.suptitle(
                "x1_t Predicted State & Error: No RTC (left - empty) vs RTC (right)", fontsize=16
            )

            # Generate actions WITHOUT RTC (plot on left column)
            self.policy.config.rtc_config.enabled = False
            with torch.no_grad():
                no_rtc_actions = self.policy.predict_action_chunk(
                    preprocessed_batch,
                    noise=noise,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_chunk_left_over,
                    viz_xt_axs=axs_xt[:, 0],  # Left column for x_t
                    viz_vt_axs=axs_vt[:, 0],  # Left column for v_t
                )

            # Generate actions WITH RTC (plot on right column)
            self.policy.config.rtc_config.enabled = True
            with torch.no_grad():
                rtc_actions = self.policy.predict_action_chunk(
                    preprocessed_batch,
                    noise=noise_clone,
                    inference_delay=inference_delay,
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
            fig_xt.savefig("denoising_xt_comparison.png", dpi=150)
            logger.info("Saved x_t denoising comparison to denoising_xt_comparison.png")
            plt.close(fig_xt)

            fig_vt.tight_layout()
            fig_vt.savefig("denoising_vt_comparison.png", dpi=150)
            logger.info("Saved v_t denoising comparison to denoising_vt_comparison.png")
            plt.close(fig_vt)

            fig_x1t.tight_layout()
            fig_x1t.savefig("denoising_x1t_comparison.png", dpi=150)
            logger.info("Saved x1_t predicted state & error comparison to denoising_x1t_comparison.png")
            plt.close(fig_x1t)

            # Create side-by-side comparison: No RTC (left) vs RTC (right)
            fig, axs = plt.subplots(6, 2, figsize=(24, 12))
            fig.suptitle("Final Action Comparison: No RTC (left) vs RTC (right)", fontsize=16)

            # Plot on left column (No RTC)
            self.axs = axs[:, 0]
            self.plot_waypoints(prev_chunk_left_over[0].cpu().numpy(), label="Ground Truth", color="green")
            self.plot_waypoints(no_rtc_actions[0].cpu().numpy(), label="No RTC", color="blue")
            for ax in self.axs:
                ax.set_title("No RTC", fontsize=12)

            # Plot on right column (RTC)
            self.axs = axs[:, 1]
            self.plot_waypoints(prev_chunk_left_over[0].cpu().numpy(), label="Ground Truth", color="green")
            self.plot_waypoints(rtc_actions[0].detach().cpu().numpy(), label="RTC", color="red")
            for ax in self.axs:
                ax.set_title("RTC", fontsize=12)

            plt.tight_layout()
            plt.savefig("final_actions_comparison.png", dpi=150)
            logger.info("Saved final actions comparison to final_actions_comparison.png")
            plt.close(fig)

            break

            prev_actions_chunk = rtc_actions
            inference_times += 1
            last_inference_i += i

            if inference_times >= self.cfg.num_iterations:
                break

        return

    def plot_waypoints(self, chunk, start_from: int = 0, color: str | None = None, label: str | None = None):
        for j in range(chunk.shape[-1]):
            self.axs[j].plot(
                np.arange(start_from, start_from + chunk.shape[0]),
                chunk[:, j],
                color=color,
                label=label,
            )
            self.axs[j].set_ylabel("Joint angle", fontsize=14)
            self.axs[j].grid()
            plt.tick_params(labelsize=14)
            self.axs[j].legend(loc="upper right", fontsize=14)
            if j == 2:
                self.axs[j].set_xlabel("Step #", fontsize=16)


@parser.wrap()
def main(cfg: RTCDatasetEvalConfig):
    """Main entry point for RTC dataset evaluation."""
    # Set random seed for reproducibility
    set_seed(cfg.seed)

    logger.info("=" * 80)
    logger.info("RTC Dataset Evaluation")
    logger.info("=" * 80)
    logger.info(f"Policy: {cfg.policy.pretrained_path}")
    logger.info(f"Dataset: {cfg.dataset.repo_id}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Num iterations: {cfg.num_iterations}")
    logger.info(f"Skip steps: {cfg.skip_steps}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info("=" * 80)

    evaluator = RTCDatasetEvaluator(cfg)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
