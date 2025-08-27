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
Evaluation metrics for RLearn (Video-Language Conditioned Reward Model).

Key metrics:
1. VOC-S (Value-Order Correlation for Success): Spearman correlation between frame indices and predicted rewards
2. Success vs Failure Detection: Model's ability to distinguish between correct and incorrect language conditions
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor
from tqdm import tqdm

from lerobot.constants import OBS_IMAGES, OBS_LANGUAGE


def compute_voc_s(
    predicted_rewards: list[np.ndarray], use_interquartile_mean: bool = True
) -> dict[str, float]:
    """
    Compute Value-Order Correlation for Success (VOC-S).

    Measures whether per-frame rewards increase as successful execution unfolds.
    For each episode, computes Spearman correlation between frame indices [1..T]
    and predicted rewards [r1..rT].

    Args:
        predicted_rewards: List of reward arrays, one per episode. Each array has shape (T,)
        use_interquartile_mean: If True, use IQM instead of mean for aggregation

    Returns:
        Dictionary with VOC-S metrics:
        - voc_s_mean: Mean Spearman correlation across episodes
        - voc_s_std: Standard deviation of correlations
        - voc_s_iqm: Interquartile mean (if use_interquartile_mean=True)
        - num_episodes: Number of episodes evaluated
        - correlations: Individual correlations per episode
    """
    if not predicted_rewards:
        return {"voc_s_mean": 0.0, "voc_s_std": 0.0, "voc_s_iqm": 0.0, "num_episodes": 0, "correlations": []}

    correlations = []

    for episode_rewards in predicted_rewards:
        if len(episode_rewards) < 2:
            # Need at least 2 points for correlation
            continue

        # Frame indices: [1, 2, ..., T]
        frame_indices = np.arange(1, len(episode_rewards) + 1)

        # Compute Spearman correlation
        try:
            correlation, p_value = spearmanr(frame_indices, episode_rewards)

            # Handle NaN correlations (e.g., all rewards are identical)
            if np.isnan(correlation):
                correlation = 0.0

            correlations.append(correlation)

        except Exception as e:
            warnings.warn(f"Failed to compute correlation for episode: {e}")
            correlations.append(0.0)

    if not correlations:
        return {"voc_s_mean": 0.0, "voc_s_std": 0.0, "voc_s_iqm": 0.0, "num_episodes": 0, "correlations": []}

    correlations = np.array(correlations)

    # Compute statistics
    voc_s_mean = float(np.mean(correlations))
    voc_s_std = float(np.std(correlations))

    # Interquartile mean: mean of values between 25th and 75th percentiles
    if use_interquartile_mean and len(correlations) >= 4:
        q25, q75 = np.percentile(correlations, [25, 75])
        iqm_mask = (correlations >= q25) & (correlations <= q75)
        voc_s_iqm = float(np.mean(correlations[iqm_mask]))
    else:
        voc_s_iqm = voc_s_mean

    return {
        "voc_s_mean": voc_s_mean,
        "voc_s_std": voc_s_std,
        "voc_s_iqm": voc_s_iqm,
        "num_episodes": len(correlations),
        "correlations": correlations.tolist(),
    }


def compute_success_failure_detection(
    correct_rewards: list[np.ndarray], incorrect_rewards: list[np.ndarray], threshold_percentile: float = 50.0
) -> dict[str, float]:
    """
    Compute success vs failure detection accuracy.

    Tests the model's ability to distinguish between correct and incorrect language conditions.
    For each episode, compares final reward under correct vs incorrect language instruction.

    Args:
        correct_rewards: List of reward arrays for episodes with correct language
        incorrect_rewards: List of reward arrays for episodes with incorrect/mismatched language
        threshold_percentile: Percentile of correct rewards to use as threshold

    Returns:
        Dictionary with detection metrics:
        - detection_accuracy: Fraction of episodes where correct > incorrect
        - mean_correct_final: Mean final reward for correct language
        - mean_incorrect_final: Mean final reward for incorrect language
        - separation_score: (mean_correct - mean_incorrect) / (std_correct + std_incorrect)
        - num_pairs: Number of episode pairs evaluated
    """
    if len(correct_rewards) != len(incorrect_rewards):
        raise ValueError("Must have same number of correct and incorrect reward sequences")

    if not correct_rewards:
        return {
            "detection_accuracy": 0.0,
            "mean_correct_final": 0.0,
            "mean_incorrect_final": 0.0,
            "separation_score": 0.0,
            "num_pairs": 0,
        }

    # Extract final rewards (last timestep of each episode)
    correct_finals = []
    incorrect_finals = []

    for correct_ep, incorrect_ep in zip(correct_rewards, incorrect_rewards, strict=False):
        if len(correct_ep) > 0 and len(incorrect_ep) > 0:
            correct_finals.append(correct_ep[-1])  # Final reward
            incorrect_finals.append(incorrect_ep[-1])  # Final reward

    if not correct_finals:
        return {
            "detection_accuracy": 0.0,
            "mean_correct_final": 0.0,
            "mean_incorrect_final": 0.0,
            "separation_score": 0.0,
            "num_pairs": 0,
        }

    correct_finals = np.array(correct_finals)
    incorrect_finals = np.array(incorrect_finals)

    # Detection accuracy: fraction where correct > incorrect
    detection_accuracy = float(np.mean(correct_finals > incorrect_finals))

    # Statistics
    mean_correct = float(np.mean(correct_finals))
    mean_incorrect = float(np.mean(incorrect_finals))
    std_correct = float(np.std(correct_finals))
    std_incorrect = float(np.std(incorrect_finals))

    # Separation score: normalized difference (clamp to prevent extreme values)
    denominator = std_correct + std_incorrect
    if denominator > 1e-6:  # Prevent division by very small numbers
        separation_score = (mean_correct - mean_incorrect) / denominator
        # Clamp to reasonable range
        separation_score = np.clip(separation_score, -100.0, 100.0)
    else:
        separation_score = 0.0

    return {
        "detection_accuracy": detection_accuracy,
        "mean_correct_final": mean_correct,
        "mean_incorrect_final": mean_incorrect,
        "separation_score": float(separation_score),
        "num_pairs": len(correct_finals),
    }


def generate_mismatched_languages(
    original_languages: list[str], mismatch_templates: list[str] | None = None
) -> list[str]:
    """
    Generate mismatched language instructions for failure detection evaluation.

    Args:
        original_languages: List of original task descriptions
        mismatch_templates: Custom mismatch templates. If None, uses defaults.

    Returns:
        List of mismatched language instructions
    """
    if mismatch_templates is None:
        mismatch_templates = ["kick the ball", "walk to the red shoes", "wave", "do nothing"]

    # For each original language, pick a random mismatch
    mismatched = []
    np.random.seed(42)  # For reproducibility

    for i, orig_lang in enumerate(original_languages):
        # Use modulo to cycle through mismatches if we have more episodes than templates
        mismatch_idx = i % len(mismatch_templates)
        mismatched.append(mismatch_templates[mismatch_idx])

    return mismatched


class RLearnEvaluator:
    """
    Comprehensive evaluator for RLearN reward models.

    Provides methods to evaluate VOC-S and success/failure detection on datasets.
    """

    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: RLearN model instance
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict_episode_rewards(self, frames: Tensor, language: str, batch_size: int = 16) -> np.ndarray:
        """
        Predict rewards for a single episode.

        Args:
            frames: Video frames tensor of shape (T, C, H, W)
            language: Language instruction string
            batch_size: Maximum sequence length to process at once

        Returns:
            Predicted rewards array of shape (T,)
        """
        T = frames.shape[0]

        # Preprocess frames to match model expectations
        processed_frames = self._preprocess_frames(frames)

        # Process in chunks if episode is very long
        if T <= batch_size:
            # Single batch
            batch = {
                OBS_IMAGES: processed_frames.unsqueeze(0).to(self.device),  # (1, T, C, H, W)
                OBS_LANGUAGE: [language],
            }

            # Use the new predict_rewards method
            values = self.model.predict_rewards(batch)  # (1, T')
            rewards = values.squeeze(0).cpu().numpy()  # (T',)

        else:
            # Process in overlapping chunks to handle very long episodes
            rewards = []
            stride = batch_size // 2  # 50% overlap

            for i in range(0, T, stride):
                end_idx = min(i + batch_size, T)
                chunk_frames = processed_frames[i:end_idx]

                batch = {OBS_IMAGES: chunk_frames.unsqueeze(0).to(self.device), OBS_LANGUAGE: [language]}

                chunk_values = self.model.predict_rewards(batch)
                chunk_rewards = chunk_values.squeeze(0).cpu().numpy()

                # For overlapping chunks, only take the first half (except for the last chunk)
                if i + batch_size < T:
                    rewards.extend(chunk_rewards[:stride])
                else:
                    rewards.extend(chunk_rewards)

            rewards = np.array(rewards[:T])  # Ensure exact length

        return rewards

    def _preprocess_frames(self, frames: Tensor) -> Tensor:
        """
        Preprocess frames to match model expectations.

        Args:
            frames: Input frames tensor of shape (T, C, H, W)

        Returns:
            Preprocessed frames tensor of shape (T, C, H', W')
        """
        import torch.nn.functional as F

        T, C, H, W = frames.shape

        # Expected input size for SigLIP2 is typically 256x256
        target_size = 256

        # Resize frames if needed
        if H != target_size or W != target_size:
            # Resize using bilinear interpolation
            frames = F.interpolate(
                frames, size=(target_size, target_size), mode="bilinear", align_corners=False
            )

        # Normalize to [0, 1] if needed
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0

        # Ensure values are in [0, 1] range
        frames = torch.clamp(frames, 0.0, 1.0)

        return frames

    def evaluate_voc_s(
        self, dataset, num_episodes: int = 100, use_interquartile_mean: bool = True
    ) -> dict[str, Any]:
        """
        Evaluate VOC-S on a dataset.

        Args:
            dataset: LeRobot dataset instance
            num_episodes: Number of episodes to evaluate (randomly sampled)
            use_interquartile_mean: Whether to compute IQM

        Returns:
            VOC-S evaluation results
        """
        print(f"Evaluating VOC-S on {num_episodes} episodes...")

        # Sample episodes
        total_episodes = dataset.num_episodes
        if num_episodes >= total_episodes:
            episode_indices = list(range(total_episodes))
        else:
            np.random.seed(42)
            episode_indices = np.random.choice(total_episodes, num_episodes, replace=False)

        predicted_rewards = []

        for ep_idx in tqdm(episode_indices, desc="Computing VOC-S"):
            try:
                # Get episode data
                ep_start = dataset.episode_data_index["from"][ep_idx].item()
                ep_end = dataset.episode_data_index["to"][ep_idx].item()
                episode_length = ep_end - ep_start

                # Get frames and language for this episode
                frames = []
                language = None

                for frame_idx in range(episode_length):
                    global_idx = ep_start + frame_idx
                    frame_data = dataset[global_idx]

                    # Extract image (assuming single camera for now)
                    if OBS_IMAGES in frame_data:
                        img = frame_data[OBS_IMAGES]
                    else:
                        # Try to find image key
                        img_keys = [k for k in frame_data.keys() if "image" in k.lower()]
                        if img_keys:
                            img = frame_data[img_keys[0]]
                        else:
                            continue

                    # Convert to tensor if needed
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)

                    # Ensure CHW format
                    if len(img.shape) == 3 and img.shape[-1] in [1, 3, 4]:
                        img = img.permute(2, 0, 1)  # HWC -> CHW

                    # Resize to expected input size (256x256 for SigLIP2) BEFORE stacking
                    if img.shape[-2:] != (256, 256):
                        import torch.nn.functional as F

                        img = F.interpolate(
                            img.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
                        ).squeeze(0)

                    # Normalize to [0, 1] if needed
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0

                    frames.append(img)

                    # Get language instruction
                    if language is None:
                        if OBS_LANGUAGE in frame_data:
                            language = frame_data[OBS_LANGUAGE]
                            if isinstance(language, list):
                                language = language[0]
                        elif "task" in frame_data:
                            language = frame_data["task"]
                        else:
                            language = ""  # Default empty language

                if not frames:
                    continue

                # Stack frames into video tensor
                frames_tensor = torch.stack(frames)  # (T, C, H, W)

                # Predict rewards
                episode_rewards = self.predict_episode_rewards(frames_tensor, language)
                predicted_rewards.append(episode_rewards)

            except Exception as e:
                warnings.warn(f"Failed to process episode {ep_idx}: {e}")
                continue

        # Compute VOC-S
        voc_results = compute_voc_s(predicted_rewards, use_interquartile_mean)

        print("VOC-S Results:")
        print(f"  Mean correlation: {voc_results['voc_s_mean']:.4f}")
        print(f"  Std correlation: {voc_results['voc_s_std']:.4f}")
        print(f"  IQM correlation: {voc_results['voc_s_iqm']:.4f}")
        print(f"  Episodes evaluated: {voc_results['num_episodes']}")

        return voc_results

    def evaluate_success_failure_detection(
        self, dataset, num_episodes: int = 100, mismatch_templates: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Evaluate success vs failure detection.

        Args:
            dataset: LeRobot dataset instance
            num_episodes: Number of episodes to evaluate
            mismatch_templates: Custom mismatch language templates

        Returns:
            Success/failure detection results
        """
        print(f"Evaluating success/failure detection on {num_episodes} episodes...")

        # Sample episodes
        total_episodes = dataset.num_episodes
        if num_episodes >= total_episodes:
            episode_indices = list(range(total_episodes))
        else:
            np.random.seed(42)
            episode_indices = np.random.choice(total_episodes, num_episodes, replace=False)

        correct_rewards = []
        incorrect_rewards = []

        # Get original languages
        original_languages = []
        for ep_idx in episode_indices:
            ep_start = dataset.episode_data_index["from"][ep_idx].item()
            frame_data = dataset[ep_start]

            if OBS_LANGUAGE in frame_data:
                lang = frame_data[OBS_LANGUAGE]
                if isinstance(lang, list):
                    lang = lang[0]
            elif "task" in frame_data:
                lang = frame_data["task"]
            else:
                lang = ""

            original_languages.append(lang)

        # Generate mismatched languages
        mismatched_languages = generate_mismatched_languages(original_languages, mismatch_templates)

        for i, ep_idx in enumerate(tqdm(episode_indices, desc="Computing detection metrics")):
            try:
                # Get episode frames (same as VOC-S evaluation)
                ep_start = dataset.episode_data_index["from"][ep_idx].item()
                ep_end = dataset.episode_data_index["to"][ep_idx].item()
                episode_length = ep_end - ep_start

                frames = []
                for frame_idx in range(episode_length):
                    global_idx = ep_start + frame_idx
                    frame_data = dataset[global_idx]

                    # Extract image
                    if OBS_IMAGES in frame_data:
                        img = frame_data[OBS_IMAGES]
                    else:
                        img_keys = [k for k in frame_data.keys() if "image" in k.lower()]
                        if img_keys:
                            img = frame_data[img_keys[0]]
                        else:
                            continue

                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)

                    if len(img.shape) == 3 and img.shape[-1] in [1, 3, 4]:
                        img = img.permute(2, 0, 1)

                    # Resize to expected input size (256x256 for SigLIP2)
                    if img.shape[-2:] != (256, 256):
                        import torch.nn.functional as F

                        img = F.interpolate(
                            img.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
                        ).squeeze(0)

                    # Normalize to [0, 1] if needed
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0

                    frames.append(img)

                if not frames:
                    continue

                frames_tensor = torch.stack(frames)

                # Predict with correct language
                correct_lang = original_languages[i]
                correct_ep_rewards = self.predict_episode_rewards(frames_tensor, correct_lang)

                # Predict with incorrect language
                incorrect_lang = mismatched_languages[i]
                incorrect_ep_rewards = self.predict_episode_rewards(frames_tensor, incorrect_lang)

                correct_rewards.append(correct_ep_rewards)
                incorrect_rewards.append(incorrect_ep_rewards)

            except Exception as e:
                warnings.warn(f"Failed to process episode {ep_idx} for detection: {e}")
                continue

        # Compute detection metrics
        detection_results = compute_success_failure_detection(correct_rewards, incorrect_rewards)

        print("Success/Failure Detection Results:")
        print(f"  Detection accuracy: {detection_results['detection_accuracy']:.4f}")
        print(f"  Mean correct final reward: {detection_results['mean_correct_final']:.4f}")
        print(f"  Mean incorrect final reward: {detection_results['mean_incorrect_final']:.4f}")
        print(f"  Separation score: {detection_results['separation_score']:.4f}")
        print(f"  Episode pairs evaluated: {detection_results['num_pairs']}")

        return detection_results

    def comprehensive_evaluation(
        self,
        dataset,
        num_episodes: int = 100,
        use_interquartile_mean: bool = True,
        mismatch_templates: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive evaluation including both VOC-S and detection metrics.

        Returns:
            Combined evaluation results
        """
        print("=" * 60)
        print("COMPREHENSIVE RLEARN EVALUATION")
        print("=" * 60)

        # VOC-S evaluation
        voc_results = self.evaluate_voc_s(
            dataset, num_episodes=num_episodes, use_interquartile_mean=use_interquartile_mean
        )

        print("\n" + "=" * 40)

        # Success/failure detection
        detection_results = self.evaluate_success_failure_detection(
            dataset, num_episodes=num_episodes, mismatch_templates=mismatch_templates
        )

        # Combined results
        results = {
            "voc_s": voc_results,
            "detection": detection_results,
            "overall_score": (
                voc_results["voc_s_iqm"] * 0.6 + detection_results["detection_accuracy"] * 0.4
            ),  # Weighted combination
        }

        print("\n" + "=" * 60)
        print(f"OVERALL EVALUATION SCORE: {results['overall_score']:.4f}")
        print("=" * 60)

        return results
