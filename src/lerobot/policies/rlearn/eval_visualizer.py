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
Visualization utilities for RLearN evaluation during training.

Creates and saves reward prediction visualizations for held-out episodes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from scipy.stats import spearmanr
from torch import Tensor

from lerobot.constants import OBS_IMAGES, OBS_LANGUAGE

# Set matplotlib backend to avoid GUI issues during training
rcParams['backend'] = 'Agg'


class RLearNEvalVisualizer:
    """
    Creates visualization plots for RLearN model evaluation during training.
    
    Generates reward prediction plots similar to the evaluation notebook but saves
    them as images for monitoring training progress.
    """
    
    def __init__(self, model, dataset, device: str = "cuda"):
        """
        Args:
            model: RLearN model instance
            dataset: LeRobot dataset instance
            device: Device to run evaluation on
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        
    def get_episode_data(self, episode_idx: int, max_frames: int = 64) -> tuple[Tensor | None, str | None, np.ndarray | None, int | None]:
        """Extract frames, language, and predict rewards for an episode."""
        try:
            # Get episode data
            ep_start = self.dataset.episode_data_index["from"][episode_idx].item()
            ep_end = self.dataset.episode_data_index["to"][episode_idx].item()
            episode_length = min(ep_end - ep_start, max_frames)

            # Collect frames and get language
            frames = []
            language = None

            for frame_idx in range(episode_length):
                global_idx = ep_start + frame_idx
                frame_data = self.dataset[global_idx]

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

                # Ensure CHW format
                if len(img.shape) == 3 and img.shape[-1] in [1, 3, 4]:
                    img = img.permute(2, 0, 1)

                # Resize to expected input size (224x224 for SigLIP2)
                if img.shape[-2:] != (224, 224):
                    import torch.nn.functional as F
                    img = F.interpolate(
                        img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                    ).squeeze(0)

                # Normalize to [0, 1] if needed
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0

                frames.append(img)

                # Get language
                if language is None:
                    if OBS_LANGUAGE in frame_data:
                        language = frame_data[OBS_LANGUAGE]
                        if isinstance(language, list):
                            language = language[0]
                    elif "task" in frame_data:
                        language = frame_data["task"]
                    else:
                        language = "No language provided"

            if not frames:
                return None, None, None, None

            frames_tensor = torch.stack(frames)
            
            # Predict rewards using the model's evaluation method
            with torch.no_grad():
                self.model.eval()
                rewards = self._predict_episode_rewards(frames_tensor, language)

            return frames_tensor, language, rewards, episode_length

        except Exception as e:
            warnings.warn(f"Error processing episode {episode_idx}: {e}")
            return None, None, None, None

    @torch.no_grad()
    def _predict_episode_rewards(self, frames: Tensor, language: str, batch_size: int = 16) -> np.ndarray:
        """
        Predict rewards for a single episode using proper temporal sequences.
        
        Args:
            frames: Video frames tensor of shape (T, C, H, W)
            language: Language instruction string
            batch_size: Maximum number of temporal sequences to process at once

        Returns:
            Predicted progress/rewards array of shape (T,)
        """
        T = frames.shape[0]
        max_seq_len = self.model.config.max_seq_len

        # Create temporal sequences for each frame
        temporal_sequences = []

        for i in range(T):
            # Create sequence ending at frame i
            seq_frames = []
            for j in range(max(0, i - max_seq_len + 1), i + 1):
                # Use frame j if available, otherwise repeat the first available frame
                frame_idx = max(0, min(j, T - 1))
                seq_frames.append(frames[frame_idx])

            # Pad sequence to max_seq_len by repeating the first frame if needed
            while len(seq_frames) < max_seq_len:
                seq_frames.insert(0, seq_frames[0])  # Prepend first frame

            # Take only the last max_seq_len frames if we have too many
            seq_frames = seq_frames[-max_seq_len:]
            temporal_sequences.append(torch.stack(seq_frames))  # (max_seq_len, C, H, W)

        # Stack all temporal sequences: (T, max_seq_len, C, H, W)
        all_sequences = torch.stack(temporal_sequences)

        # Process in batches
        rewards = []
        for i in range(0, T, batch_size):
            end_idx = min(i + batch_size, T)
            batch_sequences = all_sequences[i:end_idx].to(self.device)  # (B, max_seq_len, C, H, W)

            # Create batch for model
            batch = {
                OBS_IMAGES: batch_sequences,  # (B, T, C, H, W) format expected by model
                OBS_LANGUAGE: [language] * batch_sequences.shape[0],
            }

            # Predict rewards - model returns (B, T') but we want the last timestep for each sequence
            values = self.model.predict_rewards(batch)  # (B, T')

            # Take the last timestep prediction for each sequence (represents current frame reward)
            if values.dim() == 2:
                batch_rewards = values[:, -1].cpu().numpy()  # (B,) - last timestep
            else:
                batch_rewards = values.cpu().numpy()  # (B,) - already single timestep

            rewards.extend(batch_rewards)

        return np.array(rewards[:T])  # Ensure exact length

    def create_episode_grid_visualization(
        self, 
        episode_indices: list[int], 
        save_path: Path, 
        step: int | None = None,
        max_frames: int = 64
    ) -> dict[str, Any]:
        """
        Create a 3x3 grid visualization of episode reward predictions.
        
        Args:
            episode_indices: List of 9 episode indices to visualize
            save_path: Path to save the visualization image
            step: Training step (for title)
            max_frames: Maximum frames per episode to process
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(episode_indices) != 9:
            raise ValueError("Expected exactly 9 episode indices for 3x3 grid")
            
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        axes = axes.flatten()
        
        eval_metrics = {
            "voc_s_scores": [],
            "episode_lengths": [],
            "reward_ranges": [],
            "languages": []
        }
        
        for i, episode_idx in enumerate(episode_indices):
            ax = axes[i]
            
            frames, language, rewards, episode_length = self.get_episode_data(episode_idx, max_frames)
            
            if rewards is None:
                ax.text(
                    0.5, 0.5, f"Episode {episode_idx}\nNo data available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7)
                )
                ax.set_title(f"Episode {episode_idx} - Error", fontsize=12, pad=10)
                continue
                
            # Plot predicted rewards
            time_steps = range(len(rewards))
            ax.plot(
                time_steps, rewards, "b-", linewidth=2.5, marker="o", markersize=5, 
                label="Predicted Reward", alpha=0.8
            )
            
            # Add expected progress line (ground truth for ReWiND)
            expected_progress = np.linspace(0, 1, len(rewards))
            ax.plot(
                time_steps, expected_progress, "orange", linestyle="--", linewidth=2.5, 
                label="Expected Progress (0→1)", alpha=0.8
            )
            
            # Compute VOC-S (Value-Order Correlation for Success)
            frame_indices = np.arange(1, len(rewards) + 1)
            correlation, p_value = spearmanr(frame_indices, rewards)
            if np.isnan(correlation):
                correlation = 0.0
                
            eval_metrics["voc_s_scores"].append(correlation)
            eval_metrics["episode_lengths"].append(len(rewards))
            eval_metrics["reward_ranges"].append((rewards.min(), rewards.max()))
            eval_metrics["languages"].append(language)
            
            # Format title with language (truncated) and VOC-S
            title_lang = language[:35] + "..." if len(language) > 35 else language
            title = f'Episode {episode_idx}\n"{title_lang}"\nVOC-S: {correlation:.3f}'
            ax.set_title(title, fontsize=10, pad=15)
            
            ax.set_xlabel("Frame Index", fontsize=10)
            ax.set_ylabel("Reward", fontsize=10)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Color-coded trend indicator
            if correlation > 0.3:
                trend_text = "↗ Strong+"
                trend_color = "darkgreen"
            elif correlation > 0.1:
                trend_text = "↗ Weak+"
                trend_color = "green"
            elif correlation < -0.3:
                trend_text = "↘ Strong-"
                trend_color = "darkred"
            elif correlation < -0.1:
                trend_text = "↘ Weak-"
                trend_color = "red"
            else:
                trend_text = "→ Flat"
                trend_color = "gray"
                
            ax.text(
                0.02, 0.98, trend_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=trend_color, alpha=0.2),
                color=trend_color
            )
            
            # Add reward range info
            ax.text(
                0.98, 0.02, f"Range: [{rewards.min():.3f}, {rewards.max():.3f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.5)
            )
        
        # Add overall title
        step_text = f" - Step {step}" if step is not None else ""
        fig.suptitle(
            f"RLearN Reward Evaluation{step_text}\n"
            f"Mean VOC-S: {np.mean(eval_metrics['voc_s_scores']):.3f} | "
            f"Episodes: {len([s for s in eval_metrics['voc_s_scores'] if s != 0])}/9",
            fontsize=16, y=0.95
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Make room for suptitle
        
        # Save the figure
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to free memory
        
        # Calculate summary metrics
        valid_scores = [s for s in eval_metrics["voc_s_scores"] if s != 0]
        summary = {
            "mean_voc_s": np.mean(valid_scores) if valid_scores else 0.0,
            "std_voc_s": np.std(valid_scores) if valid_scores else 0.0,
            "num_valid_episodes": len(valid_scores),
            "total_episodes": len(episode_indices),
            "mean_episode_length": np.mean(eval_metrics["episode_lengths"]) if eval_metrics["episode_lengths"] else 0,
            "individual_scores": eval_metrics["voc_s_scores"],
            "episode_languages": eval_metrics["languages"]
        }
        
        return summary
        
    def create_comparison_visualization(
        self,
        episode_indices: list[int],
        save_path: Path,
        step: int | None = None,
        max_frames: int = 64,
        mismatch_templates: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Create correct vs incorrect language comparison visualization.
        
        Args:
            episode_indices: List of episode indices to compare (up to 6)
            save_path: Path to save the visualization image  
            step: Training step (for title)
            max_frames: Maximum frames per episode to process
            mismatch_templates: Custom mismatch templates
            
        Returns:
            Dictionary with detection metrics
        """
        if mismatch_templates is None:
            mismatch_templates = [
                "kick the ball", "clean the sink", "dance in place", 
                "wave your hand", "jump up and down", "do nothing"
            ]
            
        # Limit to 6 episodes for 2x3 grid
        episode_indices = episode_indices[:6]
        n_episodes = len(episode_indices)
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        detection_results = {
            "correct_finals": [],
            "incorrect_finals": [],
            "detection_successes": [],
            "episode_info": []
        }
        
        for i, episode_idx in enumerate(episode_indices):
            if i >= 6:  # Limit to 6 subplots
                break
                
            ax = axes[i]
            
            # Get episode data with correct language
            frames, correct_language, correct_rewards, episode_length = self.get_episode_data(episode_idx, max_frames)
            
            if correct_rewards is None:
                ax.text(
                    0.5, 0.5, f"Episode {episode_idx}\nNo data available",
                    ha="center", va="center", transform=ax.transAxes
                )
                ax.set_title(f"Episode {episode_idx} - Error")
                continue
                
            # Generate incorrect language and predict
            incorrect_language = mismatch_templates[i % len(mismatch_templates)]
            incorrect_rewards = self._predict_episode_rewards(frames, incorrect_language)
            
            # Plot both reward curves
            time_steps = range(len(correct_rewards))
            ax.plot(
                time_steps, correct_rewards, "g-", linewidth=2.5, marker="o", markersize=4,
                label=f"Correct: '{correct_language[:25]}...'" if len(correct_language) > 25 else f"Correct: '{correct_language}'"
            )
            ax.plot(
                time_steps, incorrect_rewards, "r-", linewidth=2.5, marker="s", markersize=4,
                label=f"Incorrect: '{incorrect_language}'"
            )
            
            # Calculate detection success
            final_correct = correct_rewards[-1]
            final_incorrect = incorrect_rewards[-1]
            detection_success = final_correct > final_incorrect
            
            detection_results["correct_finals"].append(final_correct)
            detection_results["incorrect_finals"].append(final_incorrect)
            detection_results["detection_successes"].append(detection_success)
            detection_results["episode_info"].append({
                "episode_idx": episode_idx,
                "correct_language": correct_language,
                "incorrect_language": incorrect_language,
                "final_correct": final_correct,
                "final_incorrect": final_incorrect
            })
            
            # Color-coded title based on detection success
            success_indicator = "✓" if detection_success else "✗"
            title_color = "darkgreen" if detection_success else "darkred"
            ax.set_title(
                f"Episode {episode_idx} {success_indicator}\nΔ: {final_correct - final_incorrect:.3f}",
                color=title_color, fontweight="bold", fontsize=11
            )
            
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Reward")
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add final reward values as text
            ax.text(
                0.98, 0.02, 
                f"Final: C={final_correct:.3f}, I={final_incorrect:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7)
            )
        
        # Hide unused subplots
        for i in range(n_episodes, 6):
            axes[i].axis('off')
        
        # Calculate summary metrics
        detection_accuracy = np.mean(detection_results["detection_successes"]) if detection_results["detection_successes"] else 0.0
        mean_correct = np.mean(detection_results["correct_finals"]) if detection_results["correct_finals"] else 0.0
        mean_incorrect = np.mean(detection_results["incorrect_finals"]) if detection_results["incorrect_finals"] else 0.0
        
        # Add overall title
        step_text = f" - Step {step}" if step is not None else ""
        fig.suptitle(
            f"RLearN Language Detection{step_text}\n"
            f"Accuracy: {detection_accuracy:.1%} | Mean Δ: {mean_correct - mean_incorrect:.3f} | "
            f"Success: {sum(detection_results['detection_successes'])}/{len(detection_results['detection_successes'])}",
            fontsize=16, y=0.95
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save the figure
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        summary = {
            "detection_accuracy": detection_accuracy,
            "mean_correct_final": mean_correct,
            "mean_incorrect_final": mean_incorrect,
            "separation_score": mean_correct - mean_incorrect,
            "num_episodes": len(detection_results["detection_successes"]),
            "individual_results": detection_results["episode_info"]
        }
        
        return summary


def select_evaluation_episodes(dataset, num_episodes: int = 9, seed: int = 42) -> list[int]:
    """
    Select a diverse set of episodes for evaluation holdout.
    
    Args:
        dataset: LeRobot dataset instance
        num_episodes: Number of episodes to select
        seed: Random seed for reproducibility
        
    Returns:
        List of episode indices
    """
    np.random.seed(seed)
    
    total_episodes = dataset.num_episodes
    if num_episodes >= total_episodes:
        return list(range(total_episodes))
    
    # Select random episodes
    episode_indices = np.random.choice(total_episodes, num_episodes, replace=False).tolist()
    
    return sorted(episode_indices)
