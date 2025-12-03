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
Inference script for SARM (Stage-Aware Reward Model).

This script loads a trained SARM model and runs inference on a dataset episode,
generating visualizations of the predicted task stages and progress over time.

Example usage:
    python scripts/visualize_sarm_predictions.py \
        --model-id username/sarm-model \
        --dataset-repo lerobot/aloha_sim_insertion_human \
        --episode-index 0 \
        --output-dir outputs/sarm_viz \
        --task-description "insert the peg into the socket"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.sarm_utils import (
    pad_state_to_max_dim,
    compute_tau,
    compute_cumulative_progress_batch,
)
from lerobot.datasets.utils import load_stats


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SARM inference and visualize predictions")
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to trained SARM model"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-repo",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g., lerobot/aloha_sim_insertion_human)"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Index of the episode to visualize (default: 0)"
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="perform the task",
        help="Task description for the reward model (default: 'perform the task')"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sarm_inference",
        help="Directory to save visualization outputs (default: outputs/sarm_inference)"
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Key for images in dataset (e.g., observation.images.image). If not specified, uses model config's image_key"
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=None,
        help="Key for joint states in dataset. If None, auto-detects from dataset"
    )
    
    # Visualization options
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Include sample frames in the visualization"
    )
    parser.add_argument(
        "--num-sample-frames",
        type=int,
        default=8,
        help="Number of sample frames to show (default: 8)"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[14, 8],
        help="Figure size as width height (default: 14 8)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/cpu, default: auto-detect)"
    )
    
    # Dual mode options
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="Which head(s) to visualize for dual-head models (default: sparse)"
    )
    
    return parser.parse_args()


def load_episode_data(
    dataset: LeRobotDataset,
    episode_index: int,
    image_key: str,
    state_key: str | None = None
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """
    Load all frames and states from a specific episode.
    
    Args:
        dataset: LeRobotDataset instance
        episode_index: Index of the episode to load
        image_key: Key for accessing images in the dataset
        state_key: Key for accessing joint states (auto-detected if None)
        
    Returns:
        Tuple of (frames, states, start_index, end_index, task_description)
    """
    # Get episode boundaries
    episode_data = dataset.meta.episodes
    start_idx = episode_data["dataset_from_index"][episode_index]
    end_idx = episode_data["dataset_to_index"][episode_index]
    
    logger.info(f"Loading episode {episode_index}: frames {start_idx} to {end_idx} ({end_idx - start_idx} frames)")
    
    # Auto-detect state key if not provided
    if state_key is None:
        first_item = dataset[start_idx]
        state_keys = [k for k in first_item.keys() if 'state' in k.lower() or 'qpos' in k.lower()]
        if state_keys:
            state_key = state_keys[0]
            logger.info(f"Auto-detected state key: {state_key}")
    
    # Get task description from the dataset if available
    task_description = None
    first_item = dataset[start_idx]
    if "task" in first_item:
        task_description = first_item["task"]
        logger.info(f"✓ Extracted task from episode {episode_index}: '{task_description}'")
    
    # Load all frames and states from the episode
    frames = []
    states = []
    for idx in tqdm(range(start_idx, end_idx), desc="Loading frames"):
        item = dataset[idx]
        
        # Get image
        img = item[image_key]
        
        # Convert to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Handle different image formats (C, H, W) or (H, W, C)
        if img.shape[0] in [1, 3]:  # Channel first
            img = np.transpose(img, (1, 2, 0))
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        frames.append(img)
        
        # Get state if available
        if state_key and state_key in item:
            state = item[state_key]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            states.append(state)
    
    frames = np.array(frames)
    states = np.array(states) if states else None
    logger.info(f"Loaded {len(frames)} frames with shape {frames[0].shape}")
    if states is not None:
        logger.info(f"Loaded states with shape {states.shape}")
    
    return frames, states, start_idx, end_idx, task_description


@torch.no_grad()
def run_inference(
    model: SARMRewardModel,
    frames: np.ndarray,
    states: Optional[np.ndarray],
    task_description: str,
    dataset_stats: dict | None = None,
    state_key: str = "observation.state",
    batch_size: int = 32,
    head_mode: str = "sparse"
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Run SARM inference on video frames and joint states.
    
    (per SARM paper Section A.4):
    - Frame 0: Initial frame of the episode (frame 0)
    - Frames 1-8: 8 consecutive frames with frame_gap spacing ending at current frame t
    Pattern: [frame_0, t-(7*gap), t-(6*gap), ..., t-gap, t]
    
    Args:
        model: SARM model
        frames: Video frames (num_frames, H, W, C) - all frames from ONE episode
        states: Joint states (num_frames, state_dim)
        task_description: Task description text
        dataset_stats: Dataset statistics for state normalization (same as training)
        state_key: Key for state in dataset_stats
        batch_size: Batch size for processing slices
        head_mode: Which head(s) to use ("sparse", "dense", or "both")
        
    Returns:
        Dict mapping head name to (progress_predictions, stage_predictions) tuple
            - progress_predictions: (num_frames,)
            - stage_predictions: (num_frames, num_stages)
    """
    logger.info("Encoding video frames with CLIP...")
    video_embeddings = model.encode_images(frames)
    
    logger.info("Encoding task description with CLIP...")
    text_embedding = model.encode_text(task_description)
    
    # Get config values
    num_frames_model = model.config.num_frames  # 9
    frame_gap = model.config.frame_gap  # 30
    dual_mode = model.config.dual_sparse_dense
    
    logger.info("Creating video slices (SARM paper: initial frame + 8 consecutive)...")
    
    # Convert to tensors
    video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
    if states is not None:
        state_embeddings = torch.tensor(states, dtype=torch.float32)
        
        # Normalize states using dataset stats (same as training processor)
        if dataset_stats is not None and state_key in dataset_stats:
            mean = torch.tensor(dataset_stats[state_key]["mean"], dtype=torch.float32)
            std = torch.tensor(dataset_stats[state_key]["std"], dtype=torch.float32)
            state_embeddings = (state_embeddings - mean) / (std + 1e-8)
            logger.info(f"✓ Applied MEAN_STD normalization to states using {state_key}")
        else:
            logger.warning("⚠ No dataset_stats provided - states not normalized (may differ from training)")
    else:
        state_embeddings = None
    
    video_slices = []
    state_slices = []
    
    for current_frame in tqdm(range(len(video_embeddings)), desc="Creating slices"):
        # Compute frame indices: [initial_frame (0), t-(7*gap), t-(6*gap), ..., t-gap, t]
        # The first delta is -100000 which clamps to episode start
        deltas = model.config.observation_delta_indices
        frame_indices = [max(0, min(current_frame + delta, len(video_embeddings) - 1)) for delta in deltas]

        video_slice = video_embeddings[frame_indices]
        video_slices.append(video_slice)
        
        if state_embeddings is not None:
            state_slice = state_embeddings[frame_indices]
            state_slices.append(state_slice)
    
    video_slices = torch.stack(video_slices)  # (num_frames, num_frames_model, 512)
    if state_embeddings is not None:
        state_slices = torch.stack(state_slices)  # (num_frames, num_frames_model, state_dim)
        # Pad states to max_state_dim (same as training processor)
        state_slices = pad_state_to_max_dim(state_slices, model.config.max_state_dim)
    else:
        state_slices = None
    
    logger.info(f"Running SARM inference on all slices (head_mode={head_mode}, dual_mode={dual_mode})...")
    
    # Initialize results dict
    results = {}
    
    # Process in batches
    if dual_mode:
        all_sparse_progress = []
        all_sparse_stages = []
        all_dense_progress = []
        all_dense_stages = []
    else:
        all_sparse_progress = []
        all_sparse_stages = []
    
    for i in tqdm(range(0, len(video_slices), batch_size), desc="Inference"):
        batch_video = video_slices[i:i + batch_size].to(model.device)
        batch_states = state_slices[i:i + batch_size].to(model.device) if state_slices is not None else None
        batch_size_actual = batch_video.shape[0]
        
        # Replicate text embedding for batch
        batch_text = text_embedding.unsqueeze(0).repeat(batch_size_actual, 1).to(model.device)
        
        # Get predictions
        if dual_mode:
            preds = model.sarm_transformer(batch_video, batch_text, batch_states, head_mode=head_mode)
            
            if head_mode in ["sparse", "both"]:
                sparse_logits, sparse_probs, sparse_progress = preds["sparse"]
                all_sparse_progress.extend(sparse_progress[:, -1, 0].cpu().numpy())
                all_sparse_stages.extend(sparse_probs[:, -1, :].cpu().numpy())
            
            if head_mode in ["dense", "both"]:
                dense_logits, dense_probs, dense_progress = preds["dense"]
                all_dense_progress.extend(dense_progress[:, -1, 0].cpu().numpy())
                all_dense_stages.extend(dense_probs[:, -1, :].cpu().numpy())
        else:
            # Single mode (sparse only)
            stage_logits, stage_probs, progress_preds = model.sarm_transformer(
                batch_video, batch_text, batch_states
            )
            
            # Extract last frame predictions (the "current" frame)
            all_sparse_progress.extend(progress_preds[:, -1, 0].cpu().numpy())
            all_sparse_stages.extend(stage_probs[:, -1, :].cpu().numpy())
    
    # Build results dict
    if head_mode in ["sparse", "both"] or not dual_mode:
        results["sparse"] = (np.array(all_sparse_progress), np.array(all_sparse_stages))
    
    if dual_mode and head_mode in ["dense", "both"]:
        results["dense"] = (np.array(all_dense_progress), np.array(all_dense_stages))
    
    return results


def compute_ground_truth_progress(
    dataset: LeRobotDataset,
    episode_index: int,
    temporal_proportions: dict[str, float],
    subtask_names_ordered: list[str],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Compute ground truth progress and stage labels for an episode using annotations.
    
    Uses SARM Paper Formula (2):
        y_t = P_{k-1} + ᾱ_k × τ_t
    
    where:
        - τ_t = (t - s_k) / (e_k - s_k) is within-subtask progress
        - P_{k-1} is cumulative prior (sum of previous subtask proportions)
        - ᾱ_k is the temporal proportion for subtask k
    
    Args:
        dataset: LeRobotDataset instance
        episode_index: Index of the episode
        temporal_proportions: Dict mapping subtask name to proportion
        subtask_names_ordered: Ordered list of subtask names (for consistent stage indexing)
        
    Returns:
        Tuple of (ground_truth_progress, ground_truth_stages) arrays, or (None, None) if no annotations
    """
    # Load episode metadata
    episodes_df = dataset.meta.episodes.to_pandas()
    
    # Check if annotations exist
    if "subtask_names" not in episodes_df.columns:
        logger.warning("No subtask_names column found in episodes metadata")
        return None, None
    
    ep_subtask_names = episodes_df.loc[episode_index, "subtask_names"]
    if ep_subtask_names is None or (isinstance(ep_subtask_names, float) and pd.isna(ep_subtask_names)):
        logger.warning(f"No annotations found for episode {episode_index}")
        return None, None
    
    subtask_start_frames = episodes_df.loc[episode_index, "subtask_start_frames"]
    subtask_end_frames = episodes_df.loc[episode_index, "subtask_end_frames"]
    
    # Get episode boundaries
    ep_start = dataset.meta.episodes["dataset_from_index"][episode_index]
    ep_end = dataset.meta.episodes["dataset_to_index"][episode_index]
    num_frames = ep_end - ep_start
    
    # Get temporal proportions as ordered list
    temporal_proportions_list = [
        temporal_proportions.get(name, 0.0) for name in subtask_names_ordered
    ]
    
    logger.info(f"Computing ground truth for {num_frames} frames using {len(ep_subtask_names)} annotated subtasks")
    logger.info(f"Subtask names in episode: {ep_subtask_names}")
    logger.info(f"Subtask start frames: {subtask_start_frames}")
    logger.info(f"Subtask end frames: {subtask_end_frames}")
    logger.info(f"Temporal proportions (ordered): {dict(zip(subtask_names_ordered, temporal_proportions_list))}")
    
    # Compute ground truth for each frame
    gt_progress = np.zeros(num_frames)
    gt_stages = np.zeros(num_frames, dtype=np.int32)
    
    for frame_rel in range(num_frames):
        # Find which subtask this frame belongs to
        found = False
        for j, (name, start_frame, end_frame) in enumerate(zip(ep_subtask_names, subtask_start_frames, subtask_end_frames)):
            if frame_rel >= start_frame and frame_rel <= end_frame:
                # Found the subtask - get its global index
                stage_idx = subtask_names_ordered.index(name) if name in subtask_names_ordered else 0
                
                # Compute τ_t using utility function
                tau = compute_tau(frame_rel, start_frame, end_frame)
                
                # Compute cumulative progress using utility function
                progress = compute_cumulative_progress_batch(tau, stage_idx, temporal_proportions_list)
                
                gt_progress[frame_rel] = progress
                gt_stages[frame_rel] = stage_idx
                found = True
                break
        
        if not found:
            # Handle frames outside annotated subtasks
            if frame_rel < subtask_start_frames[0]:
                gt_progress[frame_rel] = 0.0
                gt_stages[frame_rel] = 0
            elif frame_rel > subtask_end_frames[-1]:
                gt_progress[frame_rel] = 1.0
                gt_stages[frame_rel] = len(subtask_names_ordered) - 1
            else:
                # Between subtasks - find previous subtask
                for j in range(len(ep_subtask_names) - 1):
                    if frame_rel > subtask_end_frames[j] and frame_rel < subtask_start_frames[j + 1]:
                        name = ep_subtask_names[j]
                        stage_idx = subtask_names_ordered.index(name) if name in subtask_names_ordered else j
                        progress = compute_cumulative_progress_batch(1.0, stage_idx, temporal_proportions_list)
                        gt_progress[frame_rel] = progress
                        gt_stages[frame_rel] = stage_idx
                        break
    
    logger.info(f"✓ Ground truth computed: final={gt_progress[-1]:.3f}, max={gt_progress.max():.3f}")
    return gt_progress, gt_stages


def visualize_predictions(
    frames: np.ndarray,
    progress_predictions: np.ndarray,
    stage_predictions: np.ndarray,
    task_description: str,
    output_path: Path,
    num_sample_frames: int = 8,
    figsize: tuple = (14, 8),
    subtask_names: list[str] | None = None,
    temporal_proportions: dict[str, float] | None = None,
    ground_truth_progress: np.ndarray | None = None,
    ground_truth_stages: np.ndarray | None = None,
):
    """
    Create visualization of SARM predictions with optional ground truth comparison.
    
    Args:
        frames: Video frames (num_frames, H, W, C)
        progress_predictions: Progress predictions (num_frames,)
        stage_predictions: Stage probabilities (num_frames, num_stages)
        task_description: Task description
        output_path: Path to save the figure
        num_sample_frames: Number of frames to show
        figsize: Figure size (width, height)
        subtask_names: Optional list of subtask names for labeling
        temporal_proportions: Optional dict of temporal proportions for each subtask
        ground_truth_progress: Optional ground truth progress array (num_frames,)
        ground_truth_stages: Optional ground truth stage indices array (num_frames,)
    """
    num_stages = stage_predictions.shape[1]
    stage_colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    
    # Use subtask names if available, otherwise use generic labels
    if subtask_names is not None and len(subtask_names) == num_stages:
        stage_labels = subtask_names
    else:
        stage_labels = [f'Stage {i+1}' for i in range(num_stages)]
    
    # Create figure with progress plot, stage plot, and sample frames
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    ax_progress = fig.add_subplot(gs[0])
    ax_stages = fig.add_subplot(gs[1], sharex=ax_progress)
    ax_frames = fig.add_subplot(gs[2])
    
    frame_indices = np.arange(len(progress_predictions))
    
    # Plot 1: Progress over time
    ax_progress.plot(frame_indices, progress_predictions, linewidth=2, color='#2E86AB', label='Predicted Progress')
    ax_progress.fill_between(frame_indices, 0, progress_predictions, alpha=0.3, color='#2E86AB')
    
    # Plot ground truth if available
    if ground_truth_progress is not None:
        ax_progress.plot(frame_indices, ground_truth_progress, linewidth=2, color='#28A745', 
                        linestyle='--', label='Ground Truth Progress')
        ax_progress.fill_between(frame_indices, 0, ground_truth_progress, alpha=0.15, color='#28A745')
    
    ax_progress.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_progress.set_ylabel('Task Progress', fontsize=12)
    ax_progress.set_title(f'Task: "{task_description}"', fontsize=14, fontweight='bold')
    ax_progress.grid(True, alpha=0.3)
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc='upper left')
    
    # Add statistics box
    stats_text = (
        f'Frames: {len(progress_predictions)}\n'
        f'Final Progress: {progress_predictions[-1]:.3f}\n'
        f'Max Progress: {progress_predictions.max():.3f}\n'
        f'Mean Progress: {progress_predictions.mean():.3f}'
    )
    if ground_truth_progress is not None:
        mse = np.mean((progress_predictions - ground_truth_progress) ** 2)
        stats_text += f'\nMSE vs GT: {mse:.4f}'
        stats_text += f'\nGT Final: {ground_truth_progress[-1]:.3f}'
    
    ax_progress.text(0.98, 0.02, stats_text, transform=ax_progress.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Stage predictions (stacked area plot)
    ax_stages.stackplot(frame_indices, *[stage_predictions[:, i] for i in range(num_stages)],
                        colors=stage_colors, alpha=0.8, labels=stage_labels)
    
    # Plot ground truth stage as vertical bands or markers
    if ground_truth_stages is not None:
        # Find stage transition points in ground truth
        stage_changes = np.where(np.diff(ground_truth_stages) != 0)[0] + 1
        for change_idx in stage_changes:
            ax_stages.axvline(x=change_idx, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
            ax_progress.axvline(x=change_idx, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add small markers at bottom showing GT stage
        gt_stage_normalized = ground_truth_stages / max(num_stages - 1, 1)
        ax_stages.scatter(frame_indices[::30], np.zeros(len(frame_indices[::30])) + 0.02, 
                         c=[stage_colors[s] for s in ground_truth_stages[::30]], 
                         s=20, marker='|', alpha=0.8, label='GT Stage Markers')
    
    ax_stages.set_xlabel('Frame Index', fontsize=12)
    ax_stages.set_ylabel('Stage Probability', fontsize=12)
    ax_stages.set_ylim(0, 1)
    ax_stages.grid(True, alpha=0.3)
    
    # Adjust legend based on number of stages and label lengths
    if num_stages <= 5:
        ax_stages.legend(loc='upper left', ncol=num_stages, fontsize=8)
    else:
        ax_stages.legend(loc='upper left', ncol=3, fontsize=7)
    
    # Add vertical lines and labels for expected stage transitions (if temporal proportions available)
    if temporal_proportions is not None and subtask_names is not None:
        cumulative_progress = 0.0
        for i, name in enumerate(stage_labels):
            if name in temporal_proportions:
                # Find approximate frame where this stage should end
                stage_end_progress = cumulative_progress + temporal_proportions[name]
                
                # Find frame index closest to this progress
                progress_diffs = np.abs(progress_predictions - stage_end_progress)
                stage_end_frame = np.argmin(progress_diffs)
                
                # Draw vertical line
                ax_progress.axvline(x=stage_end_frame, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                ax_stages.axvline(x=stage_end_frame, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                
                cumulative_progress = stage_end_progress
    
    # Plot 3: Sample frames (if requested)
    frame_indices_to_show = np.linspace(0, len(frames) - 1, num_sample_frames, dtype=int)
    
    ax_frames.axis('off')
    
    # Create grid for frames
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]
    
    combined_width = frame_width * num_sample_frames
    combined_image = np.zeros((frame_height, combined_width, 3), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices_to_show):
        frame = frames[frame_idx]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        
        # Add frame to combined image
        x_start = i * frame_width
        x_end = (i + 1) * frame_width
        combined_image[:, x_start:x_end] = frame
        
        # Add frame number, progress, and stage
        progress_val = progress_predictions[frame_idx]
        stage_idx = np.argmax(stage_predictions[frame_idx])
        stage_name = stage_labels[stage_idx] if stage_idx < len(stage_labels) else f'{stage_idx+1}'
        
        # Truncate long stage names for display
        if len(stage_name) > 15:
            stage_name = stage_name[:12] + '...'
        
        label = f'Frame {frame_idx}\nProg: {progress_val:.2f}\n{stage_name}'
        
        # Draw label on image
        ax_frames.text(x_start + frame_width / 2, -10, label, 
                      ha='center', va='top', fontsize=7, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax_frames.imshow(combined_image)
    ax_frames.set_title('Sample Frames', fontsize=12, pad=20)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")
    
    plt.close()


def main():
    args = parse_args()
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading SARM model from {args.model_id}...")
    model = SARMRewardModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset_repo}...")
    dataset = LeRobotDataset(args.dataset_repo)
    logger.info(f"Dataset loaded: {len(dataset.meta.episodes)} episodes, {len(dataset)} frames")
    
    # Validate episode index
    if args.episode_index >= len(dataset.meta.episodes):
        raise ValueError(
            f"Episode index {args.episode_index} out of range. "
            f"Dataset has {len(dataset.meta.episodes)} episodes."
        )
    
    image_key = args.image_key if args.image_key is not None else model.config.image_key
    state_key = args.state_key if args.state_key is not None else model.config.state_key
    logger.info(f"Using image key: {image_key}")
    logger.info(f"Using state key: {state_key}")
    
    # Load dataset stats for state normalization (same as training)
    dataset_stats = load_stats(dataset.root)
    if dataset_stats:
        logger.info(f"✓ Loaded dataset stats from {dataset.root}")
    else:
        logger.warning("⚠ Could not load dataset stats - states will not be normalized")
    
    # Load episode data
    frames, states, start_idx, end_idx, dataset_task = load_episode_data(
        dataset, args.episode_index, image_key, state_key
    )
    
    # Use task description from dataset if available, otherwise use command-line argument
    task_description = dataset_task if dataset_task is not None else args.task_description
    logger.info(f"Using task description: '{task_description}'")
    
    # Determine head mode based on model config and user preference
    dual_mode = model.config.dual_sparse_dense
    head_mode = args.head_mode
    
    if not dual_mode and head_mode in ["dense", "both"]:
        logger.warning(f"Model is not dual-head, ignoring --head-mode={head_mode}, using 'sparse'")
        head_mode = "sparse"
    
    logger.info(f"Model dual_sparse_dense: {dual_mode}, head_mode: {head_mode}")
    
    # Run inference
    inference_results = run_inference(
        model, frames, states, task_description, 
        dataset_stats=dataset_stats, state_key=state_key,
        head_mode=head_mode
    )
    
    # Extract subtask names and temporal proportions from model config if available
    sparse_subtask_names = None
    sparse_temporal_proportions = None
    dense_subtask_names = None
    dense_temporal_proportions = None
    
    # Load sparse subtask info
    if hasattr(model.config, 'sparse_subtask_names') and model.config.sparse_subtask_names is not None:
        sparse_subtask_names = model.config.sparse_subtask_names
        logger.info(f"✓ Found {len(sparse_subtask_names)} sparse subtask names in model config: {sparse_subtask_names}")
    
    if hasattr(model.config, 'sparse_temporal_proportions') and model.config.sparse_temporal_proportions is not None:
        sparse_temporal_proportions = {
            name: prop for name, prop in zip(model.config.sparse_subtask_names, model.config.sparse_temporal_proportions)
        }
        logger.info(f"✓ Loaded sparse temporal proportions from model config")
    
    # Fallback: try to load sparse proportions from dataset meta
    if sparse_temporal_proportions is None:
        proportions_path = dataset.root / "meta" / "temporal_proportions_sparse.json"
        if not proportions_path.exists():
            # Try legacy path
            proportions_path = dataset.root / "meta" / "temporal_proportions.json"
        if proportions_path.exists():
            with open(proportions_path, 'r') as f:
                sparse_temporal_proportions = json.load(f)
                logger.info(f"✓ Loaded sparse temporal proportions from dataset: {proportions_path}")
                
                if sparse_subtask_names is None:
                    sparse_subtask_names = sorted(sparse_temporal_proportions.keys())
                    logger.info(f"✓ Extracted sparse subtask names from proportions: {sparse_subtask_names}")
    
    # Load dense subtask info (if dual mode)
    if dual_mode:
        if hasattr(model.config, 'dense_subtask_names') and model.config.dense_subtask_names is not None:
            dense_subtask_names = model.config.dense_subtask_names
            logger.info(f"✓ Found {len(dense_subtask_names)} dense subtask names in model config: {dense_subtask_names}")
        
        if hasattr(model.config, 'dense_temporal_proportions') and model.config.dense_temporal_proportions is not None:
            dense_temporal_proportions = {
                name: prop for name, prop in zip(model.config.dense_subtask_names, model.config.dense_temporal_proportions)
            }
            logger.info(f"✓ Loaded dense temporal proportions from model config")
        
        # Fallback: try to load dense proportions from dataset meta
        if dense_temporal_proportions is None:
            dense_proportions_path = dataset.root / "meta" / "temporal_proportions_dense.json"
            if dense_proportions_path.exists():
                with open(dense_proportions_path, 'r') as f:
                    dense_temporal_proportions = json.load(f)
                    logger.info(f"✓ Loaded dense temporal proportions from dataset: {dense_proportions_path}")
                    
                    if dense_subtask_names is None:
                        dense_subtask_names = sorted(dense_temporal_proportions.keys())
                        logger.info(f"✓ Extracted dense subtask names from proportions: {dense_subtask_names}")
    
    output_dir = Path(args.output_dir)
    
    # Generate visualizations for each head in inference results
    for head_name, (progress_predictions, stage_predictions) in inference_results.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {head_name.upper()} predictions...")
        logger.info(f"{'='*60}")
        
        # Select appropriate subtask info based on head
        if head_name == "sparse":
            subtask_names = sparse_subtask_names
            temporal_proportions = sparse_temporal_proportions
        else:  # dense
            subtask_names = dense_subtask_names
            temporal_proportions = dense_temporal_proportions
        
        # Compute ground truth progress if annotations are available
        ground_truth_progress = None
        ground_truth_stages = None
        
        if temporal_proportions is not None and subtask_names is not None:
            logger.info(f"Attempting to compute {head_name} ground truth progress from annotations...")
            ground_truth_progress, ground_truth_stages = compute_ground_truth_progress(
                dataset,
                args.episode_index,
                temporal_proportions,
                subtask_names
            )
            if ground_truth_progress is None:
                logger.warning(f"⚠ {head_name.capitalize()} ground truth not available - annotations may be missing for this episode")
        else:
            logger.warning(f"⚠ Cannot compute {head_name} ground truth - temporal_proportions or subtask_names not available")
        
        # Generate output paths with head suffix
        suffix = f"_{head_name}" if len(inference_results) > 1 else ""
        output_path = output_dir / f"sarm_prediction_ep{args.episode_index}{suffix}.png"
        
        visualize_predictions(
            frames,
            progress_predictions,
            stage_predictions,
            f"{task_description} ({head_name.capitalize()})" if len(inference_results) > 1 else task_description,
            output_path,
            num_sample_frames=args.num_sample_frames,
            figsize=tuple(args.figsize),
            subtask_names=subtask_names,
            temporal_proportions=temporal_proportions,
            ground_truth_progress=ground_truth_progress,
            ground_truth_stages=ground_truth_stages,
        )
        
        # Save predictions
        predictions_path = output_dir / f"predictions_ep{args.episode_index}{suffix}.npz"
        save_dict = {
            'progress': progress_predictions, 
            'stages': stage_predictions
        }
        if ground_truth_progress is not None:
            save_dict['gt_progress'] = ground_truth_progress
            save_dict['gt_stages'] = ground_truth_stages
        np.savez(predictions_path, **save_dict)
        logger.info(f"Saved {head_name} predictions to {predictions_path}")
        logger.info(f"\n{head_name.capitalize()} visualization: {output_path}")


if __name__ == "__main__":
    main()

