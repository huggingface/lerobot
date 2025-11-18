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
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel


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
    batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SARM inference on video frames and joint states.
    
    For each frame t, creates a temporal sequence of 9 frames using SARM's pattern:
    [t-240, t-210, t-180, t-150, t-120, t-90, t-60, t-30, t]
    This matches the training pattern where frames are loaded with 30-frame gaps
    relative to the current frame.
    
    Args:
        model: SARM model
        frames: Video frames (num_frames, H, W, C)
        states: Joint states (num_frames, state_dim)
        task_description: Task description text
        batch_size: Batch size for processing slices
        
    Returns:
        Tuple of (progress_predictions, stage_predictions)
            - progress_predictions: (num_frames,)
            - stage_predictions: (num_frames, num_stages)
    """
    logger.info("Encoding video frames with CLIP...")
    video_embeddings = model.encode_images(frames)
    
    logger.info("Encoding task description with MiniLM...")
    text_embedding = model.encode_text(task_description)
    
    logger.info("Creating video slices (SARM approach)...")
    # Convert to tensors
    video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
    if states is not None:
        state_embeddings = torch.tensor(states, dtype=torch.float32)
    else:
        state_embeddings = None
    
    # Create video slices: for each frame i, create a sequence using SARM's pattern
    # For SARM: 9 frames relative to current, with 30-frame gaps
    # Pattern: [current-240, current-210, ..., current-30, current]
    num_frames_model = model.config.num_frames
    frame_gap = model.config.frame_gap
    
    video_slices = []
    state_slices = []
    last_frame_indices = []
    
    for i in tqdm(range(len(video_embeddings)), desc="Creating slices"):
        # For SARM, create sequence relative to current frame (matching training pattern)
        # Pattern: [current-240, current-210, ..., current-30, current]
        # This matches observation_delta_indices: range(-240, 1, 30)
        
        # Compute frame indices for this slice (relative to current frame i)
        frame_indices = []
        for j in range(num_frames_model):
            # Start from -(num_frames_model-1) * frame_gap and go to 0
            offset = -(num_frames_model - 1 - j) * frame_gap
            idx = i + offset
            
            # Clamp to valid range [0, current_frame]
            if idx < 0:
                idx = 0  # Pad with first available frame
            
            frame_indices.append(idx)
        
        # Extract slice
        video_slice = video_embeddings[frame_indices]
        video_slices.append(video_slice)
        
        if state_embeddings is not None:
            state_slice = state_embeddings[frame_indices]
            state_slices.append(state_slice)
        
        # Track which frame index corresponds to the "current" frame
        last_frame_indices.append(min(i, len(frame_indices) - 1))
    
    video_slices = torch.stack(video_slices)  # (num_frames, num_frames_model, 512)
    if state_embeddings is not None:
        state_slices = torch.stack(state_slices)  # (num_frames, num_frames_model, state_dim)
    else:
        state_slices = None
    
    logger.info("Running SARM inference on all slices...")
    # Process in batches
    all_progress = []
    all_stages = []
    
    for i in tqdm(range(0, len(video_slices), batch_size), desc="Inference"):
        batch_video = video_slices[i:i + batch_size].to(model.device)
        batch_states = state_slices[i:i + batch_size].to(model.device) if state_slices is not None else None
        batch_size_actual = batch_video.shape[0]
        
        # Replicate text embedding for batch
        batch_text = text_embedding.unsqueeze(0).repeat(batch_size_actual, 1).to(model.device)
        
        # Get predictions
        stage_logits, stage_probs, progress_preds = model.sarm_transformer(
            batch_video, batch_text, batch_states
        )
        
        # Extract last frame predictions (the "current" frame)
        # For SARM, we take the last frame in each sequence
        batch_progress = progress_preds[:, -1, 0].cpu().numpy()
        batch_stages = stage_probs[:, -1, :].cpu().numpy()
        
        all_progress.extend(batch_progress)
        all_stages.extend(batch_stages)
    
    return np.array(all_progress), np.array(all_stages)


def visualize_predictions(
    frames: np.ndarray,
    progress_predictions: np.ndarray,
    stage_predictions: np.ndarray,
    task_description: str,
    output_path: Path,
    show_frames: bool = False,
    num_sample_frames: int = 8,
    figsize: tuple = (14, 8)
):
    """
    Create visualization of SARM predictions.
    
    Args:
        frames: Video frames (num_frames, H, W, C)
        progress_predictions: Progress predictions (num_frames,)
        stage_predictions: Stage probabilities (num_frames, num_stages)
        task_description: Task description
        output_path: Path to save the figure
        show_frames: Whether to include sample frames
        num_sample_frames: Number of frames to show
        figsize: Figure size (width, height)
    """
    num_stages = stage_predictions.shape[1]
    stage_colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    
    if show_frames:
        # Create figure with progress plot, stage plot, and sample frames
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 4))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        ax_progress = fig.add_subplot(gs[0])
        ax_stages = fig.add_subplot(gs[1], sharex=ax_progress)
        ax_frames = fig.add_subplot(gs[2])
    else:
        # Just progress and stage plots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        ax_progress = fig.add_subplot(gs[0])
        ax_stages = fig.add_subplot(gs[1], sharex=ax_progress)
    
    frame_indices = np.arange(len(progress_predictions))
    
    # Plot 1: Progress over time
    ax_progress.plot(frame_indices, progress_predictions, linewidth=2, color='#2E86AB', label='Predicted Progress')
    ax_progress.fill_between(frame_indices, 0, progress_predictions, alpha=0.3, color='#2E86AB')
    ax_progress.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_progress.set_ylabel('Task Progress', fontsize=12)
    ax_progress.set_title(f'SARM Task Progress & Stage Prediction\nTask: "{task_description}"', 
                          fontsize=14, fontweight='bold')
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
    ax_progress.text(0.98, 0.02, stats_text, transform=ax_progress.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Stage predictions (stacked area plot)
    ax_stages.stackplot(frame_indices, *[stage_predictions[:, i] for i in range(num_stages)],
                        colors=stage_colors, alpha=0.8, labels=[f'Stage {i+1}' for i in range(num_stages)])
    ax_stages.set_xlabel('Frame Index', fontsize=12)
    ax_stages.set_ylabel('Stage Probability', fontsize=12)
    ax_stages.set_ylim(0, 1)
    ax_stages.grid(True, alpha=0.3)
    ax_stages.legend(loc='upper left', ncol=num_stages, fontsize=8)
    
    # Plot 3: Sample frames (if requested)
    if show_frames:
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
            label = f'Frame {frame_idx}\nProg: {progress_val:.2f}\nStage: {stage_idx+1}'
            
            # Draw label on image
            ax_frames.text(x_start + frame_width / 2, -10, label, 
                          ha='center', va='top', fontsize=7, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax_frames.imshow(combined_image)
        ax_frames.set_title('Sample Frames', fontsize=12, pad=20)
    
    # Save figure
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
    
    # Determine which image key to use
    image_key = args.image_key if args.image_key is not None else model.config.image_key
    logger.info(f"Using image key: {image_key}")
    
    # Load episode data
    frames, states, start_idx, end_idx, dataset_task = load_episode_data(
        dataset, args.episode_index, image_key, args.state_key
    )
    
    # Use task description from dataset if available, otherwise use command-line argument
    task_description = dataset_task if dataset_task is not None else args.task_description
    logger.info(f"Using task description: '{task_description}'")
    
    # Run inference
    progress_predictions, stage_predictions = run_inference(model, frames, states, task_description)
    
    # Create visualization
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"sarm_prediction_ep{args.episode_index}.png"
    
    visualize_predictions(
        frames,
        progress_predictions,
        stage_predictions,
        task_description,
        output_path,
        show_frames=args.show_frames,
        num_sample_frames=args.num_sample_frames,
        figsize=tuple(args.figsize)
    )
    
    # Save predictions as numpy arrays
    predictions_path = output_dir / f"predictions_ep{args.episode_index}.npz"
    np.savez(predictions_path, progress=progress_predictions, stages=stage_predictions)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Dataset: {args.dataset_repo}")
    logger.info(f"Episode: {args.episode_index}")
    logger.info(f"Task: {task_description}")
    logger.info(f"Frames: {len(frames)}")
    logger.info(f"Final Progress: {progress_predictions[-1]:.3f}")
    logger.info(f"Max Progress: {progress_predictions.max():.3f}")
    logger.info(f"Mean Progress: {progress_predictions.mean():.3f}")
    logger.info(f"Most Common Stage: {np.argmax(np.sum(stage_predictions, axis=0)) + 1}")
    logger.info(f"Visualization: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

