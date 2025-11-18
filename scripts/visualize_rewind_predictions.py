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
Inference script for ReWiND Reward Model.

This script loads a trained ReWiND model and runs inference on a dataset episode,
generating visualizations of the predicted task progression over time.

Example usage:
    python scripts/visualize_rewind_predictions.py \
        --model-id username/rewind-model \
        --dataset-repo lerobot/aloha_sim_insertion_human \
        --episode-index 0 \
        --output-dir outputs/rewind_viz \
        --task-description "insert the peg into the socket"
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.rewind.modeling_rewind import ReWiNDRewardModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ReWiND inference and visualize predictions")
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to trained ReWiND model"
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
        default="outputs/rewind_inference",
        help="Directory to save visualization outputs (default: outputs/rewind_inference)"
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Key for images in dataset (e.g., observation.images.image for jaco_play). If not specified, uses model config's image_key"
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
        default=[12, 6],
        help="Figure size as width height (default: 12 6)"
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
    image_key: str
) -> tuple[np.ndarray, int, int, str]:
    """
    Load all frames from a specific episode.
    
    Args:
        dataset: LeRobotDataset instance
        episode_index: Index of the episode to load
        image_key: Key for accessing images in the dataset
        
    Returns:
        Tuple of (frames, start_index, end_index, task_description)
    """
    # Get episode boundaries
    episode_data = dataset.meta.episodes
    start_idx = episode_data["dataset_from_index"][episode_index]
    end_idx = episode_data["dataset_to_index"][episode_index]
    
    logger.info(f"Loading episode {episode_index}: frames {start_idx} to {end_idx} ({end_idx - start_idx} frames)")
    
    # Get task description from the dataset if available
    task_description = None
    first_item = dataset[start_idx]
    if "task" in first_item:
        task_description = first_item["task"]
        print(f"✓ Extracted task from episode {episode_index}: '{task_description}'")
    
    # Load all frames from the episode
    frames = []
    for idx in tqdm(range(start_idx, end_idx), desc="Loading frames"):
        item = dataset[idx]
        # Get image from the item
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
    
    frames = np.array(frames)
    logger.info(f"Loaded {len(frames)} frames with shape {frames[0].shape}")
    
    return frames, start_idx, end_idx, task_description


@torch.no_grad()
def run_inference(
    model: ReWiNDRewardModel,
    frames: np.ndarray,
    task_description: str,
    batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ReWiND inference on video frames using the original ReWiND approach.
    
    This function creates video slices for all frames at once (similar to the original
    metaworld_label_reward.py), where each slice contains frames from start up to that point.
    
    Progress Normalization (from original ReWiND dataset.py):
    - Training: progress = [1, 2, ..., N] / remaining_length
      where remaining_length = episode_end - sequence_start
    - Inference: Starting from frame 0, remaining_length = total_episode_length
      So expected progress for frame i = (i + 1) / total_episode_length
    
    This function computes both:
    1. Model predictions (what the model actually predicts)
    2. Expected progress (ground truth based on frame position)
    
    Args:
        model: ReWiND model
        frames: Video frames (num_frames, H, W, C)
        task_description: Task description text
        batch_size: Batch size for processing slices
        
    Returns:
        Tuple of:
            - Model predictions for each frame (num_frames,)
            - Expected progress for each frame (num_frames,)
    """
    total_frames = len(frames)
    
    logger.info("Encoding video frames with DINO...")
    video_embeddings = model.encode_images(frames)
    
    logger.info("Encoding task description with MiniLM...")
    text_embedding = model.encode_text(task_description)
    
    logger.info("Creating video slices (original ReWiND approach)...")
    # Convert to tensors
    video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
    
    # Create video slices: for each frame i, create a sequence of frames [0:i+1]
    # This matches the original ReWiND inference approach
    video_slices = []
    for i in tqdm(range(len(video_embeddings)), desc="Creating slices"):
        # Slice from start to current frame (inclusive)
        video_slice = video_embeddings[:i + 1]
        
        # Pad or subsample to max_length
        if model.config.subsample_video:
            video_slice = model.padding_video(video_slice, model.config.max_length)
        
        video_slices.append(video_slice)
    
    video_slices = torch.stack(video_slices)  # (num_frames, max_length, 768)
    
    # Create last_index_mask to extract the relevant prediction for each slice
    # For slice i, the last valid frame is at position min(i, max_length-1)
    max_length = model.config.max_length
    last_index_mask = torch.zeros((len(video_slices), max_length), dtype=torch.bool)
    
    for i in range(len(video_slices)):
        last_frame_idx = min(i, max_length - 1)
        last_index_mask[i, last_frame_idx] = 1
    
    logger.info("Running ReWiND inference on all slices...")
    # Process in batches
    all_progress = []
    for i in tqdm(range(0, len(video_slices), batch_size), desc="Inference"):
        batch_video = video_slices[i:i + batch_size].to(model.device)
        batch_mask = last_index_mask[i:i + batch_size].to(model.device)
        batch_size_actual = batch_video.shape[0]
        
        # Replicate text embedding for batch
        batch_text = text_embedding.unsqueeze(0).repeat(batch_size_actual, 1).to(model.device)
        
        # Get predictions for all frames in batch
        progress_preds = model.rewind_transformer(batch_video, batch_text)  # (batch, max_length, 1)
        progress_preds = progress_preds.squeeze(-1)  # (batch, max_length)
        
        # Extract predictions using the last_index_mask
        # This gets the prediction for the last valid frame in each slice
        batch_progress = progress_preds[batch_mask].cpu().numpy()
        all_progress.extend(batch_progress)
    
    predictions = np.array(all_progress)
    
    # Compute expected progress based on original ReWiND normalization
    # When starting from frame 0, remaining_length = total_episode_length
    # Expected progress for frame i = (i + 1) / total_frames
    expected_progress = np.arange(1, total_frames + 1, dtype=np.float32) / total_frames
    
    logger.info(f"Inference complete. Predicted progress range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    logger.info(f"Expected progress range: [{expected_progress.min():.3f}, {expected_progress.max():.3f}]")
    
    return predictions, expected_progress


def visualize_predictions(
    frames: np.ndarray,
    predictions: np.ndarray,
    expected_progress: np.ndarray,
    task_description: str,
    output_path: Path,
    show_frames: bool = False,
    num_sample_frames: int = 8,
    figsize: tuple = (12, 6)
):
    """
    Create visualization of ReWiND predictions with expected progress comparison.
    
    Args:
        frames: Video frames (num_frames, H, W, C)
        predictions: Model progress predictions (num_frames,)
        expected_progress: Expected progress based on frame position (num_frames,)
        task_description: Task description
        output_path: Path to save the figure
        show_frames: Whether to include sample frames
        num_sample_frames: Number of frames to show
        figsize: Figure size (width, height)
    """
    if show_frames:
        # Create figure with progress plot and sample frames
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Progress plot
        ax_progress = fig.add_subplot(gs[0])
    else:
        # Just progress plot
        fig, ax_progress = plt.subplots(1, 1, figsize=figsize)
    
    # Plot progress over time
    frame_indices = np.arange(len(predictions))
    
    # Plot expected progress (ground truth)
    ax_progress.plot(frame_indices, expected_progress, linewidth=2, color='#A8DADC', 
                    linestyle='--', label='Expected Progress (Linear)', alpha=0.7)
    
    # Plot model predictions
    ax_progress.plot(frame_indices, predictions, linewidth=2.5, color='#2E86AB', 
                    label='Model Predictions')
    ax_progress.fill_between(frame_indices, 0, predictions, alpha=0.2, color='#2E86AB')
    
    # Add reference line at 1.0
    ax_progress.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Styling
    ax_progress.set_xlabel('Frame Index', fontsize=12)
    ax_progress.set_ylabel('Task Progress', fontsize=12)
    ax_progress.set_title(f'ReWiND Task Progress Prediction\nTask: "{task_description}"', 
                          fontsize=14, fontweight='bold')
    ax_progress.grid(True, alpha=0.3)
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc='upper left')
    
    # Compute alignment metrics
    mae = np.mean(np.abs(predictions - expected_progress))
    rmse = np.sqrt(np.mean((predictions - expected_progress) ** 2))
    
    # Add statistics box
    stats_text = (
        f'Frames: {len(predictions)}\n'
        f'Model Final: {predictions[-1]:.3f}\n'
        f'Model Max: {predictions.max():.3f}\n'
        f'Model Mean: {predictions.mean():.3f}\n'
        f'MAE: {mae:.3f}\n'
        f'RMSE: {rmse:.3f}'
    )
    ax_progress.text(0.98, 0.02, stats_text, transform=ax_progress.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Show sample frames if requested
    if show_frames:
        # Select evenly spaced frames
        frame_indices_to_show = np.linspace(0, len(frames) - 1, num_sample_frames, dtype=int)
        
        # Create subplot for frames
        ax_frames = fig.add_subplot(gs[1])
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
            
            # Add frame number and progress value
            progress_val = predictions[frame_idx]
            label = f'Frame {frame_idx}\nProgress: {progress_val:.3f}'
            
            # Draw label on image
            ax_frames.text(x_start + frame_width / 2, -10, label, 
                          ha='center', va='top', fontsize=8, 
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
    logger.info(f"Loading ReWiND model from {args.model_id}...")
    model = ReWiNDRewardModel.from_pretrained(args.model_id)
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
    
    # Load episode data (this also extracts the task description from the episode)
    frames, start_idx, end_idx, dataset_task = load_episode_data(dataset, args.episode_index, image_key)
    
    # Use task description from dataset if available, otherwise use command-line argument
    task_description = dataset_task if dataset_task is not None else args.task_description
    logger.info(f"Using task description: '{task_description}'")
    
    # Run inference
    predictions, expected_progress = run_inference(model, frames, task_description)
    
    # Create visualization
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"rewind_prediction_ep{args.episode_index}.png"
    
    visualize_predictions(
        frames,
        predictions,
        expected_progress,
        task_description,
        output_path,
        show_frames=args.show_frames,
        num_sample_frames=args.num_sample_frames,
        figsize=tuple(args.figsize)
    )
    
    # Save predictions and expected progress as numpy arrays
    predictions_path = output_dir / f"predictions_ep{args.episode_index}.npy"
    expected_path = output_dir / f"expected_progress_ep{args.episode_index}.npy"
    np.save(predictions_path, predictions)
    np.save(expected_path, expected_progress)
    logger.info(f"Saved predictions array to {predictions_path}")
    logger.info(f"Saved expected progress to {expected_path}")
    
    # Compute alignment metrics
    mae = np.mean(np.abs(predictions - expected_progress))
    rmse = np.sqrt(np.mean((predictions - expected_progress) ** 2))
    correlation = np.corrcoef(predictions, expected_progress)[0, 1]
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Dataset: {args.dataset_repo}")
    logger.info(f"Episode: {args.episode_index}")
    logger.info(f"Task: {task_description}")
    logger.info(f"Frames: {len(frames)}")
    logger.info(f"\nModel Predictions:")
    logger.info(f"  Final: {predictions[-1]:.3f}")
    logger.info(f"  Max: {predictions.max():.3f}")
    logger.info(f"  Mean: {predictions.mean():.3f}")
    logger.info(f"  Std: {predictions.std():.3f}")
    logger.info(f"\nExpected Progress (Linear):")
    logger.info(f"  Final: {expected_progress[-1]:.3f}")
    logger.info(f"  Mean: {expected_progress.mean():.3f}")
    logger.info(f"\nAlignment Metrics:")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  RMSE: {rmse:.3f}")
    logger.info(f"  Correlation: {correlation:.3f}")
    logger.info(f"\nOutput:")
    logger.info(f"  Visualization: {output_path}")
    logger.info("="*60)
    
    # Diagnostic warnings
    if predictions.std() < 0.05:
        logger.warning("\n⚠ WARNING: Mode collapse detected (std < 0.05)")
        logger.warning("  Model predictions show very low variance.")
        logger.warning("  This indicates the model was likely trained with incorrect")
        logger.warning("  progress normalization (absolute indices instead of remaining length).")
    elif mae > 0.3:
        logger.warning("\n⚠ WARNING: High prediction error (MAE > 0.3)")
        logger.warning("  Model predictions deviate significantly from expected linear progress.")
        logger.warning("  Consider retraining with correct progress normalization.")
    elif correlation < 0.5:
        logger.warning("\n⚠ WARNING: Low correlation with expected progress (< 0.5)")
        logger.warning("  Model predictions don't align well with linear task progression.")
    else:
        logger.info("\n✓ Model predictions show healthy progression!")


if __name__ == "__main__":
    main()

