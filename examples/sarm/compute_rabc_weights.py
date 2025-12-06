#!/usr/bin/env python
"""
Precompute RA-BC progress deltas for a dataset using SARM.

This script processes all frames in a dataset with SARM to compute progress values,
then calculates progress deltas (progress[t + chunk_size] - progress[t]) for each frame.
The results are saved as a parquet file that can be loaded during training.

Uses the same sampling strategy as sarm_inference_visualization.py for efficient
temporal coverage with SARM's 9-frame window structure.

Usage:
    python examples/sarm/compute_rabc_weights.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4 \
        --chunk-size 50

The output is saved to the dataset's local cache directory as 'rabc_weights.parquet'.
Each row contains:
    - index: Global frame index in dataset (matches dataset's frame indexing)
    - episode_index: Episode ID
    - frame_index: Frame index within episode  
    - progress: SARM progress prediction [0, 1]
    - progress_delta: progress[t + chunk_size] - progress[t]
    - rabc_weight: Computed weight based on progress delta (paper Eq. 8-9)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def generate_strided_indices(
    ep_start: int, ep_end: int, stride: int = 30, num_window_frames: int = 9
) -> list[int]:
    """Generate frame indices ordered by window structure for efficient temporal coverage.

    For SARM, each 9-frame window is [0, second, second+30, second+60, ..., current_frame] where:
    - Frame 0: always episode start (initial frame)
    - Frames 1-8: 8 frames at stride=30 intervals, with second = current_frame - 7*stride

    Processing order (non-overlapping chunks first, then remaining):

    Chunk 0 (second_frame 1→30, current 211→240):
      [0,1,31,61,91,121,151,181,211]     ← second=1, current=211
      [0,2,32,62,92,122,152,182,212]     ← second=2, current=212
      ...
      [0,30,60,90,120,150,180,210,240]   ← second=30, current=240

    Chunk 1 (second_frame 241→270, current 451→480):
      [0,241,271,301,331,361,391,421,451] ← second=241, current=451
      [0,242,272,302,332,362,392,422,452] ← second=242, current=452
      ...

    Then remaining frames (0-210, 241-450, etc.) are filled at the end.
    """
    num_frames = ep_end - ep_start
    window_span = (num_window_frames - 2) * stride  # 7 * 30 = 210 (current - second)
    chunk_size = (num_window_frames - 1) * stride  # 8 * 30 = 240 (gap between chunk starts)

    indices = []

    # Process in chunks: chunk 0 starts at second_frame=1, chunk 1 at second_frame=241, etc.
    chunk_idx = 0
    while True:
        chunk_start_second = chunk_idx * chunk_size + 1  # 1, 241, 481, ...
        chunk_end_second = chunk_start_second + stride  # 31, 271, 511, ...

        any_valid = False
        for second_frame in range(chunk_start_second, chunk_end_second):
            current_frame = second_frame + window_span  # second + 210
            if current_frame < num_frames:
                indices.append(ep_start + current_frame)
                any_valid = True

        if not any_valid:
            break
        chunk_idx += 1

    # Fill in remaining frames (those not covered by the chunk pattern)
    covered = set(indices)
    for i in range(ep_start, ep_end):
        if i not in covered:
            indices.append(i)

    return indices


def compute_rabc_weights(
    dataset_repo_id: str,
    reward_model_path: str,
    chunk_size: int = 50,
    output_path: str | None = None,
    head_mode: str = "sparse",
    stride: int = 30,
    device: str = "cuda",
    kappa: float = 0.01,
    epsilon: float = 1e-6,
):
    """
    Compute RA-BC progress deltas for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        chunk_size: Number of frames ahead for computing progress delta
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse" or "dense")
        stride: Frame stride for SARM window sampling (default: 30)
        device: Device to run inference on
        kappa: RA-BC kappa threshold for high-quality samples
        epsilon: Small constant for numerical stability
    """
    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    logging.info(f"Loading reward model: {reward_model_path}")
    from lerobot.policies.sarm.modeling_sarm import SARM
    from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

    reward_model = SARM.from_pretrained(reward_model_path)
    reward_model.to(device)
    reward_model.eval()

    # Create preprocessor for SARM
    preprocessor, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    # Determine image and state keys
    image_key = getattr(reward_model.config, "image_key", None)
    if image_key is None:
        for key in dataset.meta.camera_keys:
            image_key = key
            break
    state_key = getattr(reward_model.config, "state_key", "observation.state")
    
    logging.info(f"Using image_key: {image_key}, state_key: {state_key}")
    logging.info(f"Dataset has {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logging.info(f"Chunk size: {chunk_size}, Stride: {stride}")

    # Storage for all frame data - pre-allocate for all frames
    total_frames = dataset.num_frames
    all_progress = np.full(total_frames, np.nan, dtype=np.float32)
    all_episode_indices = np.zeros(total_frames, dtype=np.int64)
    all_frame_indices = np.zeros(total_frames, dtype=np.int64)

    # Process all episodes using strided sampling (same as sarm_inference_visualization.py)
    for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
        ep_start = dataset.episode_data_index["from"][episode_idx].item()
        ep_end = dataset.episode_data_index["to"][episode_idx].item()
        
        # Get task description for this episode
        task = ""
        if hasattr(dataset.meta, "episodes") and dataset.meta.episodes:
            task = dataset.meta.episodes[episode_idx].get("task", "")
        if not task and hasattr(dataset.meta, "tasks") and dataset.meta.tasks:
            task = list(dataset.meta.tasks.values())[0]
        
        # Generate strided indices (same strategy as sarm_inference_visualization.py)
        strided_indices = generate_strided_indices(ep_start, ep_end, stride=stride)
        
        # Process frames in strided order
        for global_idx in tqdm(strided_indices, desc=f"Episode {episode_idx}", leave=False):
            local_idx = global_idx - ep_start
            
            # Store episode/frame indices
            all_episode_indices[global_idx] = episode_idx
            all_frame_indices[global_idx] = local_idx
            
            try:
                sample = dataset[global_idx]
                
                batch = {
                    image_key: sample[image_key],
                    "task": task,
                    "index": global_idx,
                    "episode_index": episode_idx,
                }
                if state_key in sample:
                    batch[state_key] = sample[state_key]
                
                with torch.no_grad():
                    processed = preprocessor(batch)
                    
                    video_features = processed["video_features"].to(device)
                    text_features = processed["text_features"].to(device)
                    state_features = processed.get("state_features")
                    if state_features is not None:
                        state_features = state_features.to(device)
                    
                    progress = reward_model.calculate_rewards(
                        text_features,
                        video_features,
                        state_features,
                        return_all_frames=False,
                        head_mode=head_mode,
                    )
                    
                    if isinstance(progress, tuple):
                        progress = progress[0]
                    
                    if isinstance(progress, torch.Tensor):
                        progress = progress.flatten()[0].item()
                    elif isinstance(progress, np.ndarray):
                        progress = float(progress.flatten()[0])
                    
                all_progress[global_idx] = progress
                
            except Exception as e:
                logging.warning(f"Failed to process frame {global_idx}: {e}")
                all_progress[global_idx] = np.nan

    # Create index array
    all_indices = np.arange(total_frames, dtype=np.int64)

    # Compute progress deltas per episode
    logging.info("Computing progress deltas...")
    all_progress_delta = np.full(total_frames, np.nan, dtype=np.float32)
    
    for episode_idx in range(dataset.num_episodes):
        ep_start = dataset.episode_data_index["from"][episode_idx].item()
        ep_end = dataset.episode_data_index["to"][episode_idx].item()
        
        for global_idx in range(ep_start, ep_end):
            future_idx = global_idx + chunk_size
            if future_idx < ep_end:
                current_progress = all_progress[global_idx]
                future_progress = all_progress[future_idx]
                
                if not np.isnan(current_progress) and not np.isnan(future_progress):
                    all_progress_delta[global_idx] = future_progress - current_progress

    # Compute RA-BC weights (paper Eq. 8-9)
    logging.info("Computing RA-BC weights...")
    all_rabc_weight = compute_weights_from_deltas(all_progress_delta, kappa=kappa, epsilon=epsilon)

    # Create PyArrow table
    table = pa.table({
        "index": all_indices,
        "episode_index": all_episode_indices,
        "frame_index": all_frame_indices,
        "progress": all_progress,
        "progress_delta": all_progress_delta,
        "rabc_weight": all_rabc_weight,
    })

    # Determine output path
    if output_path is None:
        dataset_path = Path(dataset.root)
        output_path = dataset_path / "rabc_weights.parquet"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    pq.write_table(table, output_path)
    logging.info(f"Saved {len(all_indices)} frame weights to {output_path}")

    # Print statistics
    valid_progress_mask = ~np.isnan(all_progress)
    valid_delta_mask = ~np.isnan(all_progress_delta)
    valid_weight_mask = ~np.isnan(all_rabc_weight)
    
    logging.info(f"Progress statistics:")
    logging.info(f"  Valid frames: {np.sum(valid_progress_mask)} / {total_frames}")
    logging.info(f"  Mean: {np.nanmean(all_progress):.4f}")
    logging.info(f"  Std:  {np.nanstd(all_progress):.4f}")
    
    valid_deltas = all_progress_delta[valid_delta_mask]
    logging.info(f"Progress delta statistics:")
    logging.info(f"  Valid frames: {len(valid_deltas)} / {total_frames}")
    logging.info(f"  Mean: {np.mean(valid_deltas):.4f}")
    logging.info(f"  Std:  {np.std(valid_deltas):.4f}")
    logging.info(f"  Min:  {np.min(valid_deltas):.4f}")
    logging.info(f"  Max:  {np.max(valid_deltas):.4f}")
    
    valid_weights = all_rabc_weight[valid_weight_mask]
    logging.info(f"RA-BC weight statistics:")
    logging.info(f"  Valid frames: {len(valid_weights)} / {total_frames}")
    logging.info(f"  Mean: {np.mean(valid_weights):.4f}")
    logging.info(f"  Zeros: {np.sum(valid_weights == 0)} / {len(valid_weights)} ({100*np.mean(valid_weights == 0):.1f}%)")
    logging.info(f"  Ones:  {np.sum(valid_weights == 1)} / {len(valid_weights)} ({100*np.mean(valid_weights == 1):.1f}%)")

    return output_path


def compute_weights_from_deltas(
    deltas: np.ndarray,
    kappa: float = 0.01,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute RA-BC weights from progress deltas.
    
    Following paper Eq. 8-9:
    - Soft weight: ˜wi = clip((ri − (µ − 2σ)) / (4σ + ε), 0, 1)
    - Final weight: wi = 1{ri > κ} + 1{0 ≤ ri ≤ κ}˜wi
    
    Args:
        deltas: Progress delta values
        kappa: Hard threshold for high-quality samples
        epsilon: Small constant for numerical stability
    
    Returns:
        Array of RA-BC weights
    """
    valid_mask = ~np.isnan(deltas)
    valid_deltas = deltas[valid_mask]
    
    if len(valid_deltas) == 0:
        return np.full_like(deltas, np.nan)
    
    # Compute statistics on valid deltas
    mean = max(np.mean(valid_deltas), 0.0)  # Clamp to non-negative
    std = max(np.std(valid_deltas), epsilon)
    
    logging.info(f"Computing weights with mean={mean:.4f}, std={std:.4f}, kappa={kappa}")
    
    # Compute soft weights
    lower_bound = mean - 2 * std
    soft_weights = (deltas - lower_bound) / (4 * std + epsilon)
    soft_weights = np.clip(soft_weights, 0.0, 1.0)
    
    # Apply paper's Eq. 9
    weights = np.zeros_like(deltas, dtype=np.float32)
    
    # High quality: ri > kappa → weight = 1
    high_quality_mask = deltas > kappa
    weights[high_quality_mask] = 1.0
    
    # Moderate quality: 0 <= ri <= kappa → weight = soft_weight
    moderate_mask = (deltas >= 0) & (deltas <= kappa)
    weights[moderate_mask] = soft_weights[moderate_mask]
    
    # Negative progress: ri < 0 → weight = 0 (already 0)
    # Invalid: NaN → weight = NaN
    weights[~valid_mask] = np.nan
    
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Precompute RA-BC weights using SARM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4

    # Custom chunk size and output path
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id my_dataset \\
        --reward-model-path my_sarm_model \\
        --chunk-size 25 \\
        --output-path ./weights/rabc_weights.parquet
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID or local path",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        required=True,
        help="Path to pretrained SARM model",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Frames ahead for progress delta (default: 50)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path. If not set, saves to dataset's cache directory",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="SARM head to use (default: sparse)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Frame stride for SARM window sampling (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.01,
        help="RA-BC kappa threshold (default: 0.01)",
    )
    
    args = parser.parse_args()
    
    output_path = compute_rabc_weights(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        chunk_size=args.chunk_size,
        output_path=args.output_path,
        head_mode=args.head_mode,
        stride=args.stride,
        device=args.device,
        kappa=args.kappa,
    )
    
    print(f"\nRA-BC weights saved to: {output_path}")
    print(f"\nTo use in training, add to your config:")
    print(f"  use_rabc: true")
    print(f"  rabc_weights_path: {output_path}")


if __name__ == "__main__":
    main()
