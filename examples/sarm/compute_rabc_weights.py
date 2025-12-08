#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
This script processes all frames in a dataset with SARM to compute progress values [0, 1].
The results are saved as a parquet file that can be loaded during training for RA-BC weighting.

Uses the same sampling strategy as sarm_inference_visualization.py for efficient
temporal coverage with SARM's 9-frame window structure.

Usage:
    python examples/sarm/compute_rabc_weights.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4

The output is saved to the dataset's local cache directory as 'sarm_progress.parquet'.
Each row contains:
    - index: Global frame index in dataset (matches dataset's frame indexing)
    - episode_index: Episode ID
    - frame_index: Frame index within episode  
    - progress_sparse: SARM sparse head progress prediction [0, 1]
    - progress_dense: SARM dense head progress prediction [0, 1] (if head_mode="both" or "dense")

During training, RABCWeights loads this file and computes:
    - progress_delta = progress[t + chunk_size] - progress[t]
    - rabc_weight based on the delta (paper Eq. 8-9)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.policies.sarm.modeling_sarm import SARM
from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors
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


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "both",
    stride: int = 30,
    device: str = "cuda",
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        stride: Frame stride for SARM window sampling (default: 30)
        device: Device to run inference on
    """
    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    logging.info(f"Loading reward model: {reward_model_path}")
    reward_model = SARM.from_pretrained(reward_model_path)
    reward_model.to(device)
    reward_model.eval()

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
    logging.info(f"Stride: {stride}, Head mode: {head_mode}")

    # Determine which heads to compute
    compute_sparse = head_mode in ("sparse", "both")
    compute_dense = head_mode in ("dense", "both")
    
    # Check if model has both heads
    has_sparse = reward_model.config.sparse_subtask_names is not None
    has_dense = reward_model.config.dense_subtask_names is not None
    
    if compute_sparse and not has_sparse:
        logging.warning("Model does not have sparse head, skipping sparse progress")
        compute_sparse = False
    if compute_dense and not has_dense:
        logging.warning("Model does not have dense head, skipping dense progress")
        compute_dense = False
    
    if not compute_sparse and not compute_dense:
        raise ValueError("No valid head mode available for this model")

    total_frames = dataset.num_frames
    all_progress_sparse = np.full(total_frames, np.nan, dtype=np.float32) if compute_sparse else None
    all_progress_dense = np.full(total_frames, np.nan, dtype=np.float32) if compute_dense else None
    all_episode_indices = np.zeros(total_frames, dtype=np.int64)
    all_frame_indices = np.zeros(total_frames, dtype=np.int64)

    for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
        ep_start = dataset.episode_data_index["from"][episode_idx].item()
        ep_end = dataset.episode_data_index["to"][episode_idx].item()
        
        # Get task description
        task = ""
        if hasattr(dataset.meta, "episodes") and dataset.meta.episodes:
            task = dataset.meta.episodes[episode_idx].get("task", "")
        if not task and hasattr(dataset.meta, "tasks") and dataset.meta.tasks:
            task = list(dataset.meta.tasks.values())[0]
        
        # Generate strided indices
        strided_indices = generate_strided_indices(ep_start, ep_end, stride=stride)
        
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
                    
                    # Compute sparse progress
                    if compute_sparse:
                        progress_sparse = reward_model.calculate_rewards(
                            text_features,
                            video_features,
                            state_features,
                            return_all_frames=False,
                            head_mode="sparse",
                        )
                        if isinstance(progress_sparse, tuple):
                            progress_sparse = progress_sparse[0]
                        if isinstance(progress_sparse, torch.Tensor):
                            progress_sparse = progress_sparse.flatten()[0].item()
                        elif isinstance(progress_sparse, np.ndarray):
                            progress_sparse = float(progress_sparse.flatten()[0])
                        all_progress_sparse[global_idx] = progress_sparse
                    
                    # Compute dense progress
                    if compute_dense:
                        progress_dense = reward_model.calculate_rewards(
                            text_features,
                            video_features,
                            state_features,
                            return_all_frames=False,
                            head_mode="dense",
                        )
                        if isinstance(progress_dense, tuple):
                            progress_dense = progress_dense[0]
                        if isinstance(progress_dense, torch.Tensor):
                            progress_dense = progress_dense.flatten()[0].item()
                        elif isinstance(progress_dense, np.ndarray):
                            progress_dense = float(progress_dense.flatten()[0])
                        all_progress_dense[global_idx] = progress_dense
                
            except Exception as e:
                logging.warning(f"Failed to process frame {global_idx}: {e}")

    all_indices = np.arange(total_frames, dtype=np.int64)

    # Create PyArrow table with progress values
    table_data = {
        "index": all_indices,
        "episode_index": all_episode_indices,
        "frame_index": all_frame_indices,
    }
    
    if compute_sparse:
        table_data["progress_sparse"] = all_progress_sparse
    if compute_dense:
        table_data["progress_dense"] = all_progress_dense
    
    table = pa.table(table_data)

    if output_path is None:
        dataset_path = Path(dataset.root)
        output_path = dataset_path / "sarm_progress.parquet"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pq.write_table(table, output_path)
    logging.info(f"Saved {len(all_indices)} frame progress values to {output_path}")

    # Print statistics
    if compute_sparse:
        valid_mask = ~np.isnan(all_progress_sparse)
        valid_progress = all_progress_sparse[valid_mask]
        logging.info(f"Sparse progress statistics:")
        logging.info(f"  Valid frames: {np.sum(valid_mask)} / {total_frames} ({100*np.mean(valid_mask):.1f}%)")
        logging.info(f"  Mean: {np.mean(valid_progress):.4f}")
        logging.info(f"  Std:  {np.std(valid_progress):.4f}")
        logging.info(f"  Min:  {np.min(valid_progress):.4f}")
        logging.info(f"  Max:  {np.max(valid_progress):.4f}")
    
    if compute_dense:
        valid_mask = ~np.isnan(all_progress_dense)
        valid_progress = all_progress_dense[valid_mask]
        logging.info(f"Dense progress statistics:")
        logging.info(f"  Valid frames: {np.sum(valid_mask)} / {total_frames} ({100*np.mean(valid_mask):.1f}%)")
        logging.info(f"  Mean: {np.mean(valid_progress):.4f}")
        logging.info(f"  Std:  {np.std(valid_progress):.4f}")
        logging.info(f"  Min:  {np.min(valid_progress):.4f}")
        logging.info(f"  Max:  {np.max(valid_progress):.4f}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute SARM progress predictions for RA-BC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (computes both sparse and dense)
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4

    # Sparse only
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id my_dataset \\
        --reward-model-path my_sarm_model \\
        --head-mode sparse
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
        "--output-path",
        type=str,
        default=None,
        help="Output path. If not set, saves to dataset's cache directory",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="both",
        choices=["sparse", "dense", "both"],
        help="SARM head to use (default: both)",
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
    
    args = parser.parse_args()
    
    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        stride=args.stride,
        device=args.device,
    )
    
    print(f"\nSARM progress values saved to: {output_path}")
    print(f"\nTo use in training, add to your config:")
    print(f"  use_rabc: true")
    print(f"  rabc_progress_path: {output_path}")
    print(f"  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
