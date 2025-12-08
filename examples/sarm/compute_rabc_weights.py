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

Supports multi-GPU parallel processing for faster computation.

Usage:
    # Single GPU
    python examples/sarm/compute_rabc_weights.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4

    # Multi-GPU (auto-detect)
    python examples/sarm/compute_rabc_weights.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4 \
        --num-gpus 4

The output is saved to the dataset's local cache directory as 'sarm_progress.parquet'.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
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


def batch_decode_episode_frames(
def process_episodes_worker(
    rank: int,
    world_size: int,
    dataset_repo_id: str,
    reward_model_path: str,
    episode_indices: list[int],
    head_mode: str,
    stride: int,
    output_dir: Path,
    total_frames: int,
):
    """Worker function to process a subset of episodes on a specific GPU."""
    device = f"cuda:{rank}"
    
    # Set up logging for this worker
    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU {rank}] %(asctime)s %(levelname)s %(message)s"
    )
    
    logging.info(f"Starting worker on {device}, processing {len(episode_indices)} episodes")
    
    # Load dataset
    dataset = LeRobotDataset(dataset_repo_id)
    
    # Load reward model on this GPU
    reward_model = SARM.from_pretrained(reward_model_path)
    reward_model.to(device)
    reward_model.eval()
    
    # Update config device before creating preprocessor so CLIP loads on correct GPU
    reward_model.config.device = device
    
    # Create preprocessor with CLIP on GPU
    preprocessor, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    
    # Ensure CLIP is on the correct GPU (belt and suspenders)
    if hasattr(preprocessor, 'clip_model'):
        preprocessor.clip_model = preprocessor.clip_model.to(device)
        preprocessor.device = torch.device(device)
    logging.info(f"CLIP model loaded on {device}")
    
    # Determine image and state keys
    image_key = getattr(reward_model.config, "image_key", None)
    if image_key is None:
        for key in dataset.meta.camera_keys:
            image_key = key
            break
    state_key = getattr(reward_model.config, "state_key", "observation.state")
    
    # Determine which heads to compute
    compute_sparse = head_mode in ("sparse", "both")
    compute_dense = head_mode in ("dense", "both")
    
    has_sparse = reward_model.config.sparse_subtask_names is not None
    has_dense = reward_model.config.dense_subtask_names is not None
    
    if compute_sparse and not has_sparse:
        compute_sparse = False
    if compute_dense and not has_dense:
        compute_dense = False
    
    # Storage arrays - only for frames this worker processes
    worker_indices = []
    worker_episode_indices = []
    worker_frame_indices = []
    worker_progress_sparse = [] if compute_sparse else None
    worker_progress_dense = [] if compute_dense else None
    
    # Process assigned episodes
    for episode_idx in tqdm(episode_indices, desc=f"GPU {rank}", position=rank):
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
        
        for global_idx in strided_indices:
            local_idx = global_idx - ep_start
            
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
                    
                    progress_sparse_val = np.nan
                    progress_dense_val = np.nan
                    
                    if compute_sparse:
                        progress = reward_model.calculate_rewards(
                            text_features, video_features, state_features,
                            return_all_frames=False, head_mode="sparse",
                        )
                        if isinstance(progress, tuple):
                            progress = progress[0]
                        if isinstance(progress, torch.Tensor):
                            progress_sparse_val = progress.flatten()[0].item()
                        elif isinstance(progress, np.ndarray):
                            progress_sparse_val = float(progress.flatten()[0])
                    
                    if compute_dense:
                        progress = reward_model.calculate_rewards(
                            text_features, video_features, state_features,
                            return_all_frames=False, head_mode="dense",
                        )
                        if isinstance(progress, tuple):
                            progress = progress[0]
                        if isinstance(progress, torch.Tensor):
                            progress_dense_val = progress.flatten()[0].item()
                        elif isinstance(progress, np.ndarray):
                            progress_dense_val = float(progress.flatten()[0])
                
                # Store results
                worker_indices.append(global_idx)
                worker_episode_indices.append(episode_idx)
                worker_frame_indices.append(local_idx)
                if compute_sparse:
                    worker_progress_sparse.append(progress_sparse_val)
                if compute_dense:
                    worker_progress_dense.append(progress_dense_val)
                    
            except Exception as e:
                logging.warning(f"Failed to process frame {global_idx}: {e}")
                worker_indices.append(global_idx)
                worker_episode_indices.append(episode_idx)
                worker_frame_indices.append(local_idx)
                if compute_sparse:
                    worker_progress_sparse.append(np.nan)
                if compute_dense:
                    worker_progress_dense.append(np.nan)
    
    # Save worker results to temp file
    table_data = {
        "index": np.array(worker_indices, dtype=np.int64),
        "episode_index": np.array(worker_episode_indices, dtype=np.int64),
        "frame_index": np.array(worker_frame_indices, dtype=np.int64),
    }
    if compute_sparse:
        table_data["progress_sparse"] = np.array(worker_progress_sparse, dtype=np.float32)
    if compute_dense:
        table_data["progress_dense"] = np.array(worker_progress_dense, dtype=np.float32)
    
    table = pa.table(table_data)
    worker_output = output_dir / f"worker_{rank}.parquet"
    pq.write_table(table, worker_output)
    
    logging.info(f"Worker {rank} saved {len(worker_indices)} frames to {worker_output}")


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "both",
    stride: int = 30,
    num_gpus: int = 1,
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        stride: Frame stride for SARM window sampling (default: 30)
        num_gpus: Number of GPUs to use for parallel processing
    """
    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    total_frames = dataset.num_frames
    num_episodes = dataset.num_episodes
    
    logging.info(f"Dataset has {num_episodes} episodes, {total_frames} frames")
    logging.info(f"Using {num_gpus} GPU(s) for parallel processing")
    
    # Determine output path
    if output_path is None:
        dataset_path = Path(dataset.root)
        output_path = dataset_path / "sarm_progress.parquet"
    else:
        output_path = Path(output_path)
    
    # Create temp directory for worker outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        if num_gpus == 1:
            # Single GPU - run directly without multiprocessing
            process_episodes_worker(
                rank=0,
                world_size=1,
                dataset_repo_id=dataset_repo_id,
                reward_model_path=reward_model_path,
                episode_indices=list(range(num_episodes)),
                head_mode=head_mode,
                stride=stride,
                output_dir=temp_path,
                total_frames=total_frames,
            )
        else:
            # Multi-GPU - split episodes across workers
            episodes_per_gpu = num_episodes // num_gpus
            episode_splits = []
            
            for i in range(num_gpus):
                start_ep = i * episodes_per_gpu
                if i == num_gpus - 1:
                    # Last GPU gets remaining episodes
                    end_ep = num_episodes
                else:
                    end_ep = (i + 1) * episodes_per_gpu
                episode_splits.append(list(range(start_ep, end_ep)))
            
            logging.info(f"Episode splits: {[len(s) for s in episode_splits]}")
            
            # Spawn workers
            mp.set_start_method('spawn', force=True)
            processes = []
            
            for rank in range(num_gpus):
                p = mp.Process(
                    target=process_episodes_worker,
                    args=(
                        rank,
                        num_gpus,
                        dataset_repo_id,
                        reward_model_path,
                        episode_splits[rank],
                        head_mode,
                        stride,
                        temp_path,
                        total_frames,
                    ),
                )
                p.start()
                processes.append(p)
            
            # Wait for all workers
            for p in processes:
                p.join()
        
        # Merge worker outputs
        logging.info("Merging worker outputs...")
        
        worker_files = sorted(temp_path.glob("worker_*.parquet"))
        if not worker_files:
            raise RuntimeError("No worker output files found")
        
        # Read and concatenate all worker tables
        tables = [pq.read_table(f) for f in worker_files]
        merged_table = pa.concat_tables(tables)
        
        # Sort by index to ensure consistent ordering
        merged_df = merged_table.to_pandas()
        merged_df = merged_df.sort_values("index").reset_index(drop=True)
        
        # Convert back to Arrow table
        final_table = pa.Table.from_pandas(merged_df, preserve_index=False)
    
    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, output_path)
    logging.info(f"Saved {len(final_table)} frame progress values to {output_path}")
    
    # Print statistics
    merged_df = final_table.to_pandas()
    
    if "progress_sparse" in merged_df.columns:
        valid_progress = merged_df["progress_sparse"].dropna()
        logging.info(f"Sparse progress statistics:")
        logging.info(f"  Valid frames: {len(valid_progress)} / {len(merged_df)} ({100*len(valid_progress)/len(merged_df):.1f}%)")
        logging.info(f"  Mean: {valid_progress.mean():.4f}")
        logging.info(f"  Std:  {valid_progress.std():.4f}")
        logging.info(f"  Min:  {valid_progress.min():.4f}")
        logging.info(f"  Max:  {valid_progress.max():.4f}")
    
    if "progress_dense" in merged_df.columns:
        valid_progress = merged_df["progress_dense"].dropna()
        logging.info(f"Dense progress statistics:")
        logging.info(f"  Valid frames: {len(valid_progress)} / {len(merged_df)} ({100*len(valid_progress)/len(merged_df):.1f}%)")
        logging.info(f"  Mean: {valid_progress.mean():.4f}")
        logging.info(f"  Std:  {valid_progress.std():.4f}")
        logging.info(f"  Min:  {valid_progress.min():.4f}")
        logging.info(f"  Max:  {valid_progress.max():.4f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute SARM progress predictions for RA-BC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4

    # Multi-GPU (4 GPUs)
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id my_dataset \\
        --reward-model-path my_sarm_model \\
        --num-gpus 4
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
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel processing (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Validate num_gpus
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        logging.warning(f"Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        args.num_gpus = available_gpus
    
    if args.num_gpus < 1:
        args.num_gpus = 1
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        stride=args.stride,
        num_gpus=args.num_gpus,
    )
    
    print(f"\nSARM progress values saved to: {output_path}")
    print(f"\nTo use in training, add to your config:")
    print(f"  use_rabc: true")
    print(f"  rabc_progress_path: {output_path}")
    print(f"  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
