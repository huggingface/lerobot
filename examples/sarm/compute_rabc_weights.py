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

Usage:
    # Single GPU
    python examples/sarm/compute_rabc_weights.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4

    # Multi-GPU (4 GPUs)
    python examples/sarm/compute_rabc_weights.py \
        --dataset-repo-id lerobot/aloha_sim_insertion_human \
        --reward-model-path pepijn223/sarm_single_uni4 \
        --num-workers 4

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
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# NOTE: torch and lerobot imports are done inside functions for multi-GPU support.
# This allows workers to set CUDA_VISIBLE_DEVICES before torch initializes CUDA.

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
    dataset,
    episode_idx: int,
    strided_indices: list[int],
    image_key: str,
) -> dict:
    """Batch decode all frames for an episode from the video file.
    
    Opens video file once and decodes all requested frames.
    
    Args:
        dataset: LeRobotDataset instance
        episode_idx: Episode index
        strided_indices: List of global frame indices to decode
        image_key: Camera/video key to use
        
    Returns:
        Dict mapping global_idx -> decoded frame tensor
    """
    from lerobot.datasets.video_utils import decode_video_frames
    
    ep = dataset.meta.episodes[episode_idx]
    ep_start = ep["dataset_from_index"]
    fps = dataset.fps
    
    # Get video file info for this episode
    from_timestamp = ep[f"videos/{image_key}/from_timestamp"]
    video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, image_key)
    
    # Convert global indices to timestamps within the video
    # Each frame is at timestamp = (global_idx - ep_start) / fps within the episode
    # Add from_timestamp to get the actual timestamp in the video file
    timestamps = []
    for global_idx in strided_indices:
        local_idx = global_idx - ep_start
        episode_ts = local_idx / fps
        video_ts = from_timestamp + episode_ts
        timestamps.append(video_ts)
    
    # Batch decode all frames at once (opens video file only once!)
    frames = decode_video_frames(
        video_path, 
        timestamps, 
        tolerance_s=dataset.tolerance_s,
        backend=dataset.video_backend,
    )
    
    # Map global indices to decoded frames
    frame_dict = {}
    for i, global_idx in enumerate(strided_indices):
        frame_dict[global_idx] = frames[i]  # frames shape: [N, C, H, W]
    
    return frame_dict


def process_episode(
    episode_idx: int,
    dataset,
    reward_model,
    preprocessor,
    image_key: str,
    state_key: str,
    stride: int,
    compute_sparse: bool,
    compute_dense: bool,
    device: str,
) -> dict:
    """Process a single episode and return progress values."""
    import time
    import torch
    
    ep = dataset.meta.episodes[episode_idx]
    ep_start = ep["dataset_from_index"]
    ep_end = ep["dataset_to_index"]
    
    # Get task description from parquet data (doesn't require video access)
    dataset._ensure_hf_dataset_loaded()
    task_idx = dataset.hf_dataset[ep_start]["task_index"]
    if hasattr(task_idx, 'item'):
        task_idx = task_idx.item()
    task = dataset.meta.tasks.iloc[task_idx].name
    
    # Generate strided indices for this episode
    strided_indices = generate_strided_indices(ep_start, ep_end, stride=stride)
    
    # BATCH DECODE: Open video file once and decode ALL frames
    t_decode = time.time()
    frame_dict = batch_decode_episode_frames(dataset, episode_idx, strided_indices, image_key)
    t_decode = time.time() - t_decode
    print(f"  Ep {episode_idx}: batch decoded {len(strided_indices)} frames in {t_decode:.2f}s ({len(strided_indices)/t_decode:.1f} fps)", flush=True)
    
    # Results for this episode
    results = {
        "indices": [],
        "episode_indices": [],
        "frame_indices": [],
        "progress_sparse": [] if compute_sparse else None,
        "progress_dense": [] if compute_dense else None,
    }
    
    # Load state data for all frames (from parquet, fast)
    state_data = {}
    if state_key:
        for global_idx in strided_indices:
            try:
                item = dataset.hf_dataset[global_idx]
                if state_key in item:
                    state_data[global_idx] = item[state_key]
            except Exception:
                pass
    
    # Process each frame (CLIP encoding + SARM inference)
    for i, global_idx in enumerate(tqdm(strided_indices, desc=f"Ep {episode_idx}", leave=False)):
        local_idx = global_idx - ep_start
        
        results["indices"].append(global_idx)
        results["episode_indices"].append(episode_idx)
        results["frame_indices"].append(local_idx)
        
        try:
            # Get pre-decoded frame from memory
            frame = frame_dict[global_idx]
            
            batch = {
                image_key: frame,
                "task": task,
                "index": global_idx,
                "episode_index": episode_idx,
            }
            if global_idx in state_data:
                batch[state_key] = state_data[global_idx]
            
            with torch.no_grad():
                t_preprocess = time.time()
                processed = preprocessor(batch)
                t_preprocess = time.time() - t_preprocess
                
                video_features = processed["video_features"].to(device)
                text_features = processed["text_features"].to(device)
                state_features = processed.get("state_features")
                if state_features is not None:
                    state_features = state_features.to(device)
                
                t_infer = time.time()
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
                    results["progress_sparse"].append(progress_sparse)
                
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
                    results["progress_dense"].append(progress_dense)
                t_infer = time.time() - t_infer
                
            # Log timing for first frame only
            if i == 0:
                print(f"  Ep {episode_idx} first frame: preprocess={t_preprocess:.2f}s, infer={t_infer:.4f}s", flush=True)
                    
        except Exception as e:
            logging.warning(f"Failed to process frame {global_idx}: {e}")
            if compute_sparse:
                results["progress_sparse"].append(np.nan)
            if compute_dense:
                results["progress_dense"].append(np.nan)
    
    # Clear frame dict to free memory
    del frame_dict
    
    return results


def worker_process_episodes(
    worker_id: int,
    gpu_id: int,
    episode_indices: list[int],
    dataset_repo_id: str,
    reward_model_path: str,
    head_mode: str,
    stride: int,
) -> list[dict]:
    """Worker function for parallel processing across GPUs."""
    import os
    import sys
    
    # Simple print with flush for debugging (logging may not work well in multiprocessing)
    def log(msg):
        print(f"[Worker {worker_id}] {msg}", flush=True)
    
    log(f"Starting on GPU {gpu_id}")
    
    # Set CUDA device for this worker BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log("Loading dataset...")
    dataset = LeRobotDataset(dataset_repo_id)
    log(f"Dataset loaded: {dataset.num_episodes} episodes")
    
    log("Loading model...")
    device = "cuda:0"  # After CUDA_VISIBLE_DEVICES, always use cuda:0
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    reward_model.to(device)
    reward_model.eval()
    log("Model loaded")
    
    log("Creating preprocessor...")

    reward_model.config.device = device
    preprocessor, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    log(f"Preprocessor created (CLIP on {device})")
    
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
    
    # Process assigned episodes
    all_results = []
    for i, ep_idx in enumerate(episode_indices):
        log(f"Processing episode {ep_idx} ({i+1}/{len(episode_indices)})")
        try:
            results = process_episode(
                ep_idx,
                dataset,
                reward_model,
                preprocessor,
                image_key,
                state_key,
                stride,
                compute_sparse,
                compute_dense,
                device,
            )
            all_results.append(results)
            log(f"Episode {ep_idx} done: {len(results['indices'])} frames")
        except Exception as e:
            log(f"ERROR: Failed episode {ep_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    log(f"Completed {len(all_results)} episodes")
    return all_results


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "both",
    stride: int = 30,
    num_workers: int = 1,
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        stride: Frame stride for SARM window sampling (default: 30)
        num_workers: Number of parallel workers (default: 1)
    """
    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_ids = list(range(min(num_workers, max(available_gpus, 1))))
    num_workers = min(num_workers, len(gpu_ids))
    
    logging.info(f"Dataset has {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logging.info(f"Using {num_workers} worker(s) on GPU(s): {gpu_ids}")
    
    # Check model config for head availability
    logging.info(f"Loading reward model to check config: {reward_model_path}")
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    
    compute_sparse = head_mode in ("sparse", "both")
    compute_dense = head_mode in ("dense", "both")
    
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
    
    del reward_model  # Free memory before spawning workers
    torch.cuda.empty_cache()
    
    total_frames = dataset.num_frames
    all_progress_sparse = np.full(total_frames, np.nan, dtype=np.float32) if compute_sparse else None
    all_progress_dense = np.full(total_frames, np.nan, dtype=np.float32) if compute_dense else None
    all_episode_indices = np.zeros(total_frames, dtype=np.int64)
    all_frame_indices = np.zeros(total_frames, dtype=np.int64)
    
    # Get all episode indices
    episode_indices = list(range(dataset.num_episodes))
    
    if num_workers > 1:
        # Multi-GPU parallel processing
        # Contiguous chunk distribution (worker 0 gets first N episodes, worker 1 gets next N, etc.)
        total_episodes = len(episode_indices)
        chunk_size = (total_episodes + num_workers - 1) // num_workers  # Ceiling division
        
        episodes_per_worker = []
        for w in range(num_workers):
            start_idx = w * chunk_size
            end_idx = min(start_idx + chunk_size, total_episodes)
            episodes_per_worker.append(episode_indices[start_idx:end_idx])
        
        logging.info(f"Distributing {total_episodes} episodes across {num_workers} workers (contiguous chunks)")
        for w in range(num_workers):
            if episodes_per_worker[w]:
                logging.info(f"  Worker {w} (GPU {gpu_ids[w]}): episodes {episodes_per_worker[w][0]}-{episodes_per_worker[w][-1]} ({len(episodes_per_worker[w])} total)")
        
        # Process in parallel
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    worker_process_episodes,
                    w,
                    gpu_ids[w],
                    episodes_per_worker[w],
                    dataset_repo_id,
                    reward_model_path,
                    head_mode,
                    stride,
                )
                for w in range(num_workers)
                if episodes_per_worker[w]
            ]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    worker_results = future.result()
                    for results in worker_results:
                        for i, global_idx in enumerate(results["indices"]):
                            all_episode_indices[global_idx] = results["episode_indices"][i]
                            all_frame_indices[global_idx] = results["frame_indices"][i]
                            if compute_sparse and results["progress_sparse"]:
                                all_progress_sparse[global_idx] = results["progress_sparse"][i]
                            if compute_dense and results["progress_dense"]:
                                all_progress_dense[global_idx] = results["progress_dense"][i]
                except Exception as e:
                    logging.error(f"Worker failed: {e}")
                    raise
    else:
        # Single GPU processing
        device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
        logging.info(f"Single worker mode on device: {device}")
        
        reward_model = SARMRewardModel.from_pretrained(reward_model_path)
        reward_model.to(device)
        reward_model.eval()
        
        reward_model.config.device = device
        preprocessor, _ = make_sarm_pre_post_processors(
            config=reward_model.config,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        logging.info(f"Preprocessor created with CLIP on {device}")
        
        image_key = getattr(reward_model.config, "image_key", None)
        if image_key is None:
            for key in dataset.meta.camera_keys:
                image_key = key
                break
        state_key = getattr(reward_model.config, "state_key", "observation.state")
        
        logging.info(f"Using image_key: {image_key}, state_key: {state_key}")
        
        for episode_idx in tqdm(episode_indices, desc="Processing episodes"):
            results = process_episode(
                episode_idx,
                dataset,
                reward_model,
                preprocessor,
                image_key,
                state_key,
                stride,
                compute_sparse,
                compute_dense,
                device,
            )
            
            for i, global_idx in enumerate(results["indices"]):
                all_episode_indices[global_idx] = results["episode_indices"][i]
                all_frame_indices[global_idx] = results["frame_indices"][i]
                if compute_sparse and results["progress_sparse"]:
                    all_progress_sparse[global_idx] = results["progress_sparse"][i]
                if compute_dense and results["progress_dense"]:
                    all_progress_dense[global_idx] = results["progress_dense"][i]
    
    # Create index array
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

    if compute_sparse:
        valid_mask = ~np.isnan(all_progress_sparse)
        valid_progress = all_progress_sparse[valid_mask]
        logging.info(f"Sparse progress statistics:")
        logging.info(f"  Valid frames: {np.sum(valid_mask)} / {total_frames} ({100*np.mean(valid_mask):.1f}%)")
        if len(valid_progress) > 0:
            logging.info(f"  Mean: {np.mean(valid_progress):.4f}")
            logging.info(f"  Std:  {np.std(valid_progress):.4f}")
            logging.info(f"  Min:  {np.min(valid_progress):.4f}")
            logging.info(f"  Max:  {np.max(valid_progress):.4f}")
    
    if compute_dense:
        valid_mask = ~np.isnan(all_progress_dense)
        valid_progress = all_progress_dense[valid_mask]
        logging.info(f"Dense progress statistics:")
        logging.info(f"  Valid frames: {np.sum(valid_mask)} / {total_frames} ({100*np.mean(valid_mask):.1f}%)")
        if len(valid_progress) > 0:
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
    # Single GPU
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4

    # Multi-GPU (4 workers)
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4 \\
        --num-workers 4
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
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for multi-GPU processing (default: 1)",
    )
    
    args = parser.parse_args()
    
    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        stride=args.stride,
        num_workers=args.num_workers,
    )
    
    print(f"\nSARM progress values saved to: {output_path}")
    print(f"\nTo use in training, add to your config:")
    print(f"  use_rabc: true")
    print(f"  rabc_progress_path: {output_path}")
    print(f"  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
