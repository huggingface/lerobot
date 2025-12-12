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
Compute SARM progress values for RA-BC (Reward-Aware Behavior Cloning) weighting.

This script processes all frames in a dataset with SARM to compute progress values [0, 1].
The results are saved as a parquet file that can be loaded during training for RA-BC weighting.

Uses multi-output extraction: each SARM query returns progress for 9 frames, so we only
need ~num_frames/30 queries instead of one per frame (~30x speedup).

Usage:
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4

The output is saved to the dataset's local cache directory as 'sarm_progress.parquet'.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors


def to_numpy_image(img) -> np.ndarray:
    """Convert image tensor to numpy uint8 (H, W, C)."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[-1]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def visualize_episode(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None
):
    """Create visualization with progress plot, stage probabilities, and sample frames.
    
    Same as sarm_inference_visualization.py
    """
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress, ax_stages, ax_frames = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # Progress plot
    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth"
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    # Stage predictions
    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    # Sample frames
    ax_frames.axis("off")
    num_sample = 8
    sample_indices = np.linspace(0, len(frames) - 1, num_sample, dtype=int)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        frame = frames[idx]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[np.argmax(stage_preds[idx])][:12]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_sparse_query_indices(ep_start: int, ep_end: int, frame_gap: int = 30) -> list[int]:
    """Generate sparse frame indices for efficient multi-output processing.

    SARM outputs progress for 9 frames per query: [0, X-120, X-90, X-60, X-30, X, X+30, X+60, X+90]
    (with frame_gap=30, deltas are [-1M, -120, -90, -60, -30, 0, +30, +60, +90])

    By querying every 30 frames, we get overlapping coverage of all frames.
    We only need ~num_frames/30 queries to cover the full episode.
    """
    num_frames = ep_end - ep_start
    min_query = frame_gap * 4  # Need at least 4*gap before current for full window

    indices = []
    # Early queries to cover frames 0 to min_query
    for early in range(0, min_query, frame_gap):
        indices.append(ep_start + early)
    # Main queries at stride = frame_gap
    for current in range(min_query, num_frames, frame_gap):
        indices.append(ep_start + current)

    return indices


def compute_frame_indices_from_query(
    query_idx: int, ep_start: int, ep_end: int, delta_indices: list[int]
) -> list[tuple[int, int]]:
    """Compute which absolute frame indices correspond to each position in the window.

    Returns list of (window_position, absolute_frame_idx) for valid frames.
    Skips position 0 (initial frame) as it's always frame 0 regardless of query.
    """
    num_frames = ep_end - ep_start
    results = []

    for pos, delta in enumerate(delta_indices):
        if pos == 0:  # Skip initial frame (always 0, handled separately)
            continue

        frame_rel = (query_idx - ep_start) + delta
        if 0 <= frame_rel < num_frames:
            results.append((pos, ep_start + frame_rel))

    return results


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str = "cuda",
    visualize: bool = True,
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        device: Device to use for inference
        visualize: Whether to save a visualization of the first episode
    """
    logging.info(f"Loading model: {reward_model_path}")
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    # Get keys and config from model
    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices
    frame_gap = reward_model.config.frame_gap

    # Load dataset to get fps
    logging.info(f"Loading dataset: {dataset_repo_id}")
    temp_dataset = LeRobotDataset(dataset_repo_id)
    fps = temp_dataset.fps

    # Build delta_timestamps and reload with temporal sampling
    delta_timestamps = {
        image_key: [idx / fps for idx in delta_indices],
        state_key: [idx / fps for idx in delta_indices],
    }
    dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)

    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    logging.info(f"Dataset: {num_episodes} episodes, {total_frames} frames, {len(delta_indices)} frames per sample")

    # Create preprocessor
    preprocess, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    # Determine which heads to compute
    dual_mode = reward_model.config.uses_dual_heads
    compute_sparse = head_mode in ("sparse", "both") or not dual_mode
    compute_dense = head_mode in ("dense", "both") and dual_mode

    # Storage arrays
    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_progress_sparse = [] if compute_sparse else None
    all_progress_dense = [] if compute_dense else None

    # Process all episodes
    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Get task description
        task = dataset[ep_start].get("task", "perform the task")

        # Use sparse query indices (~30x fewer queries)
        query_indices = generate_sparse_query_indices(ep_start, ep_end, frame_gap)

        # Dictionary to collect results (handles overlapping coverage)
        frame_results = {}

        for query_idx in tqdm(query_indices, desc=f"  Ep {episode_idx}", leave=False):
            try:
                sample = dataset[query_idx]

                batch = {
                    image_key: sample[image_key],
                    "task": task,
                    "index": query_idx,
                    "episode_index": episode_idx,
                }
                if state_key in sample:
                    batch[state_key] = sample[state_key]

                with torch.no_grad():
                    processed = preprocess(batch)
                    video_features = processed["video_features"]
                    text_features = processed["text_features"]
                    state_features = processed.get("state_features")

                    # Run model and get ALL frame outputs
                    if dual_mode:
                        preds = reward_model.sarm_transformer(
                            video_features, text_features, state_features, head_mode=head_mode
                        )
                        sparse_progress = preds["sparse"][2][0, :, 0].cpu().numpy() if compute_sparse else None
                        dense_progress = preds["dense"][2][0, :, 0].cpu().numpy() if compute_dense else None
                    else:
                        _, _, progress = reward_model.sarm_transformer(
                            video_features, text_features, state_features
                        )
                        sparse_progress = progress[0, :, 0].cpu().numpy()
                        dense_progress = None

                    # Extract progress for all 8 non-initial frames in the window
                    valid_frames = compute_frame_indices_from_query(query_idx, ep_start, ep_end, delta_indices)
                    for window_pos, abs_frame_idx in valid_frames:
                        sparse_val = float(sparse_progress[window_pos]) if sparse_progress is not None else np.nan
                        dense_val = float(dense_progress[window_pos]) if dense_progress is not None else np.nan
                        frame_results[abs_frame_idx] = (sparse_val, dense_val)

            except Exception as e:
                logging.warning(f"Failed to process query {query_idx}: {e}")

        # Handle initial frame (frame 0 has 0 progress)
        if ep_start not in frame_results:
            frame_results[ep_start] = (0.0, 0.0)

        # Convert dict results to lists
        for frame_idx in sorted(frame_results.keys()):
            sparse_val, dense_val = frame_results[frame_idx]
            local_idx = frame_idx - ep_start
            all_indices.append(frame_idx)
            all_episode_indices.append(episode_idx)
            all_frame_indices.append(local_idx)
            if compute_sparse:
                all_progress_sparse.append(sparse_val)
            if compute_dense:
                all_progress_dense.append(dense_val)

        # Visualize first episode immediately (strided sampling like sarm_inference_visualization.py)
        if episode_idx == 0 and visualize:
            num_frames = ep_end - ep_start
            viz_stride = 30  # Same as sarm_inference_visualization.py default
            
            # Generate strided indices (same as sarm_inference_visualization.py)
            strided_viz_indices = list(range(ep_start, ep_end, viz_stride))
            num_viz_frames = len(strided_viz_indices)
            
            viz_progress = np.full(num_viz_frames, np.nan)
            viz_stages = [None] * num_viz_frames
            viz_gt_progress = np.full(num_viz_frames, np.nan)
            viz_gt_stages = np.full(num_viz_frames, np.nan)
            viz_frames = []
            
            for viz_idx, frame_idx in enumerate(tqdm(strided_viz_indices, desc="  Collecting viz data", leave=False)):
                sample = dataset[frame_idx]
                viz_frames.append(to_numpy_image(sample[image_key]))
                
                batch = {
                    image_key: sample[image_key],
                    "task": task,
                    "index": frame_idx,
                    "episode_index": episode_idx,
                }
                if state_key in sample:
                    batch[state_key] = sample[state_key]
                
                with torch.no_grad():
                    processed = preprocess(batch)
                    video_features = processed["video_features"]
                    text_features = processed["text_features"]
                    state_features = processed.get("state_features")
                    
                    # Get ground truth from preprocessor
                    target_idx = 5  # Frame with delta=0 in centered pattern
                    if "sparse_progress_targets" in processed:
                        viz_gt_progress[viz_idx] = processed["sparse_progress_targets"][0, target_idx, 0].cpu().item()
                        viz_gt_stages[viz_idx] = processed["sparse_stage_labels"][0, target_idx].cpu().item()
                    
                    # Get predictions
                    if dual_mode:
                        preds = reward_model.sarm_transformer(
                            video_features, text_features, state_features, head_mode=head_mode
                        )
                        _, probs, progress = preds["sparse"] if compute_sparse else preds["dense"]
                    else:
                        _, probs, progress = reward_model.sarm_transformer(
                            video_features, text_features, state_features
                        )
                    viz_progress[viz_idx] = progress[0, -1, 0].cpu().item()
                    viz_stages[viz_idx] = probs[0, -1, :].cpu().numpy()
            
            # Get stage labels
            stage_labels = reward_model.config.sparse_subtask_names or [f"Stage {i+1}" for i in range(viz_stages[0].shape[0])]
            
            # Create visualization
            viz_path = Path("sarm_progress_ep0.png").resolve()
            visualize_episode(
                frames=np.array(viz_frames),
                progress_preds=viz_progress,
                stage_preds=np.array(viz_stages),
                title=task,
                output_path=viz_path,
                stage_labels=stage_labels,
                gt_progress=viz_gt_progress if not np.all(np.isnan(viz_gt_progress)) else None,
                gt_stages=viz_gt_stages if not np.all(np.isnan(viz_gt_stages)) else None,
            )
            print(f"\nVisualization saved to: {viz_path}\n")

    # Create output table
    table_data = {
        "index": np.array(all_indices, dtype=np.int64),
        "episode_index": np.array(all_episode_indices, dtype=np.int64),
        "frame_index": np.array(all_frame_indices, dtype=np.int64),
    }
    if compute_sparse:
        table_data["progress_sparse"] = np.array(all_progress_sparse, dtype=np.float32)
    if compute_dense:
        table_data["progress_dense"] = np.array(all_progress_dense, dtype=np.float32)

    # Sort by index
    df = pa.table(table_data).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)
    final_table = pa.Table.from_pandas(df, preserve_index=False)

    # Determine output path
    if output_path is None:
        output_path = Path(dataset.root) / "sarm_progress.parquet"
    else:
        output_path = Path(output_path)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, output_path)
    logging.info(f"Saved {len(final_table)} frame progress values to {output_path}")

    # Print statistics
    if "progress_sparse" in df.columns:
        valid = df["progress_sparse"].dropna()
        logging.info(f"Sparse progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                     f"min={valid.min():.4f}, max={valid.max():.4f}")

    if "progress_dense" in df.columns:
        valid = df["progress_dense"].dropna()
        logging.info(f"Dense progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                     f"min={valid.min():.4f}, max={valid.max():.4f}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute SARM progress values for RA-BC weighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python examples/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path pepijn223/sarm_single_uni4
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
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="SARM head to use (default: sparse)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization of first episode",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload progress file to the dataset repo on HuggingFace Hub",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        output_path=args.output_path,
        head_mode=args.head_mode,
        device=args.device,
        visualize=not args.no_visualize,
    )

    print(f"\nSARM progress values saved to: {output_path}")

    # Upload to Hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi
        
        api = HfApi()
        hub_path = "sarm_progress.parquet"
        
        print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=hub_path,
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}")
        
        print(f"\nTo use in training, add to your config:")
        print(f"  use_rabc: true")
        print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
        print(f"  rabc_head_mode: sparse  # or dense")
    else:
        print(f"\nTo use in training, add to your config:")
        print(f"  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print(f"  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
