"""
Example: Visualize SARM predictions on a dataset episode.

This example shows how to:
1. Load a trained SARM model
2. Run inference on an episode using the preprocessor pipeline
3. Visualize predicted progress and stage probabilities vs ground truth
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

MODEL_ID = "pepijn223/sarm_single_uni4"  # HuggingFace model ID or local path
DATASET_REPO = "lerobot-data-collection/three-folds-dataset-full-11-07"  # Dataset to run inference on
EPISODE_INDEX = 0  # Episode to visualize
OUTPUT_DIR = Path("outputs/sarm_inference")
HEAD_MODE = "sparse"  # "sparse", "dense", or "both" (for dual-head models)
DEVICE = "cuda"


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


@torch.no_grad()
def run_inference(
    model, dataset, preprocess, episode_index, image_key, state_key, task_description, head_mode, stride=30
):
    """Run SARM inference on an episode, returning predictions and ground truth."""
    dual_mode = model.config.uses_dual_heads
    ep_start = dataset.meta.episodes["dataset_from_index"][episode_index]
    ep_end = dataset.meta.episodes["dataset_to_index"][episode_index]
    num_frames = ep_end - ep_start

    # Generate strided sampling order for better temporal coverage
    strided_indices = generate_strided_indices(ep_start, ep_end, stride)

    # Initialize arrays to store results in original frame order
    sparse_progress = np.full(num_frames, np.nan)
    sparse_stages = [None] * num_frames
    gt_sparse_progress = np.full(num_frames, np.nan)
    gt_sparse_stages = np.full(num_frames, np.nan)
    dense_progress = np.full(num_frames, np.nan)
    dense_stages = [None] * num_frames
    gt_dense_progress = np.full(num_frames, np.nan)
    gt_dense_stages = np.full(num_frames, np.nan)
    raw_frames = [None] * num_frames

    for frame_idx in tqdm(strided_indices, desc="Processing frames (strided)"):
        local_idx = frame_idx - ep_start
        sample = dataset[frame_idx]
        # With delta_timestamps, sample[image_key] is (T, C, H, W) - take last frame for display
        raw_frames[local_idx] = to_numpy_image(sample[image_key][-1])

        # Build batch dict with observation.* keys (as expected by batch_to_transition)
        batch = {
            image_key: sample[image_key],  # (T, C, H, W) temporal window
            "task": task_description,
            "index": frame_idx,
            "episode_index": episode_index,
        }
        if state_key in sample:
            batch[state_key] = sample[state_key]  # (T, state_dim) temporal window

        processed = preprocess(batch)

        # Extract ground truth from processor output
        # Use index 5 (target frame with delta=0) instead of -1 (last frame which is +3*gap ahead)
        # Sequence structure: [initial_frame, -4*gap, -3*gap, -2*gap, -gap, 0, +gap, +2*gap, +3*gap]
        target_idx = 5  # The frame with delta=0 in the centered pattern
        if "sparse_progress_targets" in processed:
            gt_sparse_progress[local_idx] = processed["sparse_progress_targets"][0, target_idx, 0].cpu().item()
            gt_sparse_stages[local_idx] = processed["sparse_stage_labels"][0, target_idx].cpu().item()
        if "dense_progress_targets" in processed:
            gt_dense_progress[local_idx] = processed["dense_progress_targets"][0, target_idx, 0].cpu().item()
            gt_dense_stages[local_idx] = processed["dense_stage_labels"][0, target_idx].cpu().item()

        # Run model forward
        video_features = processed["video_features"]
        text_features = processed["text_features"]
        state_features = processed.get("state_features")

        if dual_mode:
            preds = model.sarm_transformer(video_features, text_features, state_features, head_mode=head_mode)
            if head_mode in ["sparse", "both"]:
                _, probs, progress = preds["sparse"]
                sparse_progress[local_idx] = progress[0, -1, 0].cpu().item()
                sparse_stages[local_idx] = probs[0, -1, :].cpu().numpy()
            if head_mode in ["dense", "both"]:
                _, probs, progress = preds["dense"]
                dense_progress[local_idx] = progress[0, -1, 0].cpu().item()
                dense_stages[local_idx] = probs[0, -1, :].cpu().numpy()
        else:
            _, probs, progress = model.sarm_transformer(video_features, text_features, state_features)
            sparse_progress[local_idx] = progress[0, -1, 0].cpu().item()
            sparse_stages[local_idx] = probs[0, -1, :].cpu().numpy()

    results = {}
    if head_mode in ["sparse", "both"] or not dual_mode:
        results["sparse"] = (
            sparse_progress,
            np.array([s for s in sparse_stages if s is not None]),
            gt_sparse_progress if not np.all(np.isnan(gt_sparse_progress)) else None,
            gt_sparse_stages if not np.all(np.isnan(gt_sparse_stages)) else None,
        )
    if dual_mode and head_mode in ["dense", "both"]:
        results["dense"] = (
            dense_progress,
            np.array([s for s in dense_stages if s is not None]),
            gt_dense_progress if not np.all(np.isnan(gt_dense_progress)) else None,
            gt_dense_stages if not np.all(np.isnan(gt_dense_stages)) else None,
        )
    return results, np.array([f for f in raw_frames if f is not None])


def visualize(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None
):
    """Create visualization with progress plot, stage probabilities, and sample frames."""
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


if __name__ == "__main__":
    device = torch.device(DEVICE)

    # Load model
    model = SARMRewardModel.from_pretrained(MODEL_ID)
    model.config.device = str(device)
    model.to(device).eval()
    print(f"Model loaded: {MODEL_ID}")

    # Build delta_timestamps from model config (same as training)
    # observation_delta_indices: [-1_000_000, -(7*gap), ..., -gap, 0] for 9 frames
    image_key = model.config.image_key
    state_key = model.config.state_key

    # Convert delta indices to timestamps (indices / fps)
    # First load dataset to get fps, then recreate with delta_timestamps
    temp_dataset = LeRobotDataset(DATASET_REPO, download_videos=True)
    fps = temp_dataset.fps
    delta_indices = model.config.observation_delta_indices
    delta_timestamps = {
        image_key: [idx / fps for idx in delta_indices],
        state_key: [idx / fps for idx in delta_indices],
    }

    # Load dataset with temporal sampling (same as training)
    dataset = LeRobotDataset(DATASET_REPO, delta_timestamps=delta_timestamps)
    print(f"Dataset: {len(dataset.meta.episodes)} episodes, {len(delta_indices)} frames per sample")

    # Create preprocessor
    preprocess, _ = make_sarm_pre_post_processors(
        config=model.config,
        dataset_stats=load_stats(dataset.root),
        dataset_meta=dataset.meta,
    )

    # Get task description
    ep_start = dataset.meta.episodes["dataset_from_index"][EPISODE_INDEX]
    task_description = dataset[ep_start].get("task", "perform the task")

    # Determine head mode
    dual_mode = model.config.uses_dual_heads
    head_mode = HEAD_MODE if dual_mode or HEAD_MODE == "sparse" else "sparse"

    # Run inference
    results, frames = run_inference(
        model, dataset, preprocess, EPISODE_INDEX, image_key, state_key, task_description, head_mode
    )

    # Visualize each head
    cfg = model.config
    for head_name, (progress_preds, stage_preds, gt_progress, gt_stages) in results.items():
        stage_labels = cfg.sparse_subtask_names if head_name == "sparse" else cfg.dense_subtask_names
        stage_labels = stage_labels or [f"Stage {i + 1}" for i in range(stage_preds.shape[1])]

        suffix = f"_{head_name}" if len(results) > 1 else ""
        output_path = OUTPUT_DIR / f"sarm_prediction_ep{EPISODE_INDEX}{suffix}.png"
        title = f"{task_description} ({head_name.capitalize()})" if len(results) > 1 else task_description

        visualize(
            frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress, gt_stages
        )

    print("Done!")
