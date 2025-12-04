"""
Example: Visualize SARM predictions on a dataset episode.

This example shows how to:
1. Load a trained SARM model
2. Run inference on an episode using the preprocessor pipeline
3. Visualize predicted progress and stage probabilities vs ground truth
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats
from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

MODEL_ID = "username/sarm-model"  # HuggingFace model ID or local path
DATASET_REPO = "lerobot/aloha_sim_insertion_human"  # Dataset to run inference on
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


@torch.no_grad()
def run_inference(model, dataset, preprocess, episode_index, image_key, state_key, task_description, head_mode):
    """Run SARM inference on an episode, returning predictions and ground truth."""
    dual_mode = model.config.uses_dual_heads
    ep_start = dataset.meta.episodes["dataset_from_index"][episode_index]
    ep_end = dataset.meta.episodes["dataset_to_index"][episode_index]

    all_sparse_progress, all_sparse_stages, gt_sparse_progress, gt_sparse_stages = [], [], [], []
    all_dense_progress, all_dense_stages, gt_dense_progress, gt_dense_stages = [], [], [], []
    raw_frames = []

    for frame_idx in tqdm(range(ep_start, ep_end), desc="Processing frames"):
        sample = dataset[frame_idx]
        raw_frames.append(to_numpy_image(sample[image_key]))

        obs = {image_key: sample[image_key]}
        if state_key in sample:
            obs[state_key] = sample[state_key]

        processed = preprocess({
            "observation": obs,
            "complementary_data": {"task": task_description, "index": frame_idx, "episode_index": episode_index},
        })
        proc_obs = processed.get("observation", processed)

        # Extract ground truth from processor output
        if "sparse_progress_targets" in proc_obs:
            gt_sparse_progress.append(proc_obs["sparse_progress_targets"][0, -1, 0].cpu().item())
            gt_sparse_stages.append(proc_obs["sparse_stage_labels"][0, -1].cpu().item())
        if "dense_progress_targets" in proc_obs:
            gt_dense_progress.append(proc_obs["dense_progress_targets"][0, -1, 0].cpu().item())
            gt_dense_stages.append(proc_obs["dense_stage_labels"][0, -1].cpu().item())

        # Run model forward
        video_features = proc_obs["video_features"]
        text_features = proc_obs["text_features"]
        state_features = proc_obs.get("state_features")

        if dual_mode:
            preds = model.sarm_transformer(video_features, text_features, state_features, head_mode=head_mode)
            if head_mode in ["sparse", "both"]:
                _, probs, progress = preds["sparse"]
                all_sparse_progress.append(progress[0, -1, 0].cpu().item())
                all_sparse_stages.append(probs[0, -1, :].cpu().numpy())
            if head_mode in ["dense", "both"]:
                _, probs, progress = preds["dense"]
                all_dense_progress.append(progress[0, -1, 0].cpu().item())
                all_dense_stages.append(probs[0, -1, :].cpu().numpy())
        else:
            _, probs, progress = model.sarm_transformer(video_features, text_features, state_features)
            all_sparse_progress.append(progress[0, -1, 0].cpu().item())
            all_sparse_stages.append(probs[0, -1, :].cpu().numpy())

    results = {}
    if head_mode in ["sparse", "both"] or not dual_mode:
        results["sparse"] = (
            np.array(all_sparse_progress), np.array(all_sparse_stages),
            np.array(gt_sparse_progress) if gt_sparse_progress else None,
            np.array(gt_sparse_stages) if gt_sparse_stages else None,
        )
    if dual_mode and head_mode in ["dense", "both"]:
        results["dense"] = (
            np.array(all_dense_progress), np.array(all_dense_stages),
            np.array(gt_dense_progress) if gt_dense_progress else None,
            np.array(gt_dense_stages) if gt_dense_stages else None,
        )
    return results, np.array(raw_frames)


def visualize(frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None):
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
        ax_progress.plot(frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth")
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    # Stage predictions
    ax_stages.stackplot(frame_indices, *[stage_preds[:, i] for i in range(num_stages)], colors=colors, alpha=0.8, labels=stage_labels)
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
        ax_frames.text(i * w + w / 2, -10, f"Frame {idx}\n{progress_preds[idx]:.2f}\n{stage_name}", ha="center", va="top", fontsize=7)
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

    # Load dataset
    dataset = LeRobotDataset(DATASET_REPO)
    print(f"Dataset: {len(dataset.meta.episodes)} episodes")

    image_key = model.config.image_key
    state_key = model.config.state_key

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
        stage_labels = stage_labels or [f"Stage {i+1}" for i in range(stage_preds.shape[1])]

        suffix = f"_{head_name}" if len(results) > 1 else ""
        output_path = OUTPUT_DIR / f"sarm_prediction_ep{EPISODE_INDEX}{suffix}.png"
        title = f"{task_description} ({head_name.capitalize()})" if len(results) > 1 else task_description

        visualize(frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress, gt_stages)

    print("Done!")

