"""
Example: Visualize subtask annotations on dataset episodes.

This example shows how to:
1. Load subtask annotations from a dataset (sparse and/or dense)
2. Extract frames from episode videos at subtask boundaries
3. Create timeline visualizations with color-coded stages
"""

from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_episodes
from lerobot.policies.sarm.sarm_utils import SubtaskAnnotation, Subtask, Timestamp

DATASET_REPO = "pepijn223/mydataset"  # HuggingFace dataset ID
ANNOTATION_TYPE = "sparse"  # "sparse", "dense", or "both"
EPISODE_INDICES = [0, 1, 2]  # Episodes to visualize (or None for random)
NUM_EPISODES = 5  # Used if EPISODE_INDICES is None
VIDEO_KEY = "observation.images.base"  # Camera key
OUTPUT_DIR = Path("./subtask_viz")

def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or SS timestamp to seconds."""
    parts = timestamp.split(":")
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else int(parts[0])


def load_annotations(dataset_path: Path, prefix: str = "sparse") -> dict[int, SubtaskAnnotation]:
    """Load subtask annotations from dataset parquet files."""
    episodes_dataset = load_episodes(dataset_path)
    if episodes_dataset is None or len(episodes_dataset) == 0:
        return {}

    col_names = f"{prefix}_subtask_names"
    col_start = f"{prefix}_subtask_start_times"
    col_end = f"{prefix}_subtask_end_times"

    # Fall back to legacy column names for sparse
    if col_names not in episodes_dataset.column_names:
        if prefix == "sparse" and "subtask_names" in episodes_dataset.column_names:
            col_names, col_start, col_end = "subtask_names", "subtask_start_times", "subtask_end_times"
        else:
            return {}

    episodes_df = episodes_dataset.to_pandas()
    annotations = {}

    for ep_idx in episodes_df.index:
        names = episodes_df.loc[ep_idx, col_names]
        if names is None or (isinstance(names, float) and pd.isna(names)):
            continue

        start_times = episodes_df.loc[ep_idx, col_start]
        end_times = episodes_df.loc[ep_idx, col_end]

        subtasks = []
        for i, name in enumerate(names):
            start_sec, end_sec = int(start_times[i]), int(end_times[i])
            subtasks.append(Subtask(
                name=name,
                timestamps=Timestamp(start=f"{start_sec // 60:02d}:{start_sec % 60:02d}",
                                    end=f"{end_sec // 60:02d}:{end_sec % 60:02d}")
            ))
        annotations[int(ep_idx)] = SubtaskAnnotation(subtasks=subtasks)

    return annotations


def extract_frame(video_path: Path, timestamp: float) -> np.ndarray | None:
    """Extract a single frame from video at given timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


def draw_timeline(ax, subtasks, total_duration, colors):
    """Draw a timeline with color-coded subtask segments."""
    bar_height, bar_y = 0.6, 0.5

    for i, subtask in enumerate(subtasks):
        start = timestamp_to_seconds(subtask.timestamps.start)
        end = timestamp_to_seconds(subtask.timestamps.end)
        color = colors[i % len(colors)]

        rect = mpatches.FancyBboxPatch(
            (start, bar_y - bar_height / 2), end - start, bar_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)

        # Add label if segment is wide enough
        duration = end - start
        if duration > total_duration * 0.06:
            ax.text((start + end) / 2, bar_y, subtask.name, ha="center", va="center",
                   fontsize=8, fontweight="bold", color="white",
                   rotation=0 if duration > total_duration * 0.12 else 45)

        if i > 0:
            ax.axvline(x=start, ymin=0.1, ymax=0.9, color="white", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.axvline(x=0, ymin=0.1, ymax=0.9, color="#00ff00", linestyle="-", linewidth=2, alpha=0.9)
    if subtasks:
        ax.axvline(x=timestamp_to_seconds(subtasks[-1].timestamps.end), ymin=0.1, ymax=0.9, color="white", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlim(-total_duration * 0.02, total_duration * 1.02)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (seconds)", fontsize=10, color="white", labelpad=5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", colors="#888888", labelsize=8)
    ax.tick_params(axis="y", left=False, labelleft=False)


def visualize_episode(ep_idx, annotation, video_path, video_start, video_end, output_path, video_key, ann_type):
    """Create visualization for a single episode with frames and timeline."""
    if annotation is None:
        print(f"No {ann_type} annotation for episode {ep_idx}")
        return
    
    subtasks = annotation.subtasks
    if not subtasks:
        print(f"No subtasks for episode {ep_idx}")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(subtasks), 10)))
    total_duration = timestamp_to_seconds(subtasks[-1].timestamps.end)

    # Extract middle frame from each subtask
    sample_frames, frame_times = [], []
    for subtask in subtasks:
        start = timestamp_to_seconds(subtask.timestamps.start)
        end = timestamp_to_seconds(subtask.timestamps.end)
        mid = (start + end) / 2
        frame_times.append(mid)
        sample_frames.append(extract_frame(video_path, video_start + mid))

    # Create figure
    fig_width = max(16, len(subtasks) * 2.5)
    fig = plt.figure(figsize=(fig_width, 10))
    fig.patch.set_facecolor("#1a1a2e")

    gs = fig.add_gridspec(2, max(len(subtasks), 1), height_ratios=[2, 1], hspace=0.3, wspace=0.1,
                          left=0.05, right=0.95, top=0.88, bottom=0.1)

    fig.suptitle(f"Episode {ep_idx} - {ann_type.capitalize()} Annotations",
                fontsize=18, fontweight="bold", color="white", y=0.96)
    fig.text(0.5, 0.91, f"Camera: {video_key} | Duration: {video_end - video_start:.1f}s | {len(subtasks)} subtasks",
            ha="center", fontsize=11, color="#888888")

    # Plot frames
    for i, (frame, subtask) in enumerate(zip(sample_frames, subtasks)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#16213e")
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, color="white", transform=ax.transAxes)
        ax.set_title(subtask.name, fontsize=10, fontweight="bold", color=colors[i % len(colors)], pad=8)
        ax.axis("off")
        ax.text(0.5, -0.08, f"t={frame_times[i]:.1f}s", ha="center", fontsize=9, color="#888888", transform=ax.transAxes)

    # Plot timeline
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_timeline.set_facecolor("#16213e")
    draw_timeline(ax_timeline, subtasks, total_duration, colors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print(f"Loading dataset: {DATASET_REPO}")
    dataset = LeRobotDataset(DATASET_REPO, download_videos=True)

    video_key = VIDEO_KEY or dataset.meta.video_keys[0]
    print(f"Using camera: {video_key}, FPS: {dataset.fps}")

    # Load annotations
    sparse_annotations, dense_annotations = {}, {}
    if ANNOTATION_TYPE in ["sparse", "both"]:
        sparse_annotations = load_annotations(dataset.root, "sparse")
        print(f"Sparse annotations: {len(sparse_annotations)} episodes")
    if ANNOTATION_TYPE in ["dense", "both"]:
        dense_annotations = load_annotations(dataset.root, "dense")
        print(f"Dense annotations: {len(dense_annotations)} episodes")

    # Determine episodes to visualize
    if ANNOTATION_TYPE == "sparse":
        available = set(sparse_annotations.keys())
    elif ANNOTATION_TYPE == "dense":
        available = set(dense_annotations.keys())
    else:
        available = set(sparse_annotations.keys()) | set(dense_annotations.keys())

    if not available:
        print("Error: No annotations found. Run subtask_annotation.py first.")
        exit(1)

    if EPISODE_INDICES:
        episodes = [ep for ep in EPISODE_INDICES if ep in available]
    else:
        import random
        episodes = sorted(random.sample(list(available), min(NUM_EPISODES, len(available))))

    print(f"Visualizing episodes: {episodes}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    for ep_idx in episodes:
        video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        video_start = float(dataset.meta.episodes[f"videos/{video_key}/from_timestamp"][ep_idx])
        video_end = float(dataset.meta.episodes[f"videos/{video_key}/to_timestamp"][ep_idx])

        if ANNOTATION_TYPE == "both":
            # Call visualize_episode for each annotation type
            for ann_type, annotations in [("sparse", sparse_annotations), ("dense", dense_annotations)]:
                output_path = OUTPUT_DIR / f"episode_{ep_idx:04d}_{ann_type}.png"
                visualize_episode(ep_idx, annotations.get(ep_idx), video_path, video_start, video_end,
                                output_path, video_key, ann_type)
        else:
            annotations = sparse_annotations if ANNOTATION_TYPE == "sparse" else dense_annotations
            output_path = OUTPUT_DIR / f"episode_{ep_idx:04d}_{ANNOTATION_TYPE}.png"
            visualize_episode(ep_idx, annotations.get(ep_idx), video_path, video_start, video_end,
                            output_path, video_key, ANNOTATION_TYPE)

    print(f"\nDone! Output: {OUTPUT_DIR.absolute()}")
