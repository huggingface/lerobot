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
SARM Subtask Annotation using local GPU (Qwen3-VL).

This script implements the annotation approach from the SARM paper using local GPU inference:
"SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation"
Paper: https://arxiv.org/pdf/2509.25358

What it does:
1. Takes videos from a LeRobot dataset
2. Uses Qwen3-VL running locally on GPU to identify when subtasks occur
3. Saves subtask timestamps to the dataset metadata
4. Optionally pushes the annotated dataset to HuggingFace Hub

SARM trains reward models that predict:
  - Stage: Which subtask is currently being executed (discrete classification)
  - Progress: How far along the subtask we are (continuous 0-1)

Supports three annotation modes:
  1. No annotations (no args): Auto-creates single sparse "task" stage covering full episode.
     Use with SARM config annotation_mode="single_stage" for simple tasks.

  2. Dense-only (--dense-only --dense-subtasks): Dense annotations from VLM, auto-generated
     single sparse "task" stage. Use with annotation_mode="dense_only".

  3. Dual mode (--sparse-subtasks + --dense-subtasks): Both sparse and dense annotations
     from VLM. Use with annotation_mode="dual".

Requirements:
  - GPU with sufficient VRAM (16GB+ recommended for 30B model)
  - `pip install transformers, torch, qwen-vl-utils`

Run with:
```bash
python examples/dataset_annotation/subtask_annotation.py \
  --repo-id your-username/your-dataset \
  --sparse-subtasks "Do ..." \
  --dense-subtasks "Do task 1, Do task 2, Do task 3" \
  --video-key observation.images.base \
  --push-to-hub
```
"""

import argparse
import json
import multiprocessing as mp
import random
import re
import subprocess
import tempfile
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Pydantic Models for SARM Subtask Annotation
class Timestamp(BaseModel):
    """Timestamp in MM:SS or SS format"""

    start: str = Field(description="Start timestamp (MM:SS or just seconds)")
    end: str = Field(description="End timestamp (MM:SS or just seconds)")


class Subtask(BaseModel):
    """Individual subtask/stage - must use EXACT names from provided list"""

    name: str = Field(description="Subtask name - MUST match one from the predefined list exactly")
    timestamps: Timestamp


class SubtaskAnnotation(BaseModel):
    """Complete annotation for a robot manipulation episode"""

    subtasks: list[Subtask] = Field(description="List of all subtasks in temporal order")


def compute_temporal_proportions(
    annotations: dict[int, Any], fps: int = 30, subtask_order: list[str] | None = None
) -> dict[str, float]:
    """
    Compute dataset-level temporal proportions (priors) for each subtask.

    Implements SARM Paper Formula (1): ᾱ_k = (1/M) × Σ_i (L_{i,k} / T_i)

    Args:
        annotations: Dict mapping episode index to SubtaskAnnotation object.
        fps: Frames per second (unused, kept for API compatibility)
        subtask_order: Optional list defining the output order of subtasks.

    Returns:
        Dict mapping subtask name to its temporal proportion (ᾱ_k), ordered by subtask_order if provided.
    """
    subtask_proportions: dict[str, list[float]] = {}

    for annotation in annotations.values():
        total_duration = 0
        durations: dict[str, int] = {}

        for subtask in annotation.subtasks:
            start_parts = subtask.timestamps.start.split(":")
            end_parts = subtask.timestamps.end.split(":")

            start_seconds = (
                int(start_parts[0]) * 60 + int(start_parts[1])
                if len(start_parts) == 2
                else int(start_parts[0])
            )
            end_seconds = (
                int(end_parts[0]) * 60 + int(end_parts[1]) if len(end_parts) == 2 else int(end_parts[0])
            )

            duration = end_seconds - start_seconds
            durations[subtask.name] = duration
            total_duration += duration

        if total_duration > 0:
            for name, duration in durations.items():
                if name not in subtask_proportions:
                    subtask_proportions[name] = []
                subtask_proportions[name].append(duration / total_duration)

    if not subtask_proportions:
        return {}

    avg_proportions = {name: sum(props) / len(props) for name, props in subtask_proportions.items()}

    total = sum(avg_proportions.values())
    if total > 0:
        avg_proportions = {name: prop / total for name, prop in avg_proportions.items()}

    # Reorder according to subtask_order if provided
    if subtask_order:
        avg_proportions = {
            name: avg_proportions.get(name, 0.0) for name in subtask_order if name in avg_proportions
        }

    return avg_proportions


def create_sarm_prompt(subtask_list: list[str]) -> str:
    subtask_str = "\n".join([f"  - {name}" for name in subtask_list])

    return textwrap.dedent(f"""\
        # Role
        You are a Robotics Vision System specializing in temporal action localization for robot manipulation. Your job is to segment a single demonstration video into distinct, non-overlapping atomic actions from a fixed subtask list.

        # Subtask Label Set (Closed Vocabulary)
        You must strictly identify the video segments using ONLY the following labels. Do not create new labels or modify existing ones:

        [
        {subtask_str}
        ]

        The video shows one successful execution of all subtasks in a logical order.

        # Ground-Truth Semantics (Very Important)
        Use **visual state changes** to define when a subtask starts and ends. Do NOT assume equal durations for the subtasks.

        - A subtask **starts** at the first frame where the robot's motion clearly initiates that subtask.
        - A subtask **ends** at the first frame where that specific action is visually completed and the manipulated object reaches a temporary, stable configuration.

        If there are short pauses or micro-motions that don't clearly correspond to a new subtask, they belong to the **current** subtask.

        # Hard Constraints & Logic
        1. **Continuous Coverage (No Gaps):**
           - The entire video duration from "00:00" to the final timestamp must be covered by subtasks.
           - There can be no gaps between subtasks.
           - If there is any idle or ambiguous time between clear actions, extend the *preceding* subtask to cover it.

        2. **Boundary Consistency:**
           - The `"end"` timestamp of one subtask must be exactly equal to the `"start"` timestamp of the next subtask.
           - Boundaries must coincide with a real visual state transition, not just a convenient time split.

        3. **Chronological Order, One Occurrence Each:**
           - This is a single successful demonstration.
           - Each subtask from the vocabulary appears **exactly once**, in the correct logical order.
           - **Durations may be very different** between subtasks. Never assume they are similar lengths. Base all boundaries only on the video.

        4. **Reject Uniform Segmentation (Important):**
           - Do NOT simply divide the video into equal or nearly equal time chunks.
           - If your boundaries would result in subtasks with similar durations (e.g. all around 5 seconds), treat this as evidence that your segmentation is wrong and refine the boundaries.
           - Only use nearly equal durations if the video truly shows each subtask taking the same amount of time (this is very rare).

        5. **Timestamps:**
           - Timestamps must be in `"MM:SS"` format.
           - The first subtask always starts at `"00:00"`.
           - The last subtask ends at the final visible frame of the video.

        # Step 1 — Textual Timeline (must do this first)
        First, write a extensive and detailed textual timeline describing what happens in the video with approximate timestamps.
        For each subtask, include:
        - its name
        - an approximate start and end time,
        - an description of the visual event at the boundary (e.g. "shirt fully folded to the left", "robot rotates folded shirt 90 degrees").

        Format this as a bullet list.

        # Step 2 — JSON Output (final answer)
        After the textual timeline, output **only** valid JSON with this structure.
        The JSON **must** be consistent with the textual timeline above:

        {{
          "subtasks": [
            {{
              "name": "EXACT_NAME_FROM_LIST",
              "timestamps": {{
                "start": "MM:SS",
                "end":   "MM:SS"
              }}
            }},
            {{
              "name": "EXACT_NAME_FROM_LIST",
              "timestamps": {{
                "start": "MM:SS",
                "end":   "MM:SS"
              }}
            }}
          ]
        }}

        Do not add any extra keys to the JSON.
        """)


class VideoAnnotator:
    """Annotates robot manipulation videos using local Qwen3-VL model on GPU"""

    def __init__(
        self,
        subtask_list: list[str],
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model: Qwen3VLMoeForConditionalGeneration | None = None,  # noqa: F821
        processor: AutoProcessor | None = None,  # noqa: F821
    ):
        """
        Initialize the video annotator with local model.

        Args:
            subtask_list: List of allowed subtask names (for consistency)
            model_name: Hugging Face model name (default: Qwen/Qwen3-VL-30B-A3B-Instruct)
            device: Device to use (cuda, cpu)
            torch_dtype: Data type for model (bfloat16, float16, float32)
            model: Pre-loaded model instance (optional, to share between annotators)
            processor: Pre-loaded processor instance (optional, to share between annotators)
        """
        self.subtask_list = subtask_list
        self.prompt = create_sarm_prompt(subtask_list)
        self.device = device

        # Use provided model/processor or load new ones
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            print(f"Using shared model on {device}")
        else:
            from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

            print(f"Loading model: {model_name}...")

            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            print(f"Model loaded successfully on {device}")

    def extract_episode_segment(
        self, file_path: Path, start_timestamp: float, end_timestamp: float, target_fps: int = 1
    ) -> Path:
        """
        Extract a specific episode segment from concatenated video.
        Uses minimal compression to preserve quality for local inference.

        Args:
            file_path: Path to the concatenated video file
            start_timestamp: Starting timestamp in seconds (within this video file)
            end_timestamp: Ending timestamp in seconds (within this video file)
            target_fps: Target FPS (default: 1 for faster processing)

        Returns:
            Path to extracted video file
        """
        # Create temporary file for extracted video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Check if ffmpeg is available
            subprocess.run(  # nosec B607
                ["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise RuntimeError("ffmpeg not found, cannot extract episode segment") from err

        try:
            # Calculate duration
            duration = end_timestamp - start_timestamp

            print(f"Extracting episode: {start_timestamp:.1f}s-{end_timestamp:.1f}s ({duration:.1f}s)")

            # Use ffmpeg to extract segment with minimal quality loss
            cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
                "-ss",
                str(start_timestamp),
                "-t",
                str(duration),
                "-r",
                str(target_fps),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-an",
                "-y",
                str(tmp_path),
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Verify the output file was created and is not empty
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                print("Video extraction failed (0 bytes) - skipping episode")
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError("FFmpeg produced empty video file")

            # Show extraction results
            file_size_mb = tmp_path.stat().st_size / (1024 * 1024)

            # Fail if file is too small (< 100KB likely means extraction failed)
            if file_size_mb < 0.1:
                print(f"Extracted video too small ({file_size_mb:.2f}MB) - skipping episode")
                tmp_path.unlink()
                raise RuntimeError(f"Video extraction produced invalid file ({file_size_mb:.2f}MB)")

            print(f"Extracted: {file_size_mb:.1f}MB ({target_fps} FPS)")

            return tmp_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed ({e})") from e

    def annotate(
        self,
        file_path: str | Path,
        fps: int,
        start_timestamp: float = 0.0,
        end_timestamp: float | None = None,
        max_retries: int = 3,
    ) -> SubtaskAnnotation:
        """Annotate a video segment using local GPU."""
        from qwen_vl_utils import process_vision_info

        file_path = Path(file_path)

        if end_timestamp is None:
            cap = cv2.VideoCapture(str(file_path))
            end_timestamp = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 1)
            cap.release()

        duration = end_timestamp - start_timestamp
        duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"

        extracted_path = self.extract_episode_segment(file_path, start_timestamp, end_timestamp, 1)
        is_extracted = extracted_path != file_path

        try:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(extracted_path), "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Video is {duration_str} (~{duration:.1f}s). Follow instructions.",
                        },
                    ],
                },
            ]

            for attempt in range(max_retries):
                try:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
                        )

                    response = self.processor.batch_decode(
                        [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
                        skip_special_tokens=True,
                    )[0].strip()

                    # Extract JSON
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0]
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0]

                    try:
                        return SubtaskAnnotation.model_validate(json.loads(response))
                    except json.JSONDecodeError:
                        match = re.search(r"\{.*\}", response, re.DOTALL)
                        if match:
                            return SubtaskAnnotation.model_validate(json.loads(match.group()))
                        raise ValueError("No JSON found") from None
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed after {max_retries} attempts") from e
                    time.sleep(1)
        finally:
            if is_extracted and extracted_path.exists():
                extracted_path.unlink()


def display_annotation(annotation: SubtaskAnnotation, episode_idx: int, fps: int, prefix: str = ""):
    """Display annotation summary."""
    subtask_summary = ", ".join(
        f"{s.name}({s.timestamps.start}-{s.timestamps.end})" for s in annotation.subtasks
    )
    print(f"Episode {episode_idx} {prefix}: {len(annotation.subtasks)} subtasks - {subtask_summary}")


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or SS timestamp to seconds"""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        return int(parts[0])


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
    import matplotlib.patches as mpatches

    bar_height, bar_y = 0.6, 0.5

    for i, subtask in enumerate(subtasks):
        start = timestamp_to_seconds(subtask.timestamps.start)
        end = timestamp_to_seconds(subtask.timestamps.end)
        color = colors[i % len(colors)]

        rect = mpatches.FancyBboxPatch(
            (start, bar_y - bar_height / 2),
            end - start,
            bar_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.85,
        )
        ax.add_patch(rect)

        # Add label if segment is wide enough
        duration = end - start
        if duration > total_duration * 0.06:
            ax.text(
                (start + end) / 2,
                bar_y,
                subtask.name,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                rotation=0 if duration > total_duration * 0.12 else 45,
            )

        if i > 0:
            ax.axvline(x=start, ymin=0.1, ymax=0.9, color="white", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.axvline(x=0, ymin=0.1, ymax=0.9, color="#00ff00", linestyle="-", linewidth=2, alpha=0.9)
    if subtasks:
        ax.axvline(
            x=timestamp_to_seconds(subtasks[-1].timestamps.end),
            ymin=0.1,
            ymax=0.9,
            color="white",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlim(-total_duration * 0.02, total_duration * 1.02)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (seconds)", fontsize=10, color="white", labelpad=5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", colors="#888888", labelsize=8)
    ax.tick_params(axis="y", left=False, labelleft=False)


def visualize_episode(
    ep_idx: int,
    annotation: SubtaskAnnotation,
    video_path: Path,
    video_start: float,
    video_end: float,
    output_path: Path,
    video_key: str,
    ann_type: str,
):
    """Create visualization for a single episode with frames and timeline."""
    import matplotlib.pyplot as plt

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

    gs = fig.add_gridspec(
        2,
        max(len(subtasks), 1),
        height_ratios=[2, 1],
        hspace=0.3,
        wspace=0.1,
        left=0.05,
        right=0.95,
        top=0.88,
        bottom=0.1,
    )

    fig.suptitle(
        f"Episode {ep_idx} - {ann_type.capitalize()} Annotations",
        fontsize=18,
        fontweight="bold",
        color="white",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        f"Camera: {video_key} | Duration: {video_end - video_start:.1f}s | {len(subtasks)} subtasks",
        ha="center",
        fontsize=11,
        color="#888888",
    )

    # Plot frames
    for i, (frame, subtask) in enumerate(zip(sample_frames, subtasks, strict=True)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#16213e")
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(
                0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, color="white", transform=ax.transAxes
            )
        ax.set_title(subtask.name, fontsize=10, fontweight="bold", color=colors[i % len(colors)], pad=8)
        ax.axis("off")
        ax.text(
            0.5,
            -0.08,
            f"t={frame_times[i]:.1f}s",
            ha="center",
            fontsize=9,
            color="#888888",
            transform=ax.transAxes,
        )

    # Plot timeline
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_timeline.set_facecolor("#16213e")
    draw_timeline(ax_timeline, subtasks, total_duration, colors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_annotations(
    dataset: LeRobotDataset,
    sparse_annotations: dict[int, SubtaskAnnotation],
    dense_annotations: dict[int, SubtaskAnnotation] | None,
    video_key: str,
    output_dir: Path,
    num_episodes: int = 5,
    annotation_type: str = "sparse",
    episode_indices: list[int] | None = None,
):
    """
    Visualize subtask annotations for a set of episodes.

    Args:
        dataset: LeRobotDataset instance
        sparse_annotations: Dict mapping episode index to sparse annotations
        dense_annotations: Dict mapping episode index to dense annotations (or None)
        video_key: Camera/video key to use
        output_dir: Directory to save visualization images
        num_episodes: Number of episodes to visualize (ignored if episode_indices provided)
        annotation_type: "sparse", "dense", or "both"
        episode_indices: Specific episode indices to visualize (optional)
    """
    # Determine available episodes based on annotation type
    if annotation_type == "sparse":
        available = set(sparse_annotations.keys())
    elif annotation_type == "dense":
        available = set(dense_annotations.keys()) if dense_annotations else set()
    else:  # both
        sparse_set = set(sparse_annotations.keys())
        dense_set = set(dense_annotations.keys()) if dense_annotations else set()
        available = sparse_set | dense_set

    if not available:
        print("Error: No annotations found to visualize.")
        return

    # Select episodes to visualize
    if episode_indices:
        episodes = sorted([e for e in episode_indices if e in available])
        missing = set(episode_indices) - available
        if missing:
            print(f"Episodes not found in annotations: {sorted(missing)}")
    else:
        episodes = sorted(random.sample(list(available), min(num_episodes, len(available))))
    print(f"Visualizing {len(episodes)} episodes: {episodes}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    for i, ep_idx in enumerate(episodes, 1):
        print(f"Processing episode {ep_idx} ({i}/{len(episodes)})")
        video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        video_start = float(dataset.meta.episodes[f"videos/{video_key}/from_timestamp"][ep_idx])
        video_end = float(dataset.meta.episodes[f"videos/{video_key}/to_timestamp"][ep_idx])

        if annotation_type == "both":
            # Visualize both sparse and dense
            for ann_type, annotations in [("sparse", sparse_annotations), ("dense", dense_annotations)]:
                if annotations and ep_idx in annotations:
                    output_path = output_dir / f"episode_{ep_idx:04d}_{ann_type}.png"
                    visualize_episode(
                        ep_idx,
                        annotations.get(ep_idx),
                        video_path,
                        video_start,
                        video_end,
                        output_path,
                        video_key,
                        ann_type,
                    )
        else:
            annotations = sparse_annotations if annotation_type == "sparse" else dense_annotations
            if annotations and ep_idx in annotations:
                output_path = output_dir / f"episode_{ep_idx:04d}_{annotation_type}.png"
                visualize_episode(
                    ep_idx,
                    annotations.get(ep_idx),
                    video_path,
                    video_start,
                    video_end,
                    output_path,
                    video_key,
                    annotation_type,
                )

    print(f"Visualizations saved to: {output_dir.absolute()}")


def save_annotations_to_dataset(
    dataset_path: Path, annotations: dict[int, SubtaskAnnotation], fps: int, prefix: str = "sparse"
):
    """Save annotations to LeRobot dataset parquet format."""
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes

    episodes_dataset = load_episodes(dataset_path)
    if not episodes_dataset or len(episodes_dataset) == 0:
        return

    episodes_df = episodes_dataset.to_pandas()
    cols = [
        f"{prefix}_{c}"
        for c in [
            "subtask_names",
            "subtask_start_times",
            "subtask_end_times",
            "subtask_start_frames",
            "subtask_end_frames",
        ]
    ]
    for col in cols:
        episodes_df[col] = None

    for ep_idx, ann in annotations.items():
        if ep_idx >= len(episodes_df):
            continue
        names, starts, ends, start_frames, end_frames = [], [], [], [], []
        for s in ann.subtasks:
            names.append(s.name)
            st, et = timestamp_to_seconds(s.timestamps.start), timestamp_to_seconds(s.timestamps.end)
            starts.append(st)
            ends.append(et)
            start_frames.append(int(st * fps))
            end_frames.append(int(et * fps))
        episodes_df.at[ep_idx, cols[0]] = names
        episodes_df.at[ep_idx, cols[1]] = starts
        episodes_df.at[ep_idx, cols[2]] = ends
        episodes_df.at[ep_idx, cols[3]] = start_frames
        episodes_df.at[ep_idx, cols[4]] = end_frames

    # Group by file and write
    for ep_idx in episodes_df.index:
        key = (
            episodes_df.loc[ep_idx, "meta/episodes/chunk_index"],
            episodes_df.loc[ep_idx, "meta/episodes/file_index"],
        )
        path = dataset_path / DEFAULT_EPISODES_PATH.format(chunk_index=key[0], file_index=key[1])
        if path.exists():
            file_df = pd.read_parquet(path)
            for col in cols + (
                [
                    "subtask_names",
                    "subtask_start_times",
                    "subtask_end_times",
                    "subtask_start_frames",
                    "subtask_end_frames",
                ]
                if prefix == "sparse"
                else []
            ):
                if col not in file_df.columns:
                    file_df[col] = None
            if ep_idx in annotations:
                for col in cols:
                    file_df.at[ep_idx, col] = episodes_df.loc[ep_idx, col]
                if prefix == "sparse":  # Legacy columns
                    for i, legacy in enumerate(
                        [
                            "subtask_names",
                            "subtask_start_times",
                            "subtask_end_times",
                            "subtask_start_frames",
                            "subtask_end_frames",
                        ]
                    ):
                        file_df.at[ep_idx, legacy] = episodes_df.loc[ep_idx, cols[i]]
            file_df.to_parquet(path, engine="pyarrow", compression="snappy")


def generate_auto_sparse_annotations(
    dataset: LeRobotDataset, episode_indices: list[int], video_key: str
) -> dict[int, SubtaskAnnotation]:
    """Auto-generate single 'task' stage annotations for all episodes."""
    annotations = {}
    for ep_idx in episode_indices:
        start = float(dataset.meta.episodes[f"videos/{video_key}/from_timestamp"][ep_idx])
        end = float(dataset.meta.episodes[f"videos/{video_key}/to_timestamp"][ep_idx])
        duration = end - start
        end_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        annotations[ep_idx] = SubtaskAnnotation(
            subtasks=[Subtask(name="task", timestamps=Timestamp(start="00:00", end=end_str))]
        )
    return annotations


def load_annotations_from_dataset(dataset_path: Path, prefix: str = "sparse") -> dict[int, SubtaskAnnotation]:
    """Load annotations from LeRobot dataset parquet files."""
    from lerobot.datasets.utils import load_episodes

    episodes_dataset = load_episodes(dataset_path)
    if not episodes_dataset or len(episodes_dataset) == 0:
        return {}

    col_names = f"{prefix}_subtask_names"
    col_start = f"{prefix}_subtask_start_times"
    col_end = f"{prefix}_subtask_end_times"

    # Fall back to legacy columns for sparse
    if col_names not in episodes_dataset.column_names:
        if prefix == "sparse" and "subtask_names" in episodes_dataset.column_names:
            col_names, col_start, col_end = "subtask_names", "subtask_start_times", "subtask_end_times"
        else:
            return {}

    df = episodes_dataset.to_pandas()
    annotations = {}
    for ep_idx in df.index:
        names = df.loc[ep_idx, col_names]
        if names is None or (isinstance(names, float) and pd.isna(names)):
            continue
        starts, ends = df.loc[ep_idx, col_start], df.loc[ep_idx, col_end]
        annotations[int(ep_idx)] = SubtaskAnnotation(
            subtasks=[
                Subtask(
                    name=n,
                    timestamps=Timestamp(
                        start=f"{int(s) // 60:02d}:{int(s) % 60:02d}",
                        end=f"{int(e) // 60:02d}:{int(e) % 60:02d}",
                    ),
                )
                for n, s, e in zip(names, starts, ends, strict=True)
            ]
        )
    return annotations


def process_single_episode(
    ep_idx: int,
    dataset_root: Path,
    dataset_meta,
    video_key: str,
    fps: int,
    annotator: VideoAnnotator,
) -> tuple[int, SubtaskAnnotation | None, str | None]:
    """Process a single episode annotation."""
    try:
        video_path = dataset_root / dataset_meta.get_video_file_path(ep_idx, video_key)
        if not video_path.exists():
            return ep_idx, None, f"Video not found: {video_path}"

        start = float(dataset_meta.episodes[f"videos/{video_key}/from_timestamp"][ep_idx])
        end = float(dataset_meta.episodes[f"videos/{video_key}/to_timestamp"][ep_idx])
        return ep_idx, annotator.annotate(video_path, fps, start, end), None
    except Exception as e:
        return ep_idx, None, str(e)


def worker_process_episodes(
    worker_id: int,
    gpu_id: int,
    episode_indices: list[int],
    repo_id: str,
    video_key: str,
    sparse_subtask_list: list[str],
    dense_subtask_list: list[str] | None,
    model_name: str,
    torch_dtype: torch.dtype,
) -> tuple[dict, dict | None]:
    """Worker for parallel processing across GPUs."""
    device = f"cuda:{gpu_id}"
    dataset = LeRobotDataset(repo_id, download_videos=False)

    sparse_annotator = VideoAnnotator(sparse_subtask_list, model_name, device, torch_dtype)
    dense_annotator = (
        VideoAnnotator(
            dense_subtask_list,
            model_name,
            device,
            torch_dtype,
            sparse_annotator.model,
            sparse_annotator.processor,
        )
        if dense_subtask_list
        else None
    )

    sparse_annotations, dense_annotations = {}, {} if dense_subtask_list else None

    for ep_idx in episode_indices:
        _, sparse_ann, err = process_single_episode(
            ep_idx, dataset.root, dataset.meta, video_key, dataset.fps, sparse_annotator
        )
        if sparse_ann:
            sparse_annotations[ep_idx] = sparse_ann

        if dense_annotator:
            _, dense_ann, _ = process_single_episode(
                ep_idx, dataset.root, dataset.meta, video_key, dataset.fps, dense_annotator
            )
            if dense_ann:
                dense_annotations[ep_idx] = dense_ann

    return sparse_annotations, dense_annotations


def main():
    parser = argparse.ArgumentParser(description="SARM-style subtask annotation using local GPU (Qwen3-VL)")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument(
        "--sparse-subtasks", type=str, default=None, help="Comma-separated sparse subtask names"
    )
    parser.add_argument(
        "--dense-subtasks", type=str, default=None, help="Comma-separated dense subtask names"
    )
    parser.add_argument(
        "--dense-only", action="store_true", help="Dense-only mode with auto-generated sparse 'task' stage"
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Episode indices to annotate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="VLM model")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already annotated episodes")
    parser.add_argument("--video-key", type=str, default=None, help="Video key (default: first available)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--output-repo-id", type=str, default=None, help="Output repo ID for push")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for multi-GPU")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None, help="GPU IDs to use")
    # Visualization options
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize existing annotations (no generation)",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of episodes to visualize (default: 5)",
    )
    parser.add_argument(
        "--visualize-type",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="Type of annotations to visualize (default: sparse)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./subtask_viz",
        help="Output directory for visualizations (default: ./subtask_viz)",
    )

    args = parser.parse_args()

    # Load dataset first (needed for both annotation and visualization)
    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, download_videos=True)
    fps = dataset.fps

    if not dataset.meta.video_keys:
        raise ValueError("No video keys found")

    video_key = (
        args.video_key if args.video_key in (dataset.meta.video_keys or []) else dataset.meta.video_keys[0]
    )
    print(f"Using camera: {video_key}, FPS: {fps}")

    # Handle visualization-only mode
    if args.visualize_only:
        print("Visualization-only mode")
        sparse_annotations = load_annotations_from_dataset(dataset.root, prefix="sparse")
        dense_annotations = load_annotations_from_dataset(dataset.root, prefix="dense")

        if not sparse_annotations and not dense_annotations:
            return print("Error: No annotations found. Run annotation first.")

        print(f"Found {len(sparse_annotations)} sparse, {len(dense_annotations)} dense annotations")

        visualize_annotations(
            dataset=dataset,
            sparse_annotations=sparse_annotations,
            dense_annotations=dense_annotations if dense_annotations else None,
            video_key=video_key,
            output_dir=Path(args.output_dir),
            num_episodes=args.num_visualizations,
            annotation_type=args.visualize_type,
            episode_indices=args.episodes,
        )
        return

    # Validate arguments for annotation mode
    if args.dense_only and not args.dense_subtasks:
        return print("Error: --dense-only requires --dense-subtasks")
    if args.dense_subtasks and not args.sparse_subtasks and not args.dense_only:
        return print("Error: --dense-subtasks requires --sparse-subtasks or --dense-only")

    sparse_subtask_list = (
        [s.strip() for s in args.sparse_subtasks.split(",")] if args.sparse_subtasks else None
    )
    dense_subtask_list = [s.strip() for s in args.dense_subtasks.split(",")] if args.dense_subtasks else None
    auto_sparse = sparse_subtask_list is None
    dense_mode = dense_subtask_list is not None
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Determine episodes
    episode_indices = args.episodes or list(range(dataset.meta.total_episodes))

    existing_annotations = load_annotations_from_dataset(dataset.root, prefix="sparse")
    if args.skip_existing:
        episode_indices = [ep for ep in episode_indices if ep not in existing_annotations]

    if not episode_indices:
        return print("All episodes already annotated!")
    print(f"Annotating {len(episode_indices)} episodes")

    # GPU setup
    gpu_ids = args.gpu_ids or list(
        range(min(args.num_workers, torch.cuda.device_count() if torch.cuda.is_available() else 1))
    )
    args.num_workers = len(gpu_ids)

    sparse_annotations = existing_annotations.copy()
    dense_annotations = {} if dense_mode else None

    # Auto-sparse mode
    if auto_sparse:
        sparse_annotations.update(generate_auto_sparse_annotations(dataset, episode_indices, video_key))
        save_annotations_to_dataset(dataset.root, sparse_annotations, fps, prefix="sparse")
        print(f"Auto-generated {len(episode_indices)} sparse 'task' annotations")

    # VLM annotation (for sparse if not auto, and for dense)
    need_vlm = (not auto_sparse) or dense_mode

    if need_vlm:
        if args.num_workers > 1 and not auto_sparse:
            # Parallel processing
            print(f"Parallel processing with {args.num_workers} workers")
            episodes_per_worker = [[] for _ in range(args.num_workers)]
            for i, ep_idx in enumerate(episode_indices):
                episodes_per_worker[i % args.num_workers].append(ep_idx)

            with ProcessPoolExecutor(
                max_workers=args.num_workers, mp_context=mp.get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        worker_process_episodes,
                        w,
                        gpu_ids[w],
                        episodes_per_worker[w],
                        args.repo_id,
                        video_key,
                        sparse_subtask_list,
                        dense_subtask_list,
                        args.model,
                        torch_dtype,
                    )
                    for w in range(args.num_workers)
                    if episodes_per_worker[w]
                ]

                for future in as_completed(futures):
                    try:
                        worker_sparse, worker_dense = future.result()
                        sparse_annotations.update(worker_sparse)
                        if dense_mode and worker_dense:
                            dense_annotations.update(worker_dense)
                        save_annotations_to_dataset(dataset.root, sparse_annotations, fps, prefix="sparse")
                        if dense_mode:
                            save_annotations_to_dataset(dataset.root, dense_annotations, fps, prefix="dense")
                    except Exception as e:
                        raise RuntimeError(f"Worker failed: {e}") from e
        else:
            # Sequential processing
            sparse_annotator = (
                VideoAnnotator(sparse_subtask_list, args.model, args.device, torch_dtype)
                if not auto_sparse and sparse_subtask_list
                else None
            )
            dense_annotator = (
                VideoAnnotator(
                    dense_subtask_list,
                    args.model,
                    args.device,
                    torch_dtype,
                    sparse_annotator.model if sparse_annotator else None,
                    sparse_annotator.processor if sparse_annotator else None,
                )
                if dense_mode
                else None
            )

            for i, ep_idx in enumerate(episode_indices):
                print(f"Episode {ep_idx} ({i + 1}/{len(episode_indices)})")

                if sparse_annotator:
                    _, sparse_ann, err = process_single_episode(
                        ep_idx, dataset.root, dataset.meta, video_key, fps, sparse_annotator
                    )
                    if sparse_ann:
                        sparse_annotations[ep_idx] = sparse_ann
                        save_annotations_to_dataset(dataset.root, sparse_annotations, fps, prefix="sparse")
                    elif err:
                        print(f"Sparse failed: {err}")

                if dense_annotator:
                    _, dense_ann, err = process_single_episode(
                        ep_idx, dataset.root, dataset.meta, video_key, fps, dense_annotator
                    )
                    if dense_ann:
                        dense_annotations[ep_idx] = dense_ann
                        save_annotations_to_dataset(dataset.root, dense_annotations, fps, prefix="dense")
                    elif err:
                        print(f"Dense failed: {err}")

    # Save temporal proportions
    def save_proportions(annotations, prefix, subtask_list=None, is_auto=False):
        props: dict[str, float] = (
            {"task": 1.0} if is_auto else compute_temporal_proportions(annotations, fps, subtask_list)
        )
        path = dataset.root / "meta" / f"temporal_proportions_{prefix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(props, f, indent=2)
        print(f"Saved {prefix} temporal proportions")

    save_proportions(sparse_annotations, "sparse", sparse_subtask_list, auto_sparse)
    if dense_mode and dense_annotations:
        save_proportions(dense_annotations, "dense", dense_subtask_list)

    print(f"\nComplete! {len(sparse_annotations)} sparse, {len(dense_annotations or {})} dense annotations")

    # Visualize annotations after generation
    if args.num_visualizations > 0:
        print(f"\nGenerating {args.num_visualizations} visualizations...")
        visualize_type = "both" if dense_mode else "sparse"
        visualize_annotations(
            dataset=dataset,
            sparse_annotations=sparse_annotations,
            dense_annotations=dense_annotations,
            video_key=video_key,
            output_dir=Path(args.output_dir),
            num_episodes=args.num_visualizations,
            annotation_type=visualize_type,
        )

    if args.push_to_hub:
        try:
            dataset.push_to_hub(push_videos=True)
            print(f"Pushed to {args.output_repo_id or args.repo_id}")
        except Exception as e:
            print(f"Push failed: {e}")


if __name__ == "__main__":
    main()
