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
Image-window subtask annotation for LeRobot datasets using Qwen VLMs.

This script assigns a subtask to each window of consecutive frames by sending
those frames as images to the VLM (instead of a video) for better accuracy.
Supports Qwen2-VL and Qwen3-VL (same models as subtask_annotate.py).

Pipeline:
1. Load a LeRobot dataset (local or Hub).
2. For each episode, slide a window over frame indices.
3. For each window, load the corresponding images (from image_key or decoded video_key).
4. Send the window of images to Qwen2-VL with the same skill prompt; get one subtask name.
5. Assign that subtask to all frames in the window.
6. Write subtasks.parquet and add subtask_index via add_features (same as subtask_annotate).

Usage:
  python -m lerobot.data_processing.annotations.subtask_annotate_image \\
    --data-dir /path/to/dataset --camera-key observation.images.base \\
    --window-size 8 --stride 8 --output-dir ./output
"""

from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from rich.console import Console

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Reuse data structures and save/load from the video-based annotator
from lerobot.data_processing.annotations.subtask_annotate import (
    EpisodeSkills,
    Skill,
    load_skill_annotations,
    save_skill_annotations,
)


def create_window_skill_prompt(
    coarse_goal: str | None = None,
    subtask_labels: list[str] | None = None,
) -> str:
    """Prompt for labeling a single window of frames with one atomic skill.
    If subtask_labels are provided, the model must choose exactly one from that list.
    """
    goal_context = f'The overall goal is: "{coarse_goal}".\n\n' if coarse_goal else ""
    if subtask_labels:
        labels_list = ", ".join(f'"{l}"' for l in subtask_labels)
        label_instruction = (
            f"You must choose exactly ONE skill from this list: [{labels_list}]. "
            "Do not create new labels. Reply with only that label.\n\n"
        )
    else:
        label_instruction = ""
    return textwrap.dedent(f"""\
        # Role
        You are a Robotics Vision System that labels short clips from robot manipulation demonstrations.

        # Task
        {goal_context}{label_instruction}The following images are consecutive frames from a single short clip of a robot demonstration.
        What single atomic manipulation skill is being performed in this clip?

        # Requirements
        - Reply with ONLY one short skill name (e.g. "pick up object", "move arm left", "release gripper").
        - No explanation, no timestamps, no JSON. Just the skill name.
        """).strip()


def _run_image_segmenter(
    self,
    images: list[PIL.Image.Image],
    coarse_goal: str | None,
    subtask_labels: list[str] | None = None,
) -> str:
    """Shared inference for Qwen2-VL and Qwen3-VL image window labeling."""
    prompt = create_window_skill_prompt(coarse_goal, subtask_labels)
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": "What single atomic skill is shown in these frames? Reply with only the skill name."})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": content},
    ]
    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = self.process_vision_info(messages)
    inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(self.device)

    with torch.no_grad():
        generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

    response = self.processor.batch_decode(
        [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
    )[0].strip()
    skill_name = response.split("\n")[0].strip().strip('."')
    return skill_name if skill_name else "unknown"


def _run_image_segmenter_batch(
    self,
    batch_images: list[list[PIL.Image.Image]],
    coarse_goal: str | None,
    subtask_labels: list[str] | None = None,
) -> list[str]:
    """Run VLM on multiple windows at once; returns one skill name per window."""
    if not batch_images:
        return []
    prompt = create_window_skill_prompt(coarse_goal, subtask_labels)
    all_texts = []
    all_image_inputs = []
    all_video_inputs = []
    for images in batch_images:
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": "What single atomic skill is shown in these frames? Reply with only the skill name."})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        all_texts.append(text)
        if image_inputs is not None:
            all_image_inputs.extend(image_inputs if isinstance(image_inputs, list) else [image_inputs])
        if video_inputs is not None:
            all_video_inputs.extend(video_inputs if isinstance(video_inputs, list) else [video_inputs])
    inputs = self.processor(
        text=all_texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(self.device)
    with torch.no_grad():
        generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
    responses = self.processor.batch_decode(
        [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
    )
    return [
        (r.split("\n")[0].strip().strip('."') or "unknown")
        for r in responses
    ]


class Qwen2VLImageSegmenter:
    """Uses Qwen2-VL to assign one skill name to a window of images (same model as subtask_annotate)."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.console = Console()
        self.device = device
        self.process_vision_info = process_vision_info
        self.console.print(f"[cyan]Loading Qwen2-VL for image-window labeling: {model_name}...[/cyan]")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.console.print(f"[green]✓ Model loaded on {device}[/green]")

    def segment_skill_from_images(
        self,
        images: list[PIL.Image.Image],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> str:
        """Return a single skill name for the given window of images."""
        return _run_image_segmenter(self, images, coarse_goal, subtask_labels)

    def segment_skill_from_images_batch(
        self,
        batch_images: list[list[PIL.Image.Image]],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[str]:
        """Return one skill name per window; processes multiple windows in one forward pass."""
        return _run_image_segmenter_batch(self, batch_images, coarse_goal, subtask_labels)


class Qwen3VLImageSegmenter:
    """Uses Qwen3-VL (MoE) to assign one skill name to a window of images."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        self.console = Console()
        self.device = device
        self.process_vision_info = process_vision_info
        self.console.print(f"[cyan]Loading Qwen3-VL for image-window labeling: {model_name}...[/cyan]")
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.console.print(f"[green]✓ Model loaded on {device}[/green]")

    def segment_skill_from_images(
        self,
        images: list[PIL.Image.Image],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> str:
        """Return a single skill name for the given window of images."""
        return _run_image_segmenter(self, images, coarse_goal, subtask_labels)

    def segment_skill_from_images_batch(
        self,
        batch_images: list[list[PIL.Image.Image]],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[str]:
        """Return one skill name per window; processes multiple windows in one forward pass."""
        return _run_image_segmenter_batch(self, batch_images, coarse_goal, subtask_labels)


def get_image_segmenter(
    model_name: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Return the appropriate image-window segmenter for the model (Qwen2-VL or Qwen3-VL)."""
    model_lower = model_name.lower()
    if "qwen3" in model_lower:
        return Qwen3VLImageSegmenter(model_name, device, torch_dtype)
    return Qwen2VLImageSegmenter(model_name, device, torch_dtype)


def frame_to_pil(frame_value) -> PIL.Image.Image:
    """Convert a single frame from dataset (tensor or PIL or path) to PIL.Image."""
    if isinstance(frame_value, PIL.Image.Image):
        return frame_value
    if isinstance(frame_value, (str, Path)):
        return PIL.Image.open(frame_value).convert("RGB")
    if hasattr(frame_value, "numpy"):
        arr = frame_value.numpy()
    else:
        arr = np.asarray(frame_value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return PIL.Image.fromarray(arr)


def _sample_window_indices(window_length: int, max_frames: int) -> list[int]:
    """Return indices into a window of length window_length, at most max_frames, in order.
    If window_length <= max_frames, returns range(window_length).
    Otherwise returns sorted random sample of max_frames indices (temporal order preserved).
    """
    if max_frames <= 0 or window_length <= max_frames:
        return list(range(window_length))
    return sorted(random.sample(range(window_length), max_frames))


class SkillAnnotatorImage:
    """Annotates episodes by sliding a window over frames and labeling each window with the VLM."""

    def __init__(
        self,
        segmenter: Qwen2VLImageSegmenter | Qwen3VLImageSegmenter,
        window_size: int = 8,
        stride: int | None = None,
        batch_size: int = 1,
        max_frames_per_window: int | None = None,
        console: Console | None = None,
    ):
        self.segmenter = segmenter
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.batch_size = max(1, batch_size)
        self.max_frames_per_window = max_frames_per_window
        self.console = console or Console()

    def annotate_dataset(
        self,
        dataset: LeRobotDataset,
        camera_key: str,
        episodes: list[int] | None = None,
        skip_existing: bool = False,
        subtask_labels: list[str] | None = None,
    ) -> dict[int, EpisodeSkills]:
        """Annotate episodes using image windows. camera_key can be an image_key or video_key."""
        episode_indices = episodes or list(range(dataset.meta.total_episodes))
        coarse_goal = self._get_coarse_goal(dataset)
        annotations: dict[int, EpisodeSkills] = {}

        if skip_existing:
            existing = load_skill_annotations(dataset.root)
            if existing and existing.get("episodes"):
                existing_eps = {int(k) for k in existing["episodes"] if existing["episodes"][k].get("skills")}
                episode_indices = [i for i in episode_indices if i not in existing_eps]

        for ep_idx in episode_indices:
            try:
                skills = self._annotate_episode(
                    dataset, ep_idx, camera_key, coarse_goal, subtask_labels
                )
                if skills:
                    annotations[ep_idx] = EpisodeSkills(
                        episode_index=ep_idx,
                        description=coarse_goal,
                        skills=skills,
                    )
                    self.console.print(f"[green]✓ Episode {ep_idx}: {len(skills)} window skills[/green]")
                else:
                    self.console.print(f"[yellow]⚠ Episode {ep_idx}: no skills[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Episode {ep_idx} failed: {e}[/red]")

        return annotations

    def _get_coarse_goal(self, dataset: LeRobotDataset) -> str:
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            return str(dataset.meta.tasks.index[0])
        return "Perform the demonstrated manipulation task."

    def _annotate_episode(
        self,
        dataset: LeRobotDataset,
        episode_index: int,
        camera_key: str,
        coarse_goal: str,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        ep = dataset.meta.episodes[episode_index]
        ep_from = int(ep["dataset_from_index"])
        ep_to = int(ep["dataset_to_index"])
        length = ep_to - ep_from
        fps = dataset.meta.fps
        if length == 0:
            return []

        # Collect full windows: (images, t_start, t_end) using frame timestamps.
        # If max_frames_per_window is set and window is larger, sample that many frames (order preserved).
        window_specs: list[tuple[list[PIL.Image.Image], float, float]] = []
        start = 0
        while start + self.window_size <= length:
            offsets = _sample_window_indices(
                self.window_size,
                self.max_frames_per_window or self.window_size,
            )
            frame_indices = [ep_from + start + i for i in offsets]
            images = []
            t_start = float(dataset[frame_indices[0]]["timestamp"].item())
            for idx in frame_indices:
                item = dataset[idx]
                images.append(frame_to_pil(item[camera_key]))
            t_end = t_start + self.window_size / fps
            window_specs.append((images, t_start, t_end))
            start += self.stride

        # Last partial window
        if start < length:
            partial_len = ep_to - (ep_from + start)
            offsets = _sample_window_indices(
                partial_len,
                self.max_frames_per_window or partial_len,
            )
            frame_indices = [ep_from + start + i for i in offsets]
            images = []
            t_start = float(dataset[frame_indices[0]]["timestamp"].item())
            for idx in frame_indices:
                item = dataset[idx]
                images.append(frame_to_pil(item[camera_key]))
            t_end = float(dataset[frame_indices[-1]]["timestamp"].item()) + 1.0 / fps
            window_specs.append((images, t_start, t_end))

        # Run in batches
        skills: list[Skill] = []
        for i in range(0, len(window_specs), self.batch_size):
            chunk = window_specs[i : i + self.batch_size]
            batch_images = [spec[0] for spec in chunk]
            if len(batch_images) > 1:
                skill_names = self.segmenter.segment_skill_from_images_batch(
                    batch_images, coarse_goal, subtask_labels
                )
            else:
                skill_names = [
                    self.segmenter.segment_skill_from_images(
                        batch_images[0], coarse_goal, subtask_labels
                    )
                ]
            for (_, t_start, t_end), name in zip(chunk, skill_names, strict=True):
                skills.append(Skill(name=name, start=t_start, end=t_end))

        return skills


def main():
    parser = argparse.ArgumentParser(
        description="Image-window subtask annotation using Qwen VLM (frames as images for better accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m lerobot.data_processing.annotations.subtask_annotate_image \\
                --data-dir /path/to/dataset --camera-key observation.images.base \\
                --window-size 8 --output-dir ./output

              python -m lerobot.data_processing.annotations.subtask_annotate_image \\
                --repo-id user/dataset --camera-key observation.images.base \\
                --window-size 6 --stride 3 --model Qwen/Qwen2-VL-7B-Instruct

              # Use Qwen3-VL (MoE)
              python -m lerobot.data_processing.annotations.subtask_annotate_image \\
                --data-dir /path/to/dataset --camera-key observation.images.base \\
                --model Qwen/Qwen3-VL-30B-A3B-Instruct
        """),
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=str, help="Path to local LeRobot dataset")
    data_group.add_argument("--repo-id", type=str, help="HuggingFace Hub dataset repository ID")

    parser.add_argument(
        "--camera-key",
        type=str,
        required=True,
        help="Image or video observation key (e.g. observation.images.base)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="VLM model: Qwen2-VL or Qwen3-VL (default: Qwen/Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Number of frames per window (default: 8)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding window (default: window_size = non-overlapping)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of windows to process in one VLM call (default: 1; increase for speed)",
    )
    parser.add_argument(
        "--max-frames-per-window",
        type=int,
        default=None,
        metavar="N",
        help="If window has more than N frames, randomly sample N frames (order kept) to avoid OOM (e.g. 16)",
    )
    parser.add_argument("--episodes", type=int, nargs="+", help="Episode indices to annotate (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes that already have annotations")
    parser.add_argument(
        "--subtask-labels",
        type=str,
        nargs="*",
        default=None,
        help="Closed vocabulary: model must choose only from these labels",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for dataset with subtask_index")
    parser.add_argument("--output-repo-id", type=str, help="Output repo id (default: <repo_id>_with_subtasks)")
    parser.add_argument("--push-to-hub", action="store_true")

    args = parser.parse_args()
    console = Console()

    # Load dataset
    console.print("[cyan]Loading dataset...[/cyan]")
    if args.data_dir:
        dataset = LeRobotDataset(repo_id="local/dataset", root=args.data_dir, download_videos=False)
    else:
        dataset = LeRobotDataset(repo_id=args.repo_id, download_videos=True)
    camera_keys = dataset.meta.camera_keys
    if args.camera_key not in camera_keys:
        console.print(f"[red]Error: camera key '{args.camera_key}' not in {camera_keys}[/red]")
        return
    console.print(f"[green]✓ Loaded dataset, {dataset.meta.total_episodes} episodes[/green]")

    # Same Qwen VLM as subtask_annotate (Qwen2-VL or Qwen3-VL), image windows instead of video
    segmenter = get_image_segmenter(args.model, args.device, torch.bfloat16)

    annotator = SkillAnnotatorImage(
        segmenter=segmenter,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        max_frames_per_window=args.max_frames_per_window,
        console=console,
    )
    annotations = annotator.annotate_dataset(
        dataset=dataset,
        camera_key=args.camera_key,
        episodes=args.episodes,
        skip_existing=args.skip_existing,
        subtask_labels=args.subtask_labels,
    )

    if not annotations:
        console.print("[yellow]No annotations to save.[/yellow]")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    output_repo_id = args.output_repo_id
    new_dataset = save_skill_annotations(dataset, annotations, output_dir, output_repo_id)

    total_skills = sum(len(a.skills) for a in annotations.values())
    console.print(f"[bold green]✓ Done.[/bold green] Episodes: {len(annotations)}, total window skills: {total_skills}")
    console.print(f"  Dataset with subtask_index: {new_dataset.root}")

    if args.push_to_hub and not args.data_dir:
        console.print("[cyan]Pushing to Hub...[/cyan]")
        try:
            new_dataset.push_to_hub(push_videos=False)
            console.print("[green]✓ Pushed.[/green]")
        except Exception as e:
            console.print(f"[red]Push failed: {e}[/red]")


if __name__ == "__main__":
    main()
