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
Automatic Skill Annotation for LeRobot Datasets.

This script performs automatic subtask/skill labeling for ANY LeRobot dataset using
Vision-Language Models (VLMs). It segments each robot demonstration into short atomic
skills (1-3 seconds each) and creates a new dataset with subtask annotations.

The pipeline:
1. Loads a LeRobot dataset (local or from HuggingFace Hub)
2. For each episode, extracts video frames
3. Uses a VLM to identify skill boundaries and labels
4. Creates a subtasks.parquet file with unique subtasks
5. Adds a subtask_index feature to the dataset

NOTE: This script does NOT modify the original tasks.parquet file. It creates a
separate subtask hierarchy while preserving the original task annotations.

Supported VLMs (modular design allows easy extension):
- Qwen2-VL (default): "Qwen/Qwen2-VL-7B-Instruct"
- Qwen3-VL: "Qwen/Qwen3-VL-30B-A3B-Instruct"

Usage:
```bash
python examples/dataset/annotate.py \
    --repo-id your-username/your-dataset \
    --video-key observation.images.base \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output-dir /path/to/output \
    --push-to-hub
```

Or with a local dataset:
```bash
python examples/dataset/annotate.py \
    --data-dir /path/to/local/dataset \
    --video-key observation.images.base \
    --output-dir /path/to/output
```

After running, you can access the subtask for any frame via:
```python
dataset = LeRobotDataset(repo_id="your/dataset_with_subtasks")
item = dataset[100]
subtask_idx = item["subtask_index"]
subtask_name = dataset.meta.subtasks.iloc[subtask_idx.item()].name
```
"""

import argparse
import json
import re
import subprocess
import tempfile
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Skill Annotation Data Structures


class Skill:
    """Represents a single atomic skill/subtask in a demonstration."""

    def __init__(self, name: str, start: float, end: float):
        self.name = name
        self.start = start  # Start timestamp in seconds
        self.end = end  # End timestamp in seconds

    def to_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(name=data["name"], start=data["start"], end=data["end"])

    def __repr__(self) -> str:
        return f"Skill(name='{self.name}', start={self.start:.2f}, end={self.end:.2f})"


class EpisodeSkills:
    """Container for all skills in an episode."""

    def __init__(self, episode_index: int, description: str, skills: list[Skill]):
        self.episode_index = episode_index
        self.description = description
        self.skills = skills

    def to_dict(self) -> dict:
        return {
            "episode_index": self.episode_index,
            "description": self.description,
            "skills": [s.to_dict() for s in self.skills],
        }


# VLM Interface (Abstract Base Class for Modularity)


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models.

    To add a new VLM:
    1. Create a subclass of BaseVLM
    2. Implement the `__init__`, `segment_skills`, and `segment_skills_batch` methods
    3. Register it in the VLM_REGISTRY dictionary
    """

    @abstractmethod
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        """Initialize the VLM with model name, device, and dtype."""
        pass

    @abstractmethod
    def segment_skills(
        self, video_path: Path, episode_duration: float, coarse_goal: str | None = None
    ) -> list[Skill]:
        """
        Segment a video into atomic skills.

        Args:
            video_path: Path to the video file
            episode_duration: Total duration of the episode in seconds
            coarse_goal: Optional high-level task description

        Returns:
            List of Skill objects representing atomic manipulation skills
        """
        pass

    @abstractmethod
    def segment_skills_batch(
        self, video_paths: list[Path], episode_durations: list[float], coarse_goal: str | None = None
    ) -> list[list[Skill]]:
        """
        Segment multiple videos into atomic skills in a single batch.

        Args:
            video_paths: List of paths to video files
            episode_durations: List of episode durations in seconds
            coarse_goal: Optional high-level task description

        Returns:
            List of skill lists, one for each video
        """
        pass


def create_skill_segmentation_prompt(coarse_goal: str | None = None) -> str:
    """Create the prompt for skill segmentation."""
    goal_context = f'The overall goal is: "{coarse_goal}"\n\n' if coarse_goal else ""

    return textwrap.dedent(f"""\
        # Role
        You are a Robotics Vision System specializing in temporal action segmentation for robot manipulation demonstrations.

        # Task
        {goal_context}Segment this robot demonstration video into short atomic manipulation skills. Each skill should:
        - Last approximately 1-3 seconds
        - Describe a clear, single action (e.g., "pick up object", "move arm left", "release gripper")
        - Have precise start and end timestamps

        # Requirements
        1. **Atomic Actions**: Each skill should be a single, indivisible action
        2. **Complete Coverage**: Skills must cover the entire video duration with no gaps
        3. **Boundary Consistency**: The end of one skill equals the start of the next
        4. **Natural Language**: Use clear, descriptive names for each skill
        5. **Timestamps**: Use seconds (float) for all timestamps



        # Output Format
        After your analysis, output ONLY valid JSON with this exact structure:

        ```json
        {{
          "skills": [
            {{"name": "skill description", "start": 0.0, "end": 1.5}},
            {{"name": "another skill", "start": 1.5, "end": 3.2}}
          ]
        }}
        ```

        The first skill must start at 0.0 and the last skill must end at the video duration.
        """)


# Qwen2-VL Implementation


class Qwen2VL(BaseVLM):
    """Qwen2-VL model for skill segmentation."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.console = Console()
        self.device = device
        self.model_name = model_name
        self.process_vision_info = process_vision_info

        self.console.print(f"[cyan]Loading Qwen2-VL model: {model_name}...[/cyan]")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.console.print(f"[green]✓ Model loaded successfully on {device}[/green]")

    def segment_skills(
        self, video_path: Path, episode_duration: float, coarse_goal: str | None = None
    ) -> list[Skill]:
        """Segment video into skills using Qwen2-VL."""
        prompt = create_skill_segmentation_prompt(coarse_goal)
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {
                        "type": "text",
                        "text": f"Video duration: {duration_str} (~{episode_duration:.1f}s). Segment into atomic skills.",
                    },
                ],
            },
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)

        response = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True,
        )[0].strip()

        return self._parse_skills_response(response)

    def segment_skills_batch(
        self, video_paths: list[Path], episode_durations: list[float], coarse_goal: str | None = None
    ) -> list[list[Skill]]:
        """Segment multiple videos into skills using Qwen2-VL in a batch."""
        prompt = create_skill_segmentation_prompt(coarse_goal)
        
        # Create messages for each video
        all_messages = []
        for video_path, duration in zip(video_paths, episode_durations):
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Video duration: {duration_str} (~{duration:.1f}s). Segment into atomic skills.",
                        },
                    ],
                },
            ]
            all_messages.append(messages)
        
        # Process all videos in batch
        all_texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in all_messages:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            all_texts.append(text)
            all_image_inputs.extend(image_inputs or [])
            all_video_inputs.extend(video_inputs or [])
        
        inputs = self.processor(
            text=all_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)

        responses = self.processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True,
        )
        
        # Parse each response
        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                if not skills:
                    self.console.print(f"[yellow]Warning: No skills parsed from response for video {idx}[/yellow]")
                all_skills.append(skills)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to parse response for video {idx}: {e}[/yellow]")
                all_skills.append([])
        
        return all_skills

    def _parse_skills_response(self, response: str) -> list[Skill]:
        """Parse the VLM response into Skill objects."""
        # Extract JSON from response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(response)
            skills_data = data.get("skills", data)
            if isinstance(skills_data, list):
                return [Skill.from_dict(s) for s in skills_data]
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                skills_data = data.get("skills", [])
                return [Skill.from_dict(s) for s in skills_data]

        raise ValueError(f"Could not parse skills from response: {response[:200]}...")


# Qwen3-VL Implementation (MoE variant)


class Qwen3VL(BaseVLM):
    """Qwen3-VL MoE model for skill segmentation."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        self.console = Console()
        self.device = device
        self.model_name = model_name
        self.process_vision_info = process_vision_info

        self.console.print(f"[cyan]Loading Qwen3-VL model: {model_name}...[/cyan]")

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.console.print(f"[green]✓ Model loaded successfully on {device}[/green]")

    def segment_skills(
        self, video_path: Path, episode_duration: float, coarse_goal: str | None = None
    ) -> list[Skill]:
        """Segment video into skills using Qwen3-VL."""
        prompt = create_skill_segmentation_prompt(coarse_goal)
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {
                        "type": "text",
                        "text": f"Video duration: {duration_str} (~{episode_duration:.1f}s). Segment into atomic skills.",
                    },
                ],
            },
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)

        response = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True,
        )[0].strip()

        return self._parse_skills_response(response)

    def segment_skills_batch(
        self, video_paths: list[Path], episode_durations: list[float], coarse_goal: str | None = None
    ) -> list[list[Skill]]:
        """Segment multiple videos into skills using Qwen3-VL in a batch."""
        prompt = create_skill_segmentation_prompt(coarse_goal)
        
        # Create messages for each video
        all_messages = []
        for video_path, duration in zip(video_paths, episode_durations):
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Video duration: {duration_str} (~{duration:.1f}s). Segment into atomic skills.",
                        },
                    ],
                },
            ]
            all_messages.append(messages)
        
        # Process all videos in batch
        all_texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in all_messages:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            all_texts.append(text)
            all_image_inputs.extend(image_inputs or [])
            all_video_inputs.extend(video_inputs or [])
        
        inputs = self.processor(
            text=all_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)

        responses = self.processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True,
        )
        
        # Parse each response
        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                if not skills:
                    self.console.print(f"[yellow]Warning: No skills parsed from response for video {idx}[/yellow]")
                all_skills.append(skills)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to parse response for video {idx}: {e}[/yellow]")
                all_skills.append([])
        
        return all_skills

    def _parse_skills_response(self, response: str) -> list[Skill]:
        """Parse the VLM response into Skill objects."""
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(response)
            skills_data = data.get("skills", data)
            if isinstance(skills_data, list):
                return [Skill.from_dict(s) for s in skills_data]
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                skills_data = data.get("skills", [])
                return [Skill.from_dict(s) for s in skills_data]
        
        raise ValueError(f"Could not parse skills from response: {response[:200]}...")


# VLM Registry - Add new VLMs here

VLM_REGISTRY: dict[str, type[BaseVLM]] = {
    # Qwen2-VL variants
    "Qwen/Qwen2-VL-2B-Instruct": Qwen2VL,
    "Qwen/Qwen2-VL-7B-Instruct": Qwen2VL,
    "Qwen/Qwen2-VL-72B-Instruct": Qwen2VL,
    # Qwen3-VL variants (MoE)
    "Qwen/Qwen3-VL-30B-A3B-Instruct": Qwen3VL,
}


def get_vlm(model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> BaseVLM:
    """
    Factory function to get the appropriate VLM based on model name.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        torch_dtype: Data type for model weights

    Returns:
        Initialized VLM instance

    Raises:
        ValueError: If model is not in registry
    """
    # Check exact match first
    if model_name in VLM_REGISTRY:
        return VLM_REGISTRY[model_name](model_name, device, torch_dtype)

    # Check for partial matches (e.g., "qwen2" in model name)
    model_lower = model_name.lower()
    if "qwen3" in model_lower:
        return Qwen3VL(model_name, device, torch_dtype)
    elif "qwen2" in model_lower or "qwen-vl" in model_lower:
        return Qwen2VL(model_name, device, torch_dtype)

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported models: {list(VLM_REGISTRY.keys())}. "
        "Or implement a new VLM class inheriting from BaseVLM."
    )


# Video Extraction Utilities

class VideoExtractor:
    """Utilities for extracting and processing video segments from LeRobot datasets."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def extract_episode_video(
        self,
        video_path: Path,
        start_timestamp: float,
        end_timestamp: float,
        target_fps: int = 1,
    ) -> Path:
        """
        Extract a specific episode segment from a concatenated video file.

        Args:
            video_path: Path to the source video file
            start_timestamp: Start time in seconds
            end_timestamp: End time in seconds
            target_fps: Target frames per second for output

        Returns:
            Path to the extracted temporary video file
        """
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        duration = end_timestamp - start_timestamp

        self.console.print(
            f"[cyan]Extracting: {start_timestamp:.1f}s - {end_timestamp:.1f}s ({duration:.1f}s)[/cyan]"
        )

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
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

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e}") from e
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install ffmpeg.")

        if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError("Video extraction produced invalid file")

        return tmp_path

    def get_video_duration(self, video_path: Path) -> float:
        """Get duration of a video file in seconds."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count / fps


# Skill Annotation Pipeline
class SkillAnnotator:
    """
    Main class for annotating LeRobot datasets with skill labels.

    This class orchestrates the full annotation pipeline:
    1. Load dataset
    2. Extract video segments for each episode
    3. Run VLM-based skill segmentation
    4. Update dataset task metadata
    """

    def __init__(
        self,
        vlm: BaseVLM,
        video_extractor: VideoExtractor | None = None,
        console: Console | None = None,
        batch_size: int = 8,
    ):
        self.vlm = vlm
        self.console = console or Console()
        self.video_extractor = video_extractor or VideoExtractor(self.console)
        self.batch_size = batch_size

    def annotate_dataset(
        self,
        dataset: LeRobotDataset,
        video_key: str,
        episodes: list[int] | None = None,
        skip_existing: bool = False,
    ) -> dict[int, EpisodeSkills]:
        """
        Annotate all episodes in a dataset with skill labels using batched processing.

        Args:
            dataset: LeRobot dataset to annotate
            video_key: Key for video observations (e.g., "observation.images.base")
            episodes: Specific episode indices to annotate (None = all)
            skip_existing: Skip episodes that already have skill annotations

        Returns:
            Dictionary mapping episode index to EpisodeSkills
        """
        episode_indices = episodes or list(range(dataset.meta.total_episodes))
        annotations: dict[int, EpisodeSkills] = {}
        failed_episodes: dict[int, str] = {}  # Track failed episodes with error messages

        # Get coarse task description if available
        coarse_goal = self._get_coarse_goal(dataset)

        # Filter out episodes that already have annotations if skip_existing is True
        if skip_existing:
            existing_annotations = load_skill_annotations(dataset.root)
            if existing_annotations and "episodes" in existing_annotations:
                # Only skip episodes that exist AND have non-empty skills
                existing_episode_indices = set()
                for idx_str, episode_data in existing_annotations["episodes"].items():
                    idx = int(idx_str)
                    # Check if skills list exists and is not empty
                    if "skills" in episode_data and episode_data["skills"]:
                        existing_episode_indices.add(idx)
                
                original_count = len(episode_indices)
                episode_indices = [ep for ep in episode_indices if ep not in existing_episode_indices]
                skipped_count = original_count - len(episode_indices)
                if skipped_count > 0:
                    self.console.print(f"[cyan]Skipping {skipped_count} episodes with existing non-empty annotations[/cyan]")

        if not episode_indices:
            self.console.print("[yellow]No episodes to annotate (all already annotated)[/yellow]")
            return annotations

        print(f"Annotating {len(episode_indices)} episodes in batches of {self.batch_size}...")

        # Process episodes in batches
        for batch_start in range(0, len(episode_indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(episode_indices))
            batch_episodes = episode_indices[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//self.batch_size + 1}/{(len(episode_indices) + self.batch_size - 1)//self.batch_size} (episodes {batch_episodes[0]} to {batch_episodes[-1]})...")

            try:
                batch_annotations = self._annotate_episodes_batch(
                    dataset, batch_episodes, video_key, coarse_goal
                )
                
                for ep_idx in batch_episodes:
                    if ep_idx in batch_annotations and batch_annotations[ep_idx]:
                        skills = batch_annotations[ep_idx]
                        annotations[ep_idx] = EpisodeSkills(
                            episode_index=ep_idx,
                            description=coarse_goal,
                            skills=skills,
                        )
                        self.console.print(
                            f"[green]✓ Episode {ep_idx}: {len(skills)} skills identified[/green]"
                        )
                    else:
                        failed_episodes[ep_idx] = "Empty or missing skills from batch processing"
                        self.console.print(f"[yellow]⚠ Episode {ep_idx}: No skills extracted, will retry[/yellow]")
            except Exception as e:
                self.console.print(f"[red]✗ Batch failed: {e}. Falling back to single-episode processing...[/red]")
                # Fallback: process episodes one by one
                for ep_idx in batch_episodes:
                    try:
                        skills = self._annotate_episode(dataset, ep_idx, video_key, coarse_goal)
                        if skills:
                            annotations[ep_idx] = EpisodeSkills(
                                episode_index=ep_idx,
                                description=coarse_goal,
                                skills=skills,
                            )
                            self.console.print(
                                f"[green]✓ Episode {ep_idx}: {len(skills)} skills identified[/green]"
                            )
                        else:
                            failed_episodes[ep_idx] = "Empty skills list from single-episode processing"
                            self.console.print(f"[yellow]⚠ Episode {ep_idx}: No skills extracted, will retry[/yellow]")
                    except Exception as ep_error:
                        failed_episodes[ep_idx] = str(ep_error)
                        self.console.print(f"[yellow]⚠ Episode {ep_idx} failed: {ep_error}, will retry[/yellow]")

        # Retry failed episodes one more time
        if failed_episodes:
            self.console.print(f"\n[cyan]Retrying {len(failed_episodes)} failed episodes...[/cyan]")
            retry_count = 0
            for ep_idx, error_msg in list(failed_episodes.items()):
                self.console.print(f"[cyan]Retry attempt for episode {ep_idx} (previous error: {error_msg})[/cyan]")
                try:
                    skills = self._annotate_episode(dataset, ep_idx, video_key, coarse_goal)
                    if skills:
                        annotations[ep_idx] = EpisodeSkills(
                            episode_index=ep_idx,
                            description=coarse_goal,
                            skills=skills,
                        )
                        self.console.print(
                            f"[green]✓ Episode {ep_idx} (retry): {len(skills)} skills identified[/green]"
                        )
                        del failed_episodes[ep_idx]
                        retry_count += 1
                    else:
                        self.console.print(f"[red]✗ Episode {ep_idx} (retry): Still no skills extracted[/red]")
                except Exception as retry_error:
                    failed_episodes[ep_idx] = str(retry_error)
                    self.console.print(f"[red]✗ Episode {ep_idx} (retry) failed: {retry_error}[/red]")
            
            if retry_count > 0:
                self.console.print(f"[green]Successfully recovered {retry_count} episodes on retry[/green]")
            
            if failed_episodes:
                self.console.print(f"\n[red]⚠ Warning: {len(failed_episodes)} episodes still failed after retry:[/red]")
                for ep_idx, error_msg in failed_episodes.items():
                    self.console.print(f"  Episode {ep_idx}: {error_msg}")

        return annotations

    def _get_coarse_goal(self, dataset: LeRobotDataset) -> str:
        """Extract or generate the coarse task description."""
        # Try to get from existing task metadata
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            # Get the first task description
            first_task = dataset.meta.tasks.index[0]
            if first_task:
                return str(first_task)

        return "Perform the demonstrated manipulation task."

    def _annotate_episodes_batch(
        self,
        dataset: LeRobotDataset,
        episode_indices: list[int],
        video_key: str,
        coarse_goal: str,
    ) -> dict[int, list[Skill]]:
        """Annotate multiple episodes with skill labels in a batch."""
        # Extract all videos for this batch
        extracted_paths = []
        durations = []
        valid_episode_indices = []
        
        for ep_idx in episode_indices:
            try:
                # Get video path and timestamps
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)
                
                if not video_path.exists():
                    self.console.print(f"[yellow]Warning: Video not found for episode {ep_idx}[/yellow]")
                    continue
                
                # Get episode timestamps from metadata
                ep = dataset.meta.episodes[ep_idx]
                start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
                end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
                duration = end_ts - start_ts
                
                # Extract episode segment to temporary file
                extracted_path = self.video_extractor.extract_episode_video(
                    video_path, start_ts, end_ts, target_fps=1
                )
                
                extracted_paths.append(extracted_path)
                durations.append(duration)
                valid_episode_indices.append(ep_idx)
                
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to extract video for episode {ep_idx}: {e}[/yellow]")
                continue
        
        if not extracted_paths:
            return {}
        
        try:
            # Run VLM skill segmentation in batch
            all_skills = self.vlm.segment_skills_batch(extracted_paths, durations, coarse_goal)
            
            # Map results back to episode indices
            results = {}
            for ep_idx, skills in zip(valid_episode_indices, all_skills):
                results[ep_idx] = skills
            
            return results
            
        finally:
            # Clean up all temporary files
            for path in extracted_paths:
                if path.exists():
                    path.unlink()

    def _annotate_episode(
        self,
        dataset: LeRobotDataset,
        episode_index: int,
        video_key: str,
        coarse_goal: str,
    ) -> list[Skill]:
        """Annotate a single episode with skill labels."""
        # Get video path and timestamps for this episode
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, video_key)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get episode timestamps from metadata
        ep = dataset.meta.episodes[episode_index]
        start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
        end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
        duration = end_ts - start_ts

        # Extract episode segment to temporary file
        extracted_path = self.video_extractor.extract_episode_video(
            video_path, start_ts, end_ts, target_fps=1
        )

        try:
            # Run VLM skill segmentation
            skills = self.vlm.segment_skills(extracted_path, duration, coarse_goal)
            return skills
        finally:
            # Clean up temporary file
            if extracted_path.exists():
                extracted_path.unlink()


# Metadata Writer - Updates per-frame task_index based on skills


def get_skill_for_timestamp(skills: list[Skill], timestamp: float) -> Skill | None:
    """
    Find which skill covers a given timestamp.

    Args:
        skills: List of skills with start/end times
        timestamp: Frame timestamp in seconds

    Returns:
        The Skill that covers this timestamp, or None if not found
    """
    for skill in skills:
        if skill.start <= timestamp < skill.end:
            return skill
        # Handle the last frame (end boundary)
        if timestamp >= skill.end and skill == skills[-1]:
            return skill
    return skills[-1] if skills else None  # Fallback to last skill


def create_subtasks_dataframe(
    annotations: dict[int, EpisodeSkills],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Create a subtasks DataFrame from skill annotations.

    Args:
        annotations: Dictionary of episode skills

    Returns:
        Tuple of (subtasks_df, skill_to_subtask_idx mapping)
    """
    console = Console()

    # Collect all unique skill names
    all_skill_names: set[str] = set()
    for episode_skills in annotations.values():
        for skill in episode_skills.skills:
            all_skill_names.add(skill.name)

    console.print(f"[cyan]Found {len(all_skill_names)} unique subtasks[/cyan]")

    # Build subtasks DataFrame
    subtask_data = []
    for i, skill_name in enumerate(sorted(all_skill_names)):
        subtask_data.append({
            "subtask": skill_name,
            "subtask_index": i,
        })

    subtasks_df = pd.DataFrame(subtask_data).set_index("subtask")

    # Build skill name to subtask_index mapping
    skill_to_subtask_idx = {
        skill_name: int(subtasks_df.loc[skill_name, "subtask_index"])
        for skill_name in all_skill_names
    }

    return subtasks_df, skill_to_subtask_idx


def save_subtasks(
    subtasks_df: pd.DataFrame,
    dataset_root: Path,
    console: Console | None = None,
) -> None:
    """Save subtasks to subtasks.parquet."""
    if console is None:
        console = Console()
    
    output_path = dataset_root / "meta" / "subtasks.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    subtasks_df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    console.print(f"[green]✓ Saved subtasks to {output_path}[/green]")


def create_subtask_index_array(
    dataset: LeRobotDataset,
    annotations: dict[int, EpisodeSkills],
    skill_to_subtask_idx: dict[str, int],
) -> np.ndarray:
    """
    Create a subtask_index array for each frame based on skill annotations.

    Args:
        dataset: The LeRobot dataset
        annotations: Dictionary of episode skills
        skill_to_subtask_idx: Mapping from skill name to subtask_index

    Returns:
        Array of subtask indices for each frame in the dataset
    """
    console = Console()

    # Array to store subtask index for each frame
    # Initialize with -1 to indicate unannotated frames
    full_dataset_length = len(dataset)
    subtask_indices = np.full(full_dataset_length, -1, dtype=np.int64)

    console.print(f"[cyan]Creating subtask_index array for {full_dataset_length} frames...[/cyan]")

    # Assign subtask_index for each annotated episode
    for ep_idx, episode_skills in annotations.items():
        skills = episode_skills.skills

        # Get episode frame range
        ep = dataset.meta.episodes[ep_idx]
        ep_from = ep["dataset_from_index"]
        ep_to = ep["dataset_to_index"]

        # Process each frame in the episode
        for frame_idx in range(ep_from, ep_to):
            frame = dataset[frame_idx]
            timestamp = frame["timestamp"].item()
            
            # Find which skill covers this timestamp
            skill = get_skill_for_timestamp(skills, timestamp)

            if skill and skill.name in skill_to_subtask_idx:
                subtask_idx = skill_to_subtask_idx[skill.name]
                subtask_indices[frame_idx] = subtask_idx

    console.print(f"[green]✓ Created subtask_index array[/green]")
    return subtask_indices


def save_skill_annotations(
    dataset: LeRobotDataset,
    annotations: dict[int, EpisodeSkills],
    output_dir: Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """
    Save skill annotations to the dataset by:
    1. Creating a subtasks.parquet file with unique subtasks
    2. Adding a subtask_index feature to the dataset
    3. Saving raw skill annotations as JSON for reference

    This function does NOT modify tasks.parquet - it keeps the original tasks intact
    and creates a separate subtask hierarchy.

    Args:
        dataset: The LeRobot dataset to annotate
        annotations: Dictionary of episode skills
        output_dir: Optional directory to save the modified dataset
        repo_id: Optional repository ID for the new dataset

    Returns:
        New dataset with subtask_index feature added
    """
    console = Console()

    if not annotations:
        console.print("[yellow]No annotations to save[/yellow]")
        return dataset

    # Step 1: Create subtasks DataFrame
    console.print("[cyan]Creating subtasks DataFrame...[/cyan]")
    subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe(annotations)
    
    # Step 2: Create subtask_index array for all frames
    console.print("[cyan]Creating subtask_index array...[/cyan]")
    subtask_indices = create_subtask_index_array(dataset, annotations, skill_to_subtask_idx)

    # Step 3: Save subtasks.parquet to the original dataset root
    save_subtasks(subtasks_df, dataset.root, console)

    # Step 4: Save the raw skill annotations as JSON for reference
    skills_path = dataset.root / "meta" / "skills.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing skills data if it exists and is not empty
    existing_skills_data = None
    if skills_path.exists():
        try:
            with open(skills_path, "r") as f:
                existing_skills_data = json.load(f)
                if existing_skills_data and len(existing_skills_data.get("episodes", {})) > 0:
                    console.print(f"[cyan]Found existing skills.json with {len(existing_skills_data.get('episodes', {}))} episodes, merging...[/cyan]")
        except (json.JSONDecodeError, IOError):
            console.print("[yellow]Warning: Could not load existing skills.json, will create new file[/yellow]")
            existing_skills_data = None
    
    # Prepare new annotations
    new_episodes = {str(ep_idx): ann.to_dict() for ep_idx, ann in annotations.items()}
    
    # Merge with existing data if available
    if existing_skills_data:
        # Preserve existing episodes that are not being updated
        merged_episodes = existing_skills_data.get("episodes", {}).copy()
        merged_episodes.update(new_episodes)
        
        # Merge skill_to_subtask_index mappings
        merged_skill_to_subtask = existing_skills_data.get("skill_to_subtask_index", {}).copy()
        merged_skill_to_subtask.update(skill_to_subtask_idx)
        
        # Use existing coarse_description if available, otherwise use new one
        coarse_desc = existing_skills_data.get("coarse_description", annotations[next(iter(annotations))].description)
        
        skills_data = {
            "coarse_description": coarse_desc,
            "skill_to_subtask_index": merged_skill_to_subtask,
            "episodes": merged_episodes,
        }
        console.print(f"[cyan]Updated {len(new_episodes)} episode(s), total episodes in skills.json: {len(merged_episodes)}[/cyan]")
    else:
        # No existing data, create new
        skills_data = {
            "coarse_description": annotations[next(iter(annotations))].description,
            "skill_to_subtask_index": skill_to_subtask_idx,
            "episodes": new_episodes,
        }

    with open(skills_path, "w") as f:
        json.dump(skills_data, f, indent=2)

    console.print(f"[green]✓ Saved skill annotations to {skills_path}[/green]")

    # Step 5: Add subtask_index feature to dataset using add_features
    console.print("[cyan]Adding subtask_index feature to dataset...[/cyan]")
    
    # Determine output directory and repo_id
    if output_dir is None:
        output_dir = dataset.root.parent / f"{dataset.root.name}_with_subtasks"
    else:
        output_dir = Path(output_dir)
    
    if repo_id is None:
        repo_id = f"{dataset.repo_id}_with_subtasks"
    
    # Add feature using dataset_tools
    feature_info = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    new_dataset = add_features(
        dataset=dataset,
        features={
            "subtask_index": (subtask_indices, feature_info),
        },
        output_dir=output_dir,
        repo_id=repo_id,
    )
    
    # Copy subtasks.parquet to new output directory
    import shutil
    shutil.copy(
        dataset.root / "meta" / "subtasks.parquet",
        output_dir / "meta" / "subtasks.parquet"
    )
    shutil.copy(
        dataset.root / "meta" / "skills.json",
        output_dir / "meta" / "skills.json"
    )
    
    console.print(f"[bold green]✓ Successfully added subtask_index feature![/bold green]")
    console.print(f"  New dataset saved to: {new_dataset.root}")
    console.print(f"  Total subtasks: {len(subtasks_df)}")
    
    return new_dataset


def load_skill_annotations(dataset_root: Path) -> dict | None:
    """Load existing skill annotations from a dataset."""
    skills_path = dataset_root / "meta" / "skills.json"
    if skills_path.exists():
        with open(skills_path) as f:
            return json.load(f)
    return None


# Main Entry Point


def main():
    """Main entry point for the skill annotation script."""
    parser = argparse.ArgumentParser(
        description="Automatic skill annotation for LeRobot datasets using VLMs (with batched processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Annotate a HuggingFace Hub dataset
              python annotate.py --repo-id user/dataset --video-key observation.images.base \\
                  --output-dir ./output

              # Annotate a local dataset with custom batch size
              python annotate.py --data-dir /path/to/dataset --video-key observation.images.base \\
                  --batch-size 16 --output-dir ./output

              # Use a specific model
              python annotate.py --repo-id user/dataset --video-key observation.images.base \\
                  --model Qwen/Qwen2-VL-7B-Instruct --output-dir ./output

              # Push annotated dataset to Hub
              python annotate.py --repo-id user/dataset --video-key observation.images.base \\
                  --output-dir ./output --push-to-hub
        """),
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=str, help="Path to local LeRobot dataset")
    data_group.add_argument("--repo-id", type=str, help="HuggingFace Hub dataset repository ID")

    # Required arguments
    parser.add_argument(
        "--video-key",
        type=str,
        required=True,
        help="Video observation key (e.g., 'observation.images.base')",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="VLM model to use for skill segmentation (default: Qwen/Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of episodes to process in each batch (default: 8)",
    )

    # Episode selection
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Specific episode indices to annotate (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes that already have annotations",
    )

    # Output options
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push annotated dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for modified dataset with subtask_index feature",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        help="Repository ID for the new dataset (default: original_repo_id_with_subtasks)",
    )

    args = parser.parse_args()
    console = Console()

    # Validate arguments
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load dataset
    console.print("[cyan]Loading dataset...[/cyan]")
    if args.data_dir:
        dataset = LeRobotDataset(repo_id="local/dataset", root=args.data_dir, download_videos=False)
    else:
        dataset = LeRobotDataset(repo_id=args.repo_id, download_videos=True)

    console.print(f"[green]✓ Loaded dataset with {dataset.meta.total_episodes} episodes[/green]")

    # Validate video key
    if args.video_key not in dataset.meta.video_keys:
        available = ", ".join(dataset.meta.video_keys)
        console.print(f"[red]Error: Video key '{args.video_key}' not found. Available: {available}[/red]")
        return

    # Initialize VLM
    console.print(f"[cyan]Initializing VLM: {args.model}...[/cyan]")
    vlm = get_vlm(args.model, args.device, torch_dtype)

    # Create annotator and run annotation
    annotator = SkillAnnotator(vlm=vlm, console=console, batch_size=args.batch_size)
    console.print(f"[cyan]Processing with batch size: {args.batch_size}[/cyan]")
    annotations = annotator.annotate_dataset(
        dataset=dataset,
        video_key=args.video_key,
        episodes=args.episodes,
        skip_existing=args.skip_existing,
    )

    # Save annotations
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_repo_id = args.output_repo_id if args.output_repo_id else None
    new_dataset = save_skill_annotations(dataset, annotations, output_dir, output_repo_id)

    # Summary
    total_skills = sum(len(ann.skills) for ann in annotations.values())
    console.print(f"\n[bold green]✓ Annotation complete![/bold green]")
    console.print(f"  Episodes annotated: {len(annotations)}")
    console.print(f"  Total subtasks identified: {total_skills}")
    console.print(f"  Dataset with subtask_index saved to: {new_dataset.root}")

    # Push to hub if requested
    if args.push_to_hub:
        if args.data_dir:
            console.print("[yellow]Warning: --push-to-hub requires --repo-id, skipping...[/yellow]")
        else:
            console.print("[cyan]Pushing to HuggingFace Hub...[/cyan]")
            try:
                new_dataset.push_to_hub(push_videos=False)
                console.print(f"[green]✓ Pushed to {output_repo_id or f'{args.repo_id}_with_subtasks'}[/green]")
            except Exception as e:
                console.print(f"[red]Push failed: {e}[/red]")


if __name__ == "__main__":
    main()
