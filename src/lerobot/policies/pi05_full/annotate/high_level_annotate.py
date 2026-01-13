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
Synthetic Data Generation for Hierarchical Policy Training.

This script generates synthetic user prompts (ℓ_t) and robot utterances (u_t) for
hierarchical policy training using Qwen VLM as the generator model (pgen).

The pipeline:
1. Loads a LeRobot dataset with skill annotations (from annotate.py)
2. For each frame, generates synthetic dialogue based on:
   - Visual context (images at time t OR video clips in video mode)
   - Current skill being performed
   - History of previous skills
   - High-level task description
3. Saves results as high-level tasks and updates dataset with task_index_high_level

Modes:
- Image Mode (default): Samples frames at intervals and sends images to the model
- Video Mode (--video-mode): Passes entire skill video clips to the model

Usage:
```bash
# Image mode (default)
python examples/dataset/annotate_pgen.py \
    --repo-id lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output-dir /path/to/output

# Video mode with batch processing
python examples/dataset/annotate_pgen.py \
    --repo-id lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --video-mode \
    --video-key observation.images.base \
    --video-batch-size 4
```
"""

import argparse
import json
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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


# Prompt Template for pgen

PGEN_PROMPT_TEMPLATE_IMAGE = textwrap.dedent("""\
    # Role
    You are a robot-assistant dialogue generator for hierarchical robot policies.
    
    # Task
    You will receive:
    - A list of images showing the current robot scene at time t
    - The high-level task: {task_description}
    - Previous skill steps completed: {skill_history}
    - The next skill to be performed by the robot: {skill_current}
    
    # Your Goal
    Generate two things that create a natural human-robot interaction:
    1. **user_prompt**: A natural-sounding user request that logically leads to the robot 
       performing the skill "{skill_current}" given the task context and history.
    2. **robot_utterance**: A natural robot reply acknowledging or clarifying the request.
    
    # Guidelines
    - The user prompt should be grounded in the visual scene and task context
    - Vary interaction types: direct commands, implicit requests, corrections, constraints
    - Examples of user prompt styles:
      * Direct: "Can you pick up the red brick?"
      * Implicit: "I need something red for the tower"
      * Negative: "Don't pick up the blue one"
      * Constraint: "Make sure to handle it gently"
      * Correction: "Actually, move to the other box instead"
    - Robot responses should be appropriate: confirmations, clarifications, or error handling
    - Use the skill history to ensure continuity (don't repeat past actions)
    - Consider world knowledge (dietary preferences, object properties, etc.)
    
    # Scenario Types (choose one that fits):
    - **specific_object**: User specifies exact object/action
    - **negative_task**: User says what NOT to do
    - **situated_correction**: User adjusts based on current state
    - **implicit_request**: User implies need without direct command
    - **constraint_based**: User adds specific constraints
    
    # Response Types (choose one that fits):
    - **confirmation**: Simple "OK, I'll do X"
    - **clarification**: "Just to confirm, you want me to..."
    - **acknowledgment**: "Got it, [doing action]"
    - **constraint_acknowledgment**: "Sure, I'll [action] while [constraint]"
    
    # Output Format
    Respond ONLY with valid JSON:
    {{
      "scenario_type": "one of the types above",
      "response_type": "one of the types above", 
      "user_prompt": "natural user request here",
      "robot_utterance": "natural robot response here"
    }}
    
    The responses must be grounded in the visual scene, the task, and the skill history.
    Make it sound like a real human-robot interaction.
    """)

PGEN_PROMPT_TEMPLATE_VIDEO = textwrap.dedent("""\
    # Role
    You are a robot-assistant dialogue generator for hierarchical robot policies.
    
    # Task
    You are watching a full robot demonstration video for the task: {task_description}
    
    For each timestamp below, generate natural human-robot dialogue that would have led to the observed behavior.
    At each timestamp, you'll see:
    - What skills have been completed so far (cumulative history)
    - What skill is currently being executed
    
    {timestamp_context}
    
    # Your Goal
    For EACH timestamp, generate:
    1. **user_prompt**: A natural user request that would lead to the robot performing the current skill
    2. **robot_utterance**: A natural robot response acknowledging the request
    
    # Guidelines
    - Watch the video from start to each timestamp to understand the context
    - Ground prompts in what's visible in the video at that time
    - Vary interaction types: direct commands, implicit requests, corrections, constraints
    - Examples of user prompt styles:
      * Direct: "Can you pick up the red brick?"
      * Implicit: "I need something red for the tower"
      * Negative: "Don't pick up the blue one"
      * Constraint: "Make sure to handle it gently"
      * Correction: "Actually, move to the other box instead"
    - Robot responses should be appropriate: confirmations, clarifications, or error handling
    - Ensure continuity across timestamps (don't contradict earlier dialogue)
    - Consider world knowledge (dietary preferences, object properties, etc.)
    
    # Scenario Types:
    - **specific_object**: User specifies exact object/action
    - **negative_task**: User says what NOT to do
    - **situated_correction**: User adjusts based on current state
    - **implicit_request**: User implies need without direct command
    - **constraint_based**: User adds specific constraints
    
    # Response Types:
    - **confirmation**: Simple "OK, I'll do X"
    - **clarification**: "Just to confirm, you want me to..."
    - **acknowledgment**: "Got it, [doing action]"
    - **constraint_acknowledgment**: "Sure, I'll [action] while [constraint]"
    
    # Output Format
    Respond ONLY with valid JSON array:
    [
      {{
        "timestamp": timestamp_value,
        "scenario_type": "one of the types above",
        "response_type": "one of the types above", 
        "user_prompt": "natural user request here",
        "robot_utterance": "natural robot response here"
      }},
      ... (one entry per timestamp)
    ]
    
    Make it sound like a real human-robot interaction grounded in the video.
    """)


def construct_prompt_image(
    task_description: str,
    skill_history: list[str],
    skill_current: str,
) -> str:
    """
    Construct the text prompt for pgen in image mode.
    
    Args:
        task_description: High-level task description
        skill_history: List of previously completed skills
        skill_current: Current skill to be performed
        
    Returns:
        Formatted prompt string
    """
    # Format skill history nicely
    if skill_history:
        history_str = ", ".join(f'"{s}"' for s in skill_history[-5:])  # Last 5 for context
        if len(skill_history) > 5:
            history_str = f"... {history_str}"
    else:
        history_str = "None (starting the task)"
    
    return PGEN_PROMPT_TEMPLATE_IMAGE.format(
        task_description=task_description,
        skill_history=history_str,
        skill_current=skill_current,
    )


def construct_prompt_video(
    task_description: str,
    timestamps_with_skills: list[dict],
) -> str:
    """
    Construct the text prompt for pgen in video mode.
    
    Args:
        task_description: High-level task description
        timestamps_with_skills: List of dicts with keys:
            - timestamp: float
            - skills_so_far: list[str]
            - current_skill: str
        
    Returns:
        Formatted prompt string
    """
    # Build timestamp context
    timestamp_lines = []
    for item in timestamps_with_skills:
        ts = item["timestamp"]
        skills_so_far = item["skills_so_far"]
        current_skill = item["current_skill"]
        
        if skills_so_far:
            skills_str = ", ".join(f'"{s}"' for s in skills_so_far)
        else:
            skills_str = "None (starting)"
        
        timestamp_lines.append(
            f"- **Timestamp {ts:.2f}s**: Skills completed: [{skills_str}] | Current skill: \"{current_skill}\""
        )
    
    timestamp_context = "\n".join(timestamp_lines)
    
    return PGEN_PROMPT_TEMPLATE_VIDEO.format(
        task_description=task_description,
        timestamp_context=timestamp_context,
    )


# Qwen VLM Interface

class QwenPgen:
    """Qwen VLM wrapper for synthetic dialogue generation."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.7,
    ):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        self.console = Console()
        self.device = device
        self.model_name = model_name
        self.temperature = temperature
        self.process_vision_info = process_vision_info
        
        self.console.print(f"[cyan]Loading Qwen model: {model_name}...[/cyan]")
        
        # Load model based on name
        if "qwen3" in model_name.lower():
            from transformers import Qwen3VLMoeForConditionalGeneration
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.console.print(f"[green]✓ Model loaded successfully on {device}[/green]")
    
    def call_qwen(
        self,
        images: list[Image.Image | str] | None = None,
        prompt: str = "",
        video: str | Path | None = None,
    ) -> dict[str, str]:
        """
        Call Qwen VLM to generate synthetic dialogue for a single request.
        
        Args:
            images: List of PIL Images or image paths (for image mode)
            prompt: Text prompt for generation
            video: Path to video file (for video mode)
            
        Returns:
            Dictionary with keys: scenario_type, response_type, user_prompt, robot_utterance
        """
        # Use batch method with single item
        results = self.call_qwen_batch(
            batch_images=[images] if images else [None],
            batch_prompts=[prompt],
            batch_videos=[video] if video else [None],
        )
        return results[0]
    
    def call_qwen_batch(
        self,
        batch_images: list[list[Image.Image | str] | None],
        batch_prompts: list[str],
        batch_videos: list[str | Path | None] | None = None,
    ) -> list[dict[str, str]]:
        """
        Call Qwen VLM to generate synthetic dialogue for a batch of requests.
        
        Args:
            batch_images: List of image lists, one per request (None for video mode)
            batch_prompts: List of text prompts, one per request
            batch_videos: List of video paths, one per request (None for image mode)
            
        Returns:
            List of dictionaries, each with keys: scenario_type, response_type, user_prompt, robot_utterance
        """
        if batch_videos is None:
            batch_videos = [None] * len(batch_images)
        
        if len(batch_images) != len(batch_prompts) or len(batch_images) != len(batch_videos):
            raise ValueError(
                f"Batch size mismatch: {len(batch_images)} image lists vs "
                f"{len(batch_prompts)} prompts vs {len(batch_videos)} videos"
            )
        
        batch_size = len(batch_images)
        if batch_size == 0:
            return []
        
        # Build messages for each item in batch
        all_messages = []
        for images, prompt, video in zip(batch_images, batch_prompts, batch_videos):
            content = []
            
            # Add video or images
            if video is not None:
                # Video mode
                content.append({"type": "video", "video": str(video), "fps": 1.0})
            elif images is not None:
                # Image mode
                for img in images:
                    if isinstance(img, str):
                        content.append({"type": "image", "image": img})
                    else:
                        # PIL Image
                        content.append({"type": "image", "image": img})
            
            content.append({"type": "text", "text": prompt})
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            all_messages.append(messages)
        
        # Process all inputs
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in all_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_inputs, video_inputs = self.process_vision_info(messages)
            all_image_inputs.append(image_inputs)
            all_video_inputs.append(video_inputs)
        
        # Flatten image and video inputs for batch processing
        # The processor expects a flat list of images across all batch items
        flat_images = []
        for img_list in all_image_inputs:
            if img_list is not None:
                if isinstance(img_list, list):
                    flat_images.extend(img_list)
                else:
                    flat_images.append(img_list)
        
        flat_videos = []
        for vid_list in all_video_inputs:
            if vid_list is not None:
                if isinstance(vid_list, list):
                    flat_videos.extend(vid_list)
                else:
                    flat_videos.append(vid_list)
        
        # Process batch
        inputs = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            videos=flat_videos if flat_videos else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.temperature,
            )
        
        # Decode responses
        responses = self.processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True,
        )
        
        # Parse all responses
        results = []
        for response in responses:
            try:
                parsed = self._parse_response(response.strip())
                results.append(parsed)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to parse response: {e}[/yellow]")
                # Return empty/default result
                results.append({
                    "scenario_type": "specific_object",
                    "response_type": "confirmation",
                    "user_prompt": "",
                    "robot_utterance": "",
                })
        
        return results
    
    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from model."""
        # Extract JSON from response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        try:
            data = json.loads(response)
            return {
                "scenario_type": data.get("scenario_type", "specific_object"),
                "response_type": data.get("response_type", "confirmation"),
                "user_prompt": data.get("user_prompt", ""),
                "robot_utterance": data.get("robot_utterance", ""),
            }
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "scenario_type": data.get("scenario_type", "specific_object"),
                    "response_type": data.get("response_type", "confirmation"),
                    "user_prompt": data.get("user_prompt", ""),
                    "robot_utterance": data.get("robot_utterance", ""),
                }
            
            raise ValueError(f"Could not parse response: {response[:200]}...")


# Annotation Pipeline

def load_skills_metadata(dataset_root: Path) -> dict | None:
    """Load skills.json metadata from annotated dataset."""
    skills_path = dataset_root / "meta" / "skills.json"
    if skills_path.exists():
        with open(skills_path) as f:
            return json.load(f)
    return None


def get_skill_at_timestamp(skills: list[dict], timestamp: float) -> str | None:
    """Find which skill covers a given timestamp."""
    for skill in skills:
        if skill["start"] <= timestamp < skill["end"]:
            return skill["name"]
        # Handle last frame
        if timestamp >= skill["end"] and skill == skills[-1]:
            return skill["name"]
    return skills[-1]["name"] if skills else None


def annotate_sample_image(
    pgen: QwenPgen,
    images: list[Image.Image | str],
    task_description: str,
    skill_history: list[str],
    skill_current: str,
) -> dict[str, str]:
    """
    Generate synthetic dialogue for a single sample using images.
    
    Args:
        pgen: Qwen model wrapper
        images: List of images at current timestep
        task_description: High-level task description
        skill_history: Previous skills completed
        skill_current: Current skill being performed
        
    Returns:
        Dictionary with generated dialogue
    """
    prompt = construct_prompt_image(task_description, skill_history, skill_current)
    result = pgen.call_qwen(images=images, prompt=prompt, video=None)
    return result


def annotate_episode_video(
    pgen: QwenPgen,
    video: str | Path,
    task_description: str,
    timestamps_with_skills: list[dict],
) -> list[dict[str, Any]]:
    """
    Generate synthetic dialogue for an entire episode using video.
    
    Args:
        pgen: Qwen model wrapper
        video: Path to episode video file
        task_description: High-level task description
        timestamps_with_skills: List of dicts with timestamp, skills_so_far, current_skill
        
    Returns:
        List of dictionaries with generated dialogue, one per timestamp
    """
    # Use batch method with single episode
    results = annotate_episodes_video_batch(
        pgen=pgen,
        batch_videos=[video],
        batch_task_descriptions=[task_description],
        batch_timestamps_with_skills=[timestamps_with_skills],
    )
    return results[0]


def annotate_episodes_video_batch(
    pgen: QwenPgen,
    batch_videos: list[str | Path],
    batch_task_descriptions: list[str],
    batch_timestamps_with_skills: list[list[dict]],
) -> list[list[dict[str, Any]]]:
    """
    Generate synthetic dialogue for multiple episodes using videos in batch.
    
    Args:
        pgen: Qwen model wrapper
        batch_videos: List of paths to episode video files
        batch_task_descriptions: List of high-level task descriptions
        batch_timestamps_with_skills: List of timestamp lists, one per episode
        
    Returns:
        List of result lists, one per episode (each containing dicts with generated dialogue)
    """
    batch_size = len(batch_videos)
    if batch_size == 0:
        return []
    
    # Build messages for each episode
    all_messages = []
    for video, task_desc, timestamps_with_skills in zip(
        batch_videos, batch_task_descriptions, batch_timestamps_with_skills
    ):
        prompt = construct_prompt_video(task_desc, timestamps_with_skills)
        
        content = [
            {"type": "video", "video": str(video), "fps": 1.0},
            {"type": "text", "text": prompt},
        ]
        
        messages = [{"role": "user", "content": content}]
        all_messages.append(messages)
    
    # Process all episodes through Qwen in batch
    all_texts = []
    all_image_inputs = []
    all_video_inputs = []
    
    for messages in all_messages:
        text = pgen.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = pgen.process_vision_info(messages)
        all_texts.append(text)
        all_image_inputs.extend(image_inputs or [])
        all_video_inputs.extend(video_inputs or [])
    
    inputs = pgen.processor(
        text=all_texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(pgen.device)
    
    with torch.no_grad():
        generated_ids = pgen.model.generate(
            **inputs,
            max_new_tokens=2048,  # Larger for multiple timestamps per episode
            do_sample=True,
            temperature=pgen.temperature,
        )
    
    responses = pgen.processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
    )
    
    # Parse each response
    all_results = []
    for response, timestamps_with_skills in zip(responses, batch_timestamps_with_skills):
        results = _parse_video_response(response.strip(), timestamps_with_skills)
        all_results.append(results)
    
    return all_results


def _parse_video_response(response: str, timestamps_with_skills: list[dict]) -> list[dict[str, Any]]:
    """Parse JSON array response from video mode."""
    # Extract JSON from response
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]
    
    try:
        data = json.loads(response)
        if not isinstance(data, list):
            # If it's a dict with a list inside
            if "annotations" in data:
                data = data["annotations"]
            elif "results" in data:
                data = data["results"]
            else:
                raise ValueError("Expected JSON array or dict with 'annotations'/'results' key")
        
        results = []
        for item in data:
            results.append({
                "timestamp": item.get("timestamp", 0.0),
                "scenario_type": item.get("scenario_type", "specific_object"),
                "response_type": item.get("response_type", "confirmation"),
                "user_prompt": item.get("user_prompt", ""),
                "robot_utterance": item.get("robot_utterance", ""),
            })
        
        return results
        
    except json.JSONDecodeError:
        # Try to find JSON array in response
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            results = []
            for item in data:
                results.append({
                    "timestamp": item.get("timestamp", 0.0),
                    "scenario_type": item.get("scenario_type", "specific_object"),
                    "response_type": item.get("response_type", "confirmation"),
                    "user_prompt": item.get("user_prompt", ""),
                    "robot_utterance": item.get("robot_utterance", ""),
                })
            return results
        
        breakpoint()
        # Fallback: return empty results for each timestamp
        print(f"Warning: Could not parse video response: {response[:200]}...")
        return [
            {
                "timestamp": ts["timestamp"],
                "scenario_type": "specific_object",
                "response_type": "confirmation",
                "user_prompt": "",
                "robot_utterance": "",
            }
            for ts in timestamps_with_skills
        ]




def _generate_synthetic_data_video_mode(
    dataset: LeRobotDataset,
    pgen: QwenPgen,
    skills_metadata: dict,
    video_key: str,
    video_extractor: VideoExtractor,
    console: Console,
    sample_interval_seconds: float = 1.0,
    batch_size: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """
    Generate synthetic dialogue data using video mode with batched VLM calls.
    
    The VLM sees full episode videos and generates dialogue for multiple
    timestamps per episode, with cumulative skill history at each timestamp.
    
    Args:
        dataset: LeRobot dataset with skill annotations
        pgen: Qwen model wrapper
        skills_metadata: Loaded skills.json metadata
        video_key: Video observation key (e.g., 'observation.images.base')
        video_extractor: VideoExtractor instance
        console: Rich console for logging
        sample_interval_seconds: Sample timestamps at this interval
        batch_size: Number of episodes to process in each VLM batch call
        
    Returns:
        Tuple of (tasks_df, task_indices_array, debug_outputs)
    """
    coarse_description = skills_metadata.get("coarse_description", "Complete the task")
    episodes = skills_metadata.get("episodes", {})
    
    # Track unique high-level tasks
    high_level_tasks = {}
    task_index_counter = 0
    
    # Array to store task index for each frame
    full_dataset_length = len(dataset)
    task_indices = np.zeros(full_dataset_length, dtype=np.int64)
    
    debug_outputs = []
    timestamps_processed = 0
    
    console.print(f"[cyan]Processing {len(episodes)} episodes in VIDEO MODE with batch_size={batch_size}...[/cyan]")
    console.print(f"[cyan]Sampling interval: {sample_interval_seconds}s[/cyan]")
    
    # Convert episodes dict to list for batching
    episode_list = list(episodes.items())
    
    # Process episodes in batches
    for batch_start in tqdm(range(0, len(episode_list), batch_size), desc="Processing episode batches"):
        batch_end = min(batch_start + batch_size, len(episode_list))
        batch_episodes = episode_list[batch_start:batch_end]
        
        # Collect data for this batch
        batch_data = []
        extracted_videos = []
        
        for episode_key, episode_data in batch_episodes:
            episode_idx = int(episode_key)
            skills = episode_data.get("skills", [])
            description = episode_data.get("description", coarse_description)
            
            if not skills:
                console.print(f"[yellow]Warning: Episode {episode_idx} has no skills[/yellow]")
                continue
            
            # Get video path and extract full episode
            extracted_path = None
            try:
                video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, video_key)
                if not video_path.exists():
                    console.print(f"[yellow]Warning: Video not found for episode {episode_idx}[/yellow]")
                    continue
                
                # Get episode timestamps
                ep = dataset.meta.episodes[episode_idx]
                episode_start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
                episode_end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
                duration = episode_end_ts - episode_start_ts
                
                # Extract FULL episode video
                extracted_path = video_extractor.extract_episode_video(
                    video_path, episode_start_ts, episode_end_ts, target_fps=1
                )
                extracted_videos.append(extracted_path)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to extract video for episode {episode_idx}: {e}[/yellow]")
                continue
            
            # Build list of timestamps to sample
            timestamps_with_skills = []
            current_time = 0.0
            
            while current_time <= duration:
                # Find which skill is active at this timestamp
                current_skill = None
                skills_so_far = []
                
                for skill in skills:
                    if skill["end"] <= current_time:
                        skills_so_far.append(skill["name"])
                    elif skill["start"] <= current_time < skill["end"]:
                        current_skill = skill["name"]
                        break
                    elif current_time >= skill["end"] and skill == skills[-1]:
                        current_skill = skill["name"]
                        break
                
                if current_skill:
                    timestamps_with_skills.append({
                        "timestamp": current_time,
                        "skills_so_far": skills_so_far.copy(),
                        "current_skill": current_skill,
                    })
                
                current_time += sample_interval_seconds
            
            if not timestamps_with_skills:
                console.print(f"[yellow]Warning: No valid timestamps for episode {episode_idx}[/yellow]")
                continue
            
            # Store batch item
            batch_data.append({
                "episode_idx": episode_idx,
                "episode_metadata": ep,
                "video_path": extracted_path,
                "task_description": description,
                "timestamps_with_skills": timestamps_with_skills,
                "skills": skills,
            })
        
        if not batch_data:
            continue
        
        # BATCHED VLM CALL for all episodes in batch
        try:
            batch_results = annotate_episodes_video_batch(
                pgen=pgen,
                batch_videos=[item["video_path"] for item in batch_data],
                batch_task_descriptions=[item["task_description"] for item in batch_data],
                batch_timestamps_with_skills=[item["timestamps_with_skills"] for item in batch_data],
            )
            
            # Process results for each episode in batch
            for item, results in zip(batch_data, batch_results):
                episode_idx = item["episode_idx"]
                ep = item["episode_metadata"]
                timestamps_with_skills = item["timestamps_with_skills"]
                description = item["task_description"]
                
                timestamps_processed += len(results)
                
                # Map results back to timestamps and create task indices
                timestamp_to_result = {}
                for result in results:
                    ts = result["timestamp"]
                    timestamp_to_result[ts] = result
                
                # Process each sampled timestamp
                for ts_info in timestamps_with_skills:
                    ts = ts_info["timestamp"]
                    result = timestamp_to_result.get(ts, {
                        "timestamp": ts,
                        "scenario_type": "specific_object",
                        "response_type": "confirmation",
                        "user_prompt": "",
                        "robot_utterance": "",
                    })
                    
                    # Create unique task key
                    task_key = (
                        result["user_prompt"],
                        result["robot_utterance"],
                        ts_info["current_skill"],
                        result["scenario_type"],
                        result["response_type"],
                    )
                    
                    # Assign or create task index
                    if task_key not in high_level_tasks:
                        high_level_tasks[task_key] = task_index_counter
                        task_index_counter += 1
                    
                    current_task_idx = high_level_tasks[task_key]
                    
                    # Find all frames at this timestamp and assign task_idx
                    ep_from = ep["dataset_from_index"]
                    ep_to = ep["dataset_to_index"]
                    
                    for frame_idx in range(ep_from, ep_to):
                        frame = dataset[frame_idx]
                        frame_ts = frame["timestamp"].item()
                        
                        # Assign to closest sampled timestamp
                        if abs(frame_ts - ts) < sample_interval_seconds / 2:
                            task_indices[frame_idx] = current_task_idx
                    
                    # Save for debugging
                    debug_outputs.append({
                        "episode_id": int(episode_idx),
                        "timestamp": float(ts),
                        "skill_current": ts_info["current_skill"],
                        "skills_so_far": ts_info["skills_so_far"],
                        "task_description": description,
                        "video_mode": True,
                        **result,
                    })
        
        finally:
            # Clean up extracted videos
            for extracted_path in extracted_videos:
                if extracted_path and extracted_path.exists():
                    extracted_path.unlink()
    
    console.print(f"[green]✓ Processed {timestamps_processed} timestamps across {len(episodes)} episodes[/green]")
    
    # Create tasks DataFrame
    tasks_data = []
    for task_key, task_idx in sorted(high_level_tasks.items(), key=lambda x: x[1]):
        user_prompt, robot_utterance, skill, scenario_type, response_type = task_key
        tasks_data.append({
            "task": f"{user_prompt} | {robot_utterance}",
            "task_index": task_idx,
            "user_prompt": user_prompt,
            "robot_utterance": robot_utterance,
            "skill": skill,
            "scenario_type": scenario_type,
            "response_type": response_type,
        })
    
    tasks_df = pd.DataFrame(tasks_data).set_index("task")
    console.print(f"[green]✓ Generated {len(high_level_tasks)} unique high-level tasks[/green]")
    
    return tasks_df, task_indices, debug_outputs


def generate_synthetic_data(
    dataset: LeRobotDataset,
    pgen: QwenPgen,
    skills_metadata: dict,
    image_keys: list[str],
    sample_interval_seconds: float = 1.0,
    console: Console | None = None,
    video_mode: bool = False,
    video_key: str | None = None,
    video_batch_size: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """
    Generate synthetic dialogue data for entire dataset.
    
    This function processes ALL frames in the dataset, but only calls the VLM
    at specified intervals (sample_interval_seconds). Frames between samples
    inherit the task_index from the most recent sample.
    
    Args:
        dataset: LeRobot dataset with skill annotations
        pgen: Qwen model wrapper
        skills_metadata: Loaded skills.json metadata
        image_keys: List of image observation keys to use (for image mode)
        sample_interval_seconds: Generate dialogue every N seconds (default: 1.0)
        console: Rich console for logging
        video_mode: If True, use video clips instead of sampled images
        video_key: Video observation key for video mode (e.g., 'observation.images.base')
        video_batch_size: Number of episodes to process in each VLM batch (video mode only)
        
    Returns:
        Tuple of (tasks_df, task_indices_array, debug_outputs)
        - tasks_df: DataFrame with high-level tasks (user_prompt, robot_utterance, etc.)
        - task_indices_array: Array of task indices for each frame (full dataset length)
        - debug_outputs: List of debug dictionaries (only for sampled frames)
    """
    if console is None:
        console = Console()
    
    # Extract metadata
    coarse_description = skills_metadata.get("coarse_description", "Complete the task")
    episodes = skills_metadata.get("episodes", {})
    
    # Track unique high-level tasks
    high_level_tasks = {}  # (user_prompt, robot_utterance, skill) -> task_index
    task_index_counter = 0  # Start at 0
    
    # Array to store task index for each frame - MUST match full dataset length
    full_dataset_length = len(dataset)
    task_indices = np.zeros(full_dataset_length, dtype=np.int64)
    
    # For debugging - save to JSONL
    debug_outputs = []
    
    # Initialize video extractor if in video mode
    video_extractor = VideoExtractor(console) if video_mode else None
    
    if video_mode:
        if video_key is None:
            raise ValueError("video_key must be provided when video_mode=True")
        console.print(f"[cyan]Using VIDEO MODE with video key: {video_key}[/cyan]")
        console.print(f"[cyan]Video batch size: {video_batch_size} episodes per VLM call[/cyan]")
        # In video mode, process episodes in batches with full videos
        return _generate_synthetic_data_video_mode(
            dataset=dataset,
            pgen=pgen,
            skills_metadata=skills_metadata,
            video_key=video_key,
            video_extractor=video_extractor,
            console=console,
            sample_interval_seconds=sample_interval_seconds,
            batch_size=video_batch_size,
        )
    
    # IMAGE MODE (original logic)
    # Track sampling
    last_sample_timestamp = {}  # episode_idx -> last sampled timestamp
    last_task_index = {}  # episode_idx -> last generated task_index
    frames_sampled = 0
    
    console.print(f"[cyan]Processing all {full_dataset_length} frames from {dataset.meta.total_episodes} episodes...[/cyan]")
    console.print(f"[cyan]Sampling interval: {sample_interval_seconds}s (fps: {dataset.meta.fps})[/cyan]")
    
    # Process each frame in the FULL dataset
    for frame_idx in tqdm(range(full_dataset_length), desc="Generating synthetic dialogue"):
        try:
            # Get frame data
            frame = dataset[frame_idx]
            episode_idx = frame["episode_index"].item()
            timestamp = frame["timestamp"].item()
            
            # Get episode skills
            episode_key = str(episode_idx)
            if episode_key not in episodes:
                console.print(f"[yellow]Warning: Episode {episode_idx} not in skills metadata[/yellow]")
                continue
            
            episode_data = episodes[episode_key]
            skills = episode_data.get("skills", [])
            description = episode_data.get("description", coarse_description)
            
            # Find current skill
            current_skill = get_skill_at_timestamp(skills, timestamp)
            if current_skill is None:
                console.print(f"[yellow]Warning: No skill found for timestamp {timestamp}[/yellow]")
                continue
            
            # Determine if we should sample this frame
            should_sample = False
            
            # Always sample first frame of an episode
            if episode_idx not in last_sample_timestamp:
                should_sample = True
                last_sample_timestamp[episode_idx] = timestamp
            else:
                # Sample if enough time has passed
                time_since_last = timestamp - last_sample_timestamp[episode_idx]
                if time_since_last >= sample_interval_seconds:
                    should_sample = True
                    last_sample_timestamp[episode_idx] = timestamp
            
            # If not sampling, reuse last task index for this episode
            if not should_sample:
                if episode_idx in last_task_index:
                    task_indices[frame_idx] = last_task_index[episode_idx]
                continue
            
            # Sample this frame - generate synthetic dialogue
            frames_sampled += 1
            
            # Build skill history (all skills before current timestamp)
            skill_history = []
            for skill in skills:
                if skill["end"] <= timestamp:
                    skill_history.append(skill["name"])
            
            # Load images
            images = []
            for img_key in image_keys:
                if img_key in frame:
                    # Frame images are tensors (C, H, W) in [0, 1]
                    img_tensor = frame[img_key]
                    if len(img_tensor.shape) == 4:  # (T, C, H, W)
                        img_tensor = img_tensor[-1]  # Take last frame
                    
                    # Convert to PIL Image
                    img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array)
                    images.append(img_pil)
            
            if not images:
                console.print(f"[yellow]Warning: No images found for frame {frame_idx}[/yellow]")
                continue
            
            # Generate synthetic dialogue
            result = annotate_sample_image(
                pgen=pgen,
                images=images,
                task_description=description,
                skill_history=skill_history,
                skill_current=current_skill,
            )
            
            # Create unique task key
            task_key = (
                result["user_prompt"],
                result["robot_utterance"],
                current_skill,
                result["scenario_type"],
                result["response_type"],
            )
            
            # Assign or create task index
            if task_key not in high_level_tasks:
                high_level_tasks[task_key] = task_index_counter
                task_index_counter += 1
            
            current_task_idx = high_level_tasks[task_key]
            task_indices[frame_idx] = current_task_idx
            last_task_index[episode_idx] = current_task_idx
            
            # Save for debugging
            debug_outputs.append({
                "episode_id": int(episode_idx),
                "frame_index": frame_idx,
                "timestamp": float(timestamp),
                "skill_current": current_skill,
                "skill_history": skill_history,
                "task_description": description,
                "sampled": True,
                **result,
            })
            
        except Exception as e:
            console.print(f"[red]Error processing frame {frame_idx}: {e}[/red]")
            continue
    
    console.print(f"[green]✓ Sampled {frames_sampled} frames out of {full_dataset_length} total ({frames_sampled/full_dataset_length*100:.1f}%)[/green]")
    
    # Create tasks DataFrame
    tasks_data = []
    for task_key, task_idx in sorted(high_level_tasks.items(), key=lambda x: x[1]):
        user_prompt, robot_utterance, skill, scenario_type, response_type = task_key
        tasks_data.append({
            "task": f"{user_prompt} | {robot_utterance}",
            "task_index": task_idx,
            "user_prompt": user_prompt,
            "robot_utterance": robot_utterance,
            "skill": skill,
            "scenario_type": scenario_type,
            "response_type": response_type,
        })
    
    tasks_df = pd.DataFrame(tasks_data).set_index("task")
    
    console.print(f"[green]✓ Generated {len(high_level_tasks)} unique high-level tasks[/green]")
    
    return tasks_df, task_indices, debug_outputs


def save_high_level_tasks(
    tasks_df: pd.DataFrame,
    dataset_root: Path,
    console: Console | None = None,
) -> None:
    """Save high-level tasks to tasks_high_level.parquet."""
    if console is None:
        console = Console()
    
    output_path = dataset_root / "meta" / "tasks_high_level.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tasks_df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    console.print(f"[green]✓ Saved high-level tasks to {output_path}[/green]")


def save_debug_outputs(
    debug_outputs: list[dict],
    dataset_root: Path,
    console: Console | None = None,
) -> None:
    """Save debug outputs to JSONL file."""
    if console is None:
        console = Console()
    
    output_path = dataset_root / "meta" / "syn_annotations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in debug_outputs:
            f.write(json.dumps(item) + "\n")
    
    console.print(f"[green]✓ Saved debug annotations to {output_path}[/green]")


# main entry point

def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic dialogue data for hierarchical robot policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Generate synthetic data for a dataset (image mode)
              python annotate_pgen.py --repo-id lerobot/svla_so101_pickplace \\
                  --model Qwen/Qwen2-VL-7B-Instruct \\
                  --output-dir ./output
              
              # Use video mode with batching (passes full episode videos)
              python annotate_pgen.py --repo-id lerobot/svla_so101_pickplace \\
                  --model Qwen/Qwen2-VL-7B-Instruct \\
                  --video-mode \\
                  --video-key observation.images.base \\
                  --video-batch-size 4
              
              # Use Qwen3 model with custom parameters
              python annotate_pgen.py --repo-id lerobot/svla_so101_pickplace \\
                  --model Qwen/Qwen3-VL-30B-A3B-Instruct \\
                  --temperature 0.8 \\
                  --batch-size 1
        """),
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=str, help="Path to local LeRobot dataset")
    data_group.add_argument("--repo-id", type=str, help="HuggingFace Hub dataset repository ID")
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="VLM model to use (default: Qwen/Qwen2-VL-7B-Instruct)",
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
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1) [currently unused]",
    )
    parser.add_argument(
        "--num-image-views-per-sample",
        type=int,
        default=1,
        help="Number of camera views to use per sample (default: 1)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Generate dialogue every N seconds (default: 1.0). Frames between samples reuse the last generated dialogue. "
             "Use larger intervals (e.g., 2.0 or 5.0) for faster processing during testing.",
    )
    parser.add_argument(
        "--video-mode",
        action="store_true",
        help="Use video input instead of sampled image frames. Passes entire skill video clips to the model.",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Video observation key for video mode (e.g., 'observation.images.base'). "
             "If not specified, uses the first available video key.",
    )
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=1,
        help="Number of episodes to process in each VLM batch call in video mode (default: 1)",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for modified dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push modified dataset to HuggingFace Hub",
    )
    # add image key
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Image observation key to use for image mode (default: None)",
    )
    
    args = parser.parse_args()
    console = Console()
    
    # Load dataset
    console.print("[cyan]Loading dataset...[/cyan]")
    if args.data_dir:
        dataset = LeRobotDataset(repo_id="local/dataset", root=args.data_dir)
        dataset_root = Path(args.data_dir)
    else:
        dataset = LeRobotDataset(repo_id=args.repo_id)
        dataset_root = dataset.root
    
    console.print(f"[green]✓ Loaded dataset with {len(dataset)} frames[/green]")
    
    # Load skills metadata
    console.print("[cyan]Loading skills metadata...[/cyan]")
    skills_metadata = load_skills_metadata(dataset_root)
    if skills_metadata is None:
        console.print("[red]Error: No skills.json found. Run annotate.py first![/red]")
        return
    
    console.print(f"[green]✓ Loaded skills for {len(skills_metadata.get('episodes', {}))} episodes[/green]")
    
    # Initialize model
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    console.print(f"[cyan]Initializing {args.model}...[/cyan]")
    pgen = QwenPgen(
        model_name=args.model,
        device=args.device,
        torch_dtype=torch_dtype,
        temperature=args.temperature,
    )
    
    # Get image keys (for image mode)
    if args.image_key:
        image_keys = [args.image_key]
    else:
        image_keys = dataset.meta.camera_keys[:args.num_image_views_per_sample]
    if not args.video_mode:
        console.print(f"[cyan]Using image keys: {image_keys}[/cyan]")
    
    # Determine video key for video mode
    video_key = None
    if args.video_mode:
        if args.video_key:
            # Use explicitly provided video key
            video_key = args.video_key
            if video_key not in dataset.meta.video_keys:
                console.print(f"[red]Error: Video key '{video_key}' not found in dataset.[/red]")
                console.print(f"[yellow]Available video keys: {', '.join(dataset.meta.video_keys)}[/yellow]")
                return
        elif dataset.meta.video_keys:
            # Use first available video key
            video_key = dataset.meta.video_keys[0]
        else:
            console.print("[red]Error: No video keys found in dataset. Cannot use video mode.[/red]")
            return
        console.print(f"[cyan]Using video key for video mode: {video_key}[/cyan]")
    
    # Generate synthetic data
    tasks_df, task_indices, debug_outputs = generate_synthetic_data(
        dataset=dataset,
        pgen=pgen,
        skills_metadata=skills_metadata,
        image_keys=image_keys,
        sample_interval_seconds=args.sample_interval,
        console=console,
        video_mode=args.video_mode,
        video_key=video_key,
        video_batch_size=args.video_batch_size,
    )
    
    # Save high-level tasks
    save_high_level_tasks(tasks_df, dataset_root, console)
    save_debug_outputs(debug_outputs, dataset_root, console)
    
    # Add task_index_high_level feature to dataset
    console.print("[cyan]Adding task_index_high_level feature to dataset...[/cyan]")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        repo_id = f"{dataset.repo_id}_with_high_level_tasks"
    else:
        output_dir = None
        repo_id = f"{dataset.repo_id}_with_high_level_tasks"
    
    # Add feature using dataset_tools
    feature_info = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    new_dataset = add_features(
        dataset=dataset,
        features={
            "task_index_high_level": (task_indices, feature_info),
        },
        output_dir=output_dir,
        repo_id=repo_id,
    )

    # copy high level tsk parquet to new output directory
    import shutil
    shutil.copy(dataset_root / "meta" / "tasks_high_level.parquet", output_dir / "meta" / "tasks_high_level.parquet")
    shutil.copy(dataset_root / "meta" / "syn_annotations.jsonl", output_dir / "meta" / "syn_annotations.jsonl")
    
    console.print(f"[bold green]✓ Successfully added task_index_high_level feature![/bold green]")
    console.print(f"  New dataset saved to: {new_dataset.root}")
    console.print(f"  Total high-level tasks: {len(tasks_df)}")
    
    # Push to hub if requested
    if args.push_to_hub:
        if args.data_dir:
            console.print("[yellow]Warning: --push-to-hub requires --repo-id, skipping...[/yellow]")
        else:
            console.print("[cyan]Pushing to HuggingFace Hub...[/cyan]")
            try:
                new_dataset.push_to_hub()
                console.print(f"[green]✓ Pushed to {repo_id}[/green]")
            except Exception as e:
                console.print(f"[red]Push failed: {e}[/red]")


if __name__ == "__main__":
    main()
