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
SARM-Style Subtask Annotation for LeRobot Datasets (Local GPU)

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

Requirements:
  - GPU with sufficient VRAM (16GB+ recommended for 30B model)
  - transformers, torch, qwen-vl-utils

Task-specific subtasks: Each task has a predefined list of subtasks. The model MUST use these exact names
to ensure consistency.

Usage:
# Install dependencies
pip install transformers torch qwen-vl-utils accelerate

# Annotate and push to hub:
python subtask_annotation_local.py \\
  --repo-id pepijn223/mydataset \\
  --subtasks "reach,grasp,lift,place" \\
  --video-key observation.images.base \\
  --push-to-hub

"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from pydantic import BaseModel, Field
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Pydantic Models for SARM-style Annotation
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


def create_sarm_prompt(subtask_list: list[str]) -> str:
    """
    Create a SARM annotation prompt with a specific subtask list.
    
    The prompt instructs the VLM to identify when each subtask occurs in the video,
    using ONLY the provided subtask names (for consistency across demonstrations).
    """
    subtask_str = "\n".join([f"  - {name}" for name in subtask_list])
    
    return f"""You are an expert video annotator. Analyze this robot manipulation video and identify when each subtask occurs.

WATCH THE ENTIRE VIDEO FIRST:


CRITICAL REQUIREMENTS:
1. You MUST use ONLY these EXACT subtask names (no variations, no other names):
{subtask_str}
2. Identify the start and end timestamp for each subtask that occurs in the video
3. Subtasks should be in chronological order
4. Timestamps should be in MM:SS format (e.g., "00:15" for 15 seconds, "01:30" for 1 minute 30 seconds)
5. Subtasks should cover the entire demonstration without gaps
6. You MUST watch the COMPLETE video from start to finish before making ANY annotations or conclusions
7. Do NOT start annotating until you have seen the entire video length
8. Only after viewing the complete video should you identify the timestamps
9. EACH SUBTASK HAPPENS ONLY ONCE in the video - do not identify the same subtask multiple times
10. Note the exact times when each subtask starts and ends, but make sure to cover the ENTIRE video timeline.

FORMAT:
Return a JSON list of subtasks with their timestamps. Each subtask must have:
- "name": One of the exact names from the list above
- "timestamps": object with "start" and "end" fields (MM:SS format)

Example structure:
{{
  "subtasks": [
    {{"name": "reach_to_object", "timestamps": {{"start": "00:00", "end": "00:05"}}}},
    {{"name": "grasp_object", "timestamps": {{"start": "00:05", "end": "00:08"}}}},
    ...
  ]
}}

Remember: Use ONLY the subtask names provided above, and cover the ENTIRE video timeline."""


class VideoAnnotator:
    """Annotates robot manipulation videos using local Qwen3-VL model on GPU"""

    def __init__(
        self,
        subtask_list: list[str],
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the video annotator with local model.

        Args:
            subtask_list: List of allowed subtask names (for consistency)
            model_name: Hugging Face model name (default: Qwen/Qwen3-VL-30B-A3B-Instruct)
            device: Device to use (cuda, cpu)
            torch_dtype: Data type for model (bfloat16, float16, float32)
        """
        self.subtask_list = subtask_list
        self.prompt = create_sarm_prompt(subtask_list)
        self.console = Console()
        self.device = device
        
        self.console.print(f"[cyan]Loading model: {model_name}...[/cyan]")
        
        # Load model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.console.print(f"[green]✓ Model loaded successfully on {device}[/green]")

    def extract_episode_segment(
        self, 
        file_path: Path, 
        start_timestamp: float,
        end_timestamp: float,
        target_fps: int = 2
    ) -> Path:
        """
        Extract a specific episode segment from concatenated video.
        Uses minimal compression to preserve quality for local inference.
        
        Args:
            file_path: Path to the concatenated video file
            start_timestamp: Starting timestamp in seconds (within this video file)
            end_timestamp: Ending timestamp in seconds (within this video file)
            target_fps: Target FPS (default: 2 for faster processing)
        
        Returns:
            Path to extracted video file
        """
        import os
        import tempfile
        import subprocess
        
        # Create temporary file for extracted video
        tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()
        
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.console.print("[yellow]Warning: ffmpeg not found, cannot extract episode segment[/yellow]")
            return file_path
        
        try:
            # Calculate duration
            duration = end_timestamp - start_timestamp
            
            self.console.print(f"[cyan]Extracting episode: {start_timestamp:.1f}s-{end_timestamp:.1f}s ({duration:.1f}s)[/cyan]")
            
            # Use ffmpeg to extract segment with minimal quality loss
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-ss', str(start_timestamp),  # Start time
                '-t', str(duration),  # Duration
                '-r', str(target_fps),  # Output FPS
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'ultrafast',  # Faster encoding
                '-crf', '23',  # Better quality (lower = better)
                '-an',  # Remove audio
                '-y',  # Overwrite output file
                str(tmp_path)
            ]
            
            # Run ffmpeg (suppress output)
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Verify the output file was created and is not empty
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                self.console.print("[red]✗ Video extraction failed (0 bytes) - skipping episode[/red]")
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError("FFmpeg produced empty video file")
            
            # Show extraction results
            file_size_mb = tmp_path.stat().st_size / (1024 * 1024)
            
            # Fail if file is too small (< 100KB likely means extraction failed)
            if file_size_mb < 0.1:
                self.console.print(f"[red]✗ Extracted video too small ({file_size_mb:.2f}MB) - skipping episode[/red]")
                tmp_path.unlink()
                raise RuntimeError(f"Video extraction produced invalid file ({file_size_mb:.2f}MB)")
            
            self.console.print(f"[green]✓ Extracted: {file_size_mb:.1f}MB ({target_fps} FPS)[/green]")
            
            return tmp_path
            
        except subprocess.CalledProcessError as e:
            self.console.print(f"[yellow]Warning: ffmpeg failed ({e})[/yellow]")
            if tmp_path.exists():
                tmp_path.unlink()
            return file_path

    def annotate(
        self, 
        file_path: str | Path, 
        fps: int, 
        start_timestamp: float = 0.0,
        end_timestamp: float | None = None,
        max_retries: int = 3
    ) -> SubtaskAnnotation:
        """
        Annotate a video file or episode segment using local GPU.

        Args:
            file_path: Path to the video file (may contain multiple concatenated episodes)
            fps: Frames per second of the video
            start_timestamp: Starting timestamp in seconds (within this video file)
            end_timestamp: Ending timestamp in seconds (within this video file)
            max_retries: Number of retries if annotation fails

        Returns:
            SubtaskAnnotation object with the results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # Calculate episode duration
        if end_timestamp is None:
            # Get video metadata (suppress AV1 warnings)
            import cv2
            import os
            import sys
            
            stderr_backup = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            
            try:
                cap = cv2.VideoCapture(str(file_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                end_timestamp = total_frames / video_fps if video_fps > 0 else 0
                cap.release()
            finally:
                sys.stderr.close()
                sys.stderr = stderr_backup
        
        duration_seconds = end_timestamp - start_timestamp
        
        duration_mins = int(duration_seconds // 60)
        duration_secs = int(duration_seconds % 60)
        duration_str = f"{duration_mins:02d}:{duration_secs:02d}"

        self.console.print(f"[cyan]Processing episode from concatenated video: {file_path.name}[/cyan]")
        self.console.print(f"[cyan]Episode timestamps: {start_timestamp:.1f}s-{end_timestamp:.1f}s ({duration_seconds:.1f}s)[/cyan]")
        self.console.print(f"[cyan]Episode duration: {duration_str}[/cyan]")

        # Extract episode segment
        extracted_path = self.extract_episode_segment(
            file_path, 
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            target_fps=2  # 2 FPS is good balance for VLM
        )
        is_extracted = extracted_path != file_path
        
        try:
            # Add video duration to prompt
            prompt_with_duration = f"""{self.prompt}

CRITICAL - VIDEO DURATION:
The video is {duration_str} long ({duration_seconds:.1f} seconds). Your annotations MUST cover the ENTIRE duration from 00:00 to {duration_str}.
Do NOT stop annotating before the video ends. Make sure your last subtask ends at {duration_str} or very close to it."""

            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(extracted_path),
                            "fps": 1.0,  # Sample at 1 FPS for analysis
                        },
                        {"type": "text", "text": prompt_with_duration},
                    ],
                }
            ]

            # Generate annotation with retries
            for attempt in range(max_retries):
                try:
                    self.console.print(f"[cyan]Generating annotation (attempt {attempt + 1}/{max_retries})...[/cyan]")
                    
                    # Prepare inputs
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
                    )
                    inputs = inputs.to(self.device)
                    
                    # Generate
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            temperature=0.1,  # Low temperature for consistent output
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    response_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    self.console.print(f"[dim]Raw response: {response_text[:200]}...[/dim]")
                    
                    # Try to extract JSON from response
                    # Sometimes models wrap JSON in markdown code blocks
                    response_text = response_text.strip()
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    # Parse response
                    import json
                    try:
                        response_dict = json.loads(response_text)
                        annotation = SubtaskAnnotation.model_validate(response_dict)
                    except json.JSONDecodeError:
                        # Try to find JSON object in response
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            response_dict = json.loads(json_match.group())
                            annotation = SubtaskAnnotation.model_validate(response_dict)
                        else:
                            raise ValueError("Could not parse JSON from model response")

                    self.console.print("[green]✓ Annotation completed successfully[/green]")
                    return annotation

                except Exception as e:
                    self.console.print(f"[yellow]⚠ Attempt {attempt + 1} failed: {e}[/yellow]")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to annotate after {max_retries} attempts") from e
                    time.sleep(1)
        
        finally:
            # Clean up temporary extracted file
            if is_extracted and extracted_path.exists():
                extracted_path.unlink()


def display_annotation(annotation: SubtaskAnnotation, console: Console, episode_idx: int, fps: int):
    """Display annotation in a nice tree format with frame indices"""
    tree = Tree(f"[bold]Episode {episode_idx} - Subtask Annotation[/bold]")

    # Subtasks
    subtasks_branch = tree.add(f"[bold cyan]Subtasks ({len(annotation.subtasks)} total)[/bold cyan]")
    for i, subtask in enumerate(annotation.subtasks, 1):
        # Calculate frame indices for display
        start_sec = timestamp_to_seconds(subtask.timestamps.start)
        end_sec = timestamp_to_seconds(subtask.timestamps.end)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        subtasks_branch.add(
            f"{i}. [cyan]{subtask.name}[/cyan]: "
            f"{subtask.timestamps.start} → {subtask.timestamps.end} "
            f"[dim](frames {start_frame}-{end_frame})[/dim]"
        )

    console.print(tree)


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or SS timestamp to seconds"""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        return int(parts[0])


def save_annotations_to_dataset(
    dataset_path: Path,
    annotations: dict[int, SubtaskAnnotation],
    fps: int,
):
    """
    Save annotations to LeRobot dataset parquet format.
    
    For each episode, stores subtask annotations with:
    - subtask_names: list of subtask names
    - subtask_start_times: list of start times (seconds)
    - subtask_end_times: list of end times (seconds)
    - subtask_start_frames: list of start frames
    - subtask_end_frames: list of end frames
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes
    
    console = Console()
    
    # Load existing episodes metadata (returns datasets.Dataset)
    episodes_dataset = load_episodes(dataset_path)
    
    if episodes_dataset is None or len(episodes_dataset) == 0:
        console.print("[red]Error: No episodes found in dataset[/red]")
        return
    
    # Convert to pandas DataFrame for easier manipulation
    episodes_df = episodes_dataset.to_pandas()
    
    # Add subtask columns to episodes dataframe
    episodes_df["subtask_names"] = None
    episodes_df["subtask_start_times"] = None
    episodes_df["subtask_end_times"] = None
    episodes_df["subtask_start_frames"] = None
    episodes_df["subtask_end_frames"] = None
    
    # Fill in annotations
    for ep_idx, annotation in annotations.items():
        if ep_idx >= len(episodes_df):
            console.print(f"[yellow]Warning: Episode {ep_idx} not found in dataset[/yellow]")
            continue
        
        subtask_names = []
        start_times = []
        end_times = []
        start_frames = []
        end_frames = []
        
        for subtask in annotation.subtasks:
            subtask_names.append(subtask.name)
            
            # Convert timestamps to seconds
            start_sec = timestamp_to_seconds(subtask.timestamps.start)
            end_sec = timestamp_to_seconds(subtask.timestamps.end)
            start_times.append(start_sec)
            end_times.append(end_sec)
            
            # Calculate frame indices from timestamps and FPS
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            start_frames.append(start_frame)
            end_frames.append(end_frame)
        
        # Store as lists in the dataframe
        episodes_df.at[ep_idx, "subtask_names"] = subtask_names
        episodes_df.at[ep_idx, "subtask_start_times"] = start_times
        episodes_df.at[ep_idx, "subtask_end_times"] = end_times
        episodes_df.at[ep_idx, "subtask_start_frames"] = start_frames
        episodes_df.at[ep_idx, "subtask_end_frames"] = end_frames
    
    # Group episodes by their chunk and file indices
    episodes_by_file = {}
    for ep_idx in episodes_df.index:
        chunk_idx = episodes_df.loc[ep_idx, "meta/episodes/chunk_index"]
        file_idx = episodes_df.loc[ep_idx, "meta/episodes/file_index"]
        key = (chunk_idx, file_idx)
        
        if key not in episodes_by_file:
            episodes_by_file[key] = []
        episodes_by_file[key].append(ep_idx)
    
    # Write back to parquet files
    for (chunk_idx, file_idx), ep_indices in episodes_by_file.items():
        episodes_path = dataset_path / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        
        if not episodes_path.exists():
            console.print(f"[yellow]Warning: Episodes file not found: {episodes_path}[/yellow]")
            continue
        
        # Read the existing parquet file
        file_df = pd.read_parquet(episodes_path)
        
        # Add subtask columns if they don't exist
        for col in ["subtask_names", "subtask_start_times", "subtask_end_times", 
                    "subtask_start_frames", "subtask_end_frames"]:
            if col not in file_df.columns:
                file_df[col] = None
        
        # Update rows that have annotations
        for ep_idx in ep_indices:
            if ep_idx in file_df.index and ep_idx in annotations:
                file_df.at[ep_idx, "subtask_names"] = episodes_df.loc[ep_idx, "subtask_names"]
                file_df.at[ep_idx, "subtask_start_times"] = episodes_df.loc[ep_idx, "subtask_start_times"]
                file_df.at[ep_idx, "subtask_end_times"] = episodes_df.loc[ep_idx, "subtask_end_times"]
                file_df.at[ep_idx, "subtask_start_frames"] = episodes_df.loc[ep_idx, "subtask_start_frames"]
                file_df.at[ep_idx, "subtask_end_frames"] = episodes_df.loc[ep_idx, "subtask_end_frames"]
        
        # Write back to parquet
        file_df.to_parquet(episodes_path, engine="pyarrow", compression="snappy")
        console.print(f"[green]✓ Updated {episodes_path.name} with {len([e for e in ep_indices if e in annotations])} annotations[/green]")
    
    console.print(f"[bold green]✓ Saved {len(annotations)} episode annotations to parquet files[/bold green]")


def load_annotations_from_dataset(dataset_path: Path) -> dict[int, SubtaskAnnotation]:
    """
    Load annotations from LeRobot dataset parquet files.
    
    Reads subtask annotations from the episodes metadata parquet files.
    """
    from lerobot.datasets.utils import load_episodes
    
    episodes_dataset = load_episodes(dataset_path)
    
    if episodes_dataset is None or len(episodes_dataset) == 0:
        return {}
    
    # Check if subtask columns exist
    if "subtask_names" not in episodes_dataset.column_names:
        return {}
    
    # Convert to pandas DataFrame for easier access
    episodes_df = episodes_dataset.to_pandas()
    
    annotations = {}
    
    for ep_idx in episodes_df.index:
        subtask_names = episodes_df.loc[ep_idx, "subtask_names"]
        
        # Skip episodes without annotations
        if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
            continue
        
        start_times = episodes_df.loc[ep_idx, "subtask_start_times"]
        end_times = episodes_df.loc[ep_idx, "subtask_end_times"]
        
        # Reconstruct SubtaskAnnotation from stored data
        subtasks = []
        for i, name in enumerate(subtask_names):
            # Convert seconds back to MM:SS format
            start_sec = int(start_times[i])
            end_sec = int(end_times[i])
            start_str = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
            end_str = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
            
            subtasks.append(
                Subtask(
                    name=name,
                    timestamps=Timestamp(start=start_str, end=end_str)
                )
            )
        
        annotations[int(ep_idx)] = SubtaskAnnotation(subtasks=subtasks)
    
    return annotations


def process_single_episode(
    ep_idx: int,
    dataset_root: Path,
    dataset_meta,
    video_key: str,
    fps: int,
    annotator: VideoAnnotator,
    console: Console,
) -> tuple[int, SubtaskAnnotation | None, str | None]:
    """
    Process a single episode annotation.
    
    Args:
        ep_idx: Episode index
        dataset_root: Dataset root path
        dataset_meta: Dataset metadata
        video_key: Video key to use
        fps: FPS of the video
        annotator: VideoAnnotator instance
        console: Console for output
    
    Returns:
        Tuple of (episode_index, annotation or None, error message or None)
    """
    try:
        # Get video path
        video_path = dataset_root / dataset_meta.get_video_file_path(ep_idx, video_key)

        if not video_path.exists():
            return ep_idx, None, f"Video not found: {video_path}"
        
        # Get video-specific timestamps (NOT global frame indices)
        video_path_key = f"videos/{video_key}/from_timestamp"
        video_path_key_to = f"videos/{video_key}/to_timestamp"
        
        start_timestamp = float(dataset_meta.episodes[video_path_key][ep_idx])
        end_timestamp = float(dataset_meta.episodes[video_path_key_to][ep_idx])
        
        # Annotate with video-specific timestamps
        annotation = annotator.annotate(
            video_path, 
            fps,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
        
        return ep_idx, annotation, None

    except Exception as e:
        return ep_idx, None, str(e)


def compute_temporal_proportions(annotations: dict[int, SubtaskAnnotation], fps: int = 30) -> dict[str, float]:
    """
    Compute average temporal proportion for each subtask across all episodes.
    This is the key insight from SARM - use semantic subtasks instead of frame indices.
    """
    # Collect all proportions per subtask
    subtask_proportions = {}
    
    for annotation in annotations.values():
        # Calculate total episode duration
        total_duration = 0
        durations = {}
        
        for subtask in annotation.subtasks:
            # Parse timestamps
            start_parts = subtask.timestamps.start.split(":")
            end_parts = subtask.timestamps.end.split(":")
            
            if len(start_parts) == 2:
                start_seconds = int(start_parts[0]) * 60 + int(start_parts[1])
            else:
                start_seconds = int(start_parts[0])
                
            if len(end_parts) == 2:
                end_seconds = int(end_parts[0]) * 60 + int(end_parts[1])
            else:
                end_seconds = int(end_parts[0])
            
            duration = end_seconds - start_seconds
            durations[subtask.name] = duration
            total_duration += duration
        
        # Calculate proportions for this episode
        if total_duration > 0:
            for name, duration in durations.items():
                if name not in subtask_proportions:
                    subtask_proportions[name] = []
                subtask_proportions[name].append(duration / total_duration)
    
    # Average across episodes
    avg_proportions = {
        name: sum(props) / len(props)
        for name, props in subtask_proportions.items()
    }
    
    # Normalize to sum to 1.0
    total = sum(avg_proportions.values())
    if total > 0:
        avg_proportions = {name: prop / total for name, prop in avg_proportions.items()}
    
    return avg_proportions


def main():
    parser = argparse.ArgumentParser(
        description="SARM-style subtask annotation using local GPU (Qwen3-VL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available cameras:
  python subtask_annotation_local.py --repo-id pepijn223/mydataset --subtasks "reach,grasp" --max-episodes 0
  
  # Annotate with specific camera:
  python subtask_annotation_local.py --repo-id pepijn223/mydataset --subtasks "reach,grasp" --video-key observation.images.top --push-to-hub
  
  # Use smaller model (7B instead of 30B):
  python subtask_annotation_local.py --repo-id pepijn223/mydataset --subtasks "reach,grasp" --video-key observation.images.top --model Qwen/Qwen2-VL-7B-Instruct --push-to-hub

Note: The 7B model requires ~16GB VRAM. Use 2B model (~8GB VRAM) if needed.
"""
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g., 'pepijn223/mydataset')",
    )
    parser.add_argument(
        "--subtasks",
        type=str,
        required=True,
        help="Comma-separated list of subtask names (e.g., 'reach,grasp,lift,place')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episode indices to annotate (default: all episodes)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to annotate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Model to use (default: Qwen/Qwen3-VL-30B-A3B-Instruct). Other options: Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2-VL-7B-Instruct",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes that already have annotations",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Camera/video key to use for annotation (e.g., 'observation.images.top'). "
             "If not specified, uses the first available video key.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push annotated dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output repository ID for push (default: same as --repo-id)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )

    args = parser.parse_args()

    # Parse subtask list
    subtask_list = [s.strip() for s in args.subtasks.split(",")]
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]SARM Subtask Annotation (Local GPU)[/bold cyan]\n"
        f"Dataset: {args.repo_id}\n"
        f"Model: {args.model}\n"
        f"Device: {args.device}\n"
        f"Subtasks: {', '.join(subtask_list)}",
        border_style="cyan"
    ))

    # Load dataset
    console.print(f"\n[cyan]Loading dataset: {args.repo_id}[/cyan]")
    dataset = LeRobotDataset(args.repo_id, download_videos=True)

    # Get FPS from dataset
    fps = dataset.fps
    console.print(f"[cyan]Dataset FPS: {fps}[/cyan]")

    # Display available cameras/video keys
    if len(dataset.meta.video_keys) == 0:
        console.print("[red]Error: No video keys found in dataset[/red]")
        return
    
    console.print(f"\n[cyan]Available cameras/video keys:[/cyan]")
    for i, vk in enumerate(dataset.meta.video_keys, 1):
        console.print(f"  {i}. {vk}")
    
    # Get video key
    if args.video_key:
        if args.video_key not in dataset.meta.video_keys:
            console.print(f"[red]Error: Video key '{args.video_key}' not found in dataset[/red]")
            console.print(f"[yellow]Available keys: {', '.join(dataset.meta.video_keys)}[/yellow]")
            return
        video_key = args.video_key
        console.print(f"\n[green]Using specified camera: {video_key}[/green]")
    else:
        video_key = dataset.meta.video_keys[0]
        console.print(f"\n[yellow]No camera specified, using first available: {video_key}[/yellow]")
        if len(dataset.meta.video_keys) > 1:
            console.print(f"[yellow]Tip: Use --video-key to specify a different camera[/yellow]")

    # Determine episodes to annotate
    if args.episodes:
        episode_indices = args.episodes
    else:
        episode_indices = list(range(dataset.meta.total_episodes))
        if args.max_episodes:
            episode_indices = episode_indices[: args.max_episodes]

    console.print(f"[cyan]Will annotate {len(episode_indices)} episodes[/cyan]")

    # Load existing annotations
    existing_annotations = load_annotations_from_dataset(dataset.root)

    if args.skip_existing and existing_annotations:
        console.print(f"[yellow]Found {len(existing_annotations)} existing annotations[/yellow]")
        episode_indices = [ep for ep in episode_indices if ep not in existing_annotations]
        console.print(f"[cyan]Will annotate {len(episode_indices)} new episodes[/cyan]")

    if not episode_indices:
        console.print("[green]All episodes already annotated![/green]")
        return

    # Initialize annotator with subtask list
    annotator = VideoAnnotator(
        subtask_list=subtask_list,
        model_name=args.model,
        device=args.device,
        torch_dtype=torch_dtype
    )

    # Annotate episodes (sequential processing)
    annotations = existing_annotations.copy()
    
    for i, ep_idx in enumerate(episode_indices):
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]Episode {ep_idx} ({i + 1}/{len(episode_indices)})[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        result_ep_idx, annotation, error = process_single_episode(
            ep_idx,
            dataset.root,
            dataset.meta,
            video_key,
            fps,
            annotator,
            console
        )
        
        if error:
            console.print(f"[red]✗ Failed to annotate episode {result_ep_idx}: {error}[/red]")
            continue
        elif annotation:
            annotations[result_ep_idx] = annotation
            display_annotation(annotation, console, result_ep_idx, fps)
            save_annotations_to_dataset(dataset.root, annotations, fps)

    # Compute temporal proportions (key SARM insight)
    console.print(f"\n[bold cyan]Computing Temporal Proportions[/bold cyan]")
    temporal_proportions = compute_temporal_proportions(annotations, fps)
    
    # Save temporal proportions
    proportions_path = dataset.root / "meta" / "temporal_proportions.json"
    proportions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(proportions_path, "w") as f:
        json.dump(temporal_proportions, f, indent=2)
    
    console.print(f"[green]✓ Saved temporal proportions to {proportions_path}[/green]")
    console.print("\n[cyan]Average temporal proportions:[/cyan]")
    for name, proportion in sorted(temporal_proportions.items(), key=lambda x: -x[1]):
        console.print(f"  {name}: {proportion:.1%}")

    # Create summary
    console.print(f"\n[bold green]{'=' * 60}[/bold green]")
    console.print(f"[bold green]Annotation Complete![/bold green]")
    console.print(f"[bold green]{'=' * 60}[/bold green]")
    console.print(f"Total episodes annotated: {len(annotations)}")
    console.print(f"Total subtasks found: {sum(len(ann.subtasks) for ann in annotations.values())}")

    # Push to hub if requested
    if args.push_to_hub:
        output_repo = args.output_repo_id if args.output_repo_id else args.repo_id
        console.print(f"\n[bold cyan]Pushing to HuggingFace Hub: {output_repo}[/bold cyan]")
        
        try:
            dataset.push_to_hub(push_videos=True)
            console.print(f"[bold green]✓ Successfully pushed to {output_repo}[/bold green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to push to hub: {e}[/red]")
            console.print("[yellow]Annotations are still saved locally[/yellow]")


if __name__ == "__main__":
    main()

