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

Supports two annotation modes:
  - Single mode (sparse only): Use --sparse-subtasks for high-level stages
  - Dual mode (sparse + dense): Use both --sparse-subtasks and --dense-subtasks

Requirements:
  - GPU with sufficient VRAM (16GB+ recommended for 30B model)
  - transformers, torch, qwen-vl-utils

Task-specific subtasks: Each task has a predefined list of subtasks. The model MUST use these exact names
to ensure consistency.

Usage:
# Install dependencies
pip install transformers torch qwen-vl-utils accelerate

# Single mode (sparse annotations only):
python subtask_annotation.py \\
  --repo-id pepijn223/mydataset \\
  --sparse-subtasks "fold1,fold2,fold3" \\
  --video-key observation.images.base \\
  --push-to-hub

# Dual mode (both sparse and dense annotations):
python subtask_annotation.py \\
  --repo-id pepijn223/mydataset \\
  --sparse-subtasks "fold1,fold2,fold3" \\
  --dense-subtasks "grab,flatten,fold_left,fold_right,rotate,fold_bottom,place" \\
  --video-key observation.images.base \\
  --push-to-hub

# Parallel processing with 4 GPUs:
python subtask_annotation.py \\
  --repo-id pepijn223/mydataset \\
  --sparse-subtasks "fold1,fold2,fold3" \\
  --video-key observation.images.base \\
  --num-workers 4 \\
  --push-to-hub

"""

import argparse
import json
import textwrap
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sarm.sarm_utils import compute_temporal_proportions
from lerobot.policies.sarm.sarm_utils import SubtaskAnnotation, Subtask, Timestamp

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
        model: "Qwen3VLMoeForConditionalGeneration | None" = None,
        processor: "AutoProcessor | None" = None,
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
        self.console = Console()
        self.device = device
        
        # Use provided model/processor or load new ones
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            self.console.print(f"[green]✓ Using shared model on {device}[/green]")
        else:
            self.console.print(f"[cyan]Loading model: {model_name}...[/cyan]")
            
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
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
        target_fps: int = 1
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
                '-ss', str(start_timestamp),  
                '-t', str(duration), 
                '-r', str(target_fps),  
                '-c:v', 'libx264',  
                '-preset', 'ultrafast',  
                '-crf', '23',  
                '-an',  
                '-y',  
                str(tmp_path)
            ]
            
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
            target_fps=1
        )
        is_extracted = extracted_path != file_path
        
        try:
            # Add video duration to prompt
            prompt_with_duration = f"""{self.prompt}

# Video Duration:
The video is {duration_str} long ({duration_seconds:.1f} seconds). Your total annotations MUST cover the ENTIRE duration from 00:00 to {duration_str}.
Do NOT stop annotating before the video ends. Make sure your last subtask ends at {duration_str}."""

            # Prepare messages for the model
            messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(extracted_path),
                        "fps": 1.0,
                    },
                    {
                        "type": "text",
                        "text": (
                            f"The video is {duration_str} long (~{duration_seconds:.1f} seconds). "
                            f"Follow the system instructions: first write the textual timeline, "
                            f"then output the JSON as specified."
                        ),
                    },
                ],
            },
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
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    response_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    self.console.print(f"[dim]Raw response: {response_text[:500]}...[/dim]")
                    
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


def display_annotation(annotation: SubtaskAnnotation, console: Console, episode_idx: int, fps: int, prefix: str = ""):
    """Display annotation in a nice tree format with frame indices"""
    title = f"[bold]Episode {episode_idx} - {prefix + ' ' if prefix else ''}Subtask Annotation[/bold]"
    tree = Tree(title)

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
    prefix: str = "sparse",
):
    """
    Save annotations to LeRobot dataset parquet format.
    
    For each episode, stores subtask annotations with columns prefixed by sparse_ or dense_:
    - {prefix}_subtask_names: list of subtask names
    - {prefix}_subtask_start_times: list of start times (seconds)
    - {prefix}_subtask_end_times: list of end times (seconds)
    - {prefix}_subtask_start_frames: list of start frames
    - {prefix}_subtask_end_frames: list of end frames
    
    Args:
        dataset_path: Path to the dataset root
        annotations: Dict mapping episode index to SubtaskAnnotation
        fps: Frames per second
        prefix: Column prefix ("sparse" or "dense")
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes
    
    console = Console()
    
    # Define column names with prefix
    col_names = f"{prefix}_subtask_names"
    col_start_times = f"{prefix}_subtask_start_times"
    col_end_times = f"{prefix}_subtask_end_times"
    col_start_frames = f"{prefix}_subtask_start_frames"
    col_end_frames = f"{prefix}_subtask_end_frames"
    
    # Also keep legacy column names for backward compatibility (sparse only)
    legacy_columns = prefix == "sparse"
    
    # Load existing episodes metadata (returns datasets.Dataset)
    episodes_dataset = load_episodes(dataset_path)
    
    if episodes_dataset is None or len(episodes_dataset) == 0:
        console.print("[red]Error: No episodes found in dataset[/red]")
        return
    
    # Convert to pandas DataFrame for easier manipulation
    episodes_df = episodes_dataset.to_pandas()
    
    # Add subtask columns to episodes dataframe
    episodes_df[col_names] = None
    episodes_df[col_start_times] = None
    episodes_df[col_end_times] = None
    episodes_df[col_start_frames] = None
    episodes_df[col_end_frames] = None
    
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
        episodes_df.at[ep_idx, col_names] = subtask_names
        episodes_df.at[ep_idx, col_start_times] = start_times
        episodes_df.at[ep_idx, col_end_times] = end_times
        episodes_df.at[ep_idx, col_start_frames] = start_frames
        episodes_df.at[ep_idx, col_end_frames] = end_frames
    
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
        all_cols = [col_names, col_start_times, col_end_times, col_start_frames, col_end_frames]
        # Also add legacy columns for sparse mode
        if legacy_columns:
            all_cols += ["subtask_names", "subtask_start_times", "subtask_end_times", 
                        "subtask_start_frames", "subtask_end_frames"]
        
        for col in all_cols:
            if col not in file_df.columns:
                file_df[col] = None
        
        # Update rows that have annotations
        for ep_idx in ep_indices:
            if ep_idx in file_df.index and ep_idx in annotations:
                file_df.at[ep_idx, col_names] = episodes_df.loc[ep_idx, col_names]
                file_df.at[ep_idx, col_start_times] = episodes_df.loc[ep_idx, col_start_times]
                file_df.at[ep_idx, col_end_times] = episodes_df.loc[ep_idx, col_end_times]
                file_df.at[ep_idx, col_start_frames] = episodes_df.loc[ep_idx, col_start_frames]
                file_df.at[ep_idx, col_end_frames] = episodes_df.loc[ep_idx, col_end_frames]
                
                # Also update legacy columns for sparse mode (backward compatibility)
                if legacy_columns:
                    file_df.at[ep_idx, "subtask_names"] = episodes_df.loc[ep_idx, col_names]
                    file_df.at[ep_idx, "subtask_start_times"] = episodes_df.loc[ep_idx, col_start_times]
                    file_df.at[ep_idx, "subtask_end_times"] = episodes_df.loc[ep_idx, col_end_times]
                    file_df.at[ep_idx, "subtask_start_frames"] = episodes_df.loc[ep_idx, col_start_frames]
                    file_df.at[ep_idx, "subtask_end_frames"] = episodes_df.loc[ep_idx, col_end_frames]
        
        # Write back to parquet
        file_df.to_parquet(episodes_path, engine="pyarrow", compression="snappy")
        console.print(f"[green]✓ Updated {episodes_path.name} with {len([e for e in ep_indices if e in annotations])} {prefix} annotations[/green]")
    
    console.print(f"[bold green]✓ Saved {len(annotations)} {prefix} episode annotations to parquet files[/bold green]")


def load_annotations_from_dataset(dataset_path: Path, prefix: str = "sparse") -> dict[int, SubtaskAnnotation]:
    """
    Load annotations from LeRobot dataset parquet files.
    
    Reads subtask annotations from the episodes metadata parquet files.
    
    Args:
        dataset_path: Path to the dataset root
        prefix: Column prefix to load ("sparse" or "dense")
        
    Returns:
        Dict mapping episode index to SubtaskAnnotation
    """
    from lerobot.datasets.utils import load_episodes
    
    episodes_dataset = load_episodes(dataset_path)
    
    if episodes_dataset is None or len(episodes_dataset) == 0:
        return {}
    
    # Define column names with prefix
    col_names = f"{prefix}_subtask_names"
    col_start_times = f"{prefix}_subtask_start_times"
    col_end_times = f"{prefix}_subtask_end_times"
    
    # Check if prefixed columns exist, fall back to legacy columns for sparse
    if col_names not in episodes_dataset.column_names:
        if prefix == "sparse" and "subtask_names" in episodes_dataset.column_names:
            # Fall back to legacy column names
            col_names = "subtask_names"
            col_start_times = "subtask_start_times"
            col_end_times = "subtask_end_times"
        else:
            return {}
    
    # Convert to pandas DataFrame for easier access
    episodes_df = episodes_dataset.to_pandas()
    
    annotations = {}
    
    for ep_idx in episodes_df.index:
        subtask_names = episodes_df.loc[ep_idx, col_names]
        
        # Skip episodes without annotations
        if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
            continue
        
        start_times = episodes_df.loc[ep_idx, col_start_times]
        end_times = episodes_df.loc[ep_idx, col_end_times]
        
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
) -> tuple[dict[int, SubtaskAnnotation], dict[int, SubtaskAnnotation] | None]:
    """
    Worker function for parallel processing across GPUs.
    
    Args:
        worker_id: Worker ID for logging
        gpu_id: GPU device ID to use
        episode_indices: List of episode indices to process
        repo_id: Dataset repo ID
        video_key: Video key to use
        sparse_subtask_list: List of sparse (high-level) subtask names
        dense_subtask_list: List of dense (fine-grained) subtask names (None for single mode)
        model_name: Model name to load
        torch_dtype: Model dtype
    
    Returns:
        Tuple of (sparse_annotations, dense_annotations)
        dense_annotations is None if dense_subtask_list is None
    """
    device = f"cuda:{gpu_id}"
    dual_mode = dense_subtask_list is not None
    
    console = Console()
    mode_str = "dual" if dual_mode else "sparse-only"
    console.print(f"[cyan]Worker {worker_id} starting on GPU {gpu_id} ({mode_str}) with {len(episode_indices)} episodes[/cyan]")
    
    dataset = LeRobotDataset(repo_id, download_videos=False)
    fps = dataset.fps
    
    # Initialize sparse annotator (loads model)
    sparse_annotator = VideoAnnotator(
        subtask_list=sparse_subtask_list,
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype
    )
    
    # Initialize dense annotator if dual mode (reuses the same model)
    dense_annotator = None
    if dual_mode:
        dense_annotator = VideoAnnotator(
            subtask_list=dense_subtask_list,
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            model=sparse_annotator.model,  # Share the model
            processor=sparse_annotator.processor,  # Share the processor
        )
    
    sparse_annotations = {}
    dense_annotations = {} if dual_mode else None
    
    for i, ep_idx in enumerate(episode_indices):
        console.print(f"[cyan]Worker {worker_id} | Episode {ep_idx} ({i+1}/{len(episode_indices)})[/cyan]")
        
        # Sparse annotation
        result_ep_idx, sparse_ann, error = process_single_episode(
            ep_idx, dataset.root, dataset.meta, video_key, fps, sparse_annotator, console
        )
        
        if error:
            console.print(f"[red]Worker {worker_id} | ✗ Sparse annotation failed for episode {result_ep_idx}: {error}[/red]")
        elif sparse_ann:
            sparse_annotations[result_ep_idx] = sparse_ann
            console.print(f"[green]Worker {worker_id} | ✓ Sparse annotation completed for episode {result_ep_idx}[/green]")
        
        # Dense annotation (if dual mode)
        if dual_mode and dense_annotator:
            _, dense_ann, dense_error = process_single_episode(
                ep_idx, dataset.root, dataset.meta, video_key, fps, dense_annotator, console
            )
            
            if dense_error:
                console.print(f"[red]Worker {worker_id} | ✗ Dense annotation failed for episode {ep_idx}: {dense_error}[/red]")
            elif dense_ann:
                dense_annotations[ep_idx] = dense_ann
                console.print(f"[green]Worker {worker_id} | ✓ Dense annotation completed for episode {ep_idx}[/green]")
    
    console.print(f"[bold green]Worker {worker_id} completed {len(sparse_annotations)} sparse annotations[/bold green]")
    if dual_mode:
        console.print(f"[bold green]Worker {worker_id} completed {len(dense_annotations)} dense annotations[/bold green]")
    
    return sparse_annotations, dense_annotations


def main():
    parser = argparse.ArgumentParser(
        description="SARM-style subtask annotation using local GPU (Qwen3-VL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single mode (sparse annotations only):
  python subtask_annotation.py \\
    --repo-id pepijn223/mydataset \\
    --sparse-subtasks "fold1,fold2,fold3" \\
    --video-key observation.images.top \\
    --push-to-hub
  
  # Dual mode (sparse + dense annotations):
  python subtask_annotation.py \\
    --repo-id pepijn223/mydataset \\
    --sparse-subtasks "fold1,fold2,fold3" \\
    --dense-subtasks "grab,flatten,fold_left,fold_right,rotate,fold_bottom,place" \\
    --video-key observation.images.top \\
    --push-to-hub
  
  # Parallel processing with 4 GPUs (4x speedup):
  python subtask_annotation.py \\
    --repo-id pepijn223/mydataset \\
    --sparse-subtasks "fold1,fold2,fold3" \\
    --video-key observation.images.top \\
    --num-workers 4 \\
    --push-to-hub

Performance remarks:
  - Each worker loads one model instance on its assigned GPU
  - The 30B model requires ~60GB VRAM per GPU
  - Use --num-workers N for N GPUs
"""
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g., 'pepijn223/mydataset')",
    )
    parser.add_argument(
        "--sparse-subtasks",
        type=str,
        required=True,
        help="Comma-separated list of sparse (high-level) subtask names (e.g., 'fold1,fold2,fold3')",
    )
    parser.add_argument(
        "--dense-subtasks",
        type=str,
        default=None,
        help="Comma-separated list of dense (fine-grained) subtask names for dual mode "
             "(e.g., 'grab,flatten,fold_left,fold_right'). If not provided, only sparse annotations are used.",
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
        help="Qwen3-VL model to use (default: Qwen/Qwen3-VL-30B-A3B-Instruct)",
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for multi-GPU processing (default: 1 for sequential). "
             "Set to number of GPUs available for parallel processing.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific GPU IDs to use (e.g., --gpu-ids 0 1 2). "
             "If not specified, uses GPUs 0 to num-workers-1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple episodes per inference (experimental, default: 1)",
    )

    args = parser.parse_args()

    sparse_subtask_list = [s.strip() for s in args.sparse_subtasks.split(",")]
    dense_subtask_list = [s.strip() for s in args.dense_subtasks.split(",")] if args.dense_subtasks else None
    dual_mode = dense_subtask_list is not None
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    console = Console()
    mode_str = "Dual (sparse + dense)" if dual_mode else "Single (sparse only)"
    panel_content = (
        "[bold cyan]SARM Subtask Annotation (Local GPU)[/bold cyan]\n"
        f"Dataset: {args.repo_id}\n"
        f"Model: {args.model}\n"
        f"Device: {args.device}\n"
        f"Mode: {mode_str}\n"
        f"Sparse subtasks: {', '.join(sparse_subtask_list)}"
    )
    if dual_mode:
        panel_content += f"\nDense subtasks: {', '.join(dense_subtask_list)}"
    
    console.print(Panel.fit(panel_content, border_style="cyan"))

    console.print(f"\n[cyan]Loading dataset: {args.repo_id}[/cyan]")
    dataset = LeRobotDataset(args.repo_id, download_videos=True)

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

    # Load existing sparse annotations
    existing_annotations = load_annotations_from_dataset(dataset.root, prefix="sparse")

    if args.skip_existing and existing_annotations:
        console.print(f"[yellow]Found {len(existing_annotations)} existing annotations[/yellow]")
        episode_indices = [ep for ep in episode_indices if ep not in existing_annotations]
        console.print(f"[cyan]Will annotate {len(episode_indices)} new episodes[/cyan]")

    if not episode_indices:
        console.print("[green]All episodes already annotated![/green]")
        return

    # Determine GPU IDs to use
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
        if len(gpu_ids) < args.num_workers:
            console.print(f"[yellow]Warning: {args.num_workers} workers requested but only {len(gpu_ids)} GPU IDs provided[/yellow]")
            args.num_workers = len(gpu_ids)
    else:
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if args.num_workers > num_gpus:
                console.print(f"[yellow]Warning: {args.num_workers} workers requested but only {num_gpus} GPUs available[/yellow]")
                args.num_workers = min(args.num_workers, num_gpus)
            gpu_ids = list(range(args.num_workers))
        else:
            console.print("[yellow]Warning: CUDA not available, using CPU (num_workers will be ignored)[/yellow]")
            args.num_workers = 1
            gpu_ids = [0]  # Dummy value for CPU

    # Annotate episodes - choose sequential or parallel mode
    sparse_annotations = existing_annotations.copy()
    dense_annotations = {} if dual_mode else None
    
    if args.num_workers > 1:
        # ===== PARALLEL PROCESSING MODE =====
        console.print(f"\n[bold cyan]Using {args.num_workers} parallel workers on GPUs: {gpu_ids}[/bold cyan]")
        
        # Split episodes across workers
        episodes_per_worker = [[] for _ in range(args.num_workers)]
        for i, ep_idx in enumerate(episode_indices):
            worker_idx = i % args.num_workers
            episodes_per_worker[worker_idx].append(ep_idx)
        
        # Show distribution
        for worker_id, episodes in enumerate(episodes_per_worker):
            console.print(f"[cyan]Worker {worker_id} (GPU {gpu_ids[worker_id]}): {len(episodes)} episodes[/cyan]")
        
        # Start parallel processing using ProcessPoolExecutor
        console.print(f"\n[bold cyan]Starting parallel annotation...[/bold cyan]")
        
        # Use 'spawn' method for CUDA compatibility (required for multi-GPU)
        mp_context = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
            # Submit all worker jobs
            futures = []
            for worker_id in range(args.num_workers):
                if not episodes_per_worker[worker_id]:
                    continue  # Skip workers with no episodes
                
                future = executor.submit(
                    worker_process_episodes,
                    worker_id,
                    gpu_ids[worker_id],
                    episodes_per_worker[worker_id],
                    args.repo_id,
                    video_key,
                    sparse_subtask_list,
                    dense_subtask_list,
                    args.model,
                    torch_dtype,
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    worker_sparse, worker_dense = future.result()
                    sparse_annotations.update(worker_sparse)
                    if dual_mode and worker_dense:
                        dense_annotations.update(worker_dense)
                    
                    # Save after each worker completes
                    save_annotations_to_dataset(dataset.root, sparse_annotations, fps, prefix="sparse")
                    if dual_mode:
                        save_annotations_to_dataset(dataset.root, dense_annotations, fps, prefix="dense")
                    console.print(f"[green]✓ Worker completed, saved {len(worker_sparse)} sparse annotations[/green]")
                    
                except Exception as e:
                    console.print(f"[red]✗ Worker failed: {e}[/red]")
        
        console.print(f"\n[bold green]Parallel processing complete! Annotated {len(sparse_annotations)} episodes[/bold green]")
        
        # Display all sparse annotations
        for ep_idx in sorted(sparse_annotations.keys()):
            if ep_idx not in existing_annotations:  # Only show newly annotated
                display_annotation(sparse_annotations[ep_idx], console, ep_idx, fps, prefix="Sparse")
    
    else:
        console.print(f"\n[bold cyan]Using sequential processing (single GPU/CPU)[/bold cyan]")
        
        # Initialize sparse annotator (loads model)
        sparse_annotator = VideoAnnotator(
            subtask_list=sparse_subtask_list,
            model_name=args.model,
            device=args.device,
            torch_dtype=torch_dtype
        )
        
        # Initialize dense annotator if dual mode (reuses the same model)
        dense_annotator = None
        if dual_mode:
            dense_annotator = VideoAnnotator(
                subtask_list=dense_subtask_list,
                model_name=args.model,
                device=args.device,
                torch_dtype=torch_dtype,
                model=sparse_annotator.model,  # Share the model
                processor=sparse_annotator.processor,  # Share the processor
            )

        # Process episodes sequentially
        for i, ep_idx in enumerate(episode_indices):
            console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
            console.print(f"[bold cyan]Episode {ep_idx} ({i + 1}/{len(episode_indices)})[/bold cyan]")
            console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

            # Sparse annotation
            console.print(f"[cyan]Annotating sparse subtasks...[/cyan]")
            result_ep_idx, sparse_ann, error = process_single_episode(
                ep_idx, dataset.root, dataset.meta, video_key, fps, sparse_annotator, console
            )
            
            if error:
                console.print(f"[red]✗ Failed sparse annotation for episode {result_ep_idx}: {error}[/red]")
            elif sparse_ann:
                sparse_annotations[result_ep_idx] = sparse_ann
                display_annotation(sparse_ann, console, result_ep_idx, fps, prefix="Sparse")
                save_annotations_to_dataset(dataset.root, sparse_annotations, fps, prefix="sparse")
            
            # Dense annotation (if dual mode)
            if dual_mode and dense_annotator:
                console.print(f"[cyan]Annotating dense subtasks...[/cyan]")
                _, dense_ann, dense_error = process_single_episode(
                    ep_idx, dataset.root, dataset.meta, video_key, fps, dense_annotator, console
                )
                
                if dense_error:
                    console.print(f"[red]✗ Failed dense annotation for episode {ep_idx}: {dense_error}[/red]")
                elif dense_ann:
                    dense_annotations[ep_idx] = dense_ann
                    display_annotation(dense_ann, console, ep_idx, fps, prefix="Dense")
                    save_annotations_to_dataset(dataset.root, dense_annotations, fps, prefix="dense")

    # Compute and save sparse temporal proportions
    console.print(f"\n[bold cyan]Computing Sparse Temporal Proportions[/bold cyan]")
    sparse_temporal_proportions = compute_temporal_proportions(sparse_annotations, fps)
    
    # Save sparse temporal proportions
    sparse_proportions_path = dataset.root / "meta" / "temporal_proportions_sparse.json"
    sparse_proportions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sparse_proportions_path, "w") as f:
        json.dump(sparse_temporal_proportions, f, indent=2)
    
    console.print(f"[green]✓ Saved sparse temporal proportions to {sparse_proportions_path}[/green]")
    console.print("\n[cyan]Sparse temporal proportions:[/cyan]")
    for name, proportion in sorted(sparse_temporal_proportions.items(), key=lambda x: -x[1]):
        console.print(f"  {name}: {proportion:.1%}")
    
    # Compute and save dense temporal proportions (if dual mode)
    if dual_mode and dense_annotations:
        console.print(f"\n[bold cyan]Computing Dense Temporal Proportions[/bold cyan]")
        dense_temporal_proportions = compute_temporal_proportions(dense_annotations, fps)
        
        dense_proportions_path = dataset.root / "meta" / "temporal_proportions_dense.json"
        with open(dense_proportions_path, "w") as f:
            json.dump(dense_temporal_proportions, f, indent=2)
        
        console.print(f"[green]✓ Saved dense temporal proportions to {dense_proportions_path}[/green]")
        console.print("\n[cyan]Dense temporal proportions:[/cyan]")
        for name, proportion in sorted(dense_temporal_proportions.items(), key=lambda x: -x[1]):
            console.print(f"  {name}: {proportion:.1%}")

    # Create summary
    console.print(f"\n[bold green]{'=' * 60}[/bold green]")
    console.print(f"[bold green]Annotation Complete![/bold green]")
    console.print(f"[bold green]{'=' * 60}[/bold green]")
    console.print(f"Sparse episodes annotated: {len(sparse_annotations)}")
    console.print(f"Sparse subtasks found: {sum(len(ann.subtasks) for ann in sparse_annotations.values())}")
    if dual_mode and dense_annotations:
        console.print(f"Dense episodes annotated: {len(dense_annotations)}")
        console.print(f"Dense subtasks found: {sum(len(ann.subtasks) for ann in dense_annotations.values())}")

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
