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
Synthetic Data Generation for Hi-Robot Style Hierarchical Policy Training.

This script generates synthetic user prompts (ℓ_t) and robot utterances (u_t) for
hierarchical policy training using Qwen VLM as the generator model (pgen).

The pipeline:
1. Loads a LeRobot dataset with skill annotations (from annotate.py)
2. For each frame, generates synthetic dialogue based on:
   - Visual context (images at time t)
   - Current skill being performed
   - History of previous skills
   - High-level task description
3. Saves results as high-level tasks and updates dataset with task_index_high_level

Usage:
```bash
python examples/dataset/annotate_pgen.py \
    --repo-id lerobot/svla_so101_pickplace \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output-dir /path/to/output \
    --batch-size 1
```
"""

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# =============================================================================
# Prompt Template for pgen
# =============================================================================

PGEN_PROMPT_TEMPLATE = textwrap.dedent("""\
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


def construct_prompt(
    task_description: str,
    skill_history: list[str],
    skill_current: str,
) -> str:
    """
    Construct the text prompt for pgen.
    
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
    
    return PGEN_PROMPT_TEMPLATE.format(
        task_description=task_description,
        skill_history=history_str,
        skill_current=skill_current,
    )


# =============================================================================
# Qwen VLM Interface
# =============================================================================

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
        images: list[Image.Image | str],
        prompt: str,
    ) -> dict[str, str]:
        """
        Call Qwen VLM to generate synthetic dialogue for a single request.
        
        Args:
            images: List of PIL Images or image paths
            prompt: Text prompt for generation
            
        Returns:
            Dictionary with keys: scenario_type, response_type, user_prompt, robot_utterance
        """
        # Use batch method with single item
        results = self.call_qwen_batch([images], [prompt])
        return results[0]
    
    def call_qwen_batch(
        self,
        batch_images: list[list[Image.Image | str]],
        batch_prompts: list[str],
    ) -> list[dict[str, str]]:
        """
        Call Qwen VLM to generate synthetic dialogue for a batch of requests.
        
        Args:
            batch_images: List of image lists, one per request
            batch_prompts: List of text prompts, one per request
            
        Returns:
            List of dictionaries, each with keys: scenario_type, response_type, user_prompt, robot_utterance
        """
        if len(batch_images) != len(batch_prompts):
            raise ValueError(f"Batch size mismatch: {len(batch_images)} image lists vs {len(batch_prompts)} prompts")
        
        batch_size = len(batch_images)
        if batch_size == 0:
            return []
        
        # Build messages for each item in batch
        all_messages = []
        for images, prompt in zip(batch_images, batch_prompts):
            content = []
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


# =============================================================================
# Annotation Pipeline
# =============================================================================

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


def annotate_sample(
    pgen: QwenPgen,
    images: list[Image.Image | str],
    task_description: str,
    skill_history: list[str],
    skill_current: str,
) -> dict[str, str]:
    """
    Generate synthetic dialogue for a single sample.
    
    Args:
        pgen: Qwen model wrapper
        images: List of images at current timestep
        task_description: High-level task description
        skill_history: Previous skills completed
        skill_current: Current skill being performed
        
    Returns:
        Dictionary with generated dialogue
    """
    prompt = construct_prompt(task_description, skill_history, skill_current)
    result = pgen.call_qwen(images, prompt)
    return result


def annotate_samples_batch(
    pgen: QwenPgen,
    batch_images: list[list[Image.Image | str]],
    batch_task_descriptions: list[str],
    batch_skill_histories: list[list[str]],
    batch_skill_currents: list[str],
) -> list[dict[str, str]]:
    """
    Generate synthetic dialogue for a batch of samples.
    
    Args:
        pgen: Qwen model wrapper
        batch_images: List of image lists, one per sample
        batch_task_descriptions: List of task descriptions
        batch_skill_histories: List of skill history lists
        batch_skill_currents: List of current skills
        
    Returns:
        List of dictionaries with generated dialogue
    """
    # Construct prompts for entire batch
    batch_prompts = []
    for task_desc, skill_hist, skill_curr in zip(
        batch_task_descriptions, batch_skill_histories, batch_skill_currents
    ):
        prompt = construct_prompt(task_desc, skill_hist, skill_curr)
        batch_prompts.append(prompt)
    
    # Process entire batch in one call
    results = pgen.call_qwen_batch(batch_images, batch_prompts)
    return results


def generate_synthetic_data(
    dataset: LeRobotDataset,
    pgen: QwenPgen,
    skills_metadata: dict,
    image_keys: list[str],
    sample_interval_seconds: float = 1.0,
    console: Console | None = None,
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
        image_keys: List of image observation keys to use
        sample_interval_seconds: Generate dialogue every N seconds (default: 1.0)
        console: Rich console for logging
        
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
            result = annotate_sample(
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


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic dialogue data for hierarchical robot policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Generate synthetic data for a dataset
              python annotate_pgen.py --repo-id lerobot/svla_so101_pickplace \\
                  --model Qwen/Qwen2-VL-7B-Instruct \\
                  --output-dir ./output
              
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
    
    # Get image keys
    image_keys = dataset.meta.camera_keys[:args.num_image_views_per_sample]
    console.print(f"[cyan]Using image keys: {image_keys}[/cyan]")
    
    # Generate synthetic data
    tasks_df, task_indices, debug_outputs = generate_synthetic_data(
        dataset=dataset,
        pgen=pgen,
        skills_metadata=skills_metadata,
        image_keys=image_keys,
        sample_interval_seconds=args.sample_interval,
        console=console,
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
    breakpoint()
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

