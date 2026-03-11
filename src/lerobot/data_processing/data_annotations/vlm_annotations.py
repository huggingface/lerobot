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

# VLM Interface (Abstract Base Class for Modularity)

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from lerobot.data_processing.data_annotations.subtask_annotations import Skill
from lerobot.utils.constants import (
    SKILL_SEGMENTATION_PROMPT_TEMPLATE,
    format_subtask_labels_section,
)


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
        self,
        video_path: Path,
        episode_duration: float,
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        """
        Segment a video into atomic skills.

        Args:
            video_path: Path to the video file
            episode_duration: Total duration of the episode in seconds
            coarse_goal: Optional high-level task description
            subtask_labels: If provided, model must choose only from these labels (closed vocabulary)

        Returns:
            List of Skill objects representing atomic manipulation skills
        """
        pass

    @abstractmethod
    def segment_skills_batch(
        self,
        video_paths: list[Path],
        episode_durations: list[float],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
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


def create_skill_segmentation_prompt(
    coarse_goal: str | None = None,
    subtask_labels: list[str] | None = None,
    duration_seconds: float | None = None,
) -> str:
    """Create the prompt for skill segmentation using the template from constants.
    duration_seconds is required. When subtask_labels is provided, uses closed-vocabulary section.
    """
    if duration_seconds is None:
        raise ValueError("duration_seconds is required for skill segmentation prompt")
    goal_context = f'The overall goal is: "{coarse_goal}"\n\n' if coarse_goal else ""
    subtask_labels_section = format_subtask_labels_section(subtask_labels) if subtask_labels else ""
    video_duration_mm_ss = f"{int(duration_seconds // 60):02d}:{int(duration_seconds % 60):02d}"
    return SKILL_SEGMENTATION_PROMPT_TEMPLATE.format(
        goal_context=goal_context,
        subtask_labels_section=subtask_labels_section,
        video_duration_seconds=duration_seconds,
        video_duration_mm_ss=video_duration_mm_ss,
    )


# Qwen2-VL Implementation


class Qwen2VL(BaseVLM):
    """Qwen2-VL model for skill segmentation."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device
        self.model_name = model_name
        self.process_vision_info = process_vision_info

        print(f"Loading Qwen2-VL model: {model_name}...")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        print(f" Model loaded successfully on {device}")

    def segment_skills(
        self,
        video_path: Path,
        episode_duration: float,
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        """Segment video into skills using Qwen2-VL."""
        prompt = create_skill_segmentation_prompt(
            coarse_goal, subtask_labels, duration_seconds=episode_duration
        )
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {
                        "type": "text",
                        "text": f"Video duration: {duration_str} (exactly {episode_duration:.1f} seconds). Segment into atomic skills. Last skill must end at {episode_duration:.1f}.",
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
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )

        response = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
        )[0].strip()

        return self._parse_skills_response(response)

    def segment_skills_batch(
        self,
        video_paths: list[Path],
        episode_durations: list[float],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[list[Skill]]:
        """Segment multiple videos into skills using Qwen2-VL in a batch."""
        # Create messages for each video (prompt includes duration so each gets correct length)
        all_messages = []
        for video_path, duration in zip(video_paths, episode_durations, strict=True):
            prompt = create_skill_segmentation_prompt(coarse_goal, subtask_labels, duration_seconds=duration)
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Video duration: {duration_str} (exactly {duration:.1f} seconds). Segment into atomic skills. Last skill must end at {duration:.1f}.",
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
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )

        responses = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
        )

        # Parse each response
        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                if not skills:
                    print(f"Warning: No skills parsed from response for video {idx}")
                all_skills.append(skills)
            except Exception as e:
                print(f"Warning: Failed to parse response for video {idx}: {e}")
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
                try:
                    data = json.loads(match.group())
                    skills_data = data.get("skills", [])
                    return [Skill.from_dict(s) for s in skills_data]
                except json.JSONDecodeError as e:
                    excerpt = response[:200]
                    raise ValueError(
                        f"Could not parse JSON from VLM response (fallback failed): {excerpt}..."
                    ) from e

        raise ValueError(f"Could not parse skills from response: {response[:200]}...")


# Qwen3-VL Implementation (MoE variant)


class Qwen3VL(BaseVLM):
    """Qwen3-VL MoE model for skill segmentation."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        self.device = device
        self.model_name = model_name
        self.process_vision_info = process_vision_info

        print(f"Loading Qwen3-VL model: {model_name}...")

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        print(f" Model loaded successfully on {device}")

    def segment_skills(
        self,
        video_path: Path,
        episode_duration: float,
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        """Segment video into skills using Qwen3-VL."""
        prompt = create_skill_segmentation_prompt(
            coarse_goal, subtask_labels, duration_seconds=episode_duration
        )
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {
                        "type": "text",
                        "text": f"Video duration: {duration_str} (exactly {episode_duration:.1f} seconds). Segment into atomic skills. Last skill must end at {episode_duration:.1f}.",
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
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )

        response = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
        )[0].strip()

        return self._parse_skills_response(response)

    def segment_skills_batch(
        self,
        video_paths: list[Path],
        episode_durations: list[float],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[list[Skill]]:
        """Segment multiple videos into skills using Qwen3-VL in a batch."""
        # Create messages for each video (prompt includes duration so each gets correct length)
        all_messages = []
        for video_path, duration in zip(video_paths, episode_durations, strict=True):
            prompt = create_skill_segmentation_prompt(coarse_goal, subtask_labels, duration_seconds=duration)
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Video duration: {duration_str} (exactly {duration:.1f} seconds). Segment into atomic skills. Last skill must end at {duration:.1f}.",
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
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )

        responses = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
        )

        # Parse each response
        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                if not skills:
                    print(f"Warning: No skills parsed from response for video {idx}")
                all_skills.append(skills)
            except Exception as e:
                print(f"Warning: Failed to parse response for video {idx}: {e}")
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


# Qwen3.5-VL Implementation (Qwen3_5ForConditionalGeneration)


class Qwen35VL(BaseVLM):
    """Qwen3.5-VL model for skill segmentation (Qwen3_5ForConditionalGeneration)."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.device = device
        self.model_name = model_name
        self.process_vision_info = process_vision_info

        print(f"Loading Qwen3.5-VL model: {model_name}...")

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        print(f" Model loaded successfully on {device}")

    def segment_skills(
        self,
        video_path: Path,
        episode_duration: float,
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        """Segment video into skills using Qwen3.5-VL."""
        prompt = create_skill_segmentation_prompt(
            coarse_goal, subtask_labels, duration_seconds=episode_duration
        )
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {
                        "type": "text",
                        "text": f"Video duration: {duration_str} (exactly {episode_duration:.1f} seconds). Segment into atomic skills. Last skill must end at {episode_duration:.1f}.",
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

        response = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return self._parse_skills_response(response)

    def segment_skills_batch(
        self,
        video_paths: list[Path],
        episode_durations: list[float],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[list[Skill]]:
        """Segment multiple videos into skills using Qwen3.5-VL in a batch."""
        all_messages = []
        for video_path, duration in zip(video_paths, episode_durations, strict=True):
            prompt = create_skill_segmentation_prompt(coarse_goal, subtask_labels, duration_seconds=duration)
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "fps": 1.0},
                        {
                            "type": "text",
                            "text": f"Video duration: {duration_str} (exactly {duration:.1f} seconds). Segment into atomic skills. Last skill must end at {duration:.1f}.",
                        },
                    ],
                },
            ]
            all_messages.append(messages)

        all_texts = []
        all_image_inputs = []
        all_video_inputs = []

        for messages in all_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

        responses = self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                if not skills:
                    print(f"Warning: No skills parsed from response for video {idx}")
                all_skills.append(skills)
            except Exception as e:
                print(f"Warning: Failed to parse response for video {idx}: {e}")
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
    # Qwen3.5-VL (Qwen3_5ForConditionalGeneration)
    "Qwen/Qwen3.5-27B": Qwen35VL,
    "Qwen/Qwen3-VL-8B-Instruct": Qwen35VL,
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
    if "qwen3.5" in model_lower:
        return Qwen35VL(model_name, device, torch_dtype)
    if "qwen3" in model_lower:
        return Qwen3VL(model_name, device, torch_dtype)
    elif "qwen2" in model_lower or "qwen-vl" in model_lower:
        return Qwen2VL(model_name, device, torch_dtype)

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported models: {list(VLM_REGISTRY.keys())}. "
        "Or implement a new VLM class inheriting from BaseVLM."
    )
