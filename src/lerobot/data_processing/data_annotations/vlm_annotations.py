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

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from lerobot.data_processing.data_annotations.subtask_annotations import Skill
from lerobot.utils.constants import (
    SKILL_SEGMENTATION_PROMPT_TEMPLATE,
    format_subtask_labels_section,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-27B"


def create_skill_segmentation_prompt(
    coarse_goal: str | None = None,
    subtask_labels: list[str] | None = None,
    duration_seconds: float | None = None,
) -> str:
    """Create the prompt for skill segmentation using the template from constants."""
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


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models used in skill segmentation.

    To add a new VLM family:
    1. Subclass BaseVLM
    2. Implement __init__, segment_skills, and segment_skills_batch
    3. Register it in get_vlm()
    """

    @abstractmethod
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        pass

    @abstractmethod
    def segment_skills(
        self,
        video_path: Path,
        episode_duration: float,
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        """Segment a single video into atomic skills."""
        pass

    @abstractmethod
    def segment_skills_batch(
        self,
        video_paths: list[Path],
        episode_durations: list[float],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[list[Skill]]:
        """Segment multiple videos into atomic skills in a single batch."""
        pass

    def _parse_skills_response(self, response: str) -> list[Skill]:
        """Parse JSON skill list from VLM response text."""
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
                try:
                    data = json.loads(match.group())
                    skills_data = data.get("skills", [])
                    return [Skill.from_dict(s) for s in skills_data]
                except json.JSONDecodeError as e:
                    raise ValueError(f"Could not parse JSON from VLM response: {response[:200]}...") from e

        raise ValueError(f"Could not parse skills from response: {response[:200]}...")


class QwenVL(BaseVLM):
    """Qwen VL model for skill segmentation (default: Qwen3.5 series).

    Uses qwen-vl-utils for video processing and the HuggingFace transformers
    Qwen3VLProcessor pipeline. Requires transformers >= 5.4.0 for correct
    video position embeddings.
    """

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.model_name = model_name
        self.process_vision_info = process_vision_info

        logger.info(f"Loading model: {model_name}...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

        logger.info(f"Model loaded on {device}")

    def _build_messages(self, video_path: Path, episode_duration: float, prompt: str) -> list[dict]:
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"
        return [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": 1.0},
                    {
                        "type": "text",
                        "text": (
                            f"Video duration: {duration_str} (exactly {episode_duration:.1f} seconds). "
                            f"Segment into atomic skills. Last skill must end at {episode_duration:.1f}."
                        ),
                    },
                ],
            },
        ]

    def _prepare_inputs(self, messages: list[dict]) -> dict:
        """Tokenize a single message and return processor inputs on device."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        image_inputs, video_inputs = self.process_vision_info(messages, return_video_metadata=True)

        videos, video_metadata = None, None
        if video_inputs:
            videos = [v[0] for v in video_inputs]
            video_metadata = [v[1] for v in video_inputs]

        return self.processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            videos_kwargs={
                "video_metadata": video_metadata,
                "do_sample_frames": False,
            },
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _decode(self, inputs, generated_ids) -> list[str]:
        return self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def segment_skills(
        self,
        video_path: Path,
        episode_duration: float,
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        prompt = create_skill_segmentation_prompt(
            coarse_goal, subtask_labels, duration_seconds=episode_duration
        )
        messages = self._build_messages(video_path, episode_duration, prompt)
        inputs = self._prepare_inputs(messages)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )

        response = self._decode(inputs, generated_ids)[0].strip()
        return self._parse_skills_response(response)

    def segment_skills_batch(
        self,
        video_paths: list[Path],
        episode_durations: list[float],
        coarse_goal: str | None = None,
        subtask_labels: list[str] | None = None,
    ) -> list[list[Skill]]:
        all_texts = []
        all_video_tuples: list[tuple] = []

        for video_path, duration in zip(video_paths, episode_durations, strict=True):
            prompt = create_skill_segmentation_prompt(coarse_goal, subtask_labels, duration_seconds=duration)
            messages = self._build_messages(video_path, duration, prompt)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            _image_inputs, video_inputs = self.process_vision_info(messages, return_video_metadata=True)
            all_texts.append(text)
            all_video_tuples.extend(video_inputs or [])

        videos, video_metadata = None, None
        if all_video_tuples:
            videos = [v[0] for v in all_video_tuples]
            video_metadata = [v[1] for v in all_video_tuples]

        inputs = self.processor(
            text=all_texts,
            videos=videos,
            videos_kwargs={
                "video_metadata": video_metadata,
                "do_sample_frames": False,
            },
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )

        responses = self._decode(inputs, generated_ids)

        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                if not skills:
                    logger.warning(f"No skills parsed for video {idx}")
                all_skills.append(skills)
            except Exception as e:
                logger.warning(f"Failed to parse response for video {idx}: {e}")
                all_skills.append([])

        return all_skills


def get_vlm(model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> BaseVLM:
    """Create a VLM instance. Defaults to QwenVL which supports the Qwen3.5 series."""
    return QwenVL(model_name, device, torch_dtype)
