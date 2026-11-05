# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch
from einops import rearrange
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLVideoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.processing_utils import Unpack

from .conversations import (
    ConversationBuilder,
    InterleavedHistoryConversationBuilder,
    build_conversation_builder,
)

DEFAULT_CONVERSATION_STYLE = InterleavedHistoryConversationBuilder.name


class RynnValueLangProcessor(Qwen3VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        use_meta=False,
        conversation_style: str = DEFAULT_CONVERSATION_STYLE,
        value_token_repeat: int = 1,
        relative_value_token_repeat: int = 1,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.use_meta = use_meta
        self.conversation_style = conversation_style
        if value_token_repeat < 1 or relative_value_token_repeat < 1:
            raise ValueError("value token repeat counts must be >= 1")
        self.value_token_repeat = int(value_token_repeat)
        self.relative_value_token_repeat = int(relative_value_token_repeat)
        self.conversation_builder: ConversationBuilder = self._build_conversation_builder()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load checkpoint assets explicitly without executing Hub model code.

        Released RynnValue processor metadata predates Transformers 5 and does
        not describe its video processor. Constructing the known Qwen
        components directly avoids both that compatibility issue and
        ``trust_remote_code``.
        """
        processor_dict, _ = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        load_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        }
        load_kwargs = {key: value for key, value in kwargs.items() if key in load_keys}
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            **load_kwargs,
        )
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
            chat_template=getattr(tokenizer, "chat_template", None),
            use_meta=kwargs.get("use_meta", processor_dict.get("use_meta", False)),
            conversation_style=kwargs.get(
                "conversation_style",
                processor_dict.get("conversation_style", DEFAULT_CONVERSATION_STYLE),
            ),
            value_token_repeat=kwargs.get("value_token_repeat", processor_dict.get("value_token_repeat", 1)),
            relative_value_token_repeat=kwargs.get(
                "relative_value_token_repeat",
                processor_dict.get("relative_value_token_repeat", 1),
            ),
        )

    @classmethod
    def from_qwen3vl(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @property
    def value_token(self):
        return "<value>"

    @property
    def value_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.value_token)

    @property
    def relative_value_token(self):
        return "<relative_value>" if "<relative_value>" in self.tokenizer.get_vocab() else None

    @property
    def relative_value_token_id(self):
        token = self.relative_value_token
        return None if token is None else self.tokenizer.convert_tokens_to_ids(token)

    def __call__(self, images=None, text=None, **kwargs: Unpack[Qwen3VLProcessorKwargs]) -> BatchFeature:
        kwargs["return_tensors"] = "pt"
        return super().__call__(images=images, text=text, **kwargs)

    def _build_conversation_builder(self):
        return build_conversation_builder(
            self.conversation_style,
            value_token=self.value_token,
            relative_value_token=self.relative_value_token,
            use_meta=self.use_meta,
            value_token_repeat=self.value_token_repeat,
            relative_value_token_repeat=self.relative_value_token_repeat,
        )

    def refresh_conversation_builder(self):
        self.conversation_builder = self._build_conversation_builder()
        return self.conversation_builder

    def _progress_targets(self, goal_timestamp, frame_timestamps, device, value_count):
        timestamps = frame_timestamps.to(device).reshape(-1)
        goal = goal_timestamp.to(device).reshape(1)
        target = (
            (goal - timestamps[-1:]).float().unsqueeze(0)
            if value_count == 1
            else (goal - timestamps).float().unsqueeze(0)
        )
        if target.shape[1] != value_count:
            raise ValueError(
                f"Conversation builder expects {value_count} <value> targets but "
                f"received {target.shape[1]} frame timestamps."
            )
        return target.clamp_min(0)

    def process_history(
        self,
        instruction: str,
        description: str | None,
        images: list[Any],
        frame_timestamps: torch.Tensor,
        goal_timestamp: torch.Tensor,
        success: bool | None = None,
        fusion: bool | None = None,
        robot_description: str | None = None,
        camera_description: str | None = None,
    ) -> BatchFeature:
        builder = self.conversation_builder
        value_count = builder.progress_value_count(len(images))
        fusion_flag = int(bool(fusion))
        content = builder.build_progress(
            instruction=instruction,
            description=description,
            num_frames=len(images),
            robot_description=robot_description,
            camera_description=camera_description,
            match_answer="No" if fusion_flag else "Yes",
            success_answer=None if success is None else ("Yes" if success else "No"),
        )
        outputs = self(text=self.apply_chat_template([{"role": "user", "content": content}]), images=images)
        input_ids = outputs["input_ids"]
        target = self._progress_targets(goal_timestamp, frame_timestamps, input_ids.device, value_count)
        outputs["value"] = target
        outputs["value_fusion_mask"] = torch.full_like(target, fusion_flag, dtype=torch.long)
        timestamps = frame_timestamps.reshape(-1).to(input_ids.device).float()
        outputs["relative_value"] = (timestamps[1:] - timestamps[:-1]).unsqueeze(0)
        outputs["labels"] = self._build_history_labels(input_ids)
        return outputs

    def _build_history_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        marker = torch.tensor(
            self.tokenizer.encode("Analysis: \n", add_special_tokens=False),
            device=input_ids.device,
        )
        labels = torch.full_like(input_ids, -100)
        for batch_index, sequence in enumerate(input_ids):
            for index in range(sequence.shape[0] - marker.shape[0], -1, -1):
                if torch.equal(sequence[index : index + marker.shape[0]], marker):
                    start = index + marker.shape[0]
                    labels[batch_index, start:] = sequence[start:]
                    break
            else:
                raise ValueError(f"'Analysis:' marker not found in sample {batch_index}.")
        return labels

    def process_episode(
        self,
        instruction: str,
        images: list[Any],
        robot_description: str | None = None,
        camera_description: str | None = None,
    ) -> BatchFeature:
        if not images:
            raise ValueError("process_episode requires at least one prediction image.")
        content = self.conversation_builder.build_progress(
            instruction=instruction,
            description=None,
            num_frames=len(images),
            robot_description=robot_description,
            camera_description=camera_description,
            is_inference=True,
        )
        outputs = self(text=self.apply_chat_template([{"role": "user", "content": content}]), images=images)
        input_ids = outputs["input_ids"]
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_positions = input_ids[0].eq(eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions):
            input_ids = input_ids[:, : eos_positions[-1]]
        outputs["input_ids"] = input_ids
        sequence_length = input_ids.shape[-1]
        outputs["attention_mask"] = outputs["attention_mask"][:, :sequence_length]
        if "mm_token_type_ids" in outputs:
            outputs["mm_token_type_ids"] = outputs["mm_token_type_ids"][:, :sequence_length]
        outputs["pixel_values"] = rearrange(outputs["pixel_values"], "(b t) d -> b t d", b=1)
        outputs["image_grid_thw"] = rearrange(outputs["image_grid_thw"], "(b t) d -> b t d", b=1)
        return outputs
