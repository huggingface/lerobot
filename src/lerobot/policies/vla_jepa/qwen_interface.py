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

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
else:
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None

from .configuration_vla_jepa import VLAJEPAConfig


class Qwen3VLInterface(torch.nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.qwen_model_name,
            torch_dtype=self._get_torch_dtype(config.torch_dtype),
        )
        self.processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        self.processor.tokenizer.padding_side = config.tokenizer_padding_side
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    @staticmethod
    def _get_torch_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float16":
            return torch.float16
        return torch.bfloat16

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        # starVLA/JEVLA checkpoints expand action tokens as action_horizon * 4,
        # independent of vj2 num_action_tokens_per_timestep. Keeping this count
        # is required for Qwen embedding/lm_head checkpoint shapes to match.
        max_action_tokens = self.config.chunk_size * 4
        tokenizer = self.processor.tokenizer
        action_tokens = []
        action_token_ids = []
        for idx in range(max_action_tokens):
            token = self.config.special_action_token.format(idx)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))

        embodied_action_token = self.config.embodied_action_token
        if embodied_action_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        # Qwen3-VL-2B ships 151936 embedding rows for a 151669-token vocab, i.e. 267 spare
        # rows, so `chunk_size * 4 + 1` added tokens fit without a resize up to chunk_size=66.
        # Past that, resizing changes `embed_tokens` and `lm_head` shapes, and the checkpoint
        # will no longer load against a model built with a smaller chunk_size (or vice versa)
        # unless those prefixes are listed in `reinit_modules`. Silent until now.
        current_rows = self.model.get_input_embeddings().weight.size(0)
        if current_rows < len(tokenizer):
            logging.warning(
                f"chunk_size={self.config.chunk_size} needs {max_action_tokens + 1} added tokens, "
                f"which exceeds the {current_rows} embedding rows of {self.config.qwen_model_name}. "
                f"Resizing to {len(tokenizer)} rows changes the shapes of "
                f"`model.qwen.model.model.language_model.embed_tokens` and "
                f"`model.qwen.model.lm_head`, so this model will not load from a checkpoint trained "
                f"with a different chunk_size unless those prefixes are in `reinit_modules`."
            )
            self.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_action_token_id

    def build_inputs(
        self,
        images: Sequence[Sequence[torch.Tensor]],
        instructions: Sequence[str],
        action_prompt: str,
        embodied_prompt: str,
    ) -> dict[str, torch.Tensor]:
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            prompt = self.config.prompt_template.format(
                instruction=instruction,
                actions=action_prompt,
                e_actions=embodied_prompt,
            )
            content = [{"type": "image", "image": img} for img in sample_images]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        # The Qwen image processor is a torchvision-backed fast processor: passing the
        # images as GPU tensors (with `device`) keeps the whole vision pipeline on-device
        # and avoids a GPU->CPU->GPU roundtrip. The image tensors are forwarded through
        # apply_chat_template untouched into Qwen3VLProcessor.__call__.
        # do_rescale=False: images already arrive as float in [0, 1] (the dataset decoder
        # yields float32/255 and VISUAL normalization is IDENTITY), so we skip the
        # processor's /255 rescale instead of round-tripping through uint8.
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": self.model.device,
                "do_rescale": False,
            },
        )
        return batch_inputs.to(self.model.device)

    @staticmethod
    def to_pixel_values(image_tensor: torch.Tensor) -> torch.Tensor:
        """Prepare an image/video tensor for the fast processors (used with do_rescale=False).

        The dataset decoder yields float32 in [0, 1] (channels-first) and VISUAL
        normalization is IDENTITY, so the tensor already arrives in [0, 1]; we pass it
        through as float and let the processors normalize (no rescale, no uint8
        quantization). A single channel is expanded to 3 to match the RGB processors.

        Works for any channels-first layout (channel dim is -3): [C, H, W], [B, C, H, W],
        [T, C, H, W], [B, V, T, C, H, W], ...
        """
        image = image_tensor.detach().float()
        if image.shape[-3] == 1:
            repeats = [1] * image.ndim
            repeats[-3] = 3
            image = image.repeat(*repeats)
        return image
