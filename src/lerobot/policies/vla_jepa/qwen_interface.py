from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
        max_action_tokens = self.config.chunk_size * self.config.num_action_tokens_per_timestep
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

        if self.model.get_input_embeddings().weight.size(0) < len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_action_token_id

    def build_inputs(
        self,
        images: Sequence[Sequence[Image.Image]],
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

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)

    @staticmethod
    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        image = image_tensor.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.float()
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255).to(torch.uint8).numpy()
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return Image.fromarray(image)
