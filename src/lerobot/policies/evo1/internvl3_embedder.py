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

import functools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoTokenizer
else:
    AutoModel = None
    AutoTokenizer = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # nosec B105
IMG_START_TOKEN = "<img>"  # nosec B105
IMG_END_TOKEN = "</img>"  # nosec B105

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=10000)
def get_target_aspect_ratio(orig_width: int, orig_height: int, image_size: int, min_num: int, max_num: int):
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    ratio_w, ratio_h = get_target_aspect_ratio(orig_width, orig_height, image_size, min_num, max_num)
    target_width = image_size * ratio_w
    target_height = image_size * ratio_h
    blocks = ratio_w * ratio_h
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVL3Embedder(nn.Module):
    """Vision-language embedder using the native HF InternVL3 model (no trust_remote_code)."""

    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B-hf",
        image_size=448,
        device="cuda",
        num_language_layers: int | None = 14,
        model_dtype: str | torch.dtype = "bfloat16",
        use_flash_attn: bool = True,
        enable_gradient_checkpointing: bool = True,
        gradient_checkpointing_use_reentrant: bool = False,
    ):
        super().__init__()
        self._requested_device = device
        self.image_size = image_size
        self.num_language_layers = num_language_layers
        self.max_text_length = 1024
        self.enable_gradient_checkpointing = bool(enable_gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = bool(gradient_checkpointing_use_reentrant)

        require_package("transformers", extra="evo1")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(model_dtype, str):
            try:
                model_dtype = getattr(torch, model_dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported EVO1 vlm_dtype '{model_dtype}'") from exc

        attn_implementation = "flash_attention_2" if (use_flash_attn and _flash_attn_available()) else "eager"
        if use_flash_attn and attn_implementation == "eager":
            logger.warning("flash_attn is not installed. Falling back to eager attention.")

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).to(self._requested_device)

        self.num_image_token = self.model.config.image_seq_length

        # Truncate language model to the requested number of layers
        layers = self.model.language_model.layers
        if self.num_language_layers is not None:
            layers = layers[: self.num_language_layers]
        self.model.language_model.layers = torch.nn.ModuleList(layers)

        self._configure_memory_features()
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def _configure_memory_features(self) -> None:
        checkpoint_kwargs = {"use_reentrant": self.gradient_checkpointing_use_reentrant}

        if not self.enable_gradient_checkpointing:
            language_model = self.model.language_model
            if hasattr(language_model, "gradient_checkpointing_disable"):
                language_model.gradient_checkpointing_disable()
            vision_tower = getattr(self.model, "vision_tower", None)
            if vision_tower is not None and hasattr(vision_tower, "encoder"):
                vision_tower.encoder.gradient_checkpointing = False
            return

        def _enable_ckpt(module: nn.Module | None) -> bool:
            if module is None:
                return False
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
                except TypeError:
                    module.gradient_checkpointing_enable()
                return True
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                return True
            return False

        enabled_any = _enable_ckpt(self.model)

        vision_tower = getattr(self.model, "vision_tower", None)
        if vision_tower is not None:
            enabled_any = _enable_ckpt(vision_tower) or enabled_any

        language_model = self.model.language_model
        enabled_any = _enable_ckpt(language_model) or enabled_any
        if hasattr(language_model, "config"):
            language_model.config.use_cache = False

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if enabled_any:
            logger.info("Gradient checkpointing enabled for InternVL3 embedder.")
        else:
            logger.warning(
                "Requested gradient checkpointing, but model does not expose checkpointing controls."
            )

    def _preprocess_single_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            pil_image = to_pil_image(image.detach().cpu())
        else:
            pil_image = image.convert("RGB")
        tiles = dynamic_preprocess(pil_image, image_size=self.image_size)
        tile_tensors = torch.stack([tvf.to_tensor(tile) for tile in tiles]).to(
            device=self.device, dtype=torch.bfloat16
        )
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=torch.bfloat16).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=torch.bfloat16).view(1, 3, 1, 1)
        return (tile_tensors - mean) / std

    def _preprocess_images(
        self,
        image_tensors_batch: Sequence[Sequence[Image.Image | torch.Tensor]],
    ) -> tuple[torch.Tensor, list[list[int]]]:
        pixel_values_list = []
        batch_num_tiles_list: list[list[int]] = []

        for image_tensors in image_tensors_batch:
            num_tiles_list: list[int] = []
            for image in image_tensors:
                tiles = self._preprocess_single_image(image)
                pixel_values_list.append(tiles)
                num_tiles_list.append(int(tiles.shape[0]))
            batch_num_tiles_list.append(num_tiles_list)

        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = torch.empty(
                0, 3, self.image_size, self.image_size, dtype=torch.bfloat16, device=self.device
            )
        return pixel_values, batch_num_tiles_list

    def _build_multimodal_prompts(
        self,
        batch_num_tiles_list: list[list[int]],
        text_prompts: Sequence[str],
    ) -> list[str]:
        prompts = []
        for num_tiles_list, text_prompt in zip(batch_num_tiles_list, text_prompts, strict=True):
            prompt_segments = []
            for i, tile_count in enumerate(num_tiles_list):
                token_count = self.num_image_token * tile_count
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                prompt_segments.append(f"Image-{i + 1}: {image_tokens}\n")
            prompts.append("".join(prompt_segments) + text_prompt.strip())
        return prompts

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors_batch: Sequence[Sequence[Image.Image | torch.Tensor]],
        image_masks: torch.Tensor,
        text_prompts: Sequence[str],
        return_cls_only: bool = True,
    ):
        pixel_values, batch_num_tiles_list = self._preprocess_images(image_tensors_batch)
        if pixel_values.shape[0] == 0:
            logger.warning("InternVL3 received an empty image batch after preprocessing.")
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError("Unable to infer hidden size for empty InternVL3 batch.")
            empty = torch.empty(0, hidden_size, device=self.device, dtype=torch.float32)
            return empty

        prompts = self._build_multimodal_prompts(batch_num_tiles_list, text_prompts)

        model_inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Zero out attention for absent images
        img_token_mask = input_ids == self.img_context_token_id
        tokens_per_tile = self.num_image_token
        for batch_index in range(input_ids.shape[0]):
            current_token_idx = 0
            img_token_locations = torch.where(img_token_mask[batch_index])[0]
            for image_index, num_tiles in enumerate(batch_num_tiles_list[batch_index]):
                num_tokens_for_image = num_tiles * tokens_per_tile
                if not bool(image_masks[batch_index, image_index].item()):
                    start_offset = current_token_idx
                    end_offset = min(current_token_idx + num_tokens_for_image, len(img_token_locations))
                    if start_offset < end_offset:
                        idxs = img_token_locations[start_offset:end_offset]
                        attention_mask[batch_index, idxs] = 0
                current_token_idx += num_tokens_for_image

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)
        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
