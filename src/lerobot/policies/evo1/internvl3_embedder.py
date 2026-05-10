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
import torchvision.transforms.functional as TF
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


def flash_attn_is_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


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
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        if isinstance(model_dtype, str):
            try:
                model_dtype = getattr(torch, model_dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported EVO1 vlm_dtype '{model_dtype}'") from exc

        resolved_use_flash_attn = bool(use_flash_attn and flash_attn_is_available())
        if use_flash_attn and not resolved_use_flash_attn:
            logger.warning("flash_attn is not installed. Falling back to standard attention.")

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            use_flash_attn=resolved_use_flash_attn,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self._requested_device)

        if hasattr(self.model.language_model, "model"):
            layers = self.model.language_model.model.layers
        else:
            layers = self.model.language_model.layers
        if self.num_language_layers is not None:
            layers = layers[: self.num_language_layers]

        if hasattr(self.model.language_model, "model"):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        self._configure_memory_features()
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def _configure_memory_features(self) -> None:
        checkpoint_kwargs = {"use_reentrant": self.gradient_checkpointing_use_reentrant}

        if not self.enable_gradient_checkpointing:
            if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
                self.model.vision_model.encoder.gradient_checkpointing = False
            language_model = getattr(self.model, "language_model", None)
            if language_model is not None:
                if hasattr(language_model, "gradient_checkpointing_disable"):
                    language_model.gradient_checkpointing_disable()
                elif hasattr(language_model, "gradient_checkpointing"):
                    language_model.gradient_checkpointing = False
                if hasattr(language_model, "model"):
                    inner = language_model.model
                    if hasattr(inner, "gradient_checkpointing_disable"):
                        inner.gradient_checkpointing_disable()
                    elif hasattr(inner, "gradient_checkpointing"):
                        inner.gradient_checkpointing = False
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

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = True
            enabled_any = True

        language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            enabled_any = _enable_ckpt(language_model) or enabled_any
            if hasattr(language_model, "model"):
                enabled_any = _enable_ckpt(language_model.model) or enabled_any
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
        tile_tensors = torch.stack([TF.to_tensor(tile) for tile in tiles]).to(
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
                token_count = self.model.num_image_token * tile_count
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                prompt_segments.append(f"Image-{i + 1}: {image_tokens}\n")
            prompts.append("".join(prompt_segments) + text_prompt.strip())
        return prompts

    def _prepare_and_fuse_embeddings(
        self,
        prompts: Sequence[str],
        vit_embeds: torch.Tensor,
        image_masks: torch.Tensor,
        batch_num_tiles_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        untruncated_ids = self.tokenizer(list(prompts), padding=False, truncation=False)["input_ids"]
        true_sequence_length = max((len(ids) for ids in untruncated_ids), default=0)
        if true_sequence_length > self.max_text_length:
            logger.warning(
                "InternVL3 prompt truncated in batch: max_length=%s actual_max_length=%s",
                self.max_text_length,
                true_sequence_length,
            )

        model_inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        img_token_mask = input_ids == self.img_context_token_id
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        batch_size, _, channels = input_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, channels).to(dtype=input_embeds.dtype, device=input_embeds.device)
        tokens_per_tile = self.model.num_image_token
        actual_vis_tokens_list = img_token_mask.sum(dim=1).tolist()

        vit_idx = 0
        for batch_index in range(batch_size):
            expected_vis_tokens = sum(batch_num_tiles_list[batch_index]) * tokens_per_tile
            mask_b = img_token_mask[batch_index]
            actual_vis_tokens = actual_vis_tokens_list[batch_index]

            item_vit_embeds = vit_embeds[vit_idx : vit_idx + expected_vis_tokens]
            vit_idx += expected_vis_tokens
            if actual_vis_tokens > 0:
                if item_vit_embeds.shape[0] < actual_vis_tokens:
                    raise ValueError(
                        f"InternVL3 produced fewer image tokens than expected for sample {batch_index}: "
                        f"got {item_vit_embeds.shape[0]}, need {actual_vis_tokens}"
                    )
                input_embeds[batch_index, mask_b] = item_vit_embeds[:actual_vis_tokens]

            current_token_idx = 0
            img_token_locations = torch.where(mask_b)[0]
            for image_index, num_tiles in enumerate(batch_num_tiles_list[batch_index]):
                num_tokens_for_image = num_tiles * tokens_per_tile
                if not bool(image_masks[batch_index, image_index].item()):
                    start_offset = current_token_idx
                    end_offset = min(current_token_idx + num_tokens_for_image, len(img_token_locations))
                    if start_offset < end_offset:
                        idxs = img_token_locations[start_offset:end_offset]
                        attention_mask[batch_index, idxs] = 0
                current_token_idx += num_tokens_for_image

        return input_embeds, attention_mask

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
            if hidden_size is None and hasattr(self.model.language_model, "config"):
                hidden_size = getattr(self.model.language_model.config, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError("Unable to infer hidden size for empty InternVL3 batch.")
            empty = torch.empty(0, hidden_size, device=self.device, dtype=torch.float32)
            return empty

        prompts = self._build_multimodal_prompts(batch_num_tiles_list, text_prompts)
        vit_embeds = self.model.extract_feature(pixel_values)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(
            prompts,
            vit_embeds,
            image_masks.to(device=self.device),
            batch_num_tiles_list,
        )

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)
        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
