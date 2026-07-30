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
import torch.nn as nn
import torchvision.transforms.functional as tvf
from torchvision.transforms.functional import InterpolationMode

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import is_flash_attn_2_available
else:
    AutoModel = None
    AutoTokenizer = None
    is_flash_attn_2_available = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # nosec B105
IMG_START_TOKEN = "<img>"  # nosec B105
IMG_END_TOKEN = "</img>"  # nosec B105

logger = logging.getLogger(__name__)


def _batched_resize_01(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Resize a batch of ``[0, 1]`` images to ``(image_size, image_size)`` on-device.

    Numerically mirrors InternVL3's reference PIL preprocessing
    (``to_pil_image`` -> ``Image.resize`` -> ``to_tensor``): the float input is quantized to uint8
    exactly as ``to_pil_image`` does, then resized with bicubic interpolation and antialiasing,
    which matches PIL's default resampler. Matching the reference pixel-for-pixel keeps the policy
    interchangeable with checkpoints produced by the upstream EVO1 preprocessing.

    Args:
        images: float tensor of shape ``(N, C, H, W)`` with values in ``[0, 1]``.

    Returns:
        float32 tensor of shape ``(N, C, image_size, image_size)`` with values in ``[0, 1]``.
    """
    # to_pil_image() quantizes float [0, 1] to uint8 (x * 255, truncated); replicate that so the
    # bicubic resample sees the same integer pixels PIL would.
    pixels_u8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
    resized = tvf.resize(
        pixels_u8, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    return resized.to(torch.float32) / 255.0


def _batched_pixel_values(
    camera_images: Sequence[torch.Tensor],
    max_views: int,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build InternVL3 ``pixel_values`` from per-camera ``[0, 1]`` image batches without leaving the device.

    Each image is resized, converted to ``dtype``, and ImageNet-normalized (a single tile per
    image), batched across the whole minibatch. Absent views (fewer cameras than ``max_views``)
    are filled with zero images; their placeholder tokens are masked out of attention downstream
    via ``_mask_absent_image_tokens``.

    Returns:
        ``pixel_values`` of shape ``(B * max_views, C, image_size, image_size)``, ordered row-major
        over ``(sample, view)`` to line up with the per-view image placeholders in the prompt.
    """
    resized: list[torch.Tensor] = []
    for image in camera_images:
        resized.append(_batched_resize_01(image.to(device=device), image_size).to(dtype))

    batch_size = resized[0].shape[0]
    channels = resized[0].shape[1]
    while len(resized) < max_views:
        resized.append(torch.zeros(batch_size, channels, image_size, image_size, dtype=dtype, device=device))

    stacked = torch.stack(resized[:max_views], dim=1)  # (B, V, C, H, W)
    mean = mean.to(device=device, dtype=dtype).view(1, 1, -1, 1, 1)
    std = std.to(device=device, dtype=dtype).view(1, 1, -1, 1, 1)
    normalized = (stacked - mean) / std
    return normalized.reshape(batch_size * max_views, channels, image_size, image_size)


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
        max_text_length: int = 1024,
        enable_gradient_checkpointing: bool = True,
        gradient_checkpointing_use_reentrant: bool = False,
        hub_kwargs: dict | None = None,
    ):
        super().__init__()
        self._requested_device = device
        self.image_size = image_size
        self.num_language_layers = num_language_layers
        self.max_text_length = max_text_length
        self.enable_gradient_checkpointing = bool(enable_gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = bool(gradient_checkpointing_use_reentrant)
        hub_kwargs = hub_kwargs or {}

        require_package("transformers", extra="evo1")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kwargs)
        if isinstance(model_dtype, str):
            try:
                model_dtype = getattr(torch, model_dtype)
            except AttributeError as exc:
                raise ValueError(f"Unsupported EVO1 vlm_dtype '{model_dtype}'") from exc
        self.model_dtype = model_dtype

        attn_implementation = (
            "flash_attention_2" if (use_flash_attn and is_flash_attn_2_available()) else "eager"
        )
        if use_flash_attn and attn_implementation == "eager":
            logger.warning(
                "Flash Attention 2 is unavailable on this runtime. Falling back to eager attention."
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            **hub_kwargs,
        ).to(self._requested_device)

        checkpoint_image_size = getattr(self.model.config.vision_config, "image_size", None)
        if isinstance(checkpoint_image_size, (list, tuple)):
            checkpoint_image_size = checkpoint_image_size[0]
        if checkpoint_image_size is not None and int(checkpoint_image_size) != int(image_size):
            raise ValueError(
                f"EVO1 image_resolution ({image_size}) must match the InternVL checkpoint's native "
                f"image size ({checkpoint_image_size}): the checkpoint's image_seq_length assumes "
                "its native resolution, so other sizes would desync the image placeholder tokens "
                "from the vision features."
            )

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

    def get_fused_image_text_embedding_batched(
        self,
        camera_images: Sequence[torch.Tensor],
        image_masks: torch.Tensor,
        text_prompts: Sequence[str],
        return_cls_only: bool = True,
    ):
        """Fused VL embedding from per-camera ``[0, 1]`` image batches (no PIL, no host round-trip).

        Args:
            camera_images: list of per-camera tensors, each shaped ``(B, C, H, W)`` in ``[0, 1]``.
            image_masks: bool tensor ``(B, max_views)`` marking present views.

        Returns:
            A ``(embeddings, valid_mask)`` tuple. With ``return_cls_only=False``, ``embeddings`` is
            ``(B, L, H)`` and ``valid_mask`` is a ``(B, L)`` bool tensor marking tokens downstream
            attention may attend to (padding and absent-view tokens are False). With
            ``return_cls_only=True``, ``embeddings`` is the pooled ``(B, H)`` last-valid-token state
            and ``valid_mask`` is None.
        """
        max_views = int(image_masks.shape[1])
        batch_size = int(image_masks.shape[0])
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.model_dtype)
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.model_dtype)
        pixel_values = _batched_pixel_values(
            camera_images, max_views, self.image_size, mean, std, self.model_dtype, self.device
        )
        # InternVL3 preprocessing uses a single tile per image (max_num=1).
        batch_num_tiles_list = [[1] * max_views for _ in range(batch_size)]
        return self._forward_vlm(
            pixel_values, batch_num_tiles_list, image_masks, text_prompts, return_cls_only
        )

    def _mask_absent_image_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_masks: torch.Tensor,
        batch_num_tiles_list: list[list[int]],
    ) -> torch.Tensor:
        """Zero attention over the image-context tokens of absent (zero-padded) views.

        Fully vectorized: runs without any host<->device synchronization.
        """
        # A single tile per image (max_num=1), so every image occupies the same number of
        # context tokens.
        tiles_per_image = (
            batch_num_tiles_list[0][0] if batch_num_tiles_list and batch_num_tiles_list[0] else 1
        )
        tokens_per_image = self.num_image_token * tiles_per_image

        image_masks = image_masks.to(device=input_ids.device).bool()
        img_token_mask = input_ids == self.img_context_token_id  # (B, L)
        # keep[b, k] tells whether the k-th image-context token (ordered view0, view1, ...) survives.
        per_token_keep = image_masks.repeat_interleave(tokens_per_image, dim=1)  # (B, V * tokens_per_image)
        # Rank each context token by its running position among the row's context tokens.
        ctx_index = img_token_mask.to(torch.long).cumsum(dim=1) - 1
        ctx_index = ctx_index.clamp(min=0, max=per_token_keep.shape[1] - 1)
        keep_here = torch.gather(per_token_keep, 1, ctx_index)  # (B, L)
        drop = img_token_mask & ~keep_here
        return attention_mask.masked_fill(drop, 0)

    def _forward_vlm(
        self,
        pixel_values: torch.Tensor,
        batch_num_tiles_list: list[list[int]],
        image_masks: torch.Tensor,
        text_prompts: Sequence[str],
        return_cls_only: bool,
    ):
        if pixel_values.shape[0] == 0:
            logger.warning("InternVL3 received an empty image batch after preprocessing.")
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError("Unable to infer hidden size for empty InternVL3 batch.")
            return torch.empty(0, hidden_size, device=self.device, dtype=torch.float32), None

        prompts = self._build_multimodal_prompts(batch_num_tiles_list, text_prompts)

        model_inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        if input_ids.shape[1] >= self.max_text_length:
            # Truncation cuts from the right, so text is dropped before image placeholders — but a
            # large max_views * image_seq_length budget can still eat into them. Fail loudly instead
            # of letting the VLM crash on a placeholder/vision-feature count mismatch.
            expected_image_tokens = self.num_image_token * sum(batch_num_tiles_list[0])
            image_token_counts = (input_ids == self.img_context_token_id).sum(dim=1)
            if not bool((image_token_counts == expected_image_tokens).all()):
                raise ValueError(
                    f"Prompt truncation at max_text_length={self.max_text_length} cut into the "
                    f"image placeholder tokens ({expected_image_tokens} expected per sample). "
                    "Increase max_text_length or reduce max_views."
                )
        attention_mask = self._mask_absent_image_tokens(
            input_ids, model_inputs["attention_mask"], image_masks, batch_num_tiles_list
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)
        valid_mask = attention_mask.to(torch.bool)
        if return_cls_only:
            # Right-padded causal decoder: the last valid token is the only one that has attended
            # to the full image + text prompt.
            positions = torch.arange(valid_mask.shape[1], device=valid_mask.device)
            last_valid = (valid_mask.long() * positions).argmax(dim=1)
            batch_index = torch.arange(fused_hidden.shape[0], device=fused_hidden.device)
            return fused_hidden[batch_index, last_valid], None
        return fused_hidden, valid_mask

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
