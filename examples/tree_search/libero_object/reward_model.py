#!/usr/bin/env python
"""Small language-conditioned visual reward model for tree-search scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    encoder_type: str = "siglip"
    encoder_model_id: str = "google/siglip-base-patch16-224"
    freeze_encoder: bool = True
    image_size: int = 224
    use_proprioception: bool = True
    proprioception_dim: int = 8
    proprioception_hidden_dim: int = 64
    scene_temporal_window: int = 1
    use_patch_text_fusion: bool = True
    text_query_ngram_window: int = 3
    text_query_ngram_stride: int = 2
    text_query_include_full: bool = True
    text_query_include_tail: bool = True
    text_query_max_count: int = 12
    text_query_stop_words: tuple[str, ...] = (
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at",
        "of",
        "with",
        "from",
        "into",
        "onto",
        "it",
        "this",
        "that",
    )
    patch_attention_dim: int = 256
    scene_summary_dim: int = 512
    wrist_summary_dim: int = 128
    head_hidden_dim: int = 512
    head_dropout: float = 0.1
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardModelConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_model_id(encoder_type: str) -> str:
    encoder_type = encoder_type.lower()
    if encoder_type == "siglip2":
        return "google/siglip2-base-patch16-224"
    if encoder_type == "siglip":
        return "google/siglip-base-patch16-224"
    if encoder_type == "clip":
        return "openai/clip-vit-base-patch32"
    if encoder_type in {"resnet18", "resnet34"}:
        return encoder_type
    raise ValueError(f"Unsupported reward encoder_type: {encoder_type}")


def image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image array, got shape {array.shape}.")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    array = np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))
    return Image.fromarray(array).convert("RGB")


def blank_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(0, 0, 0))


def normalize_text_queries(text: str, cfg: RewardModelConfig) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ]+", " ", text, flags=re.IGNORECASE)
    words = [word for word in text.split() if len(word) > 1]
    stop_words = {str(word).lower() for word in cfg.text_query_stop_words}
    content_words = [word for word in words if word.lower() not in stop_words]
    if not content_words:
        content_words = words or [str(text).strip() or "object"]

    queries: list[str] = []
    normalized_full = " ".join(content_words)
    if cfg.text_query_include_full and normalized_full:
        queries.append(normalized_full)

    window = max(1, int(cfg.text_query_ngram_window))
    stride = max(1, int(cfg.text_query_ngram_stride))
    if len(content_words) <= window:
        if normalized_full and normalized_full not in queries:
            queries.append(normalized_full)
    else:
        starts = list(range(0, len(content_words) - window + 1, stride))
        tail_start = len(content_words) - window
        if cfg.text_query_include_tail and tail_start not in starts:
            starts.append(tail_start)
        for start in starts:
            query = " ".join(content_words[start : start + window])
            if query and query not in queries:
                queries.append(query)

    max_count = max(1, int(cfg.text_query_max_count))
    return queries[:max_count] or [str(text).strip() or "object"]


class RewardBatchProcessor:
    """Converts reward samples into tensors accepted by MultiModalRewardModel."""

    def __init__(self, cfg: RewardModelConfig) -> None:
        self.cfg = cfg
        self.encoder_type = cfg.encoder_type.lower()
        logger.info(
            "Initializing reward batch processor encoder_type=%s model_id=%s use_proprioception=%s",
            cfg.encoder_type,
            cfg.encoder_model_id,
            cfg.use_proprioception,
        )
        if self.encoder_type in {"siglip2", "siglip", "clip"}:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError("Install the transformers extra to use SigLIP/SigLIP2/CLIP rewarders.") from exc

            self.processor = AutoProcessor.from_pretrained(
                cfg.encoder_model_id,
                trust_remote_code=cfg.trust_remote_code,
            )
            self.resnet_transform = None
        elif self.encoder_type in {"resnet18", "resnet34"}:
            from torchvision import transforms

            self.processor = None
            self.resnet_transform = transforms.Compose(
                [
                    transforms.Resize((cfg.image_size, cfg.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            raise ValueError(f"Unsupported reward encoder_type: {cfg.encoder_type}")

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "labels": torch.tensor([float(sample["label"]) for sample in samples], dtype=torch.float32),
            "episode_indices": torch.tensor(
                [
                    int(sample.get("dataset_id", 0)) * 1_000_000_000 + int(sample.get("episode_index", -1))
                    for sample in samples
                ],
                dtype=torch.long,
            ),
            "raw_episode_indices": torch.tensor(
                [int(sample.get("episode_index", -1)) for sample in samples],
                dtype=torch.long,
            ),
            "dataset_ids": torch.tensor(
                [int(sample.get("dataset_id", -1)) for sample in samples],
                dtype=torch.long,
            ),
            "local_indices": torch.tensor(
                [int(sample.get("local_index", -1)) for sample in samples],
                dtype=torch.long,
            ),
            "temporal_local_indices": [
                [int(ix) for ix in sample.get("temporal_local_indices", [sample.get("local_index", -1)])]
                for sample in samples
            ],
            "is_bad_sequence": torch.tensor(
                [bool(sample.get("is_bad_sequence", False)) for sample in samples],
                dtype=torch.bool,
            ),
            "source_task_orders": torch.tensor(
                [int(sample.get("source_task_order", -1)) for sample in samples],
                dtype=torch.long,
            ),
            "text_task_orders": torch.tensor(
                [int(sample.get("text_task_order", -1)) for sample in samples],
                dtype=torch.long,
            ),
            "is_text_mismatch": torch.tensor(
                [bool(sample.get("is_text_mismatch", False)) for sample in samples],
                dtype=torch.bool,
            ),
            "tasks": [str(sample.get("task", "")) for sample in samples],
            "source_tasks": [str(sample.get("source_task", sample.get("task", ""))) for sample in samples],
            "dataset_repo_ids": [str(sample.get("dataset_repo_id", "")) for sample in samples],
        }
        batch["scene_pixel_values"] = self._scene_sequences_to_tensor(
            [self._scene_images(sample) for sample in samples]
        )
        batch["wrist_pixel_values"] = self._images_to_tensor(
            [
                sample["wrist_image"] if sample.get("wrist_image") is not None else blank_image(self.cfg.image_size)
                for sample in samples
            ]
        )

        if self.encoder_type in {"siglip2", "siglip", "clip"}:
            text_inputs = self.processor(
                text=[str(sample.get("task", "")) for sample in samples],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if "input_ids" in text_inputs:
                batch["input_ids"] = text_inputs["input_ids"]
            if "attention_mask" in text_inputs:
                batch["attention_mask"] = text_inputs["attention_mask"]
            query_lists = [normalize_text_queries(str(sample.get("task", "")), self.cfg) for sample in samples]
            query_count = max(1, min(max(len(queries) for queries in query_lists), int(self.cfg.text_query_max_count)))
            padded_queries: list[str] = []
            query_mask: list[list[bool]] = []
            for queries in query_lists:
                queries = queries[:query_count]
                fallback = queries[0] if queries else str(samples[0].get("task", "object"))
                query_mask.append([ix < len(queries) for ix in range(query_count)])
                padded_queries.extend(queries + [fallback] * (query_count - len(queries)))
            query_inputs = self.processor(
                text=padded_queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if "input_ids" in query_inputs:
                batch["query_input_ids"] = query_inputs["input_ids"].view(len(samples), query_count, -1)
            if "attention_mask" in query_inputs:
                batch["query_attention_mask"] = query_inputs["attention_mask"].view(len(samples), query_count, -1)
            batch["text_query_mask"] = torch.tensor(query_mask, dtype=torch.bool)
            batch["text_queries"] = query_lists
        if self.cfg.use_proprioception:
            batch["proprioception"] = self._proprioception_to_tensor(
                [sample.get("proprioception") for sample in samples]
            )
        return batch

    def _images_to_tensor(self, images: list[Any]) -> Tensor:
        pil_images = [image_to_pil(image) for image in images]
        if self.processor is not None:
            encoded = self.processor(images=pil_images, return_tensors="pt")
            return encoded["pixel_values"]
        assert self.resnet_transform is not None
        return torch.stack([self.resnet_transform(image) for image in pil_images])

    def _scene_images(self, sample: dict[str, Any]) -> list[Any]:
        images = sample.get("scene_images")
        if images is None:
            images = [sample["scene_image"]]
        if not isinstance(images, list | tuple):
            images = [images]
        return list(images)

    def _scene_sequences_to_tensor(self, sequences: list[list[Any]]) -> Tensor:
        window = max(1, int(self.cfg.scene_temporal_window))
        normalized: list[list[Any]] = []
        for sequence in sequences:
            if not sequence:
                sequence = [blank_image(self.cfg.image_size)]
            sequence = list(sequence)[-window:]
            if len(sequence) < window:
                sequence = [sequence[0]] * (window - len(sequence)) + sequence
            normalized.append(sequence)

        flat_images = [image for sequence in normalized for image in sequence]
        flat_tensor = self._images_to_tensor(flat_images)
        return flat_tensor.view(len(normalized), window, *flat_tensor.shape[1:])

    def _proprioception_to_tensor(self, values: list[Any]) -> Tensor:
        rows: list[Tensor] = []
        for value in values:
            if value is None:
                rows.append(torch.zeros(self.cfg.proprioception_dim, dtype=torch.float32))
                continue
            tensor = torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32).flatten()
            if tensor.numel() != self.cfg.proprioception_dim:
                raise ValueError(
                    "Expected proprioception vector with "
                    f"{self.cfg.proprioception_dim} values, got shape {tuple(np.asarray(value).shape)}."
                )
            rows.append(tensor)
        return torch.stack(rows)


class MultiModalRewardModel(nn.Module):
    """Frozen multimodal encoder plus a shallow scalar reward head."""

    def __init__(self, cfg: RewardModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_type = cfg.encoder_type.lower()
        self.encoder: nn.Module
        self.resnet_feature_dim: int | None = None
        logger.info(
            "Loading reward encoder encoder_type=%s model_id=%s freeze_encoder=%s",
            cfg.encoder_type,
            cfg.encoder_model_id,
            cfg.freeze_encoder,
        )

        if self.encoder_type in {"siglip2", "siglip", "clip"}:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError("Install the transformers extra to use SigLIP/SigLIP2/CLIP rewarders.") from exc

            self.encoder = AutoModel.from_pretrained(
                cfg.encoder_model_id,
                trust_remote_code=cfg.trust_remote_code,
            )
        elif self.encoder_type in {"resnet18", "resnet34"}:
            self.encoder, self.resnet_feature_dim = self._make_resnet_encoder(self.encoder_type)
        else:
            raise ValueError(f"Unsupported reward encoder_type: {cfg.encoder_type}")
        logger.info("Reward encoder loaded.")

        if cfg.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        """
        proprioception
            [                                                                                                                                                                                           
            eef_pos_x,
            eef_pos_y,
            eef_pos_z,
            eef_axisangle_x,
            eef_axisangle_y,
            eef_axisangle_z,
            gripper_qpos_0,
            gripper_qpos_1,
            ]
        """
        self.proprioception_mlp = (
            nn.Sequential(
                nn.LayerNorm(cfg.proprioception_dim),
                nn.Linear(cfg.proprioception_dim, cfg.proprioception_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.proprioception_hidden_dim, cfg.proprioception_hidden_dim),
                nn.ReLU(inplace=True),
            )
            if cfg.use_proprioception
            else None
        )
        self.use_patch_text_fusion = bool(
            cfg.use_patch_text_fusion and self.encoder_type in {"siglip2", "siglip", "clip"}
        )
        if self.use_patch_text_fusion:
            self.patch_key = nn.LazyLinear(cfg.patch_attention_dim)
            self.patch_value = nn.LazyLinear(cfg.patch_attention_dim)
            self.query_proj = nn.LazyLinear(cfg.patch_attention_dim)
            self.scene_summary_mlp = nn.Sequential(
                nn.LayerNorm(cfg.patch_attention_dim * max(1, int(cfg.scene_temporal_window))),
                nn.Linear(cfg.patch_attention_dim * max(1, int(cfg.scene_temporal_window)), cfg.scene_summary_dim),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.scene_summary_dim, cfg.scene_summary_dim),
                nn.GELU(),
            )
        else:
            self.patch_key = None
            self.patch_value = None
            self.query_proj = None
            self.scene_summary_mlp = None

        self.wrist_mlp = nn.Sequential(
            nn.LazyLinear(cfg.wrist_summary_dim),
            nn.LayerNorm(cfg.wrist_summary_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
        )
        self.head = nn.Sequential(
            nn.LazyLinear(cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, max(64, cfg.head_hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(64, cfg.head_hidden_dim // 2), 1),
        )

    @staticmethod
    def _make_resnet_encoder(encoder_type: str) -> tuple[nn.Module, int]:
        from torchvision import models

        if encoder_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        elif encoder_type == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet encoder: {encoder_type}")
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim

    def train(self, mode: bool = True) -> "MultiModalRewardModel":
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(
        self,
        *,
        scene_pixel_values: Tensor,
        wrist_pixel_values: Tensor | None = None,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        query_input_ids: Tensor | None = None,
        query_attention_mask: Tensor | None = None,
        text_query_mask: Tensor | None = None,
        proprioception: Tensor | None = None,
    ) -> Tensor:
        if self.use_patch_text_fusion and query_input_ids is not None:
            scene_feat = self._encode_scene_sequence_with_patch_text(
                scene_pixel_values=scene_pixel_values,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                text_query_mask=text_query_mask,
            )
            append_text_to_head = False
        else:
            scene_feat = self._encode_scene_sequence(scene_pixel_values)
            append_text_to_head = True
        if wrist_pixel_values is not None:
            wrist_feat = self.wrist_mlp(self._encode_image(wrist_pixel_values))
        else:
            wrist_dim = int(self.cfg.wrist_summary_dim)
            wrist_feat = scene_feat.new_zeros((scene_feat.shape[0], wrist_dim))

        features = [scene_feat, wrist_feat]
        if append_text_to_head and self.encoder_type in {"siglip2", "siglip", "clip"} and input_ids is not None:
            text_feat = self._encode_text(input_ids=input_ids, attention_mask=attention_mask)
            features.append(text_feat)
        if self.proprioception_mlp is not None:
            if proprioception is None:
                proprioception = scene_feat.new_zeros((scene_feat.shape[0], self.cfg.proprioception_dim))
            features.append(self.proprioception_mlp(proprioception.float()))

        logits = self.head(torch.cat(features, dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    def _encode_image(self, pixel_values: Tensor) -> Tensor:
        if self.encoder_type in {"siglip2", "siglip", "clip"}:
            outputs = self.encoder.get_image_features(pixel_values=pixel_values)
        else:
            outputs = self.encoder(pixel_values)
        features = self._pooled_feature_tensor(outputs, output_name="image")
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def _encode_image_patch_tokens(self, pixel_values: Tensor) -> Tensor:
        if self.encoder_type not in {"siglip2", "siglip", "clip"}:
            raise ValueError("Patch-token image encoding is only available for SigLIP/SigLIP2/CLIP encoders.")
        vision_model = getattr(self.encoder, "vision_model", None)
        if vision_model is None:
            raise TypeError(f"Encoder {type(self.encoder).__name__} does not expose `vision_model`.")
        outputs = vision_model(pixel_values=pixel_values)
        tokens = getattr(outputs, "last_hidden_state", None)
        if not isinstance(tokens, Tensor):
            raise TypeError(f"Expected vision_model output to expose last_hidden_state, got {type(outputs).__name__}.")
        projection = getattr(self.encoder, "visual_projection", None)
        if projection is not None:
            tokens = projection(tokens)
        return torch.nn.functional.normalize(tokens.float(), dim=-1)

    def _encode_scene_sequence(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.ndim == 4:
            return self._encode_image(pixel_values)
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected scene pixel values with 4 or 5 dims, got shape {tuple(pixel_values.shape)}")

        batch_size, window = pixel_values.shape[:2]
        flat = pixel_values.reshape(batch_size * window, *pixel_values.shape[2:])
        sequence_features = self._encode_image(flat).view(batch_size, window, -1)
        current = sequence_features[:, -1]
        if window <= 1 or int(self.cfg.scene_temporal_window) <= 1:
            return current

        oldest = sequence_features[:, 0]
        context_mean = sequence_features[:, :-1].mean(dim=1)
        delta_long = current - oldest
        return torch.cat([current, delta_long, context_mean], dim=-1)

    def _encode_scene_sequence_with_patch_text(
        self,
        *,
        scene_pixel_values: Tensor,
        query_input_ids: Tensor,
        query_attention_mask: Tensor | None,
        text_query_mask: Tensor | None,
    ) -> Tensor:
        if scene_pixel_values.ndim == 4:
            scene_pixel_values = scene_pixel_values[:, None]
        if scene_pixel_values.ndim != 5:
            raise ValueError(
                f"Expected scene pixel values with 4 or 5 dims, got shape {tuple(scene_pixel_values.shape)}"
            )
        batch_size, window = scene_pixel_values.shape[:2]
        flat_images = scene_pixel_values.reshape(batch_size * window, *scene_pixel_values.shape[2:])
        patch_tokens = self._encode_image_patch_tokens(flat_images)
        patch_tokens = patch_tokens.view(batch_size, window, patch_tokens.shape[-2], patch_tokens.shape[-1])

        if query_input_ids.ndim != 3:
            raise ValueError(f"Expected query_input_ids with shape (B, Q, L), got {tuple(query_input_ids.shape)}")
        query_count = query_input_ids.shape[1]
        flat_query_ids = query_input_ids.reshape(batch_size * query_count, query_input_ids.shape[-1])
        flat_query_mask = (
            query_attention_mask.reshape(batch_size * query_count, query_attention_mask.shape[-1])
            if query_attention_mask is not None
            else None
        )
        query_features = self._encode_text(input_ids=flat_query_ids, attention_mask=flat_query_mask)
        query_features = query_features.view(batch_size, query_count, -1)

        assert self.patch_key is not None
        assert self.patch_value is not None
        assert self.query_proj is not None
        patch_key = self.patch_key(patch_tokens)
        patch_value = self.patch_value(patch_tokens)
        query = self.query_proj(query_features)

        scale = math.sqrt(max(1, patch_key.shape[-1]))
        attention_logits = torch.einsum("bqa,bwna->bwqn", query, patch_key) / scale
        patch_weights = torch.softmax(attention_logits, dim=-1)
        query_frame_features = torch.einsum("bwqn,bwna->bwqa", patch_weights, patch_value)

        if text_query_mask is None:
            query_mask = query_frame_features.new_ones((batch_size, query_count), dtype=torch.bool)
        else:
            query_mask = text_query_mask.to(device=query_frame_features.device, dtype=torch.bool)
        query_mask_float = query_mask[:, None, :, None].float()
        denom = query_mask_float.sum(dim=2).clamp_min(1.0)
        frame_features = (query_frame_features * query_mask_float).sum(dim=2) / denom

        assert self.scene_summary_mlp is not None
        expected_window = max(1, int(self.cfg.scene_temporal_window))
        if window != expected_window:
            raise ValueError(f"Expected scene temporal window {expected_window}, got {window}.")
        return self.scene_summary_mlp(frame_features.reshape(batch_size, window * frame_features.shape[-1]))

    def _encode_text(self, *, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        kwargs: dict[str, Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.encoder.get_text_features(**kwargs)
        features = self._pooled_feature_tensor(outputs, output_name="text")
        return torch.nn.functional.normalize(features.float(), dim=-1)

    @staticmethod
    def _pooled_feature_tensor(outputs: Any, *, output_name: str) -> Tensor:
        """Return the pooled encoder feature from known HF output contracts."""
        if isinstance(outputs, Tensor):
            return outputs
        pooler_output = getattr(outputs, "pooler_output", None)
        if isinstance(pooler_output, Tensor):
            return pooler_output
        raise TypeError(
            f"Expected {output_name} encoder output to be a Tensor or have a Tensor pooler_output; "
            f"got {type(outputs).__name__}."
        )


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def load_reward_model_checkpoint(
    checkpoint_path: str | bytes | Any,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MultiModalRewardModel, RewardModelConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # :)
    cfg = RewardModelConfig.from_dict(checkpoint["model_config"])
    model = MultiModalRewardModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, checkpoint
