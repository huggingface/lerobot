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

from typing import Protocol

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF
from torch import Tensor, nn

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.vita.adaptation import VitaAdaptationModule, VitaAdaptationState, VitaFastWeights
from lerobot.rewards.vita.configuration_vita import VitaConfig
from lerobot.utils.constants import OBS_STR


class VitaBackbone(Protocol):
    """Backbone interface returning image and text embeddings."""

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        """Encode batch inputs into image and text features."""
        ...


class IdentityVitaBackbone(nn.Module):
    """Lightweight backbone for pre-encoded image/text features."""

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        image_features = batch[config.image_feature_key]
        text_features = batch[config.text_feature_key]
        return image_features.float(), text_features.float()


class DummyVitaBackbone(nn.Module):
    """Deterministic test backbone with fixed linear encoders."""

    def __init__(self, image_input_dim: int, text_input_dim: int, image_output_dim: int, text_output_dim: int):
        super().__init__()
        self.image_encoder = nn.Linear(image_input_dim, image_output_dim, bias=False)
        self.text_encoder = nn.Linear(text_input_dim, text_output_dim, bias=False)
        with torch.no_grad():
            self.image_encoder.weight.copy_(torch.eye(image_output_dim, image_input_dim))
            self.text_encoder.weight.copy_(torch.eye(text_output_dim, text_input_dim))

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        image = batch[config.image_feature_key].float()
        text = batch[config.text_feature_key].float()
        return self.image_encoder(image), self.text_encoder(text)


class ClipVitaBackbone(nn.Module):
    """CLIP backbone that encodes raw images and task text."""

    def __init__(self, model_name: str):
        super().__init__()
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VITA clip backbone requires `transformers`. Install with: pip install 'lerobot[transformers-dep]'."
            ) from exc

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    def _to_image_list(self, images: Tensor) -> list:
        if images.ndim not in {4, 5}:
            raise ValueError(f"Expected raw images of shape (B,C,H,W) or (B,T,C,H,W), got {tuple(images.shape)}.")
        if images.ndim == 4:
            images = images.unsqueeze(1)
        image_list: list = []
        for sample_images in images:  # (T, C, H, W)
            for image in sample_images:
                image_list.append(TF.to_pil_image(image.detach().cpu()))
        return image_list

    def _normalize_texts(self, texts, batch_size: int, seq_len: int) -> list[str]:
        if isinstance(texts, str):
            return [texts] * (batch_size * seq_len)
        if isinstance(texts, list):
            if len(texts) == batch_size and all(isinstance(t, str) for t in texts):
                return [t for t in texts for _ in range(seq_len)]
            if len(texts) == batch_size and all(isinstance(t, list) for t in texts):
                flat_texts: list[str] = []
                for per_sample in texts:
                    if len(per_sample) != seq_len or not all(isinstance(t, str) for t in per_sample):
                        raise ValueError("Expected nested text list of shape (B, T).")
                    flat_texts.extend(per_sample)
                return flat_texts
            if len(texts) == batch_size * seq_len and all(isinstance(t, str) for t in texts):
                return texts
        raise ValueError(
            "Unsupported text format for clip backbone. Expected str, list[str] (B or B*T), or list[list[str]] (B,T)."
        )

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        images = batch[config.raw_image_key]
        texts = batch[config.raw_text_key]
        if not isinstance(images, Tensor):
            raise ValueError(f"Expected tensor images at key '{config.raw_image_key}', got {type(images)}.")

        if images.ndim == 4:
            batch_size, seq_len = images.shape[0], 1
        elif images.ndim == 5:
            batch_size, seq_len = images.shape[0], images.shape[1]
        else:
            raise ValueError(f"Expected raw images rank 4/5, got rank {images.ndim}.")

        flat_images = self._to_image_list(images)
        flat_texts = self._normalize_texts(texts, batch_size=batch_size, seq_len=seq_len)

        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=flat_texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        image_features = image_features.view(batch_size, seq_len, -1)
        text_features = text_features.view(batch_size, seq_len, -1)
        return image_features, text_features


class OpenClipVitaBackbone(nn.Module):
    """OpenCLIP backbone aligned with the VITA paper setup."""

    def __init__(self, model_name: str, pretrained: str):
        super().__init__()
        try:
            import open_clip
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VITA openclip backbone requires `open_clip_torch`. Install with: pip install open_clip_torch."
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def _prepare_images(self, images: Tensor) -> tuple[Tensor, int, int]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected raw images of shape (B,C,H,W) or (B,T,C,H,W), got {tuple(images.shape)}.")

        batch_size, seq_len = images.shape[0], images.shape[1]
        flat_images = []
        for sample_images in images:
            for image in sample_images:
                pil_image = TF.to_pil_image(image.detach().cpu())
                flat_images.append(self.preprocess(pil_image))
        return torch.stack(flat_images, dim=0), batch_size, seq_len

    def _prepare_texts(self, texts, batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        if isinstance(texts, str):
            text_list = [texts] * (batch_size * seq_len)
        elif isinstance(texts, list):
            if len(texts) == batch_size and all(isinstance(t, str) for t in texts):
                text_list = [t for t in texts for _ in range(seq_len)]
            elif len(texts) == batch_size and all(isinstance(t, list) for t in texts):
                text_list = []
                for per_sample in texts:
                    if len(per_sample) != seq_len or not all(isinstance(t, str) for t in per_sample):
                        raise ValueError("Expected nested text list of shape (B, T).")
                    text_list.extend(per_sample)
            elif len(texts) == batch_size * seq_len and all(isinstance(t, str) for t in texts):
                text_list = texts
            else:
                raise ValueError("Unsupported list text format for openclip backbone.")
        else:
            raise ValueError("Unsupported text format for openclip backbone.")
        return self.tokenizer(text_list).to(device)

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        images = batch[config.raw_image_key]
        texts = batch[config.raw_text_key]
        if not isinstance(images, Tensor):
            raise ValueError(f"Expected tensor images at key '{config.raw_image_key}', got {type(images)}.")

        image_tensor, batch_size, seq_len = self._prepare_images(images)
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        text_tokens = self._prepare_texts(texts, batch_size=batch_size, seq_len=seq_len, device=device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1).view(batch_size, seq_len, -1)
        text_features = F.normalize(text_features, dim=-1).view(batch_size, seq_len, -1)
        return image_features, text_features


class VitaRewardModel(PreTrainedRewardModel):
    """VITA reward model with sequential test-time adaptation."""

    name = "vita"
    config_class = VitaConfig

    def __init__(self, config: VitaConfig, backbone: VitaBackbone | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config
        if backbone is not None:
            self.backbone = backbone
        elif config.backbone_type == "openclip":
            self.backbone = OpenClipVitaBackbone(
                model_name=config.openclip_model_name,
                pretrained=config.openclip_pretrained,
            )
        elif config.backbone_type == "clip":
            self.backbone = ClipVitaBackbone(model_name=config.clip_model_name)
        else:
            self.backbone = IdentityVitaBackbone()

        if isinstance(self.backbone, nn.Module) and config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.key_projection = nn.Linear(config.latent_dim, config.adaptation_dim)
        self.value_projection = nn.Linear(config.latent_dim, config.adaptation_dim)
        self.query_projection = nn.Linear(config.latent_dim, config.adaptation_dim)
        self.adaptation_module = VitaAdaptationModule(config.adaptation_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(config.adaptation_dim, config.reward_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.reward_hidden_dim, 1),
        )

        self._adaptation_state: VitaAdaptationState | None = None
        self.last_adaptation_losses: Tensor | None = None

    @property
    def adaptation_state(self) -> VitaAdaptationState | None:
        return self._adaptation_state

    def reset_adaptation_state(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        if batch_size is None:
            self._adaptation_state = None
            return
        state_device = device if device is not None else self.adaptation_module.base_weight.device
        self._adaptation_state = VitaAdaptationState.initialize(self.adaptation_module, batch_size, state_device)

    def reset(self) -> None:
        self.reset_adaptation_state()

    def _ensure_sequence(self, tensor: Tensor) -> Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(1)
        if tensor.ndim == 3:
            return tensor
        raise ValueError(f"Expected tensor rank 2 or 3, got rank {tensor.ndim}")

    def _prepare_episode_starts(self, episode_start: Tensor | None, batch_size: int, seq_len: int, device) -> Tensor:
        if episode_start is None:
            return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        starts = episode_start.to(device=device, dtype=torch.bool)
        if starts.ndim == 1:
            if starts.shape[0] != batch_size:
                raise ValueError("episode_start must match batch size.")
            starts = starts.unsqueeze(1).expand(batch_size, seq_len)
        elif starts.ndim == 2:
            if starts.shape != (batch_size, seq_len):
                raise ValueError("episode_start must have shape (B, T).")
        else:
            raise ValueError("episode_start must be rank 1 or 2.")
        return starts

    def _split_support_query(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        support_len = self.config.support_len
        query_len = self.config.query_len
        total_len = support_len + query_len
        if tensor.shape[1] < total_len:
            raise ValueError(
                f"Expected at least {total_len} timesteps for support/query split, got {tensor.shape[1]}."
            )
        support = tensor[:, :support_len]
        query = tensor[:, support_len : support_len + query_len]
        return support, query

    def _sample_dissimilar_windows(self, latent: Tensor) -> list[int]:
        seq_len = latent.shape[0]
        window = min(self.config.sampling_window_size, seq_len)
        starts = list(range(0, seq_len - window + 1, self.config.sampling_stride))
        if not starts:
            starts = [0]
        windows = torch.stack([latent[s : s + window].flatten() for s in starts], dim=0)
        distances = torch.cdist(windows, windows, p=2) ** 2
        scores = distances.sum(dim=1)
        k = min(self.config.sampling_num_windows, scores.shape[0])
        topk_idx = torch.topk(scores, k=k, largest=True).indices.tolist()
        return [starts[idx] for idx in topk_idx]

    def _select_training_sequence(
        self,
        image_features: Tensor,
        text_features: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.sampling_strategy == "contiguous":
            return image_features, text_features, target

        seq_len = image_features.shape[1]
        window = min(self.config.sampling_window_size, seq_len)
        sampled_image_per_batch: list[Tensor] = []
        sampled_text_per_batch: list[Tensor] = []
        sampled_target_per_batch: list[Tensor] = []

        for batch_idx in range(image_features.shape[0]):
            latent = torch.cat([image_features[batch_idx], text_features[batch_idx]], dim=-1)
            starts = self._sample_dissimilar_windows(latent)
            sampled_image = torch.cat([image_features[batch_idx, s : s + window] for s in starts], dim=0)
            sampled_text = torch.cat([text_features[batch_idx, s : s + window] for s in starts], dim=0)
            sampled_target = torch.cat([target[batch_idx, s : s + window] for s in starts], dim=0)
            sampled_image_per_batch.append(sampled_image)
            sampled_text_per_batch.append(sampled_text)
            sampled_target_per_batch.append(sampled_target)

        return (
            torch.stack(sampled_image_per_batch, dim=0),
            torch.stack(sampled_text_per_batch, dim=0),
            torch.stack(sampled_target_per_batch, dim=0),
        )

    def _adapt_fast_weights(
        self,
        image_sequence: Tensor,
        text_sequence: Tensor,
        fast_weights: VitaFastWeights,
        adaptation_lr: float,
    ) -> tuple[VitaFastWeights, Tensor]:
        adaptation_losses: list[Tensor] = []
        for timestep in range(image_sequence.shape[1]):
            latent = torch.cat([image_sequence[:, timestep], text_sequence[:, timestep]], dim=-1)
            keys = self.key_projection(latent)
            values = self.value_projection(latent)
            fast_weights, losses = self.adaptation_module.adaptation_step(
                keys=keys,
                values=values,
                fast_weights=fast_weights,
                adaptation_lr=adaptation_lr,
                first_order=self.config.first_order,
            )
            adaptation_losses.append(losses)

        return fast_weights, torch.stack(adaptation_losses, dim=1)

    def _predict_reward_sequence(
        self, image_sequence: Tensor, text_sequence: Tensor, fast_weights: VitaFastWeights
    ) -> Tensor:
        rewards: list[Tensor] = []
        for timestep in range(image_sequence.shape[1]):
            latent = torch.cat([image_sequence[:, timestep], text_sequence[:, timestep]], dim=-1)
            queries = self.query_projection(latent)
            adapted_queries = self.adaptation_module(queries, fast_weights=fast_weights)
            rewards.append(self.reward_head(adapted_queries).squeeze(-1))
        return torch.stack(rewards, dim=1)

    def _make_progress_targets(self, batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        # Paper supervision uses normalized temporal progress y_t = t / T.
        base = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device) / float(seq_len)
        return base.unsqueeze(0).expand(batch_size, -1)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        observation = batch.get(OBS_STR, batch)
        image_features, text_features = self.backbone.encode(observation, self.config)
        image_features = self._ensure_sequence(image_features)
        text_features = self._ensure_sequence(text_features)
        if image_features.shape[:2] != text_features.shape[:2]:
            raise ValueError("Image and text features must have compatible (B, T) dimensions.")

        batch_size, seq_len, _ = image_features.shape
        device = image_features.device

        if self._adaptation_state is None or self._adaptation_state.fast_weights.w1.shape[0] != batch_size:
            self.reset_adaptation_state(batch_size=batch_size, device=device)
        assert self._adaptation_state is not None

        episode_starts = self._prepare_episode_starts(
            observation.get("episode_start"), batch_size=batch_size, seq_len=seq_len, device=device
        )

        rewards: list[Tensor] = []
        adaptation_losses: list[Tensor] = []
        for timestep in range(seq_len):
            step_starts = episode_starts[:, timestep]
            self._adaptation_state.reset_indices(module=self.adaptation_module, mask=step_starts)

            latent = torch.cat([image_features[:, timestep], text_features[:, timestep]], dim=-1)
            keys = self.key_projection(latent)
            values = self.value_projection(latent)
            queries = self.query_projection(latent)

            updated_fast_weights, losses = self.adaptation_module.adaptation_step(
                keys=keys,
                values=values,
                fast_weights=self._adaptation_state.fast_weights,
                adaptation_lr=self.config.adaptation_lr,
                first_order=True,
            )
            self._adaptation_state.fast_weights = updated_fast_weights
            self._adaptation_state.num_updates = self._adaptation_state.num_updates + 1
            adaptation_losses.append(losses)

            adapted_queries = self.adaptation_module(queries, fast_weights=self._adaptation_state.fast_weights)
            rewards.append(self.reward_head(adapted_queries).squeeze(-1))

        reward_tensor = torch.stack(rewards, dim=1)
        self.last_adaptation_losses = torch.stack(adaptation_losses, dim=1)
        if seq_len == 1:
            return reward_tensor[:, 0]
        return reward_tensor

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.config.meta_enabled:
            raise NotImplementedError("Meta-learning is disabled in config; use compute_reward() for inference.")

        observation = batch.get(OBS_STR, batch)
        image_features, text_features = self.backbone.encode(observation, self.config)
        image_features = self._ensure_sequence(image_features)
        text_features = self._ensure_sequence(text_features)

        if self.config.force_temporal_progress_targets:
            target = self._make_progress_targets(
                batch_size=image_features.shape[0],
                seq_len=image_features.shape[1],
                device=image_features.device,
            )
        else:
            target = observation.get(self.config.target_reward_key)
            if target is None:
                raise ValueError(f"Missing target reward key '{self.config.target_reward_key}' in batch.")
            target = target.float()
            if target.ndim == 1:
                target = target.unsqueeze(1)
            if target.ndim == 3 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            if target.ndim != 2:
                raise ValueError(f"Target reward must be (B,T) or (B,T,1), got shape {tuple(target.shape)}.")

        train_image, train_text, train_target = self._select_training_sequence(
            image_features=image_features,
            text_features=text_features,
            target=target,
        )

        batch_size = image_features.shape[0]
        device = image_features.device
        state = VitaAdaptationState.initialize(
            module=self.adaptation_module,
            batch_size=batch_size,
            device=device,
        )
        fast_weights = state.fast_weights

        predictions = None
        adaptation_losses = None
        for _ in range(self.config.inner_steps):
            step_predictions: list[Tensor] = []
            step_losses: list[Tensor] = []
            for timestep in range(train_image.shape[1]):
                latent = torch.cat([train_image[:, timestep], train_text[:, timestep]], dim=-1)
                keys = self.key_projection(latent)
                values = self.value_projection(latent)
                queries = self.query_projection(latent)
                fast_weights, losses = self.adaptation_module.adaptation_step(
                    keys=keys,
                    values=values,
                    fast_weights=fast_weights,
                    adaptation_lr=self.config.inner_lr,
                    first_order=self.config.first_order,
                )
                adapted_queries = self.adaptation_module(queries, fast_weights=fast_weights)
                step_predictions.append(self.reward_head(adapted_queries).squeeze(-1))
                step_losses.append(losses)
            predictions = torch.stack(step_predictions, dim=1)
            adaptation_losses = torch.stack(step_losses, dim=1)

        assert predictions is not None
        assert adaptation_losses is not None
        outer_loss = F.mse_loss(predictions, train_target)
        inner_loss = adaptation_losses.mean()
        loss = (
            self.config.outer_loss_weight * outer_loss
            + self.config.self_supervised_loss_weight * inner_loss
        )

        metrics = {
            "loss": loss.detach(),
            "outer_loss": outer_loss.detach(),
            "inner_loss": inner_loss.detach(),
            "self_supervised_loss": inner_loss.detach(),
        }
        return loss, metrics
