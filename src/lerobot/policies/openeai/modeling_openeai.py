# Copyright 2026 The OpenEAI team and The HuggingFace Inc. team. All rights reserved.
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

"""OpenEAI-VLA Policy for LeRobot.

Architecture:
  Qwen3-VL backbone (frozen/trainable) -> feat-query -> cond_embed
  DiT action head (trainable): cross-attn + flow matching
  Multi-subset branching: encoder/decoder per dataset subset
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy, T
from .blocks import (
    DiTBlock,
    create_sinusoidal_pos_embedding,
    get_1d_sincos_pos_embed_from_grid,
    make_timm_attn_mask,
)
from .configuration_openeai import OpenEAIVLAConfig

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata


class DiTPolicyHead(nn.Module):
    """DiT action prediction head with subset-based branching.

    For each subset (dataset source), maintains:
      - state_encoder[state_subset]: state_dim -> hidden_dim
      - action_encoder[action_subset]: action_dim -> hidden_dim
      - action_decoder[action_subset]: hidden_dim -> action_dim
    """

    def __init__(
        self,
        config: OpenEAIVLAConfig,
        data_dim_info: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.config = config
        self.data_dim_info = deepcopy(data_dim_info) if data_dim_info else {}
        self._initialize()

    def build_multimodal_adapter(self):
        """Vision-language -> DiT hidden dim adapter."""
        self.cond_adapter = nn.Linear(self.config.qwen_dim, self.config.hidden_dim, bias=True)

    def build_unique_encoder_decoder(self):
        """Per-subset encoder/decoder branches."""
        state_encoders = {}
        action_encoders = {}
        action_decoders = {}

        for k, v in self.data_dim_info.items():
            state_encoders[k] = nn.Sequential(
                nn.Linear(v[0], self.config.hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=True),
            )
            action_encoders[k] = nn.Sequential(
                nn.Linear(v[1], self.config.hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=True),
            )
            action_decoders[k] = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.config.hidden_dim, v[1], bias=True),
            )

        self.state_encoders = nn.ModuleDict(state_encoders)
        self.action_encoders = nn.ModuleDict(action_encoders)
        self.action_decoders = nn.ModuleDict(action_decoders)

    def _initialize(self):
        self.build_multimodal_adapter()
        self.build_unique_encoder_decoder()

        # Action sequence positional embedding (time + state + chunk)
        self.x_pos_embed = nn.Parameter(torch.zeros(1, self.config.chunk_size + 2, self.config.hidden_dim))

        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.config.hidden_dim, self.config.num_heads, self.config.ff_ratio, 0.1)
                for _ in range(self.config.n_layers)
            ]
        )

        # Preallocate condition positional embedding with sufficient length.
        # Runtime forward will slice to actual cond_len.
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, self.config.feat_length, self.config.hidden_dim))

    def initialize_weights(self):
        """Initialize linear weights with Xavier and pos embeds with sin-cos."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # x_pos_embed: time + state + chunk
        x_emb = get_1d_sincos_pos_embed_from_grid(
            self.config.hidden_dim, np.arange(self.config.chunk_size + 2)
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_emb).float().unsqueeze(0))

        # cond_pos_embed: feat_length tokens
        cond_emb = get_1d_sincos_pos_embed_from_grid(
            self.config.hidden_dim, np.arange(self.config.feat_length)
        )
        self.cond_pos_embed.data.copy_(torch.from_numpy(cond_emb).float().unsqueeze(0))

    def forward(
        self,
        noisy_action: Tensor,
        action_mask: Tensor,
        state: Tensor,
        cond_embed: Tensor,
        cond_mask: Tensor,
        timestep: Tensor,
        subset: str,
    ) -> Tensor:
        """Forward pass through DiT action head.

        Args:
            noisy_action: [B, chunk_size, action_dim]
            action_mask: [B, chunk_size] bool, True = valid
            state: [B, state_dim] (padded to max_state_dim)
            cond_embed: [B, feat_length, hidden_dim] (Qwen hidden states for feat tokens)
            cond_mask: [B, feat_length]
            timestep: [B] float in [0, 1]
            subset: dataset subset name -> selects encoder/decoder

        Returns:
            v_t: predicted velocity [B, chunk_size, action_dim]
        """
        # Per-subset encoders
        state_embed = self.state_encoders[subset](state)
        noisy_action_embed = self.action_encoders[subset](noisy_action)

        # Adapt condition with sliced positional embedding
        cond_len = cond_embed.shape[1]
        if cond_len != self.cond_pos_embed.shape[1]:
            raise ValueError(
                f"cond_embed length ({cond_len}) must match cond_pos_embed length "
                f"({self.cond_pos_embed.shape[1]} = config.feat_length). "
                f"Make sure embed_cond returns feat_length tokens."
            )
        pos = self.cond_pos_embed.to(dtype=cond_embed.dtype)
        adapted_cond = self.cond_adapter(cond_embed) + pos

        # Time embedding
        time_embed = create_sinusoidal_pos_embedding(
            timestep,
            self.config.hidden_dim,
            min_period=4e-3,
            max_period=4.0,
        )
        time_token = time_embed.unsqueeze(1)  # [B, 1, D]

        # Ensure state_embed is 2D [B, hidden_dim] before unsqueezing
        if state_embed.dim() == 3:
            state_embed = state_embed.squeeze(1)

        # Concat: [time_token, state_embed, noisy_action_embed]
        x = torch.cat([time_token, state_embed.unsqueeze(1), noisy_action_embed], dim=1)
        x = x + self.x_pos_embed

        # Attention masks
        x_pad_mask = torch.cat([torch.ones_like(action_mask[:, :2]), action_mask], dim=1)
        x_timm_mask = make_timm_attn_mask(x_pad_mask)
        cond_mask_bool = cond_mask.bool() if cond_mask is not None else None

        # DiT blocks
        for layer in self.blocks:
            x = layer(x, adapted_cond, x_timm_mask, cond_mask_bool)

        # Extract action tokens and decode
        policy_out = x[:, -self.config.chunk_size :]
        policy_out = self.action_decoders[subset](policy_out)
        return policy_out


class OpenEAIModel(nn.Module):
    """OpenEAI-VLA model: Qwen3-VL backbone + DiT action head + flow matching."""

    config: OpenEAIVLAConfig

    def __init__(
        self,
        config: OpenEAIVLAConfig,
        data_dim_info: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.config = config

        # Qwen3-VL backbone — frozen by default
        backbone_dtype = config.backbone_torch_dtype
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.qwen_path, torch_dtype=backbone_dtype
        )
        self.qwen_model.requires_grad_(not config.freeze_backbone)
        # Keep backbone in eval mode when frozen so dropout doesn't perturb features.
        if config.freeze_backbone:
            self.qwen_model.eval()

        # Learnable feat-query tokens — match backbone dtype to avoid dtype promotion on cat.
        self.feat_query = nn.Parameter(
            torch.randn(1, config.feat_length, config.qwen_dim, dtype=backbone_dtype),
            requires_grad=True,
        )

        # DiT action head
        self.policy_head = DiTPolicyHead(config, data_dim_info=data_dim_info)
        self.policy_head.initialize_weights()

    def train(self, mode: bool = True):
        """Override to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.config.freeze_backbone:
            self.qwen_model.eval()
        return self

    def embed_cond(self, batch: dict, add_feat_query: bool = True):
        """Build condition embeddings from images + text + feat-query.

        Handles both image+text and text-only batches. The feat-query
        (``add_feat_query``) is appended for the DiT head via cross-attention.

        Returns:
            cond_embed: [B, lang_seq + feat_length, qwen_dim]
            full_attention_mask: [B, lang_seq + feat_length] mask for Qwen forward
            feat_mask: [B, feat_length] mask for DiT cross-attention (None if add_feat_query=False)
        """
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
        token_embeds = self.qwen_model.get_input_embeddings()(input_ids)

        # --- Image path: insert image features into placeholder positions ---
        if "pixel_values" in batch and "image_grid_thw" in batch:
            raw_image_feature = self.qwen_model.get_image_features(
                batch["pixel_values"], batch["image_grid_thw"]
            )
            pooler = raw_image_feature.pooler_output
            image_features = torch.cat(list(pooler), dim=0) if isinstance(pooler, (list, tuple)) else pooler
            image_features = image_features.to(dtype=token_embeds.dtype)
            image_mask, _ = self.qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=token_embeds, image_features=image_features
            )
            token_embeds = token_embeds.masked_scatter(image_mask, image_features)

        if add_feat_query:
            batch_size = token_embeds.shape[0]
            feat_q = self.feat_query.expand(batch_size, -1, -1)
            feat_mask = torch.ones(
                (batch_size, self.config.feat_length),
                dtype=torch.int64,
                device=attention_mask.device,
            )
            cond_embed = torch.cat([token_embeds, feat_q], dim=1)
            full_attention_mask = torch.cat([attention_mask, feat_mask], dim=1)
        else:
            cond_embed = token_embeds
            full_attention_mask = attention_mask
            feat_mask = None

        cond_embed = cond_embed.to(dtype=self.qwen_model.dtype)
        return cond_embed, full_attention_mask, feat_mask

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timestep t ~ Beta(alpha, beta) scaled to [offset, offset + scale]."""
        beta_dist = torch.distributions.Beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta
        )
        time_beta = beta_dist.sample((batch_size,)).to(device)
        return time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset

    def compute_vla_loss(
        self,
        batch: dict,
        subset: str = "__default__",
    ) -> tuple[Tensor, dict]:
        """Compute flow matching loss for training.

        Args:
            batch: processor pipeline output with keys:
                pixel_values, image_grid_thw, input_ids, attention_mask,
                state (normalized), action (normalized)
            subset: dataset subset name for encoder/decoder selection

        Returns:
            loss: scalar action loss tensor
            info: dict with logging values (loss, action_loss, raw_losses)
        """
        batch_size = batch[OBS_STATE].shape[0]
        device = batch[OBS_STATE].device

        # Condition embedding (Qwen3 + feat_query)
        input_embed, input_mask, feat_mask = self.embed_cond(batch, add_feat_query=True)

        # Ensure backbone input is in backbone dtype
        backbone_dtype = self.config.backbone_torch_dtype
        if input_embed.dtype != backbone_dtype:
            input_embed = input_embed.to(dtype=backbone_dtype)

        # DiT head runs in batch tensor's dtype (typically FP32) for stability.
        common_dtype = torch.float32
        if hasattr(batch.get(OBS_STATE), "dtype"):
            common_dtype = batch[OBS_STATE].dtype

        qwen_res = self.qwen_model.forward(
            inputs_embeds=input_embed,
            attention_mask=input_mask,
            labels=None,
            output_hidden_states=True,
        )
        # Take last-layer hidden states for feat-query tokens; cast back to DiT dtype.
        cond_embed = qwen_res.hidden_states[-1][:, -self.config.feat_length :].to(dtype=common_dtype)

        # Action targets
        action = batch[ACTION]  # [B, chunk_size, action_dim]
        action_mask = torch.ones((batch_size, self.config.chunk_size), dtype=torch.bool, device=device)

        # Flow matching: sample noise and time
        noise = torch.randn_like(action)
        t = self.sample_time(batch_size, device).to(dtype=action.dtype, device=device)
        t_expanded = t[:, None, None]
        noisy_action = t_expanded * noise + (1 - t_expanded) * action

        # Target velocity: u_t = noise - action
        u_t = noise - action

        # DiT forward
        v_t = self.policy_head.forward(
            noisy_action=noisy_action,
            action_mask=action_mask,
            state=batch[OBS_STATE],
            cond_embed=cond_embed,
            cond_mask=feat_mask,
            timestep=t,
            subset=subset,
        )

        # MSE loss (masked)
        raw_losses = F.mse_loss(v_t, u_t, reduction="none")
        masked_losses = raw_losses * action_mask.unsqueeze(-1)
        action_loss = masked_losses.mean()

        info = {
            "loss": action_loss.item(),
            "action_loss": action_loss.item(),
            "raw_losses": masked_losses.detach().cpu(),
        }
        return action_loss, info

    @torch.no_grad()
    def sample_action(
        self,
        cond_embed: Tensor,
        cond_mask: Tensor,
        state: Tensor,
        subset: str = "__default__",
        action_dim: int | None = None,
    ) -> Tensor:
        """Sample action via iterative flow matching inversion (Euler steps)."""
        batch_size = cond_embed.shape[0]
        device = cond_embed.device
        dtype = cond_embed.dtype

        if action_dim is None:
            action_dim = self.policy_head.data_dim_info.get(subset, (0, self.config.max_action_dim))[1]
            if action_dim == 0:  # In case self.policy_head.data_dim_info['subset'] is 0
                action_dim = self.config.max_action_dim

        # Initial noise
        noisy_action = torch.randn(
            (batch_size, self.config.chunk_size, action_dim),
            dtype=dtype,
            device=device,
        )

        dt = -1.0 / self.config.denoise_steps
        dt_tensor = torch.tensor(dt, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)
        action_mask = torch.ones((batch_size, self.config.chunk_size), dtype=torch.bool, device=device)

        # Euler integration from t=1 to t=0
        for _ in range(self.config.denoise_steps):
            v_t = self.policy_head.forward(
                noisy_action=noisy_action,
                action_mask=action_mask,
                state=state,
                cond_embed=cond_embed,
                cond_mask=cond_mask,
                timestep=time.expand(batch_size),
                subset=subset,
            )
            noisy_action = noisy_action + v_t * dt_tensor
            time = time + dt_tensor
        return noisy_action


class OpenEAIPolicy(PreTrainedPolicy):
    """OpenEAI-VLA policy wrapper for LeRobot training and inference."""

    config_class = OpenEAIVLAConfig
    name = "openeai"

    def __init__(
        self,
        config: OpenEAIVLAConfig,
        data_dim_info: dict[str, tuple[int, int]] | None = None,
        dataset_meta: LeRobotDatasetMetadata | None = None,
        **kwargs,
    ):
        require_package("transformers", extra="openeai")

        super().__init__(config)
        config.validate_features()
        self.config = config

        # Derive data_dim_info from dataset metadata if available.
        # Use actual dataset dimensions, not config's padded defaults.
        if data_dim_info is None and dataset_meta is not None and hasattr(dataset_meta, "stats"):
            stats = dataset_meta.stats
            if OBS_STATE in stats and "mean" in stats[OBS_STATE]:
                actual_state_dim = stats[OBS_STATE]["mean"].shape[-1]
            else:
                actual_state_dim = config.max_state_dim
            if ACTION in stats and "mean" in stats[ACTION]:
                actual_action_dim = stats[ACTION]["mean"].shape[-1]
            else:
                actual_action_dim = config.max_action_dim
            data_dim_info = {"__default__": (actual_state_dim, actual_action_dim)}
        elif data_dim_info is None:
            data_dim_info = {"__default__": (config.max_state_dim, config.max_action_dim)}

        # Sanity check: declared output dim should match data_dim_info default action dim
        declared_action_dim = config.output_features[ACTION].shape[0]
        default_action_dim = data_dim_info["__default__"][1]
        if declared_action_dim < default_action_dim:
            # data_dim_info uses padded max dim while output_features uses actual dim — OK
            pass
        elif declared_action_dim > default_action_dim:
            warnings.warn(
                f"output_features[ACTION] dim ({declared_action_dim}) is larger than "
                f"data_dim_info['__default__'] action dim ({default_action_dim}). "
                f"This may cause shape mismatch at inference.",
                stacklevel=2,
            )

        self.model = OpenEAIModel(config, data_dim_info=data_dim_info)
        self.model.to(config.device)

        if config.gradient_checkpointing:
            self.model.qwen_model.gradient_checkpointing_enable()

        self.reset()

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        """Return parameters for optimizer.

        By default, only DiT head and feat_query are trainable.
        Qwen3 backbone is frozen unless config.freeze_backbone=False.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def reset(self):
        """Reset internal state for rollout."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action to execute in the environment.

        Note: assumes batch size is consistent across consecutive calls
        within an action chunk.
        """
        self.eval()

        if len(self._queues.get(ACTION, [])) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions given observations."""
        self.eval()

        common_dtype = torch.float32
        if hasattr(batch.get(OBS_STATE), "dtype"):
            common_dtype = batch[OBS_STATE].dtype

        # Condition embedding
        input_embed, input_mask, feat_mask = self.model.embed_cond(batch, add_feat_query=True)
        qwen_res = self.model.qwen_model.forward(
            inputs_embeds=input_embed,
            attention_mask=input_mask,
            labels=None,
            output_hidden_states=True,
        )
        cond_embed = qwen_res.hidden_states[-1][:, -self.config.feat_length :].to(dtype=common_dtype)

        actions = self.model.sample_action(
            cond_embed=cond_embed,
            cond_mask=feat_mask,
            state=batch[OBS_STATE],
            subset=batch.get("subset", "__default__"),
            action_dim=self.config.output_features[ACTION].shape[0],
        )

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training forward pass: compute loss."""
        self.train()

        subset = batch.get("subset", "__default__")
        loss, info = self.model.compute_vla_loss(batch, subset=subset)

        if reduction == "none":
            per_sample_loss = info["raw_losses"].mean(dim=(1, 2))
            return per_sample_loss, {"loss": per_sample_loss.mean().item()}
        return loss, {"loss": loss.item()}

    # ---- pretrained loading ----

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Load pretrained OpenEAI-VLA weights.

        Behavior:
            - If user explicitly provides ``data_dim_info``, use it as-is.
            - Else, infer dims from the checkpoint and (optionally) ``dataset_meta``.
                * Both available, dims agree → full load (typical inference / resume).
                * Both available, dims differ → finetune: drop encoder/decoder
                weights from checkpoint and re-initialize for dataset dims.
                * Only checkpoint available → use checkpoint dims (resume).
                * Only dataset_meta available → use dataset dims; encoder/decoder
                weights are still loaded if present and shapes match.

        Note:
            ``strict`` defaults to False to tolerate Qwen3-VL tied embedding weights.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_path = str(pretrained_name_or_path)

        # Load raw checkpoint state_dict.
        raw_sd: dict[str, torch.Tensor] | None = None
        raw_sd_error: Exception | None = None
        try:
            raw_sd = cls._load_raw_state_dict(
                model_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
        except (FileNotFoundError, OSError, KeyError, json.JSONDecodeError) as e:
            raw_sd_error = e

        # Decide data_dim_info.
        init_kwargs = {k: v for k, v in kwargs.items() if k in ("data_dim_info", "dataset_meta")}
        user_data_dim_info = init_kwargs.get("data_dim_info")
        user_dataset_meta = init_kwargs.get("dataset_meta")

        encoder_decoder_mismatch = False

        if user_data_dim_info is None:
            inferred_from_ckpt = (
                cls._infer_data_dim_info_from_state_dict(raw_sd) if raw_sd is not None else {}
            )
            inferred_from_meta = (
                cls._infer_data_dim_info_from_dataset_meta(user_dataset_meta, config)
                if user_dataset_meta is not None
                else None
            )

            if inferred_from_ckpt and inferred_from_meta:
                ckpt_default = inferred_from_ckpt.get("__default__")
                meta_default = inferred_from_meta.get("__default__")
                if ckpt_default is not None and meta_default is not None and ckpt_default != meta_default:
                    # Finetune scenario: dataset dims differ from checkpoint dims.
                    init_kwargs["data_dim_info"] = inferred_from_meta
                    init_kwargs.pop("dataset_meta", None)
                    encoder_decoder_mismatch = True
                    warnings.warn(
                        f"[OpenEAIPolicy] Checkpoint dims {ckpt_default} differ from "
                        f"dataset dims {meta_default}. Treating as finetune: "
                        f"encoder/decoder weights will be re-initialized for dataset dims.",
                        stacklevel=2,
                    )
                else:
                    # Inference / pretrain-resume: dims agree, full load.
                    init_kwargs["data_dim_info"] = inferred_from_ckpt
                    init_kwargs.pop("dataset_meta", None)
            elif inferred_from_ckpt:
                init_kwargs["data_dim_info"] = inferred_from_ckpt
                init_kwargs.pop("dataset_meta", None)
            # else: leave dataset_meta in init_kwargs (or fall back to config defaults).

        model = cls(config, **init_kwargs)

        # Load weights into model.
        if raw_sd is None:
            msg = (
                f"[OpenEAIPolicy] Failed to load weights from {pretrained_name_or_path}. "
                f"Last error: {raw_sd_error!r}"
            )
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg + " Model initialized with random weights.", stacklevel=2)
            return model

        remapped = {}
        for k, v in raw_sd.items():
            new_k = k if k.startswith("model.") else "model." + k
            remapped[new_k] = v

        if encoder_decoder_mismatch:
            dropped = []
            for k in list(remapped.keys()):
                stripped = k[len("model.") :] if k.startswith("model.") else k
                if (
                    stripped.startswith("policy_head.state_encoders.")
                    or stripped.startswith("policy_head.action_encoders.")
                    or stripped.startswith("policy_head.action_decoders.")
                ):
                    del remapped[k]
                    dropped.append(k)
            if dropped:
                warnings.warn(
                    f"[OpenEAIPolicy] Finetune mode: dropped {len(dropped)} encoder/decoder "
                    f"weights from checkpoint to reinitialize for dataset dimensions. "
                    f"This is expected when loading a model pretrained on different datasets.",
                    stacklevel=2,
                )

        try:
            missing, unexpected = model.load_state_dict(remapped, strict=False)
        except RuntimeError as e:
            if strict:
                raise
            warnings.warn(
                f"[OpenEAIPolicy] Non-strict load failed: {e}. Model partially initialized.",
                stacklevel=2,
            )
            return model

        tolerable_missing = {
            k
            for k in missing
            if k.endswith("language_model.embed_tokens.weight") or k.endswith("lm_head.weight")
        }
        real_missing = [k for k in missing if k not in tolerable_missing]

        if encoder_decoder_mismatch:
            real_missing = [
                k
                for k in real_missing
                if not (
                    ".policy_head.state_encoders." in k
                    or ".policy_head.action_encoders." in k
                    or ".policy_head.action_decoders." in k
                )
            ]

        if real_missing or unexpected:
            msg = (
                f"[OpenEAIPolicy] State dict load mismatches:\n"
                f"  Missing keys: {real_missing}\n"
                f"  Unexpected keys: {unexpected}"
            )
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, stacklevel=2)

        return model

    @staticmethod
    def _infer_data_dim_info_from_dataset_meta(
        dataset_meta, config: OpenEAIVLAConfig
    ) -> dict[str, tuple[int, int]] | None:
        """Infer per-subset (state_dim, action_dim) from dataset metadata stats.

        Mirrors the dim-inference logic in OpenEAIPolicy.__init__ so that
        from_pretrained can compare against checkpoint dims before deciding
        whether to drop encoder/decoder weights.
        """
        if dataset_meta is None or not hasattr(dataset_meta, "stats"):
            return None
        stats = dataset_meta.stats
        if OBS_STATE in stats and "mean" in stats[OBS_STATE]:
            state_dim = stats[OBS_STATE]["mean"].shape[-1]
        else:
            state_dim = config.max_state_dim
        if ACTION in stats and "mean" in stats[ACTION]:
            action_dim = stats[ACTION]["mean"].shape[-1]
        else:
            action_dim = config.max_action_dim
        return {"__default__": (state_dim, action_dim)}

    @classmethod
    def _load_raw_state_dict(
        cls,
        model_path: str,
        *,
        cache_dir=None,
        force_download=False,
        resume_download=None,
        proxies=None,
        token=None,
        local_files_only=False,
    ) -> dict[str, torch.Tensor]:
        """Load raw state dict from single-file or sharded safetensors.

        Returns the raw state_dict (without "model." prefix remapping).
        """
        # Try single-file first
        if Path(model_path).is_dir():
            single_path = Path(model_path) / "model.safetensors"
            if single_path.exists():
                return load_file(single_path)
            # Fall through to sharded
        else:
            try:
                weight_file = cached_file(
                    model_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    local_files_only=local_files_only,
                )
                return load_file(weight_file)
            except (FileNotFoundError, OSError):
                pass  # try sharded

        # Sharded fallback
        if Path(model_path).is_dir():
            index_path = Path(model_path) / "model.safetensors.index.json"
        else:
            index_path = Path(
                cached_file(
                    model_path,
                    "model.safetensors.index.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    local_files_only=local_files_only,
                )
            )

        if not index_path.exists():
            raise FileNotFoundError(f"No safetensors found at {model_path}")

        with open(index_path) as f:
            index = json.load(f)

        base_dir = index_path.parent
        merged: dict[str, torch.Tensor] = {}
        shard_files = set(index["weight_map"].values())
        for shard_file in shard_files:
            shard_path = base_dir / shard_file
            if not shard_path.exists():
                # For HF Hub, shard files need to be downloaded individually
                shard_path = Path(
                    cached_file(
                        model_path,
                        shard_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                )
            merged.update(load_file(shard_path))

        return merged

    @staticmethod
    def _infer_data_dim_info_from_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, tuple[int, int]]:
        """Infer per-subset (state_dim, action_dim) from checkpoint shapes.

        Inspects state_encoders.{subset}.0.weight (state_dim) and
        action_decoders.{subset}.2.weight (action_dim).
        """
        info: dict[str, tuple[int, int]] = {}

        # Strip optional "model." prefix when matching
        def strip_prefix(k: str) -> str:
            return k[len("model.") :] if k.startswith("model.") else k

        state_dims: dict[str, int] = {}
        action_dims: dict[str, int] = {}

        for raw_k, v in state_dict.items():
            k = strip_prefix(raw_k)
            # state_encoders.<subset>.0.weight: shape (hidden_dim, state_dim)
            if k.startswith("policy_head.state_encoders.") and k.endswith(".0.weight"):
                subset = k[len("policy_head.state_encoders.") : -len(".0.weight")]
                state_dims[subset] = v.shape[1]
            # action_decoders.<subset>.2.weight: shape (action_dim, hidden_dim)
            elif k.startswith("policy_head.action_decoders.") and k.endswith(".2.weight"):
                subset = k[len("policy_head.action_decoders.") : -len(".2.weight")]
                action_dims[subset] = v.shape[0]

        for subset in set(state_dims) | set(action_dims):
            s = state_dims.get(subset)
            a = action_dims.get(subset)
            if s is not None and a is not None:
                info[subset] = (s, a)

        return info
