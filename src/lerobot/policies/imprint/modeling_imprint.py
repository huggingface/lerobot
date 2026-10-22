#!/usr/bin/env python

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

"""Bounded IMPRINT policy.

The image stem is a small conv stand-in for the paper's frozen Wan 2.2 video
tokenizer and DINOv3 frame encoder. This module does not download those models.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import PolicyFeature
from lerobot.optim.optimizers import OptimizerParams
from lerobot.policies.common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_imprint import ImprintConfig
from .memory import (
    FullAttentionMemory,
    ImprintDecoder,
    MemoryModulation,
    RecurrentMemory,
    RecurrentState,
    memory_pretraining_loss,
)

LANG_EMBEDDING = "observation.language.embedding"
EPISODE_PROGRESS = "episode_progress"
QUERY_STEPS = "query_steps"


class TinyFrameEncoder(nn.Module):
    """One token grid per frame. This is not Wan 2.2 and not DINOv3."""

    def __init__(self, in_channels: int, visual_dim: int, tokens_per_view: int) -> None:
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.visual_dim = visual_dim
        self.conv = nn.Conv2d(in_channels, visual_dim, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(visual_dim, tokens_per_view * visual_dim)

    def forward(self, images: Tensor) -> Tensor:
        hidden = F.relu(self.conv(images))
        pooled = hidden.mean(dim=(2, 3))
        return self.proj(pooled).reshape(images.shape[0], self.tokens_per_view, self.visual_dim)


class ActionExpert(nn.Module):
    """Small flow-matching expert. Memory enters as cross-attention plus FiLM."""

    def __init__(self, config: ImprintConfig, action_dim: int) -> None:
        super().__init__()
        dim = config.memory_dim
        self.config = config
        self.in_proj = nn.Linear(action_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(
            [_ExpertLayer(dim, config.num_heads) for _ in range(config.expert_layers)]
        )
        self.out_proj = nn.Linear(dim, action_dim)

    def forward(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        obs_token: Tensor,
        memory: Tensor | None,
        memory_valid: Tensor | None,
        *,
        use_memory: bool,
    ) -> Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.memory_dim,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=noisy_actions.device,
        ).to(dtype=noisy_actions.dtype)
        tokens = self.in_proj(noisy_actions) + self.time_mlp(time_emb).unsqueeze(1)
        hidden = torch.cat([obs_token, tokens], dim=1)
        for layer in self.layers:
            hidden = layer(hidden, memory, memory_valid, use_memory=use_memory)
        return self.out_proj(hidden[:, 1:])


class _ExpertLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.modulation = MemoryModulation(dim, num_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, 4 * dim)
        self.ffn_out = nn.Linear(4 * dim, dim)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor | None,
        memory_valid: Tensor | None,
        *,
        use_memory: bool,
    ) -> Tensor:
        normalized = self.self_norm(hidden)
        attended, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        hidden = hidden + attended
        hidden = self.modulation(hidden, memory, memory_valid, enabled=use_memory)
        return hidden + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(hidden))))


class ImprintPolicy(PreTrainedPolicy):
    """IMPRINT multi-stream latent memory with a flow-matching action expert.

    ``select_action`` keeps the recurrent memory and the action queue for one
    episode. ``predict_action_chunk`` recomputes memory from the batch and does
    not advance that episode state. Call ``reset`` at every episode boundary.
    """

    config_class = ImprintConfig
    name = "imprint"
    config: ImprintConfig

    def __init__(self, config: ImprintConfig, **kwargs: object) -> None:
        del kwargs
        super().__init__(config)
        config.validate_features()
        self.config = config
        if config.training_stage == "memory" and config.align_dim != config.visual_dim:
            raise ValueError(
                "Stage-1 IMPRINT reconstructs frame tokens, so align_dim must equal "
                f"visual_dim. Got align_dim={config.align_dim} and visual_dim={config.visual_dim}. "
                "The paper align width is 768 because its targets are DINOv3 features."
            )
        image_shape = next(iter(config.image_features.values())).shape
        self.image_keys = tuple(sorted(config.image_features))
        self.num_views = len(self.image_keys)
        self.tokens_per_frame = self.num_views * config.tokens_per_view
        self.action_dim = _feature_dim(config.action_feature, "action")
        self.state_dim = _feature_dim(config.robot_state_feature, OBS_STATE)
        channels = int(image_shape[0])
        self.frame_encoder = TinyFrameEncoder(channels, config.visual_dim, config.tokens_per_view)
        if config.visual_dim == config.chunk_feature_dim:
            self.to_chunk: nn.Module = nn.Identity()
        else:
            self.to_chunk = nn.Linear(config.visual_dim, config.chunk_feature_dim)
        if config.visual_dim == config.recent_feature_dim:
            self.to_recent: nn.Module = nn.Identity()
        else:
            self.to_recent = nn.Linear(config.visual_dim, config.recent_feature_dim)
        if config.use_memory and config.encoder == "recurrent":
            self.memory: nn.Module | None = RecurrentMemory(config, self.tokens_per_frame)
        elif config.use_memory:
            self.memory = FullAttentionMemory(config)
        else:
            self.memory = None
        self.obs_visual = nn.Linear(config.visual_dim, config.memory_dim)
        self.obs_state = nn.Linear(self.state_dim, config.memory_dim)
        if config.training_stage == "joint":
            self.expert: ActionExpert | None = ActionExpert(config, self.action_dim)
            self.decoder: ImprintDecoder | None = None
        else:
            self.expert = None
            self.decoder = ImprintDecoder(config, num_patches=self.tokens_per_frame)
        self._action_queue: deque[Tensor] = deque()
        self._hist_visual: Tensor | None = None
        self._hist_state: Tensor | None = None
        self._hist_causal: Tensor | None = None
        self._committed_steps = 0
        self._last_action: Tensor | None = None
        self._recurrent_state: RecurrentState | None = None
        self.reset()

    def get_optim_params(self) -> OptimizerParams:
        modules: list[nn.Module] = [self.frame_encoder, self.to_chunk, self.to_recent]
        if self.config.training_stage == "memory":
            if self.memory is not None:
                modules.append(self.memory)
            if self.decoder is not None:
                modules.append(self.decoder)
        else:
            modules.extend([self.obs_visual, self.obs_state])
            if self.config.use_memory and self.memory is not None:
                modules.append(self.memory)
            if self.expert is not None:
                modules.append(self.expert)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    def reset(self) -> None:
        """Clear the action queue and the episode memory."""
        self._action_queue = deque()
        self._hist_visual = None
        self._hist_state = None
        self._hist_causal = None
        self._committed_steps = 0
        self._last_action = None
        self._recurrent_state = None

    def finalized_chunk_counts(self) -> Tensor:
        """Accepted recurrent chunks per batch row. Empty before the first observation."""
        if self._recurrent_state is None:
            return torch.zeros(0, dtype=torch.long)
        return self._recurrent_state.chunk_count.detach().cpu()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if self.config.training_stage == "memory":
            return self._memory_forward(batch, reduction)
        return self._joint_forward(batch, reduction)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Recompute a chunk from this batch. Does not read or write episode memory."""
        self.eval()
        if self.expert is None:
            raise RuntimeError(
                "training_stage='memory' has no action expert. Use training_stage='joint' for inference."
            )
        visual = self._encode_visual(batch)
        state = _with_time(batch[OBS_STATE], name=OBS_STATE)
        _check_aligned(visual, state)
        causal = self._causal_from_batch(batch, visual.shape[1], state.shape[0], state.device, state.dtype)
        obs = self._obs_token(visual[:, -1], state[:, -1])
        memory, valid = self._encode_memory(visual, causal, state, _language(batch, self.config.lang_dim))
        return self._sample_actions(obs, memory, valid)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action and update episode memory with this observation."""
        self.eval()
        if self.expert is None:
            raise RuntimeError(
                "training_stage='memory' has no action expert. Use training_stage='joint' for inference."
            )
        visual = self._encode_visual(batch)
        state = _with_time(batch[OBS_STATE], name=OBS_STATE)
        _check_aligned(visual, state)
        if visual.shape[1] != 1 or state.shape[1] != 1:
            raise ValueError(
                "select_action expects one environment step. "
                f"Got visual length {visual.shape[1]} and state length {state.shape[1]}."
            )
        batch_size = visual.shape[0]
        self._check_rollout_batch(batch_size, visual.device)
        if self._last_action is None:
            causal = visual.new_zeros(batch_size, self.action_dim)
        else:
            causal = self._last_action
        self._append_step(visual[:, 0], state[:, 0], causal)
        self._commit_chunks()
        if len(self._action_queue) == 0:
            assert self._hist_visual is not None
            assert self._hist_state is not None
            obs = self._obs_token(self._hist_visual[:, -1], self._hist_state[:, -1])
            memory, valid = self._memory_from_rollout(_language(batch, self.config.lang_dim))
            actions = self._sample_actions(obs, memory, valid)
            self._action_queue.extend(actions[:, : self.config.n_action_steps].transpose(0, 1))
        action = self._action_queue.popleft()
        self._last_action = action.detach()
        return action

    def _joint_forward(self, batch: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict[str, float]]:
        if self.expert is None:
            raise RuntimeError("Joint IMPRINT requires an action expert.")
        visual = self._encode_visual(batch)
        state = _with_time(batch[OBS_STATE], name=OBS_STATE)
        _check_aligned(visual, state)
        target, causal = self._split_joint_action(batch, visual.shape[1])
        obs = self._obs_token(visual[:, -1], state[:, -1])
        memory, valid = self._encode_memory(visual, causal, state, _language(batch, self.config.lang_dim))
        time = sample_time_beta(
            target.shape[0],
            target.device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )
        noise = sample_noise(target.shape, target.device)
        noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target
        velocity = self.expert(noisy, time, obs, memory, valid, use_memory=self.config.use_memory)
        per_sample = (velocity - (noise - target)).square().flatten(1).mean(dim=-1)
        logs = {"loss_action": float(per_sample.mean().detach())}
        if reduction == "none":
            return per_sample, logs
        if reduction == "mean":
            return per_sample.mean(), logs
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}.")

    def _memory_forward(self, batch: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict[str, float]]:
        if self.decoder is None or self.memory is None:
            raise RuntimeError("Stage-1 IMPRINT requires the memory encoder and decoder.")
        if EPISODE_PROGRESS not in batch:
            raise ValueError(
                "Stage-1 IMPRINT requires episode_progress with shape (B,), "
                "the current step divided by the last demonstration step."
            )
        visual = self._encode_visual(batch)
        state = _with_time(batch[OBS_STATE], name=OBS_STATE)
        _check_aligned(visual, state)
        causal = self._causal_from_batch(batch, visual.shape[1], state.shape[0], state.device, state.dtype)
        memory, _valid = self._encode_memory(visual, causal, state, _language(batch, self.config.lang_dim))
        if memory is None:
            raise RuntimeError("Stage-1 IMPRINT requires use_memory=True.")
        steps = self._query_steps(batch, visual.shape[1], state.shape[0], state.device)
        prediction = self.decoder(memory, steps)
        targets = {
            "align": _gather_time(visual, steps).detach(),
            "action": _gather_time(causal, steps).detach(),
            "proprio": _gather_time(state, steps).detach(),
            "progress": batch[EPISODE_PROGRESS].to(dtype=visual.dtype, device=visual.device),
        }
        return memory_pretraining_loss(prediction, targets, reduction)

    def _sample_actions(self, obs: Tensor, memory: Tensor | None, valid: Tensor | None) -> Tensor:
        if self.expert is None:
            raise RuntimeError("Cannot sample actions without an action expert.")
        expert = self.expert
        batch = obs.shape[0]
        noise = sample_noise((batch, self.config.chunk_size, self.action_dim), obs.device)

        def denoise(noisy: Tensor, time: Tensor) -> Tensor:
            return expert(noisy, time, obs, memory, valid, use_memory=self.config.use_memory)

        return euler_integrate(denoise, noise, self.config.num_inference_steps)

    def _encode_visual(self, batch: dict[str, Tensor]) -> Tensor:
        encoded: list[Tensor] = []
        batch_size: int | None = None
        time_steps: int | None = None
        for key in self.image_keys:
            if key not in batch:
                raise ValueError(f"IMPRINT batch is missing visual key {key!r}.")
            images = _with_time(batch[key], name=key)
            if images.ndim != 5:
                raise ValueError(f"{key} must be (B, T, C, H, W) or (B, C, H, W), got {tuple(images.shape)}.")
            current_batch, current_time = int(images.shape[0]), int(images.shape[1])
            if batch_size is None:
                batch_size = current_batch
                time_steps = current_time
            elif current_batch != batch_size or current_time != time_steps:
                raise ValueError(
                    f"{key} has batch {current_batch} and time {current_time}; "
                    f"expected batch {batch_size} and time {time_steps}."
                )
            flat = images.reshape(current_batch * current_time, *images.shape[2:])
            tokens = self.frame_encoder(flat)
            tokens = tokens.reshape(current_batch, current_time, self.config.tokens_per_view, -1)
            encoded.append(tokens)
        if not encoded or batch_size is None or time_steps is None:
            raise ValueError("IMPRINT requires at least one visual observation.")
        return torch.cat(encoded, dim=2)

    def _encode_memory(
        self, visual: Tensor, causal: Tensor, state: Tensor, lang: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.use_memory or self.memory is None:
            return None, None
        chunk_v, recent_v, chunk_a, recent_a, chunk_p, recent_p = _split_history(
            self.to_chunk(visual),
            self.to_recent(visual),
            causal,
            state,
            self.config.chunk_frames,
        )
        if isinstance(self.memory, RecurrentMemory):
            tokens, _state = self.memory.forward_sequence(
                chunk_v, chunk_a, chunk_p, recent_v, recent_a, recent_p, lang
            )
        else:
            if not isinstance(self.memory, FullAttentionMemory):
                raise TypeError(f"Unsupported memory module {type(self.memory).__name__}.")
            tokens = self.memory(
                chunk_v.flatten(1, 2),
                recent_v,
                _merge_steps(chunk_a, recent_a),
                _merge_steps(chunk_p, recent_p),
                lang,
            )
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, valid

    def _memory_from_rollout(self, lang: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
        if self._hist_visual is None or self._hist_state is None or self._hist_causal is None:
            raise RuntimeError("Rollout memory was read before any observation.")
        if not self.config.use_memory or self.memory is None:
            return None, None
        if isinstance(self.memory, RecurrentMemory):
            recent_v, recent_a, recent_p = self._recent_tail()
            if self._recurrent_state is None:
                self._recurrent_state = self.memory.init_state(
                    self._hist_visual.shape[0], self._hist_visual.device, self._hist_visual.dtype
                )
            tokens = self.memory.read(self._recurrent_state, recent_v, recent_a, recent_p, lang)
        else:
            tokens, valid = self._encode_memory(self._hist_visual, self._hist_causal, self._hist_state, lang)
            return tokens, valid
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, valid

    def _append_step(self, visual: Tensor, state: Tensor, causal: Tensor) -> None:
        visual = visual.detach()
        state = state.detach()
        causal = causal.detach()
        if self._hist_visual is None or self._hist_state is None or self._hist_causal is None:
            self._hist_visual = visual.unsqueeze(1)
            self._hist_state = state.unsqueeze(1)
            self._hist_causal = causal.unsqueeze(1)
            return
        self._hist_visual = torch.cat([self._hist_visual, visual.unsqueeze(1)], dim=1)
        self._hist_state = torch.cat([self._hist_state, state.unsqueeze(1)], dim=1)
        self._hist_causal = torch.cat([self._hist_causal, causal.unsqueeze(1)], dim=1)

    def _commit_chunks(self) -> None:
        if not isinstance(self.memory, RecurrentMemory):
            return
        if self._hist_visual is None or self._hist_causal is None or self._hist_state is None:
            return
        if self._recurrent_state is None:
            self._recurrent_state = self.memory.init_state(
                self._hist_visual.shape[0], self._hist_visual.device, self._hist_visual.dtype
            )
        frames = self.config.chunk_frames
        while self._hist_visual.shape[1] - self._committed_steps >= frames:
            start = self._committed_steps
            end = start + frames
            visual = self.to_chunk(self._hist_visual[:, start:end]).mean(dim=1)
            action = self._hist_causal[:, start:end]
            proprio = self._hist_state[:, start:end]
            self._recurrent_state = self.memory.update_incremental(
                self._recurrent_state, visual, action, proprio
            )
            self._committed_steps = end

    def _recent_tail(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._hist_visual is None or self._hist_causal is None or self._hist_state is None:
            raise RuntimeError("Recent memory was read before any observation.")
        visual = self.to_recent(self._hist_visual[:, self._committed_steps :])
        batch = visual.shape[0]
        recent_v = visual.reshape(batch, -1, visual.shape[-1])
        recent_a = self._hist_causal[:, self._committed_steps :]
        recent_p = self._hist_state[:, self._committed_steps :]
        return recent_v, recent_a, recent_p

    def _obs_token(self, visual: Tensor, state: Tensor) -> Tensor:
        pooled = visual.mean(dim=1)
        return (self.obs_visual(pooled) + self.obs_state(state)).unsqueeze(1)

    def _split_joint_action(self, batch: dict[str, Tensor], time_steps: int) -> tuple[Tensor, Tensor]:
        action = _action_tensor(batch)
        if action is None:
            raise ValueError("Joint IMPRINT training requires an action tensor.")
        if action.ndim != 3:
            raise ValueError(f"action must have shape (B, T, A), got {tuple(action.shape)}.")
        history = self.config.history_length
        expected = history + self.config.chunk_size
        if action.shape[1] != expected:
            raise ValueError(
                "Joint IMPRINT expects action shape "
                f"(B, history_length + chunk_size, A) = (B, {expected}, {self.action_dim}). "
                "The first history_length steps are causal actions for the observation window. "
                "The last chunk_size steps are the flow-matching target. "
                f"Got {tuple(action.shape)}."
            )
        if time_steps != history:
            raise ValueError(
                f"Joint IMPRINT expects {history} observation steps, got {time_steps}. "
                "That window is history_length, ending at the current frame."
            )
        return action[:, history:], action[:, :history]

    def _causal_from_batch(
        self,
        batch: dict[str, Tensor],
        time_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.config.training_stage == "memory":
            action = _action_tensor(batch)
            if action is None:
                raise ValueError("Stage-1 IMPRINT requires action history aligned with the observations.")
            if action.ndim != 3 or action.shape[1] != time_steps:
                raise ValueError(
                    "Stage-1 IMPRINT expects action shape "
                    f"(B, history_length, A) = (B, {self.config.history_length}, {self.action_dim}). "
                    f"Got {tuple(action.shape)}."
                )
            return action
        joint_length = self.config.history_length + self.config.chunk_size
        action = _action_tensor(batch)
        if action is not None and action.ndim == 3 and action.shape[1] == joint_length:
            return action[:, : self.config.history_length]
        return torch.zeros(batch_size, time_steps, self.action_dim, device=device, dtype=dtype)

    def _query_steps(
        self, batch: dict[str, Tensor], time_steps: int, batch_size: int, device: torch.device
    ) -> Tensor:
        if QUERY_STEPS in batch:
            steps = batch[QUERY_STEPS].to(dtype=torch.long)
            if steps.ndim != 2:
                raise ValueError(f"query_steps must have shape (B, Q), got {tuple(steps.shape)}.")
            return steps
        queries = min(self.config.num_query_samples, time_steps)
        return torch.randint(0, time_steps, (batch_size, queries), device=device)

    def _check_rollout_batch(self, batch_size: int, device: torch.device) -> None:
        if self._hist_visual is None:
            return
        if self._hist_visual.shape[0] != batch_size or self._hist_visual.device != device:
            raise ValueError(
                "IMPRINT episode memory was built for a different batch or device. Call reset()."
            )


def _check_aligned(visual: Tensor, state: Tensor) -> None:
    if state.ndim != 3:
        raise ValueError(f"{OBS_STATE} must have shape (B, T, D), got {tuple(state.shape)}.")
    if visual.shape[0] != state.shape[0] or visual.shape[1] != state.shape[1]:
        raise ValueError(
            "Visual and state observations must share batch and time, "
            f"got visual {tuple(visual.shape[:2])} and state {tuple(state.shape[:2])}."
        )


def _action_tensor(batch: dict[str, Tensor]) -> Tensor | None:
    """Return the action tensor. A missing key or a None placeholder means no action."""
    if ACTION not in batch or batch[ACTION] is None:
        return None
    action = batch[ACTION]
    if not isinstance(action, Tensor):
        raise ValueError(f"action must be a tensor, got {type(action).__name__}.")
    return action


def _feature_dim(feature: PolicyFeature | None, name: str) -> int:
    if feature is None:
        raise ValueError(f"IMPRINT requires {name}.")
    return int(feature.shape[0])


def _with_time(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 0 or value.shape[0] == 0:
        raise ValueError(f"{name} is empty, got {tuple(value.shape)}.")
    if name == OBS_STATE or value.ndim == 2:
        if value.ndim == 1:
            raise ValueError(f"{name} is missing a batch dimension, got {tuple(value.shape)}.")
        if value.ndim == 2:
            return value.unsqueeze(1)
        return value
    if value.ndim == 3:
        raise ValueError(f"{name} is missing a batch dimension, got {tuple(value.shape)}.")
    if value.ndim == 4:
        return value.unsqueeze(1)
    return value


def _language(batch: dict[str, Tensor], lang_dim: int) -> Tensor | None:
    if LANG_EMBEDDING not in batch:
        return None
    lang = batch[LANG_EMBEDDING]
    if lang.ndim == 1:
        lang = lang.unsqueeze(0)
    if lang.ndim != 2 or lang.shape[-1] != lang_dim:
        raise ValueError(
            f"observation.language.embedding must have shape (B, {lang_dim}), got {tuple(lang.shape)}."
        )
    return lang


def _split_history(
    chunk_visual: Tensor,
    recent_visual: Tensor,
    causal: Tensor,
    state: Tensor,
    chunk_frames: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, time_steps, tokens, _chunk_dim = chunk_visual.shape
    chunks = time_steps // chunk_frames
    final = chunks * chunk_frames
    if chunks == 0:
        chunk_v = chunk_visual[:, :0]
        chunk_a = causal[:, :0].reshape(batch, 0, chunk_frames, causal.shape[-1])
        chunk_p = state[:, :0].reshape(batch, 0, chunk_frames, state.shape[-1])
    else:
        grouped = chunk_visual[:, :final].reshape(batch, chunks, chunk_frames, tokens, -1)
        chunk_v = grouped.mean(dim=2)
        chunk_a = causal[:, :final].reshape(batch, chunks, chunk_frames, -1)
        chunk_p = state[:, :final].reshape(batch, chunks, chunk_frames, -1)
    recent_v = recent_visual[:, final:].reshape(batch, -1, recent_visual.shape[-1])
    return chunk_v, recent_v, chunk_a, causal[:, final:], chunk_p, state[:, final:]


def _merge_steps(chunk_steps: Tensor, recent_steps: Tensor) -> Tensor:
    if chunk_steps.shape[1] == 0:
        return recent_steps
    flat = chunk_steps.flatten(1, 2)
    if recent_steps.shape[1] == 0:
        return flat
    return torch.cat([flat, recent_steps], dim=1)


def _gather_time(values: Tensor, index: Tensor) -> Tensor:
    view = [index.shape[0], index.shape[1]] + [1] * (values.ndim - 2)
    expanded = index.reshape(view).expand(-1, -1, *values.shape[2:])
    return values.gather(1, expanded)
