#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Opt-in Gaussian action readout for pi0.5.

The default sampler stays the ten-step Euler integrator. A readout artifact is
loaded only when ``gaussian_readout_path`` is set. LeRobot's flow puts noise at
time 1 and uses velocity ``noise - actions``, so the raw single-step action is
``epsilon_star - action_out_proj(features)`` evaluated at time 1.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks

FORMAT = "lerobot-gar-v1"
EPSILON_SEED = 17
READOUT_TIME = 1.0
VARIANCE_HIDDEN = 128
VARIANCE_FLOOR = 1e-6
VARIANCE_LOG_WEIGHT = 0.1
PROBABILITY_CLIP = 1e-6

_RTC_KEYS = ("inference_delay", "prev_chunk_left_over", "execution_horizon")


def make_epsilon_star(chunk_size: int, max_action_dim: int, seed: int = EPSILON_SEED) -> Tensor:
    """Standard-normal probe shared by every context. Shape is ``(1, chunk, max_action_dim)``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((1, chunk_size, max_action_dim), generator=generator, dtype=torch.float32)


def normalized_command(command: float, mean: Tensor, std: Tensor) -> Tensor:
    """Map an environment command to the normalized value whose unnormalization is exact.

    The pi0.5 postprocessor inverts MEAN_STD as ``normalized * std + mean`` (no epsilon).
    """
    return (command - mean) / std


def decode_mean_std(normalized: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Invert MEAN_STD the way the policy postprocessor does."""
    return normalized * std + mean


def normal_cdf(values: Tensor) -> Tensor:
    """Standard-normal CDF. The gripper head is a probit."""
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def _tensor_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    if cpu.dtype is torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return cpu.numpy().tobytes()


def state_dict_sha256(module: nn.Module) -> str:
    """Hash a module state dict in key order, including raw dtype bytes."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def bind_processor_sha256(model: Any, digest: str) -> None:
    """Record the live processor hash the next readout load must match."""
    model._gar_processor_sha256 = digest


def collapse_label(teacher_orders: set[str], student_orders: set[str]) -> str:
    """Label mode loss only when the teacher shows two successful orders and the student drops one."""
    if len(teacher_orders) < 2:
        return "inconclusive"
    if teacher_orders - student_orders:
        return "collapse-observed"
    return "both-covered"


def flow_raw_action(epsilon: Tensor, velocity: Tensor) -> Tensor:
    """LeRobot time-1 raw action. The paper's time-0 sign is not used."""
    return epsilon - velocity


class GaussianReadout(nn.Module):
    """Affine motion residual, diagonal variance MLP, and a shared gripper probit."""

    def __init__(
        self,
        feature_dim: int,
        *,
        chunk_size: int,
        motion_dim: int,
        gripper_index: int,
        max_action_dim: int,
        action_dim: int,
        open_normalized: float,
        close_normalized: float,
        action_mean: Tensor,
        action_std: Tensor,
        gripper_threshold: float = 0.0,
        open_command: float = -1.0,
        close_command: float = 1.0,
        epsilon_seed: int = EPSILON_SEED,
    ):
        super().__init__()
        if not 0 <= motion_dim <= gripper_index < action_dim <= max_action_dim:
            raise ValueError(
                "action layout must satisfy 0 <= motion_dim <= gripper_index < action_dim <= max_action_dim, "
                f"got motion_dim={motion_dim}, gripper_index={gripper_index}, action_dim={action_dim}, "
                f"max_action_dim={max_action_dim}"
            )
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.motion_dim = motion_dim
        self.gripper_index = gripper_index
        self.max_action_dim = max_action_dim
        self.action_dim = action_dim
        self.gripper_threshold = float(gripper_threshold)
        self.open_command = float(open_command)
        self.close_command = float(close_command)
        self.epsilon_seed = int(epsilon_seed)

        self.affine_weight = nn.Parameter(torch.zeros(motion_dim, feature_dim))
        self.affine_bias = nn.Parameter(torch.zeros(chunk_size, motion_dim))
        self.var_in = nn.Linear(feature_dim, VARIANCE_HIDDEN)
        self.var_out = nn.Linear(VARIANCE_HIDDEN, chunk_size * motion_dim)
        self.gripper_weight = nn.Parameter(torch.zeros(feature_dim))
        self.gripper_bias = nn.Parameter(torch.zeros(chunk_size))

        motion_mask = torch.zeros(max_action_dim)
        motion_mask[:motion_dim] = 1
        pad_mask = torch.zeros(max_action_dim)
        pad_mask[action_dim:] = 1
        self.register_buffer("epsilon_star", make_epsilon_star(chunk_size, max_action_dim, epsilon_seed))
        self.register_buffer("motion_mask", motion_mask)
        self.register_buffer("pad_mask", pad_mask)
        self.register_buffer("open_normalized", torch.tensor(float(open_normalized), dtype=torch.float32))
        self.register_buffer("close_normalized", torch.tensor(float(close_normalized), dtype=torch.float32))
        self.register_buffer("action_mean", action_mean.detach().float().reshape(action_dim).cpu())
        self.register_buffer("action_std", action_std.detach().float().reshape(action_dim).cpu())

    def variance(self, features: Tensor) -> Tensor:
        """Diagonal variance from features averaged across the chunk. Shape ``(B, T, motion)``."""
        pooled = features.mean(dim=1)
        hidden = F.gelu(self.var_in(pooled))
        positive = F.softplus(self.var_out(hidden)) + VARIANCE_FLOOR
        return positive.view(features.shape[0], self.chunk_size, self.motion_dim)

    def continuous_mean(self, features: Tensor, raw_action: Tensor) -> Tensor:
        """Shared per-position affine residual added to the raw single-step motion."""
        residual = torch.einsum("btw,mw->btm", features, self.affine_weight) + self.affine_bias
        return raw_action[..., : self.motion_dim] + residual

    def gripper_latent(self, features: Tensor) -> Tensor:
        """Shared feature weight and one bias per action position."""
        return torch.einsum("btw,w->bt", features, self.gripper_weight) + self.gripper_bias

    def sample_action(self, features: Tensor, raw_action: Tensor) -> Tensor:
        """Draw motion noise and a gripper latent. Padding stays zero."""
        features = features.float()
        raw_action = raw_action.float()
        mean = self.continuous_mean(features, raw_action)
        scale = self.variance(features).sqrt()
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype)
        motion = mean + scale * noise
        latent = self.gripper_latent(features)
        command_noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype)
        close = (latent + command_noise) >= 0
        gripper = torch.where(
            close, self.close_normalized.to(mean.dtype), self.open_normalized.to(mean.dtype)
        )
        action = raw_action.new_zeros(features.shape[0], self.chunk_size, self.max_action_dim)
        action[..., : self.motion_dim] = motion
        action[..., self.gripper_index] = gripper
        return action


def solve_affine_residual(
    features: Tensor, raw_motion: Tensor, teacher_mean: Tensor
) -> tuple[Tensor, Tensor]:
    """Least squares for a weight shared across positions and a bias per position.

    ``torch.linalg.pinv`` runs on ``features.device``. Padded action coordinates are not in
    ``raw_motion`` or ``teacher_mean``.
    """
    if features.ndim != 3 or raw_motion.shape != teacher_mean.shape:
        raise ValueError("affine fit expects features (N, T, W) and matching motion tensors")
    count, horizon, width = features.shape
    motion_dim = raw_motion.shape[-1]
    design_features = features.reshape(count * horizon, width).to(dtype=torch.float64)
    position = torch.arange(horizon, device=features.device).repeat(count)
    position_eye = torch.eye(horizon, dtype=torch.float64, device=features.device)
    design = torch.cat([design_features, position_eye[position]], dim=1)
    target = (teacher_mean - raw_motion).reshape(count * horizon, motion_dim).to(dtype=torch.float64)
    beta = torch.linalg.pinv(design) @ target
    weight = beta[:width].transpose(0, 1).to(dtype=torch.float32)
    bias = beta[width:].to(dtype=torch.float32)
    return weight, bias


def variance_loss(predicted: Tensor, target: Tensor, scale: Tensor) -> Tensor:
    """Normalized squared variance error plus a log penalty. ``scale`` is the training-set factor."""
    squared = (predicted - target).pow(2).mean()
    ratio = predicted.clamp_min(VARIANCE_FLOOR).log() - target.clamp_min(VARIANCE_FLOOR).log()
    return squared / scale + VARIANCE_LOG_WEIGHT * ratio.pow(2).mean()


def gripper_loss(latent: Tensor, frequency: Tensor) -> Tensor:
    """Binary cross-entropy on probit probabilities clamped away from 0 and 1."""
    probability = normal_cdf(latent).clamp(PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return F.binary_cross_entropy(probability, frequency.float())


def _cycle_indices(count: int, batch_size: int, generator: torch.Generator):
    while True:
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            yield order[start : start + batch_size]


def _restore(module: nn.Module, state: dict[str, Tensor], device: torch.device) -> None:
    module.load_state_dict({key: value.to(device) for key, value in state.items()})


def _snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def fit_variance_head(
    readout: GaussianReadout,
    features: Tensor,
    target: Tensor,
    tune_features: Tensor,
    tune_target: Tensor,
    *,
    updates: int = 2000,
    batch_size: int = 32,
    seed: int = EPSILON_SEED,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
) -> int:
    """AdamW on the variance MLP. Returns the selected update, including 0 for the initialization."""
    features = features.detach().float()
    target = target.detach().float().clamp_min(VARIANCE_FLOOR)
    tune_features = tune_features.detach().float()
    tune_target = tune_target.detach().float().clamp_min(VARIANCE_FLOOR)
    scale = target.mean().pow(2).clamp_min(VARIANCE_FLOOR).detach()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    for parameter in list(readout.var_in.parameters()) + list(readout.var_out.parameters()):
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        list(readout.var_in.parameters()) + list(readout.var_out.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    device = features.device
    best_state = _snapshot(readout)
    best_loss = float("inf")
    selected = 0

    def tuning_loss() -> float:
        readout.eval()
        with torch.no_grad():
            return float(variance_loss(readout.variance(tune_features), tune_target, scale))

    best_loss = tuning_loss()
    batches = _cycle_indices(features.shape[0], batch_size, generator)
    for update in range(1, updates + 1):
        index = next(batches).to(device)
        readout.train()
        optimizer.zero_grad(set_to_none=True)
        loss = variance_loss(readout.variance(features[index]), target[index], scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(readout.var_in.parameters()) + list(readout.var_out.parameters()),
            grad_clip,
        )
        optimizer.step()
        if update % 100 == 0 or update == updates:
            current = tuning_loss()
            if current < best_loss:
                best_loss = current
                best_state = _snapshot(readout)
                selected = update
    _restore(readout, best_state, device)
    readout.eval()
    return selected


def fit_gripper_head(
    readout: GaussianReadout,
    features: Tensor,
    frequency: Tensor,
    tune_features: Tensor,
    tune_frequency: Tensor,
    *,
    updates: int = 2000,
    batch_size: int = 32,
    seed: int = EPSILON_SEED,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
) -> int:
    """AdamW on the gripper probit. Returns the selected update."""
    features = features.detach().float()
    frequency = frequency.detach().float()
    tune_features = tune_features.detach().float()
    tune_frequency = tune_frequency.detach().float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    readout.gripper_weight.requires_grad_(True)
    readout.gripper_bias.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [readout.gripper_weight, readout.gripper_bias], lr=lr, weight_decay=weight_decay
    )
    device = features.device
    best_state = _snapshot(readout)
    readout.eval()
    with torch.no_grad():
        best_loss = float(gripper_loss(readout.gripper_latent(tune_features), tune_frequency))
    selected = 0
    batches = _cycle_indices(features.shape[0], batch_size, generator)
    for update in range(1, updates + 1):
        index = next(batches).to(device)
        readout.train()
        optimizer.zero_grad(set_to_none=True)
        loss = gripper_loss(readout.gripper_latent(features[index]), frequency[index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([readout.gripper_weight, readout.gripper_bias], grad_clip)
        optimizer.step()
        if update % 100 == 0 or update == updates:
            readout.eval()
            with torch.no_grad():
                current = float(gripper_loss(readout.gripper_latent(tune_features), tune_frequency))
            if current < best_loss:
                best_loss = current
                best_state = _snapshot(readout)
                selected = update
    _restore(readout, best_state, device)
    readout.eval()
    return selected


def save_readout(
    readout: GaussianReadout,
    path: str | Path,
    *,
    teacher_sha256: str,
    processor_sha256: str,
) -> None:
    """Write the readout tensors and the hashes needed to reject a mismatched teacher."""
    tensors = {key: value.detach().cpu().contiguous() for key, value in readout.state_dict().items()}
    metadata = {
        "format": FORMAT,
        "teacher_sha256": teacher_sha256,
        "processor_sha256": processor_sha256,
        "feature_dim": str(readout.feature_dim),
        "chunk_size": str(readout.chunk_size),
        "motion_dim": str(readout.motion_dim),
        "gripper_index": str(readout.gripper_index),
        "max_action_dim": str(readout.max_action_dim),
        "action_dim": str(readout.action_dim),
        "epsilon_seed": str(readout.epsilon_seed),
        "time": str(READOUT_TIME),
        "gripper_threshold": str(readout.gripper_threshold),
        "open_command": str(readout.open_command),
        "close_command": str(readout.close_command),
    }
    save_file(tensors, str(path), metadata=metadata)


def _require_meta(metadata: dict[str, str], key: str) -> str:
    if key not in metadata:
        raise ValueError(f"readout artifact is missing metadata field {key}")
    return metadata[key]


def load_gaussian_readout(path: str | Path, teacher: Any, processor_sha256: str) -> GaussianReadout:
    """Load a readout and reject a hash, shape, mask, or probe that does not match this teacher."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {}
        for key in handle.keys():  # noqa: SIM118
            tensors[key] = handle.get_tensor(key)
    if _require_meta(metadata, "format") != FORMAT:
        raise ValueError(f"unsupported readout format {metadata.get('format')!r}")
    teacher_hash = state_dict_sha256(teacher)
    if _require_meta(metadata, "teacher_sha256") != teacher_hash:
        raise ValueError("readout teacher hash does not match the loaded weights")
    if _require_meta(metadata, "processor_sha256") != processor_sha256:
        raise ValueError("readout processor hash does not match the bound processors")
    if _require_meta(metadata, "time") != str(READOUT_TIME):
        raise ValueError("readout artifact was not fit at LeRobot time 1")
    if int(_require_meta(metadata, "epsilon_seed")) != EPSILON_SEED:
        raise ValueError("readout probe seed is not 17")

    feature_dim = int(_require_meta(metadata, "feature_dim"))
    chunk_size = int(_require_meta(metadata, "chunk_size"))
    motion_dim = int(_require_meta(metadata, "motion_dim"))
    gripper_index = int(_require_meta(metadata, "gripper_index"))
    max_action_dim = int(_require_meta(metadata, "max_action_dim"))
    action_dim = int(_require_meta(metadata, "action_dim"))
    config = teacher.config
    projection = teacher.action_out_proj
    if feature_dim != projection.in_features:
        raise ValueError(
            f"readout feature width {feature_dim} does not match action_out_proj {projection.in_features}"
        )
    if chunk_size != config.chunk_size or max_action_dim != config.max_action_dim:
        raise ValueError("readout chunk or padded action size does not match the policy config")

    # Initialization is overwritten by the artifact; preserve the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        readout = GaussianReadout(
            feature_dim,
            chunk_size=chunk_size,
            motion_dim=motion_dim,
            gripper_index=gripper_index,
            max_action_dim=max_action_dim,
            action_dim=action_dim,
            open_normalized=0.0,
            close_normalized=1.0,
            action_mean=torch.zeros(action_dim),
            action_std=torch.ones(action_dim),
            gripper_threshold=float(_require_meta(metadata, "gripper_threshold")),
            open_command=float(_require_meta(metadata, "open_command")),
            close_command=float(_require_meta(metadata, "close_command")),
            epsilon_seed=EPSILON_SEED,
        )
    readout.load_state_dict(tensors, strict=True)
    expected_probe = make_epsilon_star(chunk_size, max_action_dim, EPSILON_SEED)
    if not torch.equal(readout.epsilon_star, expected_probe):
        raise ValueError("readout fixed noise does not match the seed-17 probe")
    motion_mask = torch.zeros(max_action_dim)
    motion_mask[:motion_dim] = 1
    pad_mask = torch.zeros(max_action_dim)
    pad_mask[action_dim:] = 1
    if not torch.equal(readout.motion_mask, motion_mask) or not torch.equal(readout.pad_mask, pad_mask):
        raise ValueError("readout motion or padding mask does not match the declared layout")
    if not torch.isfinite(readout.open_normalized) or not torch.isfinite(readout.close_normalized):
        raise ValueError("readout gripper codes are not finite")
    if torch.equal(readout.open_normalized, readout.close_normalized):
        raise ValueError("readout open and close codes are identical")
    readout.eval()
    return readout


def _reject_unsupported(
    model: Any, noise: Tensor | None, num_steps: int | None, kwargs: dict[str, Any]
) -> None:
    unknown = set(kwargs) - set(_RTC_KEYS)
    if unknown:
        raise ValueError(f"Gaussian readout received unsupported arguments: {sorted(unknown)}")
    if noise is not None:
        raise ValueError("Gaussian readout rejects explicit starting noise")
    if num_steps is not None:
        raise ValueError("Gaussian readout rejects an explicit step count")
    config = model.config
    if config.compile_model:
        raise ValueError("Gaussian readout rejects compilation")
    if config.use_visual_memory or config.use_proprioceptive_memory:
        raise ValueError("Gaussian readout rejects memory modes")
    if model._rtc_enabled() or any(kwargs.get(key) is not None for key in _RTC_KEYS):
        raise ValueError("Gaussian readout rejects real-time chunking")


def encode_prefix(
    model: Any,
    images: Any,
    img_masks: Any,
    tokens: Tensor,
    masks: Tensor,
    states: Tensor | None = None,
    state_masks: Tensor | None = None,
) -> tuple[Tensor, Any]:
    """Prefix prefill copied from the Euler sampler, including its eager attention flag."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    mask_dtype = prefix_embs.dtype if model.use_typed_attention_masks else None
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=mask_dtype)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values


def project_velocity(model: Any, features: Tensor) -> Tensor:
    """Apply ``action_out_proj`` in FP32 without writing the teacher weights."""
    projection = model.action_out_proj
    bias = None if projection.bias is None else projection.bias.detach().float()
    return F.linear(features.float(), projection.weight.detach().float(), bias)


def evaluate_time_one_features(
    model: Any,
    images: Any,
    img_masks: Any,
    tokens: Tensor,
    masks: Tensor,
    states: Tensor | None,
    state_masks: Tensor | None,
    epsilon: Tensor,
) -> tuple[Tensor, Tensor]:
    """One suffix evaluation at time 1. Returns features and the raw single-step action."""
    prefix_pad_masks, past_key_values = encode_prefix(
        model, images, img_masks, tokens, masks, states, state_masks
    )
    batch = tokens.shape[0]
    device = tokens.device
    probe = epsilon.to(device=device, dtype=torch.float32).expand(batch, -1, -1)
    timestep = torch.ones(batch, dtype=torch.float32, device=device)
    features = model.suffix_features(prefix_pad_masks, past_key_values, probe, timestep)
    raw = flow_raw_action(probe, project_velocity(model, features))
    return features, raw


def sample_euler_from_cache(
    model: Any,
    prefix_pad_masks: Tensor,
    past_key_values: Any,
    noise: Tensor,
    num_steps: int,
) -> Tensor:
    """Euler integration that reuses one prefix cache. Each step clones that cache in ``denoise_step``."""
    return euler_integrate(
        lambda x_t, timestep: model.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep),
        noise,
        num_steps,
    )


def _cached_readout(model: Any, path: str) -> GaussianReadout:
    processor_sha256 = getattr(model, "_gar_processor_sha256", None)
    if not isinstance(processor_sha256, str) or not processor_sha256:
        raise ValueError("Gaussian readout requires a bound processor hash")
    cache = getattr(model, "_gar_cache", None)
    if cache is not None and cache[0] == path and cache[1] == processor_sha256:
        return cache[2]
    readout = load_gaussian_readout(path, model, processor_sha256)
    model._gar_cache = (path, processor_sha256, readout)
    return readout


@torch.no_grad()
def sample_gaussian_actions(
    model: Any,
    images: Any,
    img_masks: Any,
    tokens: Tensor,
    masks: Tensor,
    *,
    states: Tensor | None = None,
    state_masks: Tensor | None = None,
    noise: Tensor | None = None,
    num_steps: int | None = None,
    **kwargs: Any,
) -> Tensor:
    """Sample one action chunk from the readout. Rejects modes the Euler sampler still accepts."""
    _reject_unsupported(model, noise, num_steps, kwargs)
    path = model.config.gaussian_readout_path
    if not isinstance(path, str) or not path.strip():
        raise ValueError("gaussian_readout_path must be a file path when the readout sampler is selected")
    readout = _cached_readout(model, path)
    device = tokens.device
    readout = readout.to(device)
    features, raw = evaluate_time_one_features(
        model, images, img_masks, tokens, masks, states, state_masks, readout.epsilon_star
    )
    return readout.sample_action(features, raw)


def diagnose_two_mode_obstacle() -> dict[str, Any]:
    """CPU fixture: a mean between two obstacle corridors is not a third successful mode.

    The teacher executes both corridors. The collapsed policy executes their coordinate-wise
    mean, hits the obstacle, and loses both modes. A teacher that only shows one mode is
    inconclusive even if the student copies it.
    """
    start = torch.tensor([0.0, 0.0])
    upper = torch.tensor([3.0, 1.2])
    lower = torch.tensor([3.0, -1.2])
    obstacle_center = torch.tensor([1.2, 0.0])
    obstacle_radius = 0.35
    goal_radius = 0.3
    steps = 25

    def trajectory(target: Tensor) -> Tensor:
        direction = target - start
        step = direction / steps
        return start + step * torch.arange(1, steps + 1, dtype=torch.float32)[:, None]

    upper_path = trajectory(upper)
    lower_path = trajectory(lower)
    mean_path = 0.5 * (upper_path + lower_path)

    def hits_obstacle(path: Tensor) -> bool:
        return bool(torch.linalg.vector_norm(path - obstacle_center, dim=-1).le(obstacle_radius).any())

    def mode_of(path: Tensor) -> str:
        if hits_obstacle(path):
            return "neither"
        end = path[-1]
        if torch.linalg.vector_norm(end - upper) <= goal_radius:
            return "upper"
        if torch.linalg.vector_norm(end - lower) <= goal_radius:
            return "lower"
        return "neither"

    def between_targets(path: Tensor) -> bool:
        end = path[-1]
        midpoint = 0.5 * (upper + lower)
        return bool(hits_obstacle(path) and torch.linalg.vector_norm(end - midpoint) < 0.25)

    teacher_modes = {mode_of(upper_path), mode_of(lower_path)}
    mean_mode = mode_of(mean_path)
    one_mode_teacher = {"upper"}
    return {
        "teacher_modes": sorted(teacher_modes),
        "mean_mode": mean_mode,
        "mean_between_targets": between_targets(mean_path),
        "teacher_hits_obstacle": hits_obstacle(upper_path) or hits_obstacle(lower_path),
        "lost_modes": sorted(teacher_modes - {mean_mode}),
        "collapse_label": collapse_label(teacher_modes, set() if mean_mode == "neither" else {mean_mode}),
        "one_mode_label": collapse_label(one_mode_teacher, {"upper"}),
        "mean_endpoint_y": float(mean_path[-1, 1]),
    }
