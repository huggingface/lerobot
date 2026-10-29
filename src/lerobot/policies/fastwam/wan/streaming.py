# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Adapted from Streaming-WAM by Xuyao Huang et al.
# Source revision: ff36d6e66514afd65bf957a06a0a2410f182b0e3
# Copyright 2026 Shaohua Pan. Apache-2.0; see ../LICENSE.streaming-wam.

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .model import rope_params, sinusoidal_embedding_1d
from .modular import ActionDiT, FastWAM, MoT, MoTLayer
from .video_dit import WanVideoDiT

AC_STREAM_ACTION_HORIZON = 32
AC_STREAM_CONDITION_SLOTS = 16
AC_STREAM_DELAY = 8


def action_consistency_boundary(sample: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """Recover a clean action from flow velocity with batch or token sigma."""
    shape = (-1, 1, 1) if sigma.ndim == 1 else (*sigma.shape, 1)
    return (sample - sigma.to(sample).reshape(shape) * velocity).to(sample.dtype)


def video_consistency_boundary(sample: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """Apply the released joint-consistency video boundary (sigma_data=0.5)."""
    sigma = sigma.to(sample).view(sigma.shape[0], 1, sigma.shape[1], 1, 1)
    predicted = sample - sigma * velocity
    scale = sigma.float()
    skip = 0.25 / (scale.square() + 0.25)
    out = scale * 0.5 / (scale.square() + 0.25).sqrt()
    return (skip.to(sample.dtype) * sample + out.to(sample.dtype) * predicted).to(sample.dtype)


def _build_video_mask(
    video_seq_len: int,
    video_tokens_per_frame: int,
    device: torch.device,
) -> Tensor:
    first_frame = min(max(int(video_tokens_per_frame), 1), int(video_seq_len))
    mask = torch.ones(
        (int(video_seq_len), int(video_seq_len)),
        dtype=torch.bool,
        device=device,
    )
    mask[:first_frame, first_frame:] = False
    return mask


def build_ac_stream_policy_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    known_prefix_length: int,
    device: torch.device,
) -> Tensor:
    """Build the checkpoint's directed D0 or D8 video/policy mask."""

    if int(action_seq_len) != AC_STREAM_ACTION_HORIZON:
        raise ValueError(
            f"AC-Stream policy stream requires {AC_STREAM_ACTION_HORIZON} actions, got {action_seq_len}"
        )
    if int(known_prefix_length) not in (0, AC_STREAM_DELAY):
        raise ValueError(f"AC-Stream prefix length must be 0 or {AC_STREAM_DELAY}")
    video_seq_len = int(video_seq_len)
    action_seq_len = int(action_seq_len)
    action_start = video_seq_len
    prefix_end = action_start + int(known_prefix_length)
    total = video_seq_len + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:video_seq_len, :video_seq_len] = _build_video_mask(
        video_seq_len,
        video_tokens_per_frame,
        device,
    )
    if known_prefix_length:
        first_frame = int(video_tokens_per_frame)
        mask[action_start:prefix_end, :first_frame] = True
        mask[action_start:prefix_end, action_start:prefix_end] = True
    mask[prefix_end:, :] = True
    return mask


def build_ac_stream_condition_mask(
    *,
    video_seq_len: int,
    condition_seq_len: int,
    video_tokens_per_frame: int,
    device: torch.device,
) -> Tensor:
    """Build the fixed 16-slot condition graph used by D0 and D8."""

    if int(condition_seq_len) != AC_STREAM_CONDITION_SLOTS:
        raise ValueError(
            f"AC-Stream condition stream requires {AC_STREAM_CONDITION_SLOTS} slots, got {condition_seq_len}"
        )
    video_seq_len = int(video_seq_len)
    condition_seq_len = int(condition_seq_len)
    tokens_per_frame = int(video_tokens_per_frame)
    if video_seq_len != 3 * tokens_per_frame:
        raise ValueError(
            "AC-Stream video tokens must contain exactly z0/z1/z2; got "
            f"video_seq_len={video_seq_len}, tokens_per_frame={tokens_per_frame}"
        )
    condition_start = video_seq_len
    prefix_end = condition_start + AC_STREAM_DELAY
    total = video_seq_len + condition_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:video_seq_len, :video_seq_len] = _build_video_mask(
        video_seq_len,
        tokens_per_frame,
        device,
    )
    mask[tokens_per_frame : 2 * tokens_per_frame, condition_start:] = True
    mask[condition_start:prefix_end, :tokens_per_frame] = True
    mask[condition_start:prefix_end, condition_start:prefix_end] = True
    mask[prefix_end:, :tokens_per_frame] = True
    mask[prefix_end:, condition_start:] = True
    return mask


class ACStreamSlotEncoder(nn.Module):
    """Represent known clean actions and structure-only unknown slots."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_embedding = nn.Embedding(2, self.hidden_dim)
        nn.init.normal_(self.state_embedding.weight, std=self.hidden_dim**-0.5)

    def forward(self, action_tokens: Tensor, known_mask: Tensor) -> Tensor:
        if tuple(action_tokens.shape[1:]) != (
            AC_STREAM_CONDITION_SLOTS,
            self.hidden_dim,
        ):
            raise ValueError(
                "AC-Stream action slots must have shape "
                f"[B,{AC_STREAM_CONDITION_SLOTS},{self.hidden_dim}], got "
                f"{tuple(action_tokens.shape)}"
            )
        if tuple(known_mask.shape) != tuple(action_tokens.shape[:2]):
            raise ValueError(
                f"AC-Stream known mask must be {tuple(action_tokens.shape[:2])}, "
                f"got {tuple(known_mask.shape)}"
            )
        known_mask = known_mask.to(device=action_tokens.device, dtype=torch.bool)
        content = torch.where(
            known_mask.unsqueeze(-1),
            action_tokens,
            torch.zeros_like(action_tokens),
        )
        state = self.state_embedding(known_mask.to(dtype=torch.long))
        return content + state.to(dtype=action_tokens.dtype)


def _reference_norms(expert: nn.Module) -> None:
    """Keep the release's BF16 LayerNorm convention confined to this opt-in path."""
    for block in expert.blocks:
        for name in ("norm1", "norm2", "norm3"):
            norm = getattr(block, name)
            setattr(
                block,
                name,
                nn.LayerNorm(norm.normalized_shape, eps=norm.eps, elementwise_affine=norm.elementwise_affine),
            )


class StreamingActionDiT(ActionDiT):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _reference_norms(self)
        # Non-registered complex RoPE tables must survive meta construction.
        with torch.device("cpu"):
            self.freqs = rope_params(1024, self.attn_head_dim)

    def pre_dit(
        self, action_tokens: Tensor, timestep: Tensor, context: Tensor, context_mask: Tensor | None = None
    ) -> dict[str, Any]:
        batch, length, dim = action_tokens.shape
        if dim != self.action_dim or timestep.ndim not in (1, 2):
            raise ValueError("Invalid streaming action or timestep geometry.")
        if timestep.shape[0] not in (1, batch) or (timestep.ndim == 2 and timestep.shape[1] != length):
            raise ValueError("Streaming timesteps must be [B] or [B,T].")
        timestep = timestep.expand(batch, *timestep.shape[1:])
        action_tokens = action_tokens.to(self.action_encoder.weight)
        context = context.to(self.text_embedding[0].weight)
        time = sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1)).to(action_tokens.dtype)
        t = self.time_embedding(time)
        shape = (batch, length) if timestep.ndim == 2 else (batch,)
        t_mod = self.time_projection(t).reshape(*shape, 6, self.hidden_dim)
        if context_mask is not None:
            context_mask = context_mask.to(action_tokens.device).unsqueeze(1).expand(batch, length, -1)
        return {
            "tokens": self.action_encoder(action_tokens),
            "freqs": self.freqs[:length].view(length, 1, -1).to(action_tokens.device),
            "t_mod": t_mod,
            "context": self.text_embedding(context),
            "context_mask": context_mask,
        }


class StreamingVideoHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, patch_size: tuple[int, ...], eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, out_dim * patch_size[0] * patch_size[1] * patch_size[2])
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        shift, scale = (self.modulation.unsqueeze(0) + t.unsqueeze(2)).chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))


class StreamingVideoDiT(WanVideoDiT):
    def __init__(self, **kwargs: Any) -> None:
        action_hidden_dim = kwargs.pop("stream_action_hidden_dim", 1024)
        super().__init__(**kwargs)
        _reference_norms(self)
        self.head = StreamingVideoHead(self.hidden_dim, self.out_dim, self.patch_size, kwargs["eps"])
        self.rtc_1step_selfatt_z1_slot_encoder = ACStreamSlotEncoder(action_hidden_dim)

    def pre_dit(
        self,
        x: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
        action: Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = True,
    ) -> dict[str, Any]:
        # The inherited flow-loss API passes clean actions here. They do not
        # enter video text context; AC-Stream conditioning has its own stream.
        x, timestep, context_mask = self._validate_forward_inputs(x, timestep, context, context_mask, None)
        x = x.to(self.patch_embedding.weight)
        context = context.to(self.text_embedding[0].weight)
        patches = self.patchify(x)
        batch, _, frames, height, width = patches.shape
        per_frame = height * width
        times = x.new_ones(batch, frames, per_frame) * timestep.to(x.dtype).view(batch, 1, 1)
        times[:, 0] = 0
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, times.reshape(-1)).to(x.dtype))
        t = t.reshape(batch, frames * per_frame, self.hidden_dim)
        dims = (
            self.attn_head_dim - 2 * (self.attn_head_dim // 3),
            self.attn_head_dim // 3,
            self.attn_head_dim // 3,
        )
        freqs_f, freqs_h, freqs_w = [rope_params(1024, dim).to(x.device) for dim in dims]
        freqs = torch.cat(
            (
                freqs_f[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                freqs_h[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freqs_w[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ),
            dim=-1,
        )
        return {
            "tokens": patches.flatten(2).transpose(1, 2).contiguous(),
            "freqs": freqs.reshape(frames * per_frame, 1, -1),
            "t": t,
            "t_mod": self.time_projection(t).unflatten(2, (6, self.hidden_dim)),
            "context": self.text_embedding(context),
            "context_mask": context_mask.unsqueeze(1).expand(batch, frames * per_frame, -1),
            "meta": {
                "tokens_per_frame": per_frame,
                "grid_sizes": torch.tensor([[frames, height, width]] * batch, device=x.device),
            },
        }


class StreamingMoT(MoT):
    def forward_streaming(self, states: dict[str, dict[str, Any]], delay: int) -> dict[str, Tensor]:
        """Directed eager attention with shared action blocks and a z1 residual."""
        tokens = {name: state["tokens"] for name, state in states.items()}
        video_len = tokens["video"].shape[1]
        per_frame = states["video"]["meta"]["tokens_per_frame"]
        policy_mask = build_ac_stream_policy_mask(
            video_seq_len=video_len,
            action_seq_len=32,
            video_tokens_per_frame=per_frame,
            known_prefix_length=delay,
            device=tokens["video"].device,
        )
        condition_mask = build_ac_stream_condition_mask(
            video_seq_len=video_len,
            condition_seq_len=16,
            video_tokens_per_frame=per_frame,
            device=tokens["video"].device,
        )
        for layer in self.layers:
            io = {
                name: layer._build_expert_attention_io(
                    "video" if name == "video" else "action",
                    token,
                    states[name]["freqs"],
                    states[name]["t_mod"],
                )
                for name, token in tokens.items()
            }
            qv, kv, vv = io["video"][:3]
            qp, kp, vp = io["action"][:3]
            qc, kc, vc = io["condition"][:3]
            # The reference uses SDPA in the parameter dtype.
            attend = layer._mixed_attention
            video = attend(qv, kv, vv, policy_mask[:video_len, :video_len])
            policy = attend(qp, torch.cat((kv, kp), 1), torch.cat((vv, vp), 1), policy_mask[video_len:])
            condition = attend(qc, torch.cat((kv, kc), 1), torch.cat((vv, vc), 1), condition_mask[video_len:])
            z1 = slice(per_frame, 2 * per_frame)
            residual = attend(qv[:, z1], kc, vc, condition_mask[z1, video_len:])
            video = torch.cat(
                (video[:, :per_frame], video[:, z1] + bool(delay) * residual, video[:, 2 * per_frame :]), 1
            )
            for name, output in (("video", video), ("action", policy), ("condition", condition)):
                block = layer.blocks["video" if name == "video" else "action"]
                _, _, _, x, gate, shift, scale, gate_mlp, _ = io[name]
                tokens[name] = MoTLayer._apply_expert_post_block(
                    block,
                    x,
                    output,
                    gate,
                    shift,
                    scale,
                    gate_mlp,
                    {"context": states[name]["context"], "mask": states[name]["context_mask"]},
                )
        return {"video": tokens["video"], "action": tokens["action"]}


class StreamingWAM(FastWAM):
    """Eager, one-step sampler for the released AC-Stream model."""

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> Tensor:
        # The pinned LIBERO evaluator's checkpoint_format='fastwam' path casts
        # byte pixels before scaling. LeRobot supplies those pixels in [0,1].
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        pixels = (input_image.float() * 255.0).round().to(device=self.device, dtype=self.torch_dtype)
        video = (pixels * (2.0 / 255.0) - 1.0)[0].unsqueeze(1)
        latents = self.vae.encode(
            [video], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if isinstance(latents, list):
            latents = latents[0].unsqueeze(0)
        return latents.to(device=self.device, dtype=self.torch_dtype)

    def streaming_velocity(
        self,
        video: Tensor,
        action: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Tensor,
        previous: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        delay = 0 if previous is None else 8
        times = timestep.reshape(1, 1).expand(1, 32).clone()
        if previous is not None:
            if previous.shape != action.shape:
                raise ValueError("Streaming prefix target must match [1,32,D].")
            action = action.clone()
            action[:, :8] = previous[:, :8].to(action)
            times[:, :8] = 0
        condition = action.new_zeros(1, 16, action.shape[-1])
        if previous is not None:
            condition[:, :8] = previous[:, :8].to(action)
        cv = self.video_expert.pre_dit(video, timestep, context, context_mask)
        ca = self.action_expert.pre_dit(action, times, context, context_mask)
        cc = self.action_expert.pre_dit(condition, condition.new_zeros(1, 16), context, context_mask)
        known = torch.arange(16, device=action.device).unsqueeze(0) < delay
        cc["tokens"] = self.video_expert.rtc_1step_selfatt_z1_slot_encoder(cc["tokens"], known)
        output = self.mot.forward_streaming({"video": cv, "action": ca, "condition": cc}, delay)
        return self.video_expert.post_dit(output["video"], cv), self.action_expert.post_dit(
            output["action"], ca
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: str | None,
        input_image: Tensor,
        action_horizon: int,
        proprio: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        seed: int | None = None,
        num_inference_steps: int = 1,
        previous_action_chunk: Tensor | None = None,
        video_noise: Tensor | None = None,
        action_noise: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if action_horizon != 32 or num_inference_steps != 1 or kwargs.get("compile_action_infer", False):
            raise ValueError("Streaming-WAM supports eager H32 one-step inference only.")
        if kwargs.get("text_cfg_scale", 1.0) != 1.0 or kwargs.get("rand_device", "cpu") != "cpu":
            raise ValueError("Streaming-WAM requires text_cfg_scale=1 and CPU noise generation.")
        self.eval()
        image, height, width = self._normalize_infer_input_image(input_image)
        if self.proprio_encoder is not None and proprio is None:
            raise ValueError("Streaming-WAM requires proprioception for this checkpoint.")
        context, context_mask = self._prepare_infer_context(
            prompt, context, context_mask, self._normalize_infer_proprio(proprio)
        )
        if context.shape[0] != 1:
            raise ValueError("Streaming-WAM requires a single context batch.")
        first = self._encode_input_image_latents_tensor(image, tiled=kwargs.get("tiled", False))
        video = (
            self._make_video_latents(9, height, width, seed, "cpu")
            if video_noise is None
            else video_noise.clone().to(first)
        )
        action = (
            self._make_action_latents(32, seed, "cpu")
            if action_noise is None
            else action_noise.clone().to(first)
        )
        if video.shape != (1, first.shape[1], 3, first.shape[3], first.shape[4]) or action.shape != (
            1,
            32,
            self.action_expert.action_dim,
        ):
            raise ValueError("Streaming noise must match the batch-one released geometry.")
        video[:, :, :1] = first
        timestep = self.infer_action_scheduler.build_inference_schedule(1, self.device, action.dtype)[0][:1]
        previous = None if previous_action_chunk is None else previous_action_chunk.to(action)
        if previous is not None:
            if previous.shape != action.shape:
                raise ValueError("Streaming previous_action_chunk must be [1,32,D].")
            action[:, :8] = previous[:, :8]
        video_velocity, velocity = self.streaming_velocity(
            video, action, timestep, context, context_mask, previous
        )
        sigma = timestep / self.infer_action_scheduler.num_train_timesteps
        action_sigma = sigma.reshape(1, 1).expand(1, 32).clone()
        if previous is not None:
            action_sigma[:, :8] = 0
        result = action_consistency_boundary(action, velocity, action_sigma)
        video_sigma = sigma.reshape(1, 1).expand(1, 3).clone()
        video_sigma[:, 0] = 0
        video_result = video_consistency_boundary(video, video_velocity, video_sigma)
        video_result[:, :, :1] = first
        if previous is not None:
            result[:, :8] = previous[:, :8]
        return {
            "action": result[0].detach().to(device="cpu", dtype=torch.float32),
            "video_latents": video_result,
        }
