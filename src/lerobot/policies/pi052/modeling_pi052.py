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

"""PI0.5 with joint flow/text training and hierarchical language inference."""

from __future__ import annotations

import json
import logging
import types
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Unpack

import torch
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import functional
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)
from lerobot.utils.import_utils import require_package

from ..pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch as PI05PytorchBase,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from ..pretrained import PreTrainedPolicy, T
from .configuration_pi052 import PI052Config

logger = logging.getLogger(__name__)

_SAFETENSORS_FILE = "model.safetensors"
_SAFETENSORS_INDEX = "model.safetensors.index.json"


def _resolve_weight_files(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool,
    resume_download: bool | None,
    proxies: dict | None,
    token: str | bool | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
) -> list[Path]:
    model_id = str(pretrained_name_or_path)
    local_dir = Path(model_id)
    load_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "force_download": force_download,
        "resume_download": resume_download,
        "proxies": proxies,
        "token": token,
        "local_files_only": local_files_only,
    }

    if local_dir.is_dir():
        index_path = local_dir / _SAFETENSORS_INDEX
        single_path = local_dir / _SAFETENSORS_FILE
    else:
        resolved_index = cached_file(
            model_id,
            _SAFETENSORS_INDEX,
            _raise_exceptions_for_missing_entries=False,
            **load_kwargs,
        )
        index_path = Path(resolved_index) if resolved_index is not None else None
        single_path = None
        if index_path is None:
            resolved_file = cached_file(model_id, _SAFETENSORS_FILE, **load_kwargs)
            single_path = Path(resolved_file) if resolved_file is not None else None

    if index_path is None or not index_path.is_file():
        if single_path is None or not single_path.is_file():
            raise FileNotFoundError(f"No {_SAFETENSORS_FILE} found in {model_id!r}.")
        return [single_path]

    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError(f"Invalid safetensors index without a weight_map: {index_path}")
    if local_dir.is_dir():
        files = [local_dir / name for name in shard_names]
    else:
        files = []
        for name in shard_names:
            resolved_file = cached_file(model_id, name, **load_kwargs)
            if resolved_file is None:
                raise FileNotFoundError(f"Checkpoint shard {name!r} not found in {model_id!r}.")
            files.append(Path(resolved_file))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
    return files


def _load_weight_files(files: list[Path]) -> dict[str, Tensor]:
    state_dict: dict[str, Tensor] = {}
    for path in files:
        shard = load_file(path)
        overlap = state_dict.keys() & shard.keys()
        if overlap:
            raise ValueError(f"Duplicate checkpoint keys in {path}: {sorted(overlap)[:5]}")
        state_dict.update(shard)
    return state_dict


class PI05Pytorch(PI05PytorchBase):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing_disable()
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _prepare_attention_masks_4d(self, att_2d_masks, dtype=None):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # SigLIP checkpoints its encoder layers internally. An outer tower
        # checkpoint would recreate every layer activation at once in backward.
        img_embs = [self.paligemma_with_expert.embed_image(img) for img in images]

        for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            # GemmaTextScaledWordEmbedding already applies sqrt(hidden_size); do not scale twice.
            return self.paligemma_with_expert.embed_language_tokens(tokens)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = functional.silu(x)
            x = self.time_mlp_out(x)
            return functional.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # Build the constant suffix mask on-device to avoid a per-step host sync.
        n = len(att_masks)
        att_masks = torch.zeros(n, dtype=embs.dtype, device=embs.device)
        att_masks[0] = 1
        att_masks = att_masks[None, :].expand(bsize, n)

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # The prefix has no gradient path to a flow-only loss under KI.
        suppress_prefix = bool(getattr(self.config, "knowledge_insulation", False))
        with torch.no_grad() if suppress_prefix else nullcontext():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks
            )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)

        # Transformer and vision layers own their checkpoint boundaries.
        ki_kwargs = {"suppress_prefix_grads": True} if suppress_prefix else {}
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            **ki_kwargs,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return functional.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks, dtype=prefix_embs.dtype
        )
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        # Precompute timesteps on-device to avoid a host sync per denoising step.
        times = torch.tensor([1.0 + s * dt for s in range(num_steps)], dtype=torch.float32, device=device)

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt  # Python float kept for the RTC branch below
            time_tensor = times[step].expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=suffix_embs.dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # Crop appended suffix K/V after each step instead of copying the read-only prefix cache.
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        past_key_values.crop(prefix_len)

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


# FAST tokens occupy the high vocabulary range and must be masked during text generation.
_FAST_ACTION_VOCAB_SIZE = 2048


_HF_KERNELS_ENABLED = False


def _enable_hf_kernels() -> None:
    """Patch supported PaliGemma operations before constructing the model."""
    global _HF_KERNELS_ENABLED
    if _HF_KERNELS_ENABLED:
        return
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_paligemma  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "PI052: liger-kernel is not installed; skipping fused Triton "
            "kernels. Install with ``pip install liger-kernel``."
        )
        return
    apply_liger_kernel_to_paligemma(
        rope=True,
        geglu=True,
        # Liger LayerNorm regresses at SigLIP shapes; RoPE and GeGLU remain enabled.
        layer_norm=False,
        rms_norm=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    _HF_KERNELS_ENABLED = True
    logger.info("PI052: HF kernels (Liger) enabled — rope, geglu fused.")


def _reduce_action_loss(per_sample: Tensor, predict_actions_t: Tensor | None, reduction: str) -> Tensor:
    """Mask non-action samples and apply the requested batch reduction."""
    if predict_actions_t is None:
        return per_sample if reduction == "none" else per_sample.mean()
    mask = predict_actions_t.to(per_sample.dtype)
    if reduction == "none":
        return per_sample * mask
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


# Materialized logits win at VLA token counts; larger dense targets use Liger.
_LOGITS_CE_MAX_POSITIONS = 2048


def _lin_ce_small(
    flat_hidden: Tensor,
    lm_head_weight: Tensor,
    flat_labels: Tensor,
    z_loss_weight: float = 0.0,
) -> Tensor:
    """Small-N linear CE on materialized logits (see ``_lin_ce_flat``)."""
    logits = (flat_hidden @ lm_head_weight.t()).float()
    n_valid = (flat_labels != -100).sum().clamp(min=1)
    loss = functional.cross_entropy(logits, flat_labels, ignore_index=-100, reduction="sum") / n_valid
    if z_loss_weight > 0:
        lse = torch.logsumexp(logits, dim=-1)
        valid = (flat_labels != -100).to(lse.dtype)
        loss = loss + float(z_loss_weight) * (lse.square() * valid).sum() / n_valid
    return loss


# Built lazily so importing this module does not invoke Dynamo.
_compiled_lin_ce_small = None


def _get_compiled_lin_ce_small():
    global _compiled_lin_ce_small
    if _compiled_lin_ce_small is None:
        _compiled_lin_ce_small = torch.compile(_lin_ce_small, dynamic=False)
    return _compiled_lin_ce_small


def _lin_ce_flat(
    flat_hidden: Tensor,
    lm_head_weight: Tensor,
    flat_labels: Tensor,
    z_loss_weight: float = 0.0,
    compiled: bool = False,
) -> Tensor:
    """Dispatch sparse targets to fixed logits buckets and dense targets to Liger."""
    if flat_hidden.shape[0] > _LOGITS_CE_MAX_POSITIONS:
        valid = flat_labels != -100
        compact_hidden = flat_hidden[valid]
        compact_labels = flat_labels[valid]
        compact_rows = compact_hidden.shape[0]

        if compact_rows == 0:
            return _lin_ce_flat(
                functional.pad(compact_hidden, (0, 0, 0, 1)),
                lm_head_weight,
                functional.pad(compact_labels, (0, 1), value=-100),
                z_loss_weight,
                compiled=compiled,
            )

        # Fixed power-of-two buckets avoid shape churn while keeping sparse
        # supervision on the materialized-logits path.
        bucket_rows = 1 << (compact_rows - 1).bit_length()
        if bucket_rows < flat_hidden.shape[0]:
            weighted_losses = []
            for start in range(0, compact_rows, _LOGITS_CE_MAX_POSITIONS):
                end = min(start + _LOGITS_CE_MAX_POSITIONS, compact_rows)
                rows = end - start
                chunk_rows = 1 << (rows - 1).bit_length()
                hidden_chunk = compact_hidden[start:end]
                labels_chunk = compact_labels[start:end]
                pad_rows = chunk_rows - rows
                if pad_rows:
                    hidden_chunk = functional.pad(hidden_chunk, (0, 0, 0, pad_rows))
                    labels_chunk = functional.pad(labels_chunk, (0, pad_rows), value=-100)
                chunk_loss = _lin_ce_flat(
                    hidden_chunk,
                    lm_head_weight,
                    labels_chunk,
                    z_loss_weight,
                    compiled=compiled,
                )
                weighted_losses.append(chunk_loss * rows)
            return torch.stack(weighted_losses).sum() / compact_rows

    if flat_hidden.shape[0] <= _LOGITS_CE_MAX_POSITIONS:
        fn = _get_compiled_lin_ce_small() if compiled else _lin_ce_small
        return fn(flat_hidden, lm_head_weight, flat_labels, z_loss_weight)

    # Keep Liger optional for inference-only installations.
    from liger_kernel.transformers.fused_linear_cross_entropy import (  # noqa: PLC0415
        LigerFusedLinearCrossEntropyLoss,
    )

    loss_fn = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        lse_square_scale=float(z_loss_weight),
        reduction="mean",
    )
    return loss_fn(lm_head_weight, flat_hidden, flat_labels)


def _shifted_lin_ce(
    hidden: Tensor,
    lm_head_weight: Tensor,
    labels: Tensor,
    z_loss_weight: float = 0.0,
    compiled: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Compute next-token CE through the shape-aware linear-CE dispatcher."""
    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().long()
    if reduction == "none":
        return torch.stack(
            [
                _lin_ce_flat(
                    sample_hidden.to(lm_head_weight.dtype),
                    lm_head_weight,
                    sample_labels,
                    z_loss_weight,
                    compiled=compiled,
                )
                for sample_hidden, sample_labels in zip(shift_hidden, shift_labels, strict=True)
            ]
        )
    batch_size, target_length, hidden_size = shift_hidden.shape
    flat_hidden = shift_hidden.reshape(batch_size * target_length, hidden_size)
    flat_labels = shift_labels.reshape(batch_size * target_length)
    # Match the dtype the eager path used: cast hidden to the lm_head's
    # weight dtype so bf16 weights see bf16 activations.
    flat_hidden = flat_hidden.to(lm_head_weight.dtype)
    return _lin_ce_flat(flat_hidden, lm_head_weight, flat_labels, z_loss_weight, compiled=compiled)


def _mark_target_span_causal(
    prefix_att_masks: Tensor, text_labels: Tensor, lang_start: int, lang_end: int
) -> Tensor:
    """Make supervised language targets causal while leaving prompts bidirectional."""
    att = prefix_att_masks.clone()
    n = min(text_labels.shape[1], lang_end - lang_start)
    if n <= 0:
        return att
    target = text_labels[:, :n] != -100  # (B, n) bool
    seg = att[:, lang_start : lang_start + n].bool()
    att[:, lang_start : lang_start + n] = seg | target
    return att


def _fast_lin_ce(
    hidden: Tensor,
    lm_head_weight: Tensor,
    action_tokens: Tensor,
    action_code_mask: Tensor,
    predict_actions_t: Tensor | None,
    compiled: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Compute FAST token CE over the enabled action-code positions."""
    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_targets = action_tokens[:, 1:].contiguous().long()
    shift_valid = action_code_mask[:, 1:].contiguous().bool()
    if predict_actions_t is not None:
        sample_mask = predict_actions_t[:, None].expand_as(shift_valid)
        shift_valid = shift_valid & sample_mask
    # Encode the mask with ignore_index to avoid a host sync and preserve graph capture.
    shift_targets = torch.where(shift_valid, shift_targets, torch.full_like(shift_targets, -100))

    if reduction == "none":
        return torch.stack(
            [
                _lin_ce_flat(
                    sample_hidden.to(lm_head_weight.dtype),
                    lm_head_weight,
                    sample_labels,
                    compiled=compiled,
                )
                for sample_hidden, sample_labels in zip(shift_hidden, shift_targets, strict=True)
            ]
        )
    batch_size, target_length, hidden_size = shift_hidden.shape
    flat_hidden = shift_hidden.reshape(batch_size * target_length, hidden_size).to(lm_head_weight.dtype)
    flat_labels = shift_targets.reshape(batch_size * target_length)
    return _lin_ce_flat(flat_hidden, lm_head_weight, flat_labels, compiled=compiled)


# ----------------------------------------------------------------------
# Knowledge insulation helpers
# ----------------------------------------------------------------------
# Action queries consume detached VLM K/V. Flow-only callers may additionally
# suppress the now-dead prefix graph without changing forward values.


# Consumer GPUs need smaller FlexAttention backward tiles at head_dim=256.
_FLEX_SHRUNK_TILES = {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}
_flex_kernel_options: dict[int, dict | None] = {}
_flex_fns: tuple | None | bool = None


def _get_flex_kernel_options(device: torch.device) -> dict | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if device_index not in _flex_kernel_options:
        smem = torch.cuda.get_device_properties(
            device_index
        ).shared_memory_per_block_optin  # spellchecker:disable-line
        _flex_kernel_options[device_index] = _FLEX_SHRUNK_TILES if smem < 128 * 1024 else None
    return _flex_kernel_options[device_index]


def _get_flex_fns(device: torch.device):
    """Return compiled FlexAttention helpers when available."""
    global _flex_fns
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if _flex_fns is None:
        try:
            from torch.nn.attention.flex_attention import (  # noqa: PLC0415
                create_block_mask,
                flex_attention,
            )

            _flex_fns = (
                torch.compile(flex_attention, dynamic=False),
                torch.compile(create_block_mask, dynamic=False),
            )
            _get_flex_kernel_options(device)
        except Exception as exc:
            logger.warning("PI052: FlexAttention unavailable (%s); using SDPA.", exc)
            _flex_fns = False
    return _flex_fns or None


class _FlexMaskBuilder:
    """Build KI masks while retaining stable compiled mask callables."""

    def __init__(self):
        self._key = None

    def build(self, prefix_pad, prefix_att, non_fast_prefix_len, k, chunk):
        _, create_bm = _get_flex_fns(prefix_pad.device)
        b, p = prefix_pad.shape
        a = k * chunk
        device = prefix_pad.device
        key = (b, p, a, int(non_fast_prefix_len), device)
        if self._key != key:
            self._key = key
            self._pad = torch.empty(b, p, dtype=torch.bool, device=device)
            self._cum = torch.empty(b, p, dtype=torch.long, device=device)
            pad, cum = self._pad, self._cum
            nf = int(non_fast_prefix_len)

            def vlm_rows(bi, h, q_idx, kv_idx):
                kv_p = kv_idx.clamp(max=p - 1)
                ok = (cum[bi, kv_p] <= cum[bi, q_idx]) & pad[bi, kv_p] & pad[bi, q_idx]
                return (kv_idx < p) & ok

            def action_rows(bi, h, q_idx, kv_idx):
                kv_p = kv_idx.clamp(max=p - 1)
                to_prefix = (kv_idx < nf) & pad[bi, kv_p]
                same_block = (q_idx // chunk) == ((kv_idx - p) // chunk)
                return to_prefix | ((kv_idx >= p) & same_block)

            self._vlm_mod, self._action_mod = vlm_rows, action_rows

        self._pad.copy_(prefix_pad)
        self._cum.copy_(torch.cumsum(prefix_att.to(torch.long), dim=1))
        s = p + a
        bm_vlm = create_bm(self._vlm_mod, B=b, H=None, Q_LEN=p, KV_LEN=s, device=device)
        bm_action = create_bm(self._action_mod, B=b, H=None, Q_LEN=a, KV_LEN=s, device=device)
        return bm_vlm, bm_action


# Lazily loaded FlashRT AdaRMS backend; unsupported cases use eager PyTorch.
_flashrt_adarms_cache = None


def _get_adarms_backend():
    global _flashrt_adarms_cache
    if _flashrt_adarms_cache is None:
        try:
            from kernels import get_kernel  # noqa: PLC0415

            _flashrt_adarms_cache = get_kernel("flashrt/flashrt-adarms-train", revision="v1")
        except Exception as exc:
            logger.warning(
                "PI052: flashrt-adarms-train unavailable (%s); using the eager norm path.",
                exc,
            )
            _flashrt_adarms_cache = False
    return _flashrt_adarms_cache or None


def _adarms_norm(backend, norm, x, cond):
    """PiGemmaRMSNorm through the fused kernel when a backend is present."""
    if backend is not None:
        if cond is not None and norm.dense is not None:
            return backend.adarms(x, norm.dense(cond), norm.eps, True)
        if norm.dense is None:
            return backend.adarms(x, norm.weight, norm.eps, False)
    return norm(x, cond=cond)


def _manual_attention_part(qs, ks, vs, m, scale):
    """Materialized-logits GQA with an FP32 softmax."""
    batch_size, num_heads, query_length, head_dim = qs.shape
    num_kv_heads = ks.shape[1]
    if num_kv_heads != num_heads:
        groups = num_heads // num_kv_heads
        grouped_queries = qs.reshape(batch_size, num_kv_heads, groups * query_length, head_dim)
        logits = (grouped_queries @ ks.transpose(-1, -2)).reshape(batch_size, num_heads, query_length, -1)
    else:
        logits = qs @ ks.transpose(-1, -2)
    logits = logits * scale + m
    p = logits.float().softmax(dim=-1).to(qs.dtype)
    out = (
        (p.reshape(batch_size, num_kv_heads, groups * query_length, -1) @ vs).reshape(
            batch_size, num_heads, query_length, head_dim
        )
        if num_kv_heads != num_heads
        else p @ vs
    )
    return out.transpose(1, 2).contiguous()


# Knowledge insulation keeps the forward equivalent while detaching VLM K/V for action-query gradients.
_manual_attention = None


def _get_manual_attention():
    """Load the Hub implementation, with the inline function as fallback."""
    global _manual_attention
    if _manual_attention is None:
        part = _manual_attention_part
        try:
            from kernels import get_kernel  # noqa: PLC0415

            _hub = getattr(
                get_kernel("flashrt/flashrt-flex-attention-train"),
                "manual_attention_part",
                None,
            )
            if _hub is not None:

                def part(qs, ks, vs, m, scale, _hub=_hub):
                    return _hub(qs, ks, vs, m, scale).transpose(1, 2).contiguous()

                logger.info("PI052: manual attention backed by flashrt-flex-attention-train (Hub).")
        except Exception as exc:
            logger.info(
                "PI052: flashrt-flex-attention-train unavailable (%s); using the inline manual-attention path.",
                exc,
            )
        _manual_attention = torch.compile(part, dynamic=False)
    return _manual_attention


def _compute_layer_ki(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_embeddings,
    adarms_cond,
    paligemma,
    gemma_expert,
    suppress_prefix_grads=False,
    flex_masks=None,
    adarms_backend=None,
    manual_attention=False,
):
    from transformers.models.gemma import modeling_gemma  # noqa: PLC0415

    # ``_gated_residual`` is LeRobot's adaRMSNorm helper, not a Transformers symbol.
    from ..pi_gemma import _gated_residual  # noqa: PLC0415

    def _vlm_ctx(i):
        return torch.no_grad() if (i == 0 and suppress_prefix_grads) else nullcontext()

    models = [paligemma.model.language_model, gemma_expert.model]
    query_states, key_states, value_states, gates = [], [], [], []

    vlm_len = inputs_embeds[0].shape[1]

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        with _vlm_ctx(i):
            hidden_states, gate = _adarms_norm(
                adarms_backend, layer.input_layernorm, hidden_states, adarms_cond[i]
            )
            gates.append(gate)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(q)
        key_states.append(k)
        value_states.append(v)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling

    # Split queries / K / V at the VLM-vs-action boundary.
    q_vlm = query_states[:, :, :vlm_len, :]
    q_action = query_states[:, :, vlm_len:, :]
    k_vlm = key_states[:, :, :vlm_len, :]
    k_action = key_states[:, :, vlm_len:, :]
    v_vlm = value_states[:, :, :vlm_len, :]
    v_action = value_states[:, :, vlm_len:, :]

    # Detach VLM K/V *only* on the path the action queries use.
    k_for_vlm = key_states
    v_for_vlm = value_states
    k_for_action = torch.cat([k_vlm.detach(), k_action], dim=2)
    v_for_action = torch.cat([v_vlm.detach(), v_action], dim=2)

    if flex_masks is not None:
        flex_attn, _ = _get_flex_fns(query_states.device)
        bm_vlm, bm_action = flex_masks
        n_rep = paligemma.model.language_model.layers[layer_idx].self_attn.num_key_value_groups
        with _vlm_ctx(0):
            att_vlm = flex_attn(
                q_vlm,
                k_for_vlm,
                v_for_vlm,
                block_mask=bm_vlm,
                scale=scaling,
                enable_gqa=n_rep > 1,
                kernel_options=_get_flex_kernel_options(query_states.device),
            ).transpose(1, 2)
        att_action = flex_attn(
            q_action,
            k_for_action,
            v_for_action,
            block_mask=bm_action,
            scale=scaling,
            enable_gqa=n_rep > 1,
            kernel_options=_get_flex_kernel_options(query_states.device),
        ).transpose(1, 2)
    else:
        mask_for_vlm = attention_mask[:, :, :vlm_len, :]
        mask_for_action = attention_mask[:, :, vlm_len:, :]
        # SDPA requires the additive bias to match each query dtype.
        if mask_for_vlm.dtype != q_vlm.dtype:
            mask_for_vlm = mask_for_vlm.to(dtype=q_vlm.dtype)
        if mask_for_action.dtype != q_action.dtype:
            mask_for_action = mask_for_action.to(dtype=q_action.dtype)

        if manual_attention:
            manual_fn = _get_manual_attention()
            if manual_attention == "action":
                from ..pi_gemma import sdpa_attention_forward  # noqa: PLC0415

                with _vlm_ctx(0):
                    att_vlm, _ = sdpa_attention_forward(
                        paligemma.model.language_model.layers[layer_idx].self_attn,
                        q_vlm,
                        k_for_vlm,
                        v_for_vlm,
                        mask_for_vlm,
                        scaling,
                    )
            else:
                with _vlm_ctx(0):
                    att_vlm = manual_fn(q_vlm, k_for_vlm, v_for_vlm, mask_for_vlm, scaling)
            att_action = manual_fn(q_action, k_for_action, v_for_action, mask_for_action, scaling)
        else:
            from ..pi_gemma import sdpa_attention_forward  # noqa: PLC0415

            with _vlm_ctx(0):
                att_vlm, _ = sdpa_attention_forward(
                    paligemma.model.language_model.layers[layer_idx].self_attn,
                    q_vlm,
                    k_for_vlm,
                    v_for_vlm,
                    mask_for_vlm,
                    scaling,
                )
            att_action, _ = sdpa_attention_forward(
                paligemma.model.language_model.layers[layer_idx].self_attn,
                q_action,
                k_for_action,
                v_for_action,
                mask_for_action,
                scaling,
            )
    att = torch.cat([att_vlm, att_action], dim=1)

    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att = att.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end = start + hidden_states.shape[1]
        if att.dtype != layer.self_attn.o_proj.weight.dtype:
            att = att.to(layer.self_attn.o_proj.weight.dtype)
        with _vlm_ctx(i):
            out_emb = layer.self_attn.o_proj(att[:, start:end])
            pa_norm = layer.post_attention_layernorm
            if adarms_backend is not None:
                if adarms_cond[i] is not None and pa_norm.dense is not None:
                    after_first, out_emb, gate = adarms_backend.resgate_adarms(
                        hidden_states,
                        out_emb,
                        gates[i],
                        pa_norm.dense(adarms_cond[i]),
                        pa_norm.eps,
                        True,
                    )
                else:
                    after_first, out_emb, gate = adarms_backend.resgate_adarms(
                        hidden_states, out_emb, gates[i], pa_norm.weight, pa_norm.eps, False
                    )
            else:
                out_emb = _gated_residual(hidden_states, out_emb, gates[i])
                after_first = out_emb
                out_emb, gate = pa_norm(out_emb, cond=adarms_cond[i])
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)
            out_emb = _gated_residual(after_first, out_emb, gate)
        outputs_embeds.append(out_emb)
        start = end
    return outputs_embeds


def _paligemma_forward_ki(
    self,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    adarms_cond=None,
    suppress_prefix_grads=False,
    flex_masks=None,
    adarms_backend=None,
    manual_attention=False,
):
    """Run dual-expert layers through KI and defer single-expert calls."""
    from ..pi_gemma import layernorm_forward  # noqa: PLC0415

    if adarms_cond is None:
        adarms_cond = [None, None]

    # Single-expert paths: defer to the original forward saved in
    # PI052Policy.__init__.
    if inputs_embeds[0] is None or inputs_embeds[1] is None:
        return self._pi052_orig_forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )

    models = [self.paligemma.model.language_model, self.gemma_expert.model]
    num_layers = self.paligemma.config.text_config.num_hidden_layers

    # RoPE values are shared by every layer.
    position_embeddings = self.paligemma.model.language_model.rotary_emb(inputs_embeds[0], position_ids)

    use_gc = (
        hasattr(self.gemma_expert.model, "gradient_checkpointing")
        and self.gemma_expert.model.gradient_checkpointing
        and self.training
    ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

    for layer_idx in range(num_layers):
        if use_gc:
            inputs_embeds = torch.utils.checkpoint.checkpoint(
                _compute_layer_ki,
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_embeddings,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
                paligemma=self.paligemma,
                gemma_expert=self.gemma_expert,
                suppress_prefix_grads=suppress_prefix_grads,
                flex_masks=flex_masks,
                adarms_backend=adarms_backend,
                manual_attention=manual_attention,
            )
        else:
            inputs_embeds = _compute_layer_ki(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_embeddings,
                adarms_cond,
                paligemma=self.paligemma,
                gemma_expert=self.gemma_expert,
                suppress_prefix_grads=suppress_prefix_grads,
                flex_masks=flex_masks,
                adarms_backend=adarms_backend,
                manual_attention=manual_attention,
            )

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        with torch.no_grad() if (i == 0 and suppress_prefix_grads) else nullcontext():
            out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
        outputs_embeds.append(out_emb)
    return [outputs_embeds[0], outputs_embeds[1]], None


class PI052Policy(PI05Policy):
    """π0.5 with the PaliGemma LM head re-enabled.

    It inherits unchanged PI0.5 policy behavior and replaces the core model with
    the joint flow/text implementation below.
    """

    config_class = PI052Config
    name = "pi052"

    def __init__(self, config: PI052Config, **kwargs: Any) -> None:
        # Patch before constructing Gemma/SigLIP layers; the operation is optional and idempotent.
        _enable_hf_kernels()

        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

        # Re-enable layers PI0.5 freezes when text supervision is requested.
        if config.text_loss_weight > 0 and config.unfreeze_lm_head:
            self._unfreeze_lm_head()

        # Bind knowledge insulation per instance so stock PI0.5 policies remain unchanged.
        if getattr(config, "knowledge_insulation", False):
            backbone = self.model.paligemma_with_expert
            backbone._pi052_orig_forward = backbone.forward
            backbone.forward = types.MethodType(_paligemma_forward_ki, backbone)
            logger.info(
                "PI052: knowledge insulation enabled — action→VLM K/V gradients are blocked in attention."
            )
            if config.use_flashrt_adarms:
                self._flashrt_adarms = _get_adarms_backend()
                if self._flashrt_adarms is not None:
                    logger.info("PI052: FlashRT adaRMS training kernels enabled.")

        if config.use_compiled_vision:
            _tower = self.model.paligemma_with_expert.paligemma.model.vision_tower
            _tower_eager_fwd = _tower.forward
            _tower_compiled_fwd = torch.compile(_tower_eager_fwd, dynamic=False)

            def _tower_dispatch(*args, _e=_tower_eager_fwd, _c=_tower_compiled_fwd, **kwargs):
                if torch.is_grad_enabled():
                    return _e(*args, **kwargs)
                return _c(*args, **kwargs)

            _tower.forward = _tower_dispatch
            logger.info("PI052: SigLIP vision tower compiled for no-grad passes.")

        # Cache the fixed K-repeat action mask outside the training step.
        if config.flow_num_repeats > 1:
            self.register_buffer(
                "_flow_block_diag",
                torch.block_diag(
                    *[
                        torch.ones(
                            config.chunk_size, config.chunk_size, dtype=torch.bool, device=config.device
                        )
                        for _ in range(config.flow_num_repeats)
                    ]
                ),
                persistent=False,
            )

        # Size per-environment inference state lazily.
        self.last_subtasks: list[str] | None = None
        self.last_subtasks_raw: list[str] | None = None
        self.last_subtasks_source: list[str] | None = None
        self._last_good_subtasks: list[str | None] | None = None

    def reset(self):
        """Reset action and high-level inference state."""
        # inlined PI05Policy.reset
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self.last_subtasks = None
        self.last_subtasks_raw = None
        self.last_subtasks_source = None
        self._last_good_subtasks = None
        # Counts action chunks since the last subtask (re)generation, so the
        # subtask can be held across several chunks (see subtask_replan_steps).
        self._subtask_chunk_counter = 0

    def apply_flashrt_fp8_mlp(self, batch: dict[str, Tensor], *, safety: float = 1.05) -> bool:
        """Opt-in: swap every Gemma + SigLIP MLP to FlashRT fused FP8 kernels.

        Calibrates static activation scales once on ``batch`` (one representative
        observation, already through the preprocessor) and swaps the MLP modules
        in place. Returns False (no-op, BF16 kept) if the kernels are missing.
        Gated by ``config.use_flashrt_fp8_mlp`` — see flashrt_fp8.py.
        """
        from .flashrt_fp8 import apply_fp8_mlp  # noqa: PLC0415

        return apply_fp8_mlp(self, batch, safety=safety)

    def _unfreeze_lm_head(self) -> None:
        """Walk the PaliGemma submodules and re-enable gradients on
        ``lm_head`` + the immediately preceding norm / last text-model
        layer that ``PI05Policy`` typically freezes."""
        backbone = self.model.paligemma_with_expert.paligemma
        if hasattr(backbone, "lm_head"):
            for p in backbone.lm_head.parameters():
                p.requires_grad_(True)
        # Discover terminal text layers dynamically across Transformers versions.
        text_model = getattr(backbone, "model", None)
        text_model = getattr(text_model, "language_model", text_model)
        if text_model is None:
            return
        norm = getattr(text_model, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)
        layers = getattr(text_model, "layers", None)
        if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
            for p in layers[-1].parameters():
                p.requires_grad_(True)

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Compute the enabled flow, text and FAST training losses."""
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported loss reduction: {reduction!r}")
        text_labels = batch.get("text_labels")
        predict_actions_t = batch.get("predict_actions")

        # Delegate only unannotated batches; PI0.5 ignores recipe action-routing masks.
        if (
            text_labels is None
            and predict_actions_t is None
            and not getattr(self.config, "enable_fast_action_loss", False)
        ):
            return self._pi05_flow_forward(batch, reduction=reduction)

        # Compute the host-side action-routing decision once for both flow and FAST.
        predict_any = predict_actions_t is None or bool(predict_actions_t.any().item())
        run_flow = self.config.flow_loss_weight > 0 and predict_any
        run_text = self.config.text_loss_weight > 0 and text_labels is not None

        loss_dict: dict[str, Any] = {}
        total: Tensor | None = None

        # Decide which losses fire this step.
        run_fast = (
            getattr(self.config, "enable_fast_action_loss", False)
            and self.config.fast_action_loss_weight > 0
            and predict_any
        )
        action_tokens = action_mask = action_code_mask = None
        if run_fast:
            from lerobot.utils.constants import (  # noqa: PLC0415
                ACTION_CODE_TOKEN_MASK,
                ACTION_TOKEN_MASK,
                ACTION_TOKENS,
            )

            action_tokens = batch.get(ACTION_TOKENS)
            action_mask = batch.get(ACTION_TOKEN_MASK)
            action_code_mask = batch.get(ACTION_CODE_TOKEN_MASK)
            if action_tokens is None or action_mask is None or action_code_mask is None:
                run_fast = False

        # Flow uses one fused prefix/suffix pass; text-only batches skip the suffix.
        if run_flow:
            flow_loss, text_loss, fast_loss = self._compute_all_losses_fused(
                batch,
                text_labels=text_labels if run_text else None,
                action_tokens=action_tokens if run_fast else None,
                action_mask=action_mask if run_fast else None,
                action_code_mask=action_code_mask if run_fast else None,
                predict_actions_t=predict_actions_t,
                reduction=reduction,
            )
            loss_dict["flow_loss"] = flow_loss.detach().mean()
            total = self.config.flow_loss_weight * flow_loss
            if text_loss is not None:
                loss_dict["text_loss"] = text_loss.detach().mean()
                total = total + self.config.text_loss_weight * text_loss
            if fast_loss is not None:
                loss_dict["fast_action_loss"] = fast_loss.detach().mean()
                total = total + self.config.fast_action_loss_weight * fast_loss
        elif run_text or run_fast:
            text_loss, fast_loss = self._compute_text_and_fast_loss(
                batch,
                text_labels=text_labels if run_text else None,
                action_tokens=action_tokens if run_fast else None,
                action_mask=action_mask if run_fast else None,
                action_code_mask=action_code_mask if run_fast else None,
                predict_actions_t=predict_actions_t,
                reduction=reduction,
            )
            if text_loss is not None:
                loss_dict["text_loss"] = text_loss.detach().mean()
                weighted = self.config.text_loss_weight * text_loss
                total = weighted if total is None else total + weighted
            if fast_loss is not None:
                loss_dict["fast_action_loss"] = fast_loss.detach().mean()
                weighted = self.config.fast_action_loss_weight * fast_loss
                total = weighted if total is None else total + weighted

        if total is None:
            # Both flow and text disabled — make this an obvious bug
            # rather than a silent zero loss.
            raise RuntimeError(
                "PI052Policy.forward: both flow_loss_weight and "
                "text_loss_weight are 0 (or text_labels missing) — "
                "nothing to train."
            )

        # Keep metrics detached on-device until logging to avoid extra CUDA synchronization.
        loss_dict["loss"] = total.detach().mean()
        return total, loss_dict

    def _compute_all_losses_fused(
        self,
        batch: dict[str, Tensor],
        text_labels: Tensor | None,
        action_tokens: Tensor | None,
        action_mask: Tensor | None,
        action_code_mask: Tensor | None,
        predict_actions_t: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Compute flow, text and FAST losses from one shared prefix."""
        # ---- preamble (mirrors PI05Pytorch.forward) ------------------
        actions = self.prepare_action(batch)

        # Flow-only KI steps have no live gradient path through the prefix.
        suppress_prefix_grads = (
            text_labels is None
            and action_tokens is None
            and getattr(self.config, "knowledge_insulation", False)
        )

        # ---- prefix: images + language + (optional FAST) -------------
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        with torch.no_grad() if suppress_prefix_grads else nullcontext():
            prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        non_fast_prefix_len = prefix_embs.shape[1]  # images + language only

        # Make supervised text causal rather than a bidirectional copy task.
        if text_labels is not None:
            lang_start = non_fast_prefix_len - text_labels.shape[1]
            if lang_start >= 0:
                prefix_att = _mark_target_span_causal(
                    prefix_att, text_labels, lang_start, non_fast_prefix_len
                )

        fast_len = 0
        if action_tokens is not None and action_mask is not None:
            # Gemma embedding already applies its hidden-size scale.
            fast_emb = self.model.paligemma_with_expert.embed_language_tokens(action_tokens)
            fast_len = action_tokens.shape[1]
            ones_att = torch.ones(
                (action_tokens.shape[0], fast_len),
                dtype=torch.bool,
                device=prefix_embs.device,
            )
            prefix_embs = torch.cat([prefix_embs, fast_emb], dim=1)
            prefix_pad = torch.cat([prefix_pad, action_mask.to(prefix_pad.dtype)], dim=1)
            prefix_att = torch.cat([prefix_att, ones_att], dim=1)

        # Amortized flow reuses one VLM prefix across fresh denoising targets.
        num_repeats = int(getattr(self.config, "flow_num_repeats", 1))
        if num_repeats > 1:
            prefix_out, flow_loss = self._amortized_prefix_and_flow(
                actions,
                prefix_embs,
                prefix_pad,
                prefix_att,
                non_fast_prefix_len,
                fast_len,
                predict_actions_t,
                num_repeats,
                suppress_prefix_grads=suppress_prefix_grads,
                reduction=reduction,
            )
        else:
            prefix_out, flow_loss = self._combined_prefix_and_flow(
                actions,
                prefix_embs,
                prefix_pad,
                prefix_att,
                non_fast_prefix_len,
                fast_len,
                predict_actions_t,
                suppress_prefix_grads=suppress_prefix_grads,
                reduction=reduction,
            )

        text_loss, fast_loss = self._prefix_ce_losses(
            prefix_out,
            text_labels,
            action_tokens,
            action_code_mask,
            fast_len,
            predict_actions_t,
            reduction,
        )
        return flow_loss, text_loss, fast_loss

    def _combined_prefix_and_flow(
        self,
        actions: Tensor,
        prefix_embs: Tensor,
        prefix_pad: Tensor,
        prefix_att: Tensor,
        non_fast_prefix_len: int,
        fast_len: int,
        predict_actions_t: Tensor | None,
        suppress_prefix_grads: bool = False,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor]:
        """Run the single-repeat combined prefix and action path."""
        from lerobot.utils.constants import ACTION  # noqa: PLC0415

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ---- suffix: noisy actions ----------------------------------
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.model.embed_suffix(x_t, time)

        # ---- bf16 alignment (mirrors PI05Pytorch.forward) -----------
        first_layer = self.model.paligemma_with_expert.paligemma.model.language_model.layers[0]
        if first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        # Block suffix-to-FAST attention to prevent trivial action leakage.
        if fast_len > 0:
            fast_start = non_fast_prefix_len
            fast_end = non_fast_prefix_len + fast_len  # = prefix_pad.shape[1]
            att_2d_masks[:, fast_end:, fast_start:fast_end] = False

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        if fast_len > 0:
            # Position flow parallel to FAST so its RoPE offsets match inference without FAST.
            non_fast_valid = prefix_pad[:, :non_fast_prefix_len].sum(dim=1, keepdim=True)
            suffix_pos = non_fast_valid + torch.cumsum(suffix_pad, dim=1) - 1
            position_ids = torch.cat([position_ids[:, : prefix_pad.shape[1]], suffix_pos], dim=1)
        att_2d_masks_4d = self.model._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)

        # ---- forward (capture BOTH expert outputs) ------------------
        ki_kwargs = self._ki_forward_kwargs(suppress_prefix_grads=suppress_prefix_grads)
        (prefix_out, suffix_out), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            **ki_kwargs,
        )

        # ---- flow loss (mirrors PI05Pytorch.forward) ----------------
        suffix_out_slice = suffix_out[:, -self.model.config.chunk_size :].to(dtype=torch.float32)
        v_t = self.model.action_out_proj(suffix_out_slice)
        flow_per_dim = functional.mse_loss(u_t, v_t, reduction="none")
        # Truncate to the actual action dimensionality (PI05 pads
        # internally to max_action_dim).
        original_action_dim = self.config.output_features[ACTION].shape[0]
        flow_per_dim = flow_per_dim[:, :, :original_action_dim]
        per_sample_flow = flow_per_dim.mean(dim=(1, 2))
        flow_loss = _reduce_action_loss(per_sample_flow, predict_actions_t, reduction)
        return prefix_out, flow_loss

    def _ki_forward_kwargs(self, suppress_prefix_grads: bool = False, flex_masks=None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if suppress_prefix_grads:
            kwargs["suppress_prefix_grads"] = True
        if flex_masks is not None:
            kwargs["flex_masks"] = flex_masks
        adarms_backend = getattr(self, "_flashrt_adarms", None)
        if adarms_backend is not None:
            kwargs["adarms_backend"] = adarms_backend
        if self.config.use_manual_attention:
            kwargs["manual_attention"] = self.config.manual_attention_scope
        return kwargs

    def _amortized_prefix_and_flow(
        self,
        actions: Tensor,
        prefix_embs: Tensor,
        prefix_pad: Tensor,
        prefix_att: Tensor,
        non_fast_prefix_len: int,
        fast_len: int,
        predict_actions_t: Tensor | None,
        num_repeats: int,
        suppress_prefix_grads: bool = False,
        reduction: str = "mean",
    ) -> tuple[Tensor, Tensor]:
        """Run K independent action draws against one shared VLM prefix."""
        from lerobot.utils.constants import ACTION  # noqa: PLC0415

        model = self.model
        k = num_repeats
        chunk = self.config.chunk_size
        batch_size, prefix_len = prefix_pad.shape

        first_layer = model.paligemma_with_expert.paligemma.model.language_model.layers[0]
        use_bf16 = first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16
        if use_bf16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # ---- K suffix blocks: independent noise/time draws ----------
        # Embed all independent K draws in one flattened batch.
        noise = model.sample_noise((k * batch_size, *actions.shape[1:]), actions.device)
        time = model.sample_time(k * batch_size, actions.device)
        actions_rep = actions.repeat(k, 1, 1)  # (k*B, chunk, motor_dim)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions_rep
        u_t = (noise - actions_rep).view(k, batch_size, chunk, -1).transpose(0, 1)  # (B, k, chunk, motor)
        s_embs, suffix_pad, suffix_att, adarms = model.embed_suffix(x_t, time)
        if use_bf16:
            s_embs = s_embs.to(dtype=torch.bfloat16)
        suffix_pad = suffix_pad[:batch_size]
        suffix_att = suffix_att[:batch_size]
        suffix_embs = (
            s_embs.view(k, batch_size, chunk, -1).transpose(0, 1).reshape(batch_size, k * chunk, -1)
        )  # (B, k*chunk, D)
        # Broadcast each draw's AdaRMS condition over its action chunk.
        adarms_cond = (
            adarms.view(k, batch_size, 1, adarms.shape[-1])
            .expand(k, batch_size, chunk, adarms.shape[-1])
            .transpose(0, 1)
            .reshape(batch_size, k * chunk, adarms.shape[-1])
        )  # (B, k*chunk, cond_dim)

        # Prefix rows cannot see action blocks; each action block sees only itself and the prefix.
        use_flex = (
            self.config.use_flex_attention
            and getattr(self.config, "knowledge_insulation", False)
            and not getattr(self, "_flex_attention_disabled", False)
            and _get_flex_fns(prefix_pad.device) is not None
        )
        flex_masks = None
        if use_flex:
            try:
                if not hasattr(self, "_flex_mask_builder"):
                    self._flex_mask_builder = _FlexMaskBuilder()
                flex_masks = self._flex_mask_builder.build(
                    prefix_pad, prefix_att, non_fast_prefix_len, k, chunk
                )
            except Exception as exc:
                logger.warning("PI052: FlexAttention initialization failed (%s); using SDPA.", exc)
                self._flex_attention_disabled = True
        if flex_masks is not None:
            att_2d_4d = None
        else:
            device = prefix_pad.device
            prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)  # (B, P, P)
            prefix_rows = torch.cat(
                [
                    prefix_att_2d,
                    torch.zeros(batch_size, prefix_len, k * chunk, dtype=torch.bool, device=device),
                ],
                dim=2,
            )

            action_to_prefix = prefix_pad[:, None, :].expand(batch_size, k * chunk, prefix_len).clone()
            if fast_len > 0:
                action_to_prefix[:, :, non_fast_prefix_len:prefix_len] = False
            action_to_action = self._flow_block_diag[None].expand(batch_size, k * chunk, k * chunk)
            action_rows = torch.cat([action_to_prefix, action_to_action], dim=2)

            att_2d = torch.cat([prefix_rows, action_rows], dim=1)  # (B, P + k*chunk, P + k*chunk)
            att_2d_4d = model._prepare_attention_masks_4d(att_2d, dtype=prefix_embs.dtype)

        # Restart every independent flow block after the non-FAST prefix to match inference RoPE.
        if fast_len > 0:
            prefix_offsets = prefix_pad[:, :non_fast_prefix_len].sum(dim=-1)[:, None]
        else:
            prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
        block_positions = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1  # (B, chunk)
        position_ids = torch.cat([torch.cumsum(prefix_pad, dim=1) - 1, block_positions.repeat(1, k)], dim=1)

        ki_kwargs = self._ki_forward_kwargs(suppress_prefix_grads, flex_masks)
        (prefix_out, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            **ki_kwargs,
        )

        # ---- flow loss averaged over the K blocks -------------------
        # Project all blocks together before averaging their losses.
        original_action_dim = self.config.output_features[ACTION].shape[0]
        v_t = model.action_out_proj(suffix_out.to(dtype=torch.float32))
        v_t = v_t.view(batch_size, k, chunk, -1)  # (B, k, chunk, motor)
        flow_per_dim = functional.mse_loss(u_t, v_t, reduction="none")[..., :original_action_dim]
        per_sample_flow = flow_per_dim.mean(dim=(1, 2, 3))
        flow_loss = _reduce_action_loss(per_sample_flow, predict_actions_t, reduction)
        return prefix_out, flow_loss

    def _prefix_ce_losses(
        self,
        prefix_out: Tensor | None,
        text_labels: Tensor | None,
        action_tokens: Tensor | None,
        action_code_mask: Tensor | None,
        fast_len: int,
        predict_actions_t: Tensor | None,
        reduction: str = "mean",
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute enabled text and FAST losses from the shared prefix output."""
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head

        text_loss: Tensor | None = None
        if text_labels is not None and prefix_out is not None:
            lang_len = text_labels.shape[1]
            if fast_len > 0:
                text_hidden = prefix_out[:, -(fast_len + lang_len) : -fast_len, :]
            else:
                text_hidden = prefix_out[:, -lang_len:, :]
            # Liger avoids materializing the full vocabulary logits tensor.
            text_loss = _shifted_lin_ce(
                text_hidden,
                lm_head.weight,
                text_labels,
                z_loss_weight=getattr(self.config, "text_ce_z_loss_weight", 0.0),
                compiled=self.config.use_compiled_text_ce,
                reduction=reduction,
            )

        fast_loss: Tensor | None = None
        if fast_len > 0 and prefix_out is not None and action_code_mask is not None:
            fast_hidden = prefix_out[:, -fast_len:, :]
            fast_loss = _fast_lin_ce(
                fast_hidden,
                lm_head.weight,
                action_tokens,
                action_code_mask,
                predict_actions_t,
                compiled=self.config.use_compiled_text_ce,
                reduction=reduction,
            )

        return text_loss, fast_loss

    def _compute_text_and_fast_loss(
        self,
        batch: dict[str, Tensor],
        text_labels: Tensor | None,
        action_tokens: Tensor | None,
        action_mask: Tensor | None,
        action_code_mask: Tensor | None,
        predict_actions_t: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor | None, Tensor | None]:
        """Single prefix forward → text CE + FAST CE.

        Embed [images, language] (and FAST when requested) once, run
        one backbone forward, then slice the resulting hidden states
        at the language and FAST positions to compute both CE losses.
        Bit-equivalent to running the two losses in separate forwards
        because the segment-aware ``make_att_2d_masks`` keeps FAST
        tokens invisible to language tokens, so adding FAST to the
        prefix doesn't perturb the hidden states at language positions.

        Returns ``(text_loss, fast_loss)``. Either can be ``None`` if
        the caller doesn't want that head.
        """

        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Make supervised text causal before appending FAST tokens.
        if text_labels is not None:
            lang_start = prefix_embs.shape[1] - text_labels.shape[1]
            if lang_start >= 0:
                prefix_att = _mark_target_span_causal(
                    prefix_att, text_labels, lang_start, prefix_embs.shape[1]
                )

        fast_len = 0
        if action_tokens is not None and action_mask is not None:
            # embed_language_tokens already applies the Gemma sqrt(hidden) scale (tf>=5.4.0);
            # do not scale FAST action tokens again (would double-scale).
            fast_emb = self.model.paligemma_with_expert.embed_language_tokens(action_tokens)

            fast_len = action_tokens.shape[1]
            ones_att = torch.ones(
                (action_tokens.shape[0], fast_len),
                dtype=torch.bool,
                device=prefix_embs.device,
            )
            full_embs = torch.cat([prefix_embs, fast_emb], dim=1)
            full_pad = torch.cat([prefix_pad, action_mask.to(prefix_pad.dtype)], dim=1)
            full_att = torch.cat([prefix_att, ones_att], dim=1)
        else:
            full_embs = prefix_embs
            full_pad = prefix_pad
            full_att = prefix_att

        att_2d = make_att_2d_masks(full_pad, full_att)
        position_ids = torch.cumsum(full_pad, dim=1) - 1
        att_2d_4d = self.model._prepare_attention_masks_4d(att_2d, dtype=full_embs.dtype)

        (vlm_out, _), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[full_embs, None],
            use_cache=False,
        )
        if vlm_out is None:
            raise RuntimeError("PI052 text+fast loss: VLM forward returned no hidden states.")

        lm_head = self.model.paligemma_with_expert.paligemma.lm_head

        text_loss: Tensor | None = None
        if text_labels is not None:
            lang_len = text_labels.shape[1]
            # embed_prefix lays out as [images, language]; with FAST
            # appended the full sequence is [images, language, FAST].
            if fast_len > 0:
                text_hidden = vlm_out[:, -(fast_len + lang_len) : -fast_len, :]
            else:
                text_hidden = vlm_out[:, -lang_len:, :]
            text_loss = _shifted_lin_ce(
                text_hidden,
                lm_head.weight,
                text_labels,
                z_loss_weight=getattr(self.config, "text_ce_z_loss_weight", 0.0),
                compiled=self.config.use_compiled_text_ce,
                reduction=reduction,
            )

        fast_loss: Tensor | None = None
        if action_tokens is not None and action_code_mask is not None and fast_len > 0:
            fast_hidden = vlm_out[:, -fast_len:, :]
            fast_loss = _fast_lin_ce(
                fast_hidden,
                lm_head.weight,
                action_tokens,
                action_code_mask,
                predict_actions_t,
                compiled=self.config.use_compiled_text_ce,
                reduction=reduction,
            )

        return text_loss, fast_loss

    def select_message(
        self,
        batch: dict[str, Tensor],
        *,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer: Any = None,
        suppress_loc_tokens: bool = False,
        use_kv_cache: bool = True,
    ) -> str:
        """Generate text continuation from a multimodal prefix (used by the runtime CLI).

        ``suppress_loc_tokens=True`` masks PaliGemma's reserved ``<locDDDD>`` ids
        ([256000, 257024)) before sampling — the pretraining prior drifts back to
        them on small text-CE budgets. Pass ``True`` for subtask/memory/plan,
        ``False`` for VQA (spatial answers legitimately emit ``<loc>``).
        """
        self.eval()

        if tokenizer is None:
            from transformers import AutoTokenizer  # noqa: PLC0415

            from .inference.pi052_adapter import _get_loc_tokenizer  # noqa: PLC0415
            from .text_processor_pi052 import register_paligemma_loc_tokens  # noqa: PLC0415

            tok_name = getattr(self.config, "tokenizer_name", None) or "google/paligemma-3b-pt-224"
            tokenizer = _get_loc_tokenizer(tok_name, AutoTokenizer, register_paligemma_loc_tokens)
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        special_ids: set[int] = set()
        try:
            for sid in tokenizer.all_special_ids or []:
                if sid is not None:
                    special_ids.add(int(sid))
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        if eos_token_id is not None:
            special_ids.add(int(eos_token_id))

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )

        device = prefix_embs.device
        bsize = prefix_embs.shape[0]
        ones_step = torch.ones((bsize, 1), dtype=torch.bool, device=device)

        current_embs = prefix_embs
        current_pad = prefix_pad_masks
        current_att = prefix_att_masks
        generated: list[int] = []
        new_emb = None

        # Cache the image-heavy prefix; disabling the cache retains the full-recompute parity path.
        cache = None

        backbone = self.model.paligemma_with_expert
        lm_head = backbone.paligemma.lm_head

        # Use q_proj's dtype because norms and embeddings may remain fp32 while SDPA queries are bf16.
        backbone_dtype = backbone.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype

        for _ in range(max_new_tokens):
            if cache is None:
                # Run the full bidirectional prefix initially or whenever caching is disabled.
                step_embs = current_embs
                att_2d = make_att_2d_masks(current_pad, current_att)
                position_ids = torch.cumsum(current_pad, dim=1) - 1
                att_2d_4d = self.model._prepare_attention_masks_4d(att_2d, dtype=backbone_dtype)
            else:
                # Incremental decoding feeds only the last token while retaining prefix padding masks.
                step_embs = new_emb
                att_2d = current_pad[:, None, :]
                att_2d_4d = self.model._prepare_attention_masks_4d(att_2d, dtype=backbone_dtype)
                position_ids = (torch.cumsum(current_pad, dim=1) - 1)[:, -1:]
            (vlm_out, _), new_cache = backbone.forward(
                attention_mask=att_2d_4d,
                position_ids=position_ids,
                past_key_values=cache,
                inputs_embeds=[step_embs, None],
                use_cache=use_kv_cache,
            )
            if use_kv_cache:
                cache = new_cache
            if vlm_out is None:
                break
            last = vlm_out[:, -1:].to(lm_head.weight.dtype)
            logits_step = lm_head(last)[:, -1]  # (B, V)
            if special_ids and len(generated) < min_new_tokens:
                for sid in special_ids:
                    logits_step[..., sid] = float("-inf")
            # Suppress FAST-only vocabulary that otherwise leaks into generated text.
            vocab_size = logits_step.shape[-1]
            fast_skip = int(getattr(self.config, "fast_skip_tokens", 128))
            fast_lo = vocab_size - 1 - fast_skip - (_FAST_ACTION_VOCAB_SIZE - 1)
            if 0 < fast_lo < 256000:
                logits_step[..., fast_lo:256000] = float("-inf")
            if suppress_loc_tokens:
                logits_step[..., 256000:257024] = float("-inf")
            next_ids = self._sample_next_token(logits_step, temperature, top_p)
            tok_id = int(next_ids[0].item())
            generated.append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id:
                break

            # embed_language_tokens already applies the Gemma sqrt(hidden) scale (tf>=5.4.0).
            new_emb = backbone.embed_language_tokens(next_ids.unsqueeze(0))
            # Both paths track valid keys, but only recompute retains full embedding history.
            current_pad = torch.cat([current_pad, ones_step], dim=1)
            if not use_kv_cache:
                current_embs = torch.cat([current_embs, new_emb], dim=1)
                current_att = torch.cat([current_att, ones_step], dim=1)

        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not decoded and generated:
            try:
                self._last_select_message_debug = (
                    f"raw_ids={generated[:16]} "
                    f"decoded_w_special={tokenizer.decode(generated, skip_special_tokens=False)!r}"
                )
            except Exception:  # noqa: BLE001
                self._last_select_message_debug = f"raw_ids={generated[:16]}"
        else:
            self._last_select_message_debug = ""
        return decoded

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select an action via PI052's high-level → low-level inference path.

        At action-chunk boundaries, first generate a low-level subtask from
        the high-level task prompt. Then retokenize that subtask as the
        low-level action prompt before sampling the action chunk. This keeps
        the public policy API identical to PI05 (`Tensor` action out), while
        matching the PI052 training/runtime conditioning more closely.
        """
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        if len(self._action_queue) == 0:
            action_batch = self._with_low_level_subtask_prompt(batch)
            actions = self.predict_action_chunk(action_batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def _with_low_level_subtask_prompt(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        from .inference.pi052_adapter import _build_text_batch  # noqa: PLC0415
        from .text_processor_pi052 import discretize_state_str  # noqa: PLC0415

        n = self._batch_size_from_observation(batch)
        self._ensure_subtask_state(n)
        tasks = self._tasks_from_batch(batch, n)
        # Mirror training by appending the already normalized state to low-level prompts.
        state_all = batch.get(OBS_STATE)

        # Hold subtasks for the configured interval to match training and avoid rapid replanning.
        replan = int(getattr(self.config, "subtask_replan_steps", 0) or 0)
        hold_chunks = max(1, round(replan / self.config.n_action_steps)) if replan > 0 else 1
        regenerate = self._subtask_chunk_counter % hold_chunks == 0 or not any(self.last_subtasks or [])
        self._subtask_chunk_counter += 1

        # Generate and batch one independently conditioned subtask per environment.
        rows: list[tuple[Tensor, Tensor | None]] = []
        tokenizer = None
        for i in range(n):
            if regenerate or not self.last_subtasks[i]:
                obs_i = self._slice_observation(batch, i)
                subtask = self._generate_low_level_subtask(obs_i, tasks[i], i)
            else:
                # Hold the previously generated subtask; only the state in the
                # prompt below is refreshed to the current observation.
                subtask = self.last_subtasks[i]
            content = subtask
            if torch.is_tensor(state_all):
                content = f"{subtask}, State: {discretize_state_str(state_all[i])};"
            text_batch = _build_text_batch(
                self,
                [{"role": "user", "content": content}],
                add_generation_prompt=False,
            )
            rows.append((text_batch["lang_tokens"], text_batch["lang_masks"]))
            tokenizer = text_batch["tokenizer"]

        tokens, masks = self._stack_token_rows(rows, tokenizer)

        out = dict(batch)
        out[OBS_LANGUAGE_TOKENS] = tokens
        out[OBS_LANGUAGE_ATTENTION_MASK] = masks
        return out

    def _generate_low_level_subtask(self, obs_i: dict[str, Tensor], task: str, i: int) -> str:
        from .inference.pi052_adapter import _generate_with_policy  # noqa: PLC0415

        msg = ""
        if task:
            msg = _generate_with_policy(
                self,
                [{"role": "user", "content": task}],
                observation=obs_i,
                label=f"eval subtask gen[{i}]",
                suppress_loc_tokens=True,
            )
        self.last_subtasks_raw[i] = msg or ""

        # Feed the generated subtask verbatim, matching low-level training.
        if msg:
            subtask = " ".join(msg.strip().split())
            self._last_good_subtasks[i] = subtask
            self.last_subtasks[i] = subtask
            self.last_subtasks_source[i] = "generated"
            logger.info("PI052 eval subtask[%d]: %r (task=%r)", i, subtask, task)
            return subtask

        # Reuse the last valid subtask, or derive an initial imperative, when generation fails.
        debug = getattr(self, "_last_select_message_debug", "") or ""
        if not task:
            reason = "No task string was available in the batch."
        else:
            reason = f"Empty generated subtask. {debug}".strip()
        if self._last_good_subtasks[i]:
            subtask = self._last_good_subtasks[i]
            source = "reuse_last"
        else:
            subtask = self._fallback_subtask_from_task(task)
            source = "fallback_task"
        self.last_subtasks[i] = subtask
        self.last_subtasks_source[i] = source
        logger.info(
            "PI052 eval subtask[%d] fallback (%s): %s | final=%r task=%r",
            i,
            source,
            reason,
            subtask,
            task,
        )
        return subtask

    def _ensure_subtask_state(self, n: int) -> None:
        """(Re)allocate per-env subtask buffers when the env count is first seen."""
        if self.last_subtasks is not None and len(self.last_subtasks) == n:
            return
        self.last_subtasks = ["" for _ in range(n)]
        self.last_subtasks_raw = ["" for _ in range(n)]
        self.last_subtasks_source = ["unset" for _ in range(n)]
        self._last_good_subtasks = [None for _ in range(n)]

    @staticmethod
    def _slice_observation(batch: dict[str, Tensor], i: int) -> dict[str, Tensor]:
        """Slice the per-env observation tensors for env ``i`` (images/state).

        Language keys are excluded so high-level generation uses the freshly
        tokenized task prompt, not the preprocessor's low-level fallback tokens.
        """
        out: dict[str, Tensor] = {}
        for k, v in batch.items():
            if not (isinstance(k, str) and k.startswith("observation.")):
                continue
            if k.startswith("observation.language"):
                continue
            if torch.is_tensor(v):
                out[k] = v[i : i + 1]
        return out

    @staticmethod
    def _stack_token_rows(rows: list[tuple[Tensor, Tensor | None]], tokenizer: Any) -> tuple[Tensor, Tensor]:
        """Right-pad per-env ``(1, L_i)`` token/mask rows and stack to ``(n, L)``.

        Right-padding with a False attention mask matches the training-time
        tokenizer (``padding_side="right"``), so the action expert treats pad
        positions as masked.
        """
        max_len = max(t.shape[1] for t, _ in rows)
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        tok_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        for tokens, masks in rows:
            length = tokens.shape[1]
            if masks is None:
                masks = torch.ones((1, length), dtype=torch.bool, device=tokens.device)
            if length < max_len:
                pad = max_len - length
                tokens = torch.cat(
                    [tokens, torch.full((1, pad), pad_id, dtype=tokens.dtype, device=tokens.device)],
                    dim=1,
                )
                masks = torch.cat(
                    [masks, torch.zeros((1, pad), dtype=masks.dtype, device=masks.device)],
                    dim=1,
                )
            tok_rows.append(tokens)
            mask_rows.append(masks)
        return torch.cat(tok_rows, dim=0), torch.cat(mask_rows, dim=0)

    @staticmethod
    def _fallback_subtask_from_task(task: str) -> str:
        target = PI052Policy._navigation_target_from_task(task)
        if target:
            return f"go to {target}"
        if task.lower().startswith("open the stand mixer head"):
            return "pull stand mixer head"
        return task

    @staticmethod
    def _navigation_target_from_task(task: str) -> str:
        prefix = "navigate to "
        lower = task.lower().strip()
        if not lower.startswith(prefix):
            return ""
        return lower[len(prefix) :].strip().rstrip(".")

    @staticmethod
    def _tasks_from_batch(batch: dict[str, Any], n: int) -> list[str]:
        """Return one task string per env, padded/truncated to ``n``."""
        task = batch.get("task")
        if isinstance(task, list):
            raw = list(task)
        elif task is None:
            raw = []
        else:
            raw = [task]
        tasks: list[str] = []
        for t in raw:
            if hasattr(t, "item"):
                t = t.item()
            tasks.append(t if isinstance(t, str) else "")
        if len(tasks) < n:
            tasks += [tasks[-1] if tasks else ""] * (n - len(tasks))
        return tasks[:n]

    @staticmethod
    def _batch_size_from_observation(batch: dict[str, Any]) -> int:
        state = batch.get("observation.state")
        if torch.is_tensor(state) and state.ndim > 0:
            return int(state.shape[0])
        for key, value in batch.items():
            if isinstance(key, str) and key.startswith("observation.images.") and torch.is_tensor(value):
                return int(value.shape[0])
        return 1

    @staticmethod
    def _sample_next_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_ix = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_p, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_p = sorted_p.masked_fill(mask, 0.0)
            sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            choice = torch.multinomial(sorted_p, num_samples=1)
            return sorted_ix.gather(-1, choice).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # PI0.5 flow-only fallback for unannotated batches.
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
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load a PI05/PI052 checkpoint, including sharded safetensors checkpoints."""
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

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

        model = cls(config, **kwargs)
        files = _resolve_weight_files(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        fixed_state_dict = model._fix_pytorch_state_dict_keys(_load_weight_files(files), model.config)
        remapped_state_dict = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed_state_dict.items()
        }

        lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
        embed_tokens_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        if lm_head_key not in remapped_state_dict and embed_tokens_key in remapped_state_dict:
            remapped_state_dict[lm_head_key] = remapped_state_dict[embed_tokens_key].clone().float()
        elif lm_head_key in remapped_state_dict:
            remapped_state_dict[lm_head_key] = remapped_state_dict[lm_head_key].float()

        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)
        if not strict:
            if missing_keys:
                logger.warning("Missing PI052 checkpoint keys: %s", missing_keys)
            if unexpected_keys:
                logger.warning("Unexpected PI052 checkpoint keys: %s", unexpected_keys)
        model.to(config.device)
        model.eval()
        return model

    def get_optim_params(self):
        """Return policy parameters, optionally split into LR-scaled groups.

        Three orthogonal multipliers scale the base ``optimizer_lr``:
        ``lm_head_lr_scale`` (PaliGemma ``lm_head`` + tied ``embed_tokens``),
        ``backbone_lr_scale`` (the rest of the PaliGemma tower), and
        ``action_expert_lr_scale`` (the Gemma expert + action/time projection
        heads). The cosine scheduler multiplies every group by the same lambda
        each step so the ratios are preserved across decay. When all three are
        ``1.0`` this returns ``self.parameters()`` (back-compat with existing
        checkpoints and configs).
        """
        head_scale = float(getattr(self.config, "lm_head_lr_scale", 1.0))
        backbone_scale = float(getattr(self.config, "backbone_lr_scale", 1.0))
        expert_scale = float(getattr(self.config, "action_expert_lr_scale", 1.0))
        if head_scale == 1.0 and backbone_scale == 1.0 and expert_scale == 1.0:
            return self.parameters()

        # Keep the tied LM projection and embeddings in the same optimizer group.
        head_substrings = (
            "paligemma_with_expert.paligemma.lm_head.",
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.",
        )
        backbone_substring = "paligemma_with_expert.paligemma."
        head_params: list[torch.nn.Parameter] = []
        backbone_params: list[torch.nn.Parameter] = []
        expert_params: list[torch.nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(s in name for s in head_substrings):
                head_params.append(p)
            elif backbone_substring in name:
                backbone_params.append(p)
            else:
                expert_params.append(p)
        base_lr = float(self.config.optimizer_lr)
        groups: list[dict[str, object]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_scale, "name": "backbone"})
        if expert_params:
            groups.append({"params": expert_params, "lr": base_lr * expert_scale, "name": "action_expert"})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr * head_scale, "name": "lm_head"})
        # Sanity: a non-trivial head scale that matches no params would silently
        # do nothing — surface that fast.
        if head_scale != 1.0 and not head_params:
            raise RuntimeError(
                "lm_head_lr_scale != 1.0 but no parameters matched the LM-head "
                f"name patterns: {head_substrings!r}. Did the underlying PaliGemma "
                "module rename?"
            )
        logging.info(
            "PI052Policy LR groups (base=%.3g): backbone=%.3g (×%.3g, n=%d), "
            "action_expert=%.3g (×%.3g, n=%d), lm_head=%.3g (×%.3g, n=%d)",
            base_lr,
            base_lr * backbone_scale,
            backbone_scale,
            len(backbone_params),
            base_lr * expert_scale,
            expert_scale,
            len(expert_params),
            base_lr * head_scale,
            head_scale,
            len(head_params),
        )
        return groups

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Guard before first-observation FP8 calibration to prevent recursive prediction.
        if self.config.use_flashrt_fp8_mlp and not getattr(self, "_fp8_applied", False):
            self._fp8_applied = True
            self.apply_flashrt_fp8_mlp(batch)

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def _pi05_flow_forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
