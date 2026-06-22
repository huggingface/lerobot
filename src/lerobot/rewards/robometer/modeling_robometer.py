# Copyright 2026 Anthony Liang, Yigit Korkmaz, Stephen Tu, Erdem Bıyık, Jesse Zhang
# and The HuggingFace Inc. team. All rights reserved.
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

"""ROBOMETER: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons.

Paper:         https://arxiv.org/abs/2603.02115
Project:       https://robometer.github.io
Original code: https://github.com/aliang8/robometer
Model:         https://huggingface.co/robometer/Robometer-4B

Robometer is a general-purpose, video-language-input reward model built on
``Qwen/Qwen3-VL-4B-Instruct``. It is trained with a dual reward-prediction
objective:

- A frame-level progress loss anchoring reward magnitude on expert data.
- A trajectory-comparison preference loss imposing global ordering constraints
  across trajectories sharing the same instruction.

To support downstream RL it also predicts a frame-level binary success. The
training prompt inserts three learnable tokens:

- ``<|prog_token|>`` after each frame to read per-frame progress and success.
- ``<|pref_token|>`` at the end to read pairwise preference (training-only).
- ``<|split_token|>`` between two trajectories in preference samples
  (training-only).

Progress is modeled as a categorical distribution over ``progress_discrete_bins``
uniformly-spaced centers in ``[0, 1]`` (C51-style), and the continuous estimate
is recovered as the softmax-weighted mean of those centers — see
:func:`convert_bins_to_continuous`.

This LeRobot port is **inference-only**: the preference head is preserved in
the state dict for byte-equivalence with the published ``Robometer-4B``
checkpoint but is not queried by :meth:`RobometerRewardModel.compute_reward`,
which returns the last-frame progress (clamped to ``[0, 1]``) or sigmoid'd
success probability depending on :attr:`RobometerConfig.reward_output`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.utils.constants import OBS_PREFIX
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModelForImageTextToText
else:
    AutoModelForImageTextToText = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Namespace for Robometer's pre-encoded Qwen-VL observation tensors.
ROBOMETER_FEATURE_PREFIX = f"{OBS_PREFIX}robometer."
ROBOMETER_QWEN_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "mm_token_type_ids",
)
ROBOMETER_METADATA_KEYS = (
    "prog_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "video_merge_size",
)
ROBOMETER_INPUT_KEYS = ROBOMETER_QWEN_INPUT_KEYS + ROBOMETER_METADATA_KEYS


def convert_bins_to_continuous(bin_logits: Tensor) -> Tensor:
    """Collapse per-bin logits into a single value in ``[0, 1]``.

    The discrete progress head outputs ``num_bins`` logits per frame. Bins are
    evenly spaced centers in ``[0, 1]``; the continuous prediction is the
    softmax-weighted mean of those centers.
    """
    bin_probs = torch.softmax(bin_logits, dim=-1)
    num_bins = bin_logits.shape[-1]
    bin_centers = torch.linspace(0.0, 1.0, num_bins, device=bin_logits.device, dtype=bin_logits.dtype)
    return (bin_probs * bin_centers).sum(dim=-1)


def _squeeze_last_safe(x: Tensor) -> Tensor:
    """Drop a trailing singleton dim only when present."""
    return x.squeeze(-1) if x.ndim > 1 and x.shape[-1] == 1 else x


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unknown torch dtype: {name!r}")


class RobometerPredictionHead(nn.Sequential):
    """Small MLP head used for Robometer's progress / success / preference outputs."""

    def __init__(self, hidden_dim: int, output_size: int, *, dropout: float, with_sigmoid: bool) -> None:
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size),
        ]
        if with_sigmoid:
            layers.append(nn.Sigmoid())
        super().__init__(*layers)


def decode_progress_outputs(
    progress_logits: Tensor | None,
    success_logits: Tensor | None,
    *,
    is_discrete_mode: bool,
) -> dict[str, list[list[float]]]:
    """Decode RBM head outputs into per-frame floats.

    Args:
        progress_logits: ``(B, T)`` (continuous) or ``(B, T, num_bins)`` (discrete).
        success_logits: ``(B, T)`` raw logits, ``sigmoid``-ed to probabilities.
        is_discrete_mode: if True the progress logits get a softmax over bins
            and are projected onto bin centers via :func:`convert_bins_to_continuous`.

    Returns:
        Dict with ``progress_pred`` and ``success_probs``, each a list of
        length ``B`` of per-frame float lists.
    """
    progress_pred: list[list[float]] = []
    success_probs: list[list[float]] = []

    if progress_logits is not None:
        for sample_logits in progress_logits:
            if is_discrete_mode:
                continuous = convert_bins_to_continuous(sample_logits.detach().float().cpu())
                progress_pred.append(continuous.flatten().tolist())
            else:
                progress_pred.append(sample_logits.detach().float().cpu().flatten().tolist())

    if success_logits is not None:
        for sample_logits in success_logits:
            success_probs.append(torch.sigmoid(sample_logits.detach().float().cpu()).flatten().tolist())

    return {"progress_pred": progress_pred, "success_probs": success_probs}


class RobometerRewardModel(PreTrainedRewardModel):
    """Robometer (RBM) reward model — inference-only LeRobot port.

    Wraps a Qwen-VL backbone (default: ``Qwen/Qwen3-VL-4B-Instruct``) with three
    prediction heads from the paper (progress, success, preference). At
    inference time only the progress and success heads are queried; the
    preference head is kept on the module so the published ``Robometer-4B``
    safetensors load unchanged.
    """

    name = "robometer"
    config_class = RobometerConfig

    def __init__(self, config: RobometerConfig, *, dropout: float = 0.1) -> None:
        require_package("transformers", extra="robometer")
        super().__init__(config)
        self.config = config

        # Two backbone-build paths (EO-1 style, branched on ``pretrained_path``):
        #
        #   - Fresh training (``pretrained_path is None``): download the base
        #     Qwen weights and resize the embed table to match
        #     ``vlm_config.text_config.vocab_size`` — populated deterministically
        #     in ``RobometerConfig.__post_init__`` as
        #     ``len(tokenizer) + len(ROBOMETER_SPECIAL_TOKENS)``
        #
        #   - Loading a saved checkpoint (``pretrained_path`` is set): rebuild
        #     the empty architecture from ``vlm_config`` via
        #     ``AutoModelForImageTextToText.from_config`` so the subsequent
        #     ``model.safetensors`` load is a direct fill of the right shape —
        #     no redundant Qwen weight download.
        torch_dtype = _torch_dtype(config.torch_dtype)
        if config.pretrained_path is None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                config.base_model_id,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            target_vocab = config.vlm_config["text_config"]["vocab_size"]
            self.model.resize_token_embeddings(target_vocab)
        else:
            self.model = AutoModelForImageTextToText.from_config(
                config.vlm_backbone_config,
                dtype=torch_dtype,
                trust_remote_code=True,
            )

        # All Qwen-VL backbones Robometer supports expose `text_config.hidden_size`.
        # Falls back to the top-level `hidden_size` so future non-multimodal
        # variants would still resolve.
        backbone_config = self.model.config
        text_config = getattr(backbone_config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None) if text_config is not None else None
        if hidden_size is None:
            hidden_size = getattr(backbone_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError(
                f"Could not infer hidden_size from backbone config of {config.base_model_id}"
            )
        hidden_dim = int(hidden_size)

        # Robometer's three prediction heads + frame-pool attention.
        progress_output = config.progress_discrete_bins if config.use_discrete_progress else 1
        self.progress_head = RobometerPredictionHead(
            hidden_dim,
            progress_output,
            dropout=dropout,
            with_sigmoid=not config.use_discrete_progress,
        )
        self.preference_head = RobometerPredictionHead(hidden_dim, 1, dropout=dropout, with_sigmoid=False)
        self.success_head = RobometerPredictionHead(hidden_dim, 1, dropout=dropout, with_sigmoid=False)
        self.frame_pool_attn = nn.Linear(hidden_dim, 1, bias=False)

        # Match the dtype of the loaded base model so weight loading is a no-op cast.
        model_dtype = next(self.model.parameters()).dtype
        self.progress_head.to(dtype=model_dtype)
        self.preference_head.to(dtype=model_dtype)
        self.success_head.to(dtype=model_dtype)
        self.frame_pool_attn.to(dtype=model_dtype)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = {
            key: batch[f"{ROBOMETER_FEATURE_PREFIX}{key}"]
            for key in ROBOMETER_INPUT_KEYS
            if f"{ROBOMETER_FEATURE_PREFIX}{key}" in batch
        }
        if "input_ids" not in inputs:
            raise KeyError(
                f"Robometer batch missing pre-encoded inputs (expected "
                f"`{ROBOMETER_FEATURE_PREFIX}input_ids`). Make sure the "
                "RobometerEncoderProcessorStep ran before `compute_reward`."
            )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        self.eval()
        with torch.no_grad():
            progress_logits, success_logits = self._compute_rbm_logits(inputs)

        decoded = decode_progress_outputs(
            progress_logits,
            success_logits,
            is_discrete_mode=self.config.use_discrete_progress,
        )
        values = (
            decoded["success_probs"] if self.config.reward_output == "success" else decoded["progress_pred"]
        )

        rewards = torch.stack([torch.as_tensor(seq, dtype=torch.float32)[-1] for seq in values])
        if self.config.reward_output == "success":
            rewards = (rewards > self.config.success_threshold).float()
        else:
            # Match upstream Robometer's ``extract_rewards_from_output``: per-frame
            # progress predictions are clamped to ``[0, 1]`` before being returned.
            rewards = rewards.clamp(0.0, 1.0)
        return rewards.to(self.config.device or "cpu")

    def _compute_rbm_logits(
        self,
        inputs: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        """Run the Qwen3-VL backbone and apply Robometer's heads.

        ``inputs`` is the encoded batch produced by
        :class:`RobometerEncoderProcessorStep`. It carries Qwen tensors as well
        as Robometer-specific metadata (``prog_token_id``,
        ``vision_start_token_id``, ``vision_end_token_id``, ``video_merge_size``)
        — the metadata is popped here so the rest can be forwarded straight to
        the Qwen model.

        Returns ``(progress_logits, success_logits)``. Shapes:

        - ``progress_logits``: ``(B, T)`` (continuous) or ``(B, T, num_bins)`` (discrete).
        - ``success_logits``: ``(B, T)`` raw logits (sigmoid happens at decode time).
        """
        prog_token_id = inputs.pop("prog_token_id", None)
        vision_start_token_id = inputs.pop("vision_start_token_id", None)
        vision_end_token_id = inputs.pop("vision_end_token_id", None)
        video_merge_size = inputs.pop("video_merge_size", 14)

        # Qwen3-VL doesn't reliably populate `last_hidden_state`; ask for the
        # full hidden-state tuple and take the last layer. This matches the
        # `is_qwen3` path in upstream Robometer's `RBM.forward_qwen` (main).
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_state = (
            outputs.hidden_states[-1]
            if getattr(outputs, "hidden_states", None)
            else outputs.last_hidden_state
        )

        input_ids = inputs["input_ids"]
        if self.config.use_per_frame_progress_token:
            if prog_token_id is None:
                raise KeyError("`prog_token_id` missing in batch (run RobometerEncoderProcessorStep first)")
            return self._process_token_extraction(hidden_state, input_ids, prog_token_id=prog_token_id)
        if self.config.use_multi_image:
            if vision_start_token_id is None or vision_end_token_id is None:
                raise KeyError(
                    "`vision_start_token_id` / `vision_end_token_id` missing in batch "
                    "(run RobometerEncoderProcessorStep first)"
                )
            return self._process_multi_image_frames(
                hidden_state,
                input_ids,
                start_id=vision_start_token_id,
                end_id=vision_end_token_id,
            )
        video_grid_thw = inputs.get("video_grid_thw")
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is required for video-mode Robometer inference")
        if vision_start_token_id is None:
            raise KeyError("`vision_start_token_id` missing in batch")
        return self._process_video_frames(
            hidden_state,
            input_ids,
            video_grid_thw,
            start_id=vision_start_token_id,
            merge_size=video_merge_size,
        )

    def _apply_heads_to_hidden_states(self, frame_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Apply progress + success heads to a tensor of frame embeddings."""
        progress_out = self.progress_head(frame_embeddings)
        progress = progress_out if self.config.use_discrete_progress else _squeeze_last_safe(progress_out)
        success = _squeeze_last_safe(self.success_head(frame_embeddings))
        return progress, success

    def _process_token_extraction(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        *,
        prog_token_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Per-frame progress/success from ``<|prog_token|>`` positions."""
        token_mask = input_ids == prog_token_id
        batch_indices, positions = token_mask.nonzero(as_tuple=True)
        if positions.numel() == 0:
            raise ValueError("`<|prog_token|>` not found in any sequence")

        per_sample_hidden = [
            hidden_state[i, positions[batch_indices == i]] for i in range(input_ids.shape[0])
        ]
        progress_list, success_list = [], []
        for embeddings in per_sample_hidden:
            if embeddings.shape[0] == 0:
                raise ValueError("`<|prog_token|>` missing in a sequence")
            progress, success = self._apply_heads_to_hidden_states(embeddings)
            progress_list.append(progress)
            success_list.append(success)

        return torch.stack(progress_list), torch.stack(success_list)

    def _process_multi_image_frames(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        *,
        start_id: int,
        end_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Per-frame progress/success in multi-image mode (Qwen-VL)."""
        progress_list, success_list = [], []
        for batch_idx in range(input_ids.shape[0]):
            seq_ids = input_ids[batch_idx]
            seq_hidden = hidden_state[batch_idx]
            frame_embeddings = self._extract_hidden_states_from_token_pairs(
                seq_hidden, seq_ids, start_id, end_id
            )
            progress, success = self._apply_heads_to_hidden_states(frame_embeddings)
            progress_list.append(progress)
            success_list.append(success)

        return torch.stack(progress_list), torch.stack(success_list)

    def _extract_hidden_states_from_token_pairs(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        start_id: int,
        end_id: int,
    ) -> Tensor:
        start_positions = (input_ids == start_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == end_id).nonzero(as_tuple=True)[0]
        if start_positions.numel() == 0:
            raise ValueError("`<|vision_start|>` not found in sequence")
        if start_positions.numel() != end_positions.numel():
            raise ValueError(
                f"Mismatched vision token counts: {start_positions.numel()} start vs "
                f"{end_positions.numel()} end"
            )

        frames: list[Tensor] = []
        for start, end in zip(start_positions.tolist(), end_positions.tolist(), strict=True):
            if start >= end:
                raise ValueError(f"Invalid vision token pair: start={start} end={end}")
            patch_tokens = hidden_state[start + 1 : end]
            if patch_tokens.shape[0] == 0:
                frames.append((hidden_state[start] + hidden_state[end]) / 2.0)
                continue

            pooling = self.config.frame_pooling
            if pooling == "mean":
                frames.append(patch_tokens.mean(dim=0))
            elif pooling == "boundary":
                frames.append(patch_tokens[-1])
            else:  # attention
                scores = (
                    self.frame_pool_attn(patch_tokens).squeeze(-1)
                    / self.config.frame_pooling_attn_temperature
                )
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                frames.append((weights * patch_tokens).sum(dim=0))

        return torch.stack(frames)

    def _process_video_frames(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        video_grid_thw: Tensor,
        *,
        start_id: int,
        merge_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Per-frame progress/success in video mode (Qwen-VL)."""
        progress_list, success_list = [], []
        for batch_idx in range(input_ids.shape[0]):
            seq_ids = input_ids[batch_idx]
            seq_hidden = hidden_state[batch_idx]
            start_positions = (seq_ids == start_id).nonzero(as_tuple=True)[0]
            if start_positions.numel() == 0:
                raise ValueError("`<|vision_start|>` not found in sequence")
            t_dim, h_dim, w_dim = (int(x) for x in video_grid_thw[batch_idx].tolist())
            tokens_per_frame = (h_dim * w_dim) // (merge_size**2)

            cursor = start_positions[0].item()
            frame_embeddings: list[Tensor] = []
            for _ in range(t_dim):
                if self.config.average_temporal_patches:
                    patch = seq_hidden[cursor : cursor + tokens_per_frame]
                    frame_embeddings.append(patch.mean(dim=0))
                else:
                    frame_embeddings.append(seq_hidden[cursor + tokens_per_frame])
                cursor += tokens_per_frame

            stacked = torch.stack(frame_embeddings)
            progress, success = self._apply_heads_to_hidden_states(stacked)
            progress_list.append(progress)
            success_list.append(success)

        return torch.stack(progress_list), torch.stack(success_list)
