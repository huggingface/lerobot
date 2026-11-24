# Copyright 2026 Alibaba DAMO Academy and The HuggingFace Inc. team. All rights reserved.
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

# modeling_rynn_value_lang.py

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .attention_impl import pred_slot_isolated_eager  # noqa: F401
from .configuration_rynn_value_lang import RynnValueLangConfig
from .value_heads import build_value_head
from .value_tokenizer import ValueTokenizer


@dataclass
class _ValueOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    pred_value: torch.Tensor | None = None
    value_logits: torch.Tensor | None = None
    entropy: torch.Tensor | None = None


@dataclass
class _RelativeValueOutput:
    pred_value: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@dataclass
class RynnValueLangOutputWithPast(ModelOutput):
    past_key_values: tuple | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    rope_deltas: torch.LongTensor | None = None
    value: _ValueOutput | None = None
    relative: _RelativeValueOutput | None = None
    lang_loss: torch.Tensor | None = None
    logits: torch.FloatTensor | None = None
    cached_pred_key: torch.Tensor | None = None


class RynnValueLangModel(Qwen3VLForConditionalGeneration):
    config: RynnValueLangConfig
    model_type = "rynn_value_lang"

    def __init__(self, config: RynnValueLangConfig):
        super().__init__(config)

        input_dim = config.text_config.hidden_size
        output_dim = config.value_tokenizer_config.bins

        self.value_tokenizer = ValueTokenizer.from_config(config.value_tokenizer_config, device="cpu")
        num_value_heads = config.num_value_heads

        if config.relative_value_head_config is not None:
            self.relative_value_tokenizer = ValueTokenizer.from_config(
                config.relative_value_tokenizer_config, device="cpu"
            )
            relative_output_dim = config.relative_value_tokenizer_config.bins
        else:
            self.relative_value_tokenizer = self.value_tokenizer
            relative_output_dim = output_dim

        value_repeat = int(getattr(config, "value_token_repeat", 1))
        relative_repeat = int(getattr(config, "relative_value_token_repeat", 1))

        self.value_heads = None
        if config.value_head_config is not None:
            self.value_heads = nn.ModuleList(
                [
                    build_value_head(
                        config=config.value_head_config,
                        input_dim=input_dim * value_repeat,
                        output_dim=output_dim,
                    )
                    for _ in range(num_value_heads)
                ]
            )

        # Separate single head for relative-value prediction (used in StepRelative mode).
        self.relative_value_head = None
        if config.relative_value_head_config is not None:
            self.relative_value_head = build_value_head(
                config=config.relative_value_head_config,
                input_dim=input_dim * relative_repeat,
                output_dim=relative_output_dim,
            )

        self.post_init()

    @classmethod
    def from_qwen3vl(
        cls,
        pretrained_model_name_or_path: str,
        config: RynnValueLangConfig | None = None,
        value_tokenizer_config=None,
        value_head_config=None,
        num_value_heads: int = 1,
        **kwargs,
    ) -> "RynnValueLangModel":
        """
        Build a RynnValueLangModel from a Qwen3VL checkpoint.

        If `config` is not provided, a RynnValueLangConfig is created from the
        underlying Qwen3VL config.
        """
        if config is None:
            config = RynnValueLangConfig.from_qwen3vl(
                pretrained_model_name_or_path,
                value_tokenizer_config=value_tokenizer_config,
                value_head_config=value_head_config,
                num_value_heads=num_value_heads,
            )

        return cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )

    def _compute_value_loss(
        self,
        logits: torch.Tensor,
        target_value: torch.Tensor,
        fusion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.float()
        if logits.shape[-1] != self.value_tokenizer.n_bins:
            raise ValueError(
                f"Expected logits last dim == n_bins ({self.value_tokenizer.n_bins}), got {logits.shape[-1]}."
            )

        target_value = self.value_tokenizer._to_tensor(target_value, device=logits.device)
        target_dist = self.value_tokenizer.encode(target_value).to(
            device=logits.device,
            dtype=logits.dtype,
        )

        # Replace fused slots with a uniform (max-entropy) target so the model
        # is trained to be maximally uncertain when the instruction doesn't
        # match the video.
        if fusion_mask is not None:
            mask = fusion_mask.to(device=target_dist.device).bool().view(*target_dist.shape[:-1])
            if mask.any():
                uniform = torch.full_like(target_dist, 1.0 / self.value_tokenizer.n_bins)
                target_dist = torch.where(mask.unsqueeze(-1), uniform, target_dist)

        n_extra = logits.ndim - target_dist.ndim
        if n_extra > 0:
            target_dist = target_dist.view(*([1] * n_extra), *target_dist.shape).expand_as(logits)

        log_probs = functional.log_softmax(logits, dim=-1)

        return -(target_dist * log_probs).sum(dim=-1)

    def _compute_relative_value_loss(self, logits: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if logits.shape[-1] != self.relative_value_tokenizer.n_bins:
            raise ValueError(
                f"Expected logits last dim == n_bins ({self.relative_value_tokenizer.n_bins}), "
                f"got {logits.shape[-1]}."
            )

        target_value = self.relative_value_tokenizer._to_tensor(target_value, device=logits.device)
        target_dist = self.relative_value_tokenizer.encode(target_value).to(
            device=logits.device,
            dtype=logits.dtype,
        )

        n_extra = logits.ndim - target_dist.ndim
        if n_extra > 0:
            target_dist = target_dist.view(*([1] * n_extra), *target_dist.shape).expand_as(logits)

        log_probs = functional.log_softmax(logits, dim=-1)

        return -(target_dist * log_probs).sum(dim=-1)

    @staticmethod
    def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    def _gather_by_token_id(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        token_id: int | None,
    ) -> tuple[torch.Tensor | None, torch.Size | None]:
        """Gather hidden states at every occurrence of ``token_id``.

        Returns ``(flattened, prefix_shape)`` where ``flattened`` is
        ``(total_keep, D)``, or ``(None, None)`` when no occurrence is found.
        """
        if input_ids is None or token_id is None or token_id < 0:
            return None, None

        mask = input_ids.eq(token_id)
        if not mask.any():
            return None, None

        batch_indices, positions = mask.nonzero(as_tuple=True)
        flattened = hidden_states[batch_indices, positions].contiguous()
        return flattened, flattened.shape[:-1]

    def _compute_value_outputs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        value: torch.Tensor | None,
        fusion_mask: torch.Tensor | None = None,
    ) -> _ValueOutput:
        flattened, prefix_shape = self._gather_by_token_id(
            hidden_states,
            input_ids,
            self.config.value_token_id,
        )
        if flattened is None:
            return _ValueOutput()

        # Concat value_token_repeat consecutive <value> hidden states into a
        # single prediction slot before feeding the value head. The processor
        # emits R copies per slot; the model flattens them into one (R*D) vector.
        repeat = int(getattr(self.config, "value_token_repeat", 1))
        if repeat > 1:
            total_keep = flattened.shape[0]
            if total_keep % repeat != 0:
                raise ValueError(
                    f"Number of <value> tokens ({total_keep}) is not divisible "
                    f"by value_token_repeat ({repeat}). Check that the "
                    "conversation builder is emitting R copies per slot."
                )
            flattened = flattened.view(total_keep // repeat, repeat, -1).reshape(total_keep // repeat, -1)
            prefix_shape = flattened.shape[:-1]

        if self.value_heads is not None:
            stacked = torch.stack(
                [head(flattened) for head in self.value_heads],
                dim=0,
            )
            value_logits = stacked.view(len(self.value_heads), *prefix_shape, -1)
            loss = (
                self._compute_value_loss(value_logits, value, fusion_mask=fusion_mask)
                if value is not None
                else None
            )
            return _ValueOutput(
                loss=loss,
                logits=value_logits,
                pred_value=self.value_tokenizer.decode_from_bins(value_logits),
                value_logits=value_logits,
                entropy=self._compute_entropy(value_logits),
            )

        logits = self.lm_head(flattened)
        logits = logits[..., -self.value_tokenizer.n_bins :].view(*prefix_shape, -1)
        loss = self._compute_value_loss(logits, value, fusion_mask=fusion_mask) if value is not None else None
        return _ValueOutput(
            loss=loss,
            logits=logits,
            pred_value=self.value_tokenizer.decode_from_bins(logits),
            entropy=self._compute_entropy(logits),
        )

    def _compute_relative_value_outputs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        relative_value: torch.Tensor | None,
    ) -> _RelativeValueOutput:
        if self.relative_value_head is None:
            return _RelativeValueOutput()

        flattened, prefix_shape = self._gather_by_token_id(
            hidden_states,
            input_ids,
            self.config.relative_value_token_id,
        )
        if flattened is None:
            return _RelativeValueOutput()

        # Same concat story as ``_compute_value_outputs``: the processor may
        # emit R copies of <relative_value> per prediction slot; flatten the R
        # hidden states into one (R*D) vector so the head sees one input per slot.
        repeat = int(getattr(self.config, "relative_value_token_repeat", 1))
        if repeat > 1:
            total_keep = flattened.shape[0]
            if total_keep % repeat != 0:
                raise ValueError(
                    f"Number of <relative_value> tokens ({total_keep}) is not "
                    f"divisible by relative_value_token_repeat ({repeat}). "
                    "Check that the conversation builder is emitting R copies "
                    "per slot."
                )
            flattened = flattened.view(total_keep // repeat, repeat, -1).reshape(total_keep // repeat, -1)
            prefix_shape = flattened.shape[:-1]

        r_logits = self.relative_value_head(flattened).view(*prefix_shape, -1)
        loss = (
            self._compute_relative_value_loss(r_logits, relative_value)
            if relative_value is not None
            else None
        )
        return _RelativeValueOutput(
            pred_value=self.relative_value_tokenizer.decode_from_bins(r_logits),
            loss=loss,
            logits=r_logits,
        )

    def _build_pred_slot_extras(
        self,
        input_ids: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cached_pred_key: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor | None]:
        if input_ids is None:
            return {}, cached_pred_key

        # Generation with KV cache: only need is_pred_key over the full key
        # dimension to keep pred keys invisible to generated text tokens.
        if cached_pred_key is not None:
            new_is_pred = self._is_pred_token(input_ids)
            is_pred_key = torch.cat([cached_pred_key, new_is_pred], dim=1)
            return {"is_pred_key": is_pred_key}, is_pred_key

        # Prefill: compute full extras.
        is_value_query = input_ids.eq(self.config.value_token_id)
        if self.config.relative_value_token_id is not None:
            is_special_query = is_value_query | input_ids.eq(self.config.relative_value_token_id)
        else:
            is_special_query = is_value_query

        prev_ids = torch.nn.functional.pad(input_ids[:, :-1], (1, 0), value=-1)
        prev_special = torch.nn.functional.pad(is_special_query[:, :-1], (1, 0), value=False)
        same_as_prev = is_special_query & prev_special & input_ids.eq(prev_ids)
        is_slot_start = is_special_query & ~same_as_prev
        slot_running = is_slot_start.long().cumsum(dim=1) - 1
        pred_slot_id = torch.where(
            is_special_query,
            slot_running,
            torch.full_like(slot_running, -1),
        )

        extras = {
            "is_pred_key": is_special_query,
            "pred_slot_id": pred_slot_id,
        }

        return extras, is_special_query

    def _is_pred_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.eq(self.config.value_token_id)
        if self.config.relative_value_token_id is not None:
            mask = mask | input_ids.eq(self.config.relative_value_token_id)
        return mask

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        model_kwargs["cached_pred_key"] = outputs.get("cached_pred_key", None)
        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        value: torch.Tensor | None = None,
        relative_value: torch.Tensor | None = None,
        value_fusion_mask: torch.Tensor | None = None,
        cached_pred_key: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor | None = 0,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RynnValueLangOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        drop_extras, cached_pred_key = self._build_pred_slot_extras(
            input_ids, past_key_values, cached_pred_key
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **drop_extras,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Language logits. Compute lazily to keep training memory unchanged for
        # non-language strategies. Cases:
        #   * ``labels is not None``     — training with LM supervision; needs
        #     full-sequence logits for the causal LM loss.
        #   * ``logits_to_keep != 0``    — inference / generation path;
        #     ``GenerationMixin`` sets this to 1 so we only project the last
        #     position onto the vocab.
        #   * otherwise                  — skip entirely; keeps the original
        #     behaviour for strategies without ``labels``.
        # lm_head is excluded from FSDP wrapping, so it can be called directly.
        logits = None
        lang_loss = None
        want_logits = labels is not None or (not (isinstance(logits_to_keep, int) and logits_to_keep == 0))
        if want_logits:
            if labels is not None:
                logits = self.lm_head(hidden_states)
                lang_loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.text_config.vocab_size,
                )
            else:
                slice_indices = (
                    slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                )
                logits = self.lm_head(hidden_states[:, slice_indices, :])

        value_output = self._compute_value_outputs(
            hidden_states, input_ids, value, fusion_mask=value_fusion_mask
        )
        relative_output = self._compute_relative_value_outputs(hidden_states, input_ids, relative_value)

        return RynnValueLangOutputWithPast(
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            rope_deltas=outputs.rope_deltas,
            value=value_output,
            relative=relative_output,
            lang_loss=lang_loss,
            logits=logits,
            cached_pred_key=cached_pred_key,
        )
