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

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from . import attention_impl as _attention_impl  # noqa: F401
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
    past_key_values: Cache | tuple | None = None
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
        self.value_tokenizer = ValueTokenizer.from_config(config.value_tokenizer_config)
        if config.relative_value_head_config is not None:
            self.relative_value_tokenizer = ValueTokenizer.from_config(config.relative_value_tokenizer_config)
            relative_output_dim = config.relative_value_tokenizer_config.bins
        else:
            self.relative_value_tokenizer = self.value_tokenizer
            relative_output_dim = output_dim
        self.value_heads = None
        if config.value_head_config is not None:
            self.value_heads = nn.ModuleList(
                [
                    build_value_head(
                        config.value_head_config,
                        input_dim=input_dim * config.value_token_repeat,
                        output_dim=output_dim,
                    )
                    for _ in range(config.num_value_heads)
                ]
            )
        self.relative_value_head = None
        if config.relative_value_head_config is not None:
            self.relative_value_head = build_value_head(
                config.relative_value_head_config,
                input_dim=input_dim * config.relative_value_token_repeat,
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
        if config is None:
            config = RynnValueLangConfig.from_qwen3vl(
                pretrained_model_name_or_path,
                value_tokenizer_config=value_tokenizer_config,
                value_head_config=value_head_config,
                num_value_heads=num_value_heads,
            )
        return cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)

    def _compute_value_loss(self, logits, target_value, fusion_mask=None):
        logits = logits.float()
        if logits.shape[-1] != self.value_tokenizer.n_bins:
            raise ValueError(f"Expected logits last dim == n_bins ({self.value_tokenizer.n_bins}).")
        target_dist = self.value_tokenizer.encode(
            self.value_tokenizer._to_tensor(target_value, device=logits.device)
        ).to(device=logits.device, dtype=logits.dtype)
        if fusion_mask is not None:
            mask = fusion_mask.to(target_dist.device).bool().view(*target_dist.shape[:-1])
            if mask.any():
                target_dist = torch.where(
                    mask.unsqueeze(-1),
                    torch.full_like(target_dist, 1.0 / self.value_tokenizer.n_bins),
                    target_dist,
                )
        if (n_extra := logits.ndim - target_dist.ndim) > 0:
            target_dist = target_dist.view(*([1] * n_extra), *target_dist.shape).expand_as(logits)
        return -(target_dist * functional.log_softmax(logits, dim=-1)).sum(dim=-1)

    def _compute_relative_value_loss(self, logits, target_value):
        logits = logits.float()
        if logits.shape[-1] != self.relative_value_tokenizer.n_bins:
            raise ValueError(f"Expected logits last dim == n_bins ({self.relative_value_tokenizer.n_bins}).")
        target_dist = self.relative_value_tokenizer.encode(
            self.relative_value_tokenizer._to_tensor(target_value, device=logits.device)
        ).to(device=logits.device, dtype=logits.dtype)
        if (n_extra := logits.ndim - target_dist.ndim) > 0:
            target_dist = target_dist.view(*([1] * n_extra), *target_dist.shape).expand_as(logits)
        return -(target_dist * functional.log_softmax(logits, dim=-1)).sum(dim=-1)

    @staticmethod
    def _compute_entropy(logits):
        probs = torch.softmax(logits.float(), dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    @staticmethod
    def _gather_by_token_id(hidden_states, input_ids, token_id):
        if input_ids is None or token_id is None or token_id < 0:
            return None, None
        mask = input_ids.eq(token_id)
        if not mask.any():
            return None, None
        batch_indices, positions = mask.nonzero(as_tuple=True)
        flattened = hidden_states[batch_indices, positions].contiguous()
        return flattened, flattened.shape[:-1]

    @staticmethod
    def _concat_slot_tokens(flattened, prefix_shape, repeat, token_name):
        if repeat > 1:
            total_keep = flattened.shape[0]
            if total_keep % repeat:
                raise ValueError(
                    f"Number of {token_name} tokens ({total_keep}) is not divisible by repeat ({repeat})."
                )
            flattened = flattened.view(total_keep // repeat, repeat, -1).reshape(total_keep // repeat, -1)
            prefix_shape = flattened.shape[:-1]
        return flattened, prefix_shape

    def _compute_value_outputs(self, hidden_states, input_ids, value, fusion_mask=None):
        flattened, prefix_shape = self._gather_by_token_id(
            hidden_states, input_ids, self.config.value_token_id
        )
        if flattened is None:
            return _ValueOutput()
        flattened, prefix_shape = self._concat_slot_tokens(
            flattened, prefix_shape, self.config.value_token_repeat, "<value>"
        )
        if self.value_heads is not None:
            value_logits = torch.stack([head(flattened) for head in self.value_heads], dim=0).view(
                len(self.value_heads), *prefix_shape, -1
            )
            loss = self._compute_value_loss(value_logits, value, fusion_mask) if value is not None else None
            return _ValueOutput(
                loss=loss,
                logits=value_logits,
                pred_value=self.value_tokenizer.decode_from_bins(value_logits),
                value_logits=value_logits,
                entropy=self._compute_entropy(value_logits),
            )
        logits = self.lm_head(flattened)[..., -self.value_tokenizer.n_bins :].view(*prefix_shape, -1)
        return _ValueOutput(
            loss=self._compute_value_loss(logits, value, fusion_mask) if value is not None else None,
            logits=logits,
            pred_value=self.value_tokenizer.decode_from_bins(logits),
            entropy=self._compute_entropy(logits),
        )

    def _compute_relative_value_outputs(self, hidden_states, input_ids, relative_value):
        if self.relative_value_head is None:
            return _RelativeValueOutput()
        flattened, prefix_shape = self._gather_by_token_id(
            hidden_states, input_ids, self.config.relative_value_token_id
        )
        if flattened is None:
            return _RelativeValueOutput()
        flattened, prefix_shape = self._concat_slot_tokens(
            flattened, prefix_shape, self.config.relative_value_token_repeat, "<relative_value>"
        )
        logits = self.relative_value_head(flattened).view(*prefix_shape, -1)
        return _RelativeValueOutput(
            pred_value=self.relative_value_tokenizer.decode_from_bins(logits),
            loss=(
                self._compute_relative_value_loss(logits, relative_value)
                if relative_value is not None
                else None
            ),
            logits=logits,
        )

    def _is_pred_token(self, input_ids):
        mask = input_ids.eq(self.config.value_token_id)
        if self.config.relative_value_token_id is not None:
            mask |= input_ids.eq(self.config.relative_value_token_id)
        return mask

    def _build_pred_slot_extras(self, input_ids, past_key_values=None, cached_pred_key=None):
        del past_key_values
        if input_ids is None:
            return {}, cached_pred_key
        if cached_pred_key is not None:
            is_pred_key = torch.cat([cached_pred_key, self._is_pred_token(input_ids)], dim=1)
            return {"is_pred_key": is_pred_key}, is_pred_key
        is_special = self._is_pred_token(input_ids)
        previous_ids = functional.pad(input_ids[:, :-1], (1, 0), value=-1)
        previous_special = functional.pad(is_special[:, :-1], (1, 0), value=False)
        slot_start = is_special & ~(is_special & previous_special & input_ids.eq(previous_ids))
        running = slot_start.long().cumsum(dim=1) - 1
        slot_id = torch.where(is_special, running, torch.full_like(running, -1))
        return {"is_pred_key": is_special, "pred_slot_id": slot_id}, is_special

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        model_kwargs["cached_pred_key"] = outputs.get("cached_pred_key")
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
        cache_position: torch.LongTensor | None = None,
        value: torch.Tensor | None = None,
        relative_value: torch.Tensor | None = None,
        value_fusion_mask: torch.Tensor | None = None,
        cached_pred_key: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RynnValueLangOutputWithPast:
        del return_dict
        extras, cached_pred_key = self._build_pred_slot_extras(input_ids, past_key_values, cached_pred_key)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **extras,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = lang_loss = None
        if labels is not None:
            logits = self.lm_head(hidden_states)
            lang_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )
        elif not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, indices])
        return RynnValueLangOutputWithPast(
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=outputs.rope_deltas,
            value=self._compute_value_outputs(hidden_states, input_ids, value, value_fusion_mask),
            relative=self._compute_relative_value_outputs(hidden_states, input_ids, relative_value),
            lang_loss=lang_loss,
            logits=logits,
            cached_pred_key=cached_pred_key,
        )
