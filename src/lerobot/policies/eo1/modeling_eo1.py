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

from __future__ import annotations

import contextlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import create_sinusoidal_pos_embedding, pad_vector
from ..pretrained import PreTrainedPolicy
from .configuration_eo1 import EO1Config
from .processor_eo1 import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_STATE_TOKEN,
    EO1_SPECIAL_TOKENS,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    SYSTEM_MESSAGE,
    TASK_VLA_TOKEN,
)

if TYPE_CHECKING or _transformers_available:
    from transformers.activations import ACT2FN
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from transformers.utils import torch_compilable_check
else:
    ACT2FN = None
    Qwen2_5_VLForConditionalGeneration = None
    Qwen2_5_VLProcessor = None
    torch_compilable_check = None

logger = logging.getLogger(__name__)


class EO1Policy(PreTrainedPolicy):
    """EO1 policy wrapper for LeRobot robot-only training/evaluation."""

    config_class = EO1Config
    name = "eo1"
    #: EO-1's trained subtask wording; the runtime fills it with the operator's goal.
    PROMPT_TEMPLATES = {"subtask": "{task}\nPredict the next action in language."}  # noqa: RUF012

    def __init__(self, config: EO1Config, **kwargs):
        require_package("transformers", extra="eo1")
        super().__init__(config)
        config.validate_features()
        self.config = config

        if config.pretrained_path is None:
            # Initialize from pretrained VLM
            vlm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.vlm_base,
                dtype=config.dtype,
                attn_implementation=config.attn_implementation,
            )
        else:
            vlm_backbone = Qwen2_5_VLForConditionalGeneration._from_config(
                config.vlm_backbone_config,
                dtype=config.vlm_backbone_config.dtype if config.dtype == "auto" else config.dtype,
            )

        self.model = EO1VisionFlowMatchingModel(config, vlm_backbone)
        self._text_processor = None
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @staticmethod
    def _get_model_inputs(batch: dict[str, Tensor], excluded_keys: set[str]) -> dict[str, Tensor]:
        return {key: value for key, value in batch.items() if key not in excluded_keys}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        state = self.prepare_state(batch[OBS_STATE])
        actions = self.prepare_action(batch[ACTION])
        model_inputs = self._get_model_inputs(batch, {OBS_STATE, ACTION})
        outputs = self.model(states=state, action=actions, **model_inputs)
        loss = None
        if outputs.flow_loss is not None:
            loss = self.config.flow_loss_weight * outputs.flow_loss
        if outputs.text_loss is not None:
            weighted_text_loss = self.config.text_loss_weight * outputs.text_loss
            loss = weighted_text_loss if loss is None else loss + weighted_text_loss
        if loss is None:
            raise RuntimeError(
                "EO-1 batch produced neither action nor text supervision. "
                "Check the selected recipe and target annotations."
            )

        loss_dict = {"loss": float(loss.detach())}
        if outputs.flow_loss is not None:
            loss_dict["flow_loss"] = float(outputs.flow_loss.detach())
        if outputs.text_loss is not None:
            loss_dict["text_loss"] = float(outputs.text_loss.detach())
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        states = self.prepare_state(batch[OBS_STATE])
        model_inputs = self._get_model_inputs(batch, {OBS_STATE})
        actions = self.model.sample_actions(states=states, **model_inputs).to(torch.float32)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def prepare_state(self, state: Tensor) -> Tensor:
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, action: Tensor) -> Tensor:
        return pad_vector(action, self.config.max_action_dim)

    def _get_text_processor(self):
        if self._text_processor is None:
            self._text_processor = Qwen2_5_VLProcessor.from_pretrained(
                self.config.vlm_base,
                use_fast=self.config.use_fast_processor,
                fix_mistral_regex=True,
            )
            self._text_processor.tokenizer.add_tokens(EO1_SPECIAL_TOKENS, special_tokens=True)
        return self._text_processor

    @staticmethod
    def _batch_tasks(task: str | list[str] | None, batch_size: int) -> list[str]:
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, list) and len(task) == batch_size and all(isinstance(v, str) for v in task):
            return task
        raise ValueError(f"EO-1 expected exactly {batch_size} task strings.")

    def _runtime_images(self, batch: dict[str, Any], batch_size: int) -> list[list[dict[str, Any]]]:
        image_keys = [key for key in self.config.image_features if key in batch]
        if not image_keys:
            raise ValueError("EO-1 text generation requires at least one observation image.")
        rows: list[list[dict[str, Any]]] = []
        for row in range(batch_size):
            blocks = []
            for key in image_keys:
                image = batch[key]
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                image = image[row].detach().cpu()
                if image.is_floating_point():
                    image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
                blocks.append({"type": "image", "image": image})
            rows.append(blocks)
        return rows

    def prepare_runtime_action_batch(self, batch: dict[str, Any], task: str | list[str]) -> dict[str, Any]:
        """Rebuild the EO-1 action prompt from the runtime's current subtask."""
        state = batch[OBS_STATE]
        batch_size = state.shape[0]
        tasks = self._batch_tasks(task, batch_size)
        image_rows = self._runtime_images(batch, batch_size)
        messages = []
        for row, instruction in enumerate(tasks):
            messages.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {
                        "role": "user",
                        "content": [
                            *image_rows[row],
                            {
                                "type": "text",
                                "text": (
                                    f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"
                                    f"{instruction}{TASK_VLA_TOKEN}"
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{ACTION_START_TOKEN}"
                                    f"{DEFAULT_ACTION_TOKEN * self.config.chunk_size}"
                                    f"{ACTION_END_TOKEN}"
                                ),
                            }
                        ],
                    },
                ]
            )

        processor = self._get_text_processor()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "padding_side": "left",
                "min_pixels": self.config.image_min_pixels,
                "max_pixels": self.config.image_max_pixels,
            },
        )
        device = state.device
        return {
            OBS_STATE: state,
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
            "pixel_values": inputs["pixel_values"].to(device),
            "image_grid_thw": inputs["image_grid_thw"].to(device),
            "mm_token_type_ids": inputs["mm_token_type_ids"].to(device),
            "state_token_id": processor.tokenizer.convert_tokens_to_ids(DEFAULT_STATE_TOKEN),
            "action_token_id": processor.tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN),
        }

    def generate_text(self, batch: dict[str, Tensor], prompt: str) -> str:
        """Answer `prompt` about the current observation with EO-1's own text head."""
        return self._one_text(batch, kind="vqa", user_text=prompt)

    def _one_text(self, batch: dict[str, Tensor], *, kind: str, user_text: str | None = None) -> str:
        outputs = self.generate_texts(
            batch,
            kind=kind,
            user_text=user_text,
            temperature=self.config.text_temperature,
            top_p=self.config.text_top_p,
        )
        if len(outputs) != 1:
            raise ValueError(f"The interactive runtime expected one EO-1 text output, got {len(outputs)}.")
        return outputs[0]

    @torch.no_grad()
    def generate_texts(
        self,
        batch: dict[str, Any],
        *,
        kind: str = "vqa",
        user_text: str | list[str] | None = None,
        max_new_tokens: int = 100,
        min_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate EO-1 captions, VQA answers, or subtasks for a whole batch."""
        self.eval()
        allowed_kinds = {"vqa", "caption", "grounding", "text", "subtask"}
        if kind not in allowed_kinds:
            raise ValueError(f"Unsupported EO-1 text kind: {kind!r}.")

        state = batch[OBS_STATE]
        batch_size = state.shape[0]
        tasks = self._batch_tasks(batch.get("task", [""] * batch_size), batch_size)
        if user_text is None:
            if kind == "caption":
                prompts = ["Describe the image in one sentence."] * batch_size
            elif kind == "subtask":
                prompts = [f"{task}\nPredict the next action in language." for task in tasks]
            else:
                prompts = tasks
        elif isinstance(user_text, str):
            prompts = [user_text] * batch_size
        elif len(user_text) == batch_size and all(isinstance(value, str) for value in user_text):
            prompts = user_text
        else:
            raise ValueError(f"EO-1 expected exactly {batch_size} text prompts.")

        image_rows = self._runtime_images(batch, batch_size)
        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {
                    "role": "user",
                    "content": [*image_rows[row], {"type": "text", "text": prompt}],
                },
            ]
            for row, prompt in enumerate(prompts)
        ]
        processor = self._get_text_processor()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "padding_side": "left",
                "min_pixels": self.config.image_min_pixels,
                "max_pixels": self.config.image_max_pixels,
            },
        )
        prompt_length = inputs["input_ids"].shape[1]
        device = state.device
        inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, Tensor)}
        do_sample = temperature > 0
        tokenizer = processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        generated = self.model.vlm_backbone.generate(**inputs, **generation_kwargs)
        return [
            text.strip()
            for text in processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def get_optim_params(self) -> dict:
        return self.parameters()


class EO1VisionActionProjector(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_layer: str = "linear",
        bias: bool = True,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ):
        layers = []
        in_dim = in_channels
        hidden_channels = [in_dim] * (num_layers - 1) + [out_channels]
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device))
            layers.append(ACT2FN[activation_layer])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=dtype, device=device))
        super().__init__(*layers)

    @property
    def dtype(self):
        return self[0].weight.dtype


@dataclass
class EO1Output:
    loss: Tensor | None = None
    flow_loss: Tensor | None = None
    text_loss: Tensor | None = None


class EO1VisionFlowMatchingModel(nn.Module):
    def __init__(
        self,
        config: EO1Config,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration | None = None,
    ):
        require_package("transformers", extra="eo1")
        super().__init__()

        self.config = config
        # Preserve the backbone dtype selected at construction time so Qwen's fp32 rotary buffers stay intact.
        self.vlm_backbone = vlm_backbone
        self.hidden_size = self.vlm_backbone.config.text_config.hidden_size
        max_state_dim = config.max_state_dim
        max_action_dim = config.max_action_dim
        self.state_proj = nn.Linear(max_state_dim, self.hidden_size, dtype=torch.float32)
        self.action_in_proj = nn.Linear(max_action_dim, self.hidden_size, dtype=torch.float32)
        self.action_out_proj = EO1VisionActionProjector(
            self.hidden_size,
            max_action_dim,
            config.num_action_layers,
            config.action_act,
            dtype=torch.float32,
        )
        self.action_time_mlp_in = nn.Linear(self.hidden_size * 2, self.hidden_size, dtype=torch.float32)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float32)
        self.gradient_checkpointing_enabled = False

    def get_input_embeddings(self):
        return self.vlm_backbone.get_input_embeddings()

    def flow_head_autocast_context(self):
        if self.config.force_fp32_autocast:
            return torch.autocast(
                device_type=self.state_proj.weight.device.type,
                enabled=False,
            )
        return contextlib.nullcontext()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the Qwen2.5-VL backbone."""
        self.gradient_checkpointing_enabled = True
        self.vlm_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Enabled gradient checkpointing for EO1VisionFlowMatchingModel")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the Qwen2.5-VL backbone."""
        self.gradient_checkpointing_enabled = False
        self.vlm_backbone.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for EO1VisionFlowMatchingModel")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Apply manual gradient checkpointing to EO1 flow-head computations when training."""
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(
            bsize,
            device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
        state_features: torch.FloatTensor | None = None,
        action_features: torch.FloatTensor | None = None,
        *,
        state_token_id: int,
        action_token_id: int,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """Return EO1 state/action placeholder masks, following Qwen's multimodal mask style."""
        if input_ids is None:
            special_state_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(state_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_state_mask = special_state_mask.all(-1)
            special_action_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(action_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_action_mask = special_action_mask.all(-1)
        else:
            special_state_mask = input_ids == state_token_id
            special_action_mask = input_ids == action_token_id

        n_state_tokens = special_state_mask.sum()
        special_state_mask = (
            special_state_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        if state_features is not None:
            torch_compilable_check(
                inputs_embeds[special_state_mask].numel() == state_features.numel(),
                f"State features and state tokens do not match, tokens: {n_state_tokens}, features: {state_features.shape[0]}",
            )

        n_action_tokens = special_action_mask.sum()
        special_action_mask = (
            special_action_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        if action_features is not None:
            torch_compilable_check(
                inputs_embeds[special_action_mask].numel() == action_features.numel(),
                f"Action features and action tokens do not match, tokens: {n_action_tokens}, features: {action_features.shape[0]}",
            )

        return special_state_mask, special_action_mask

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        states: torch.Tensor,
        *,
        state_token_id: int,
        action_token_id: int,
    ) -> torch.FloatTensor:
        """Embed the EO1 prefix tokens before native Qwen injects multimodal features."""

        # Get the input embeddings for the input IDs
        def input_embed_func(input_ids: torch.LongTensor) -> torch.FloatTensor:
            return self.get_input_embeddings()(input_ids)

        inputs_embeds = self._apply_checkpoint(input_embed_func, input_ids)

        # Project the states to the hidden size
        def state_proj_func(states: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                states = states.to(dtype=self.state_proj.weight.dtype)
                return self.state_proj(states)

        state_embs = self._apply_checkpoint(state_proj_func, states)
        state_mask, _ = self.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            state_features=state_embs,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        state_embs = state_embs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_embs)
        return inputs_embeds

    def embed_suffix(
        self,
        timestep: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.FloatTensor:
        """Embed the suffix"""

        def action_proj_func(noisy_actions: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
                return self.action_in_proj(noisy_actions)

        action_embs = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_embs = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=action_embs.device,
        )
        time_embs = time_embs.to(dtype=action_embs.dtype)
        time_embs = time_embs[:, None, :].expand_as(action_embs)
        action_time_embs = torch.cat([action_embs, time_embs], dim=2)

        def mlp_func(action_time_embs: torch.Tensor) -> torch.FloatTensor:
            with self.flow_head_autocast_context():
                action_time_embs = action_time_embs.to(dtype=self.action_time_mlp_in.weight.dtype)
                action_time_embs = self.action_time_mlp_in(action_time_embs)
                action_time_embs = F.silu(action_time_embs)
                return self.action_time_mlp_out(action_time_embs)

        action_time_embs = self._apply_checkpoint(mlp_func, action_time_embs)
        return action_time_embs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        states: torch.FloatTensor | None = None,
        action: torch.FloatTensor | None = None,
        action_is_pad: torch.BoolTensor | None = None,
        text_labels: torch.LongTensor | None = None,
        *,
        state_token_id: int,
        action_token_id: int,
        **kwargs,
    ) -> EO1Output:
        """Run EO-1 flow and sparse assistant-token supervision in one sequence."""

        # 1. Build the EO1 prefix with state placeholders resolved.
        inputs_embeds = self.embed_prefix(
            input_ids,
            states=states,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )

        # 2. Sample the diffusion target only for rows carrying action placeholders.
        _, action_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        action_token_mask = action_mask[..., 0]
        action_rows = action_token_mask.any(dim=-1)
        u_t = None
        if action_rows.any():
            active_action = action[action_rows]
            time = self.sample_time(active_action.shape[0], inputs_embeds.device)
            noise = self.sample_noise(active_action.shape, inputs_embeds.device)
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * active_action
            u_t = noise - active_action
            action_time_embs = self.embed_suffix(time, x_t)
            expected_tokens = int(action_token_mask.sum().item())
            if expected_tokens != action_time_embs.shape[0] * action_time_embs.shape[1]:
                raise ValueError("EO-1 requires one action placeholder per supervised horizon step.")
            inputs_embeds = inputs_embeds.masked_scatter(
                action_mask,
                action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        # 3. Optionally drop padded action tokens from backbone attention.
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        active_action_is_pad = None
        if action_rows.any() and not self.config.supervise_padding_actions:
            active_action_is_pad = action_is_pad[action_rows].to(
                device=inputs_embeds.device, dtype=torch.bool
            )
            action_padding_mask = torch.zeros_like(action_token_mask)
            action_padding_mask = action_padding_mask.masked_scatter(
                action_token_mask,
                active_action_is_pad.reshape(-1),
            )
            attention_mask = attention_mask.masked_fill(action_padding_mask, 0)

        # 4. Run the Qwen backbone on the fused EO1 sequence.
        def vlm_forward_func(
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            inputs_embeds: torch.FloatTensor,
            pixel_values: torch.Tensor | None,
            image_grid_thw: torch.LongTensor | None,
            mm_token_type_ids: torch.IntTensor | None,
        ) -> torch.FloatTensor:
            outputs = self.vlm_backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

        hidden_states = self._apply_checkpoint(
            vlm_forward_func,
            input_ids,
            attention_mask,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
        )
        text_loss = None
        if text_labels is not None:
            shifted_labels = text_labels[:, 1:].contiguous()
            text_mask = shifted_labels != -100
            if text_mask.any():
                text_hidden = hidden_states[:, :-1][text_mask]
                text_hidden = text_hidden.to(self.vlm_backbone.lm_head.weight.dtype)
                text_logits = self.vlm_backbone.lm_head(text_hidden).float()
                text_loss = F.cross_entropy(text_logits, shifted_labels[text_mask].to(text_logits.device))

        flow_loss = None
        if action_rows.any():
            assert u_t is not None

            def action_out_proj_func(action_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
                with self.flow_head_autocast_context():
                    action_hidden_states = action_hidden_states.to(dtype=self.action_out_proj.dtype)
                    return self.action_out_proj(action_hidden_states)

            v_t = self._apply_checkpoint(action_out_proj_func, hidden_states[action_token_mask])
            v_t = v_t.reshape(u_t.shape).to(dtype=u_t.dtype)
            losses = F.mse_loss(u_t, v_t, reduction="none")
            if not self.config.supervise_padding_action_dims:
                original_action_dim = self.config.output_features[ACTION].shape[0]
                losses = losses[..., :original_action_dim]
            if not self.config.supervise_padding_actions:
                losses = losses[~active_action_is_pad]
            flow_loss = losses.mean()

        losses = [value for value in (flow_loss, text_loss) if value is not None]
        return EO1Output(loss=sum(losses) if losses else None, flow_loss=flow_loss, text_loss=text_loss)

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        states: torch.Tensor | None = None,
        *,
        state_token_id: int,
        action_token_id: int,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""
        if states is None:
            raise ValueError("states are required for EO1 action sampling.")
        if mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids are required for EO1 action sampling.")

        # 1. Resolve the left-padded rollout prompt and locate the action span.
        chunk_size = self.config.chunk_size

        inputs_embeds = self.embed_prefix(
            input_ids,
            states=states,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        ).clone()
        _, action_placeholder_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        action_mask = action_placeholder_mask[..., 0]
        token_counts = action_mask.sum(dim=1)
        if not torch.all(token_counts == chunk_size):
            raise ValueError(
                f"Each sample must contain exactly {chunk_size} action tokens, got {token_counts.tolist()}."
            )
        if action_mask.ne(action_mask[:1]).any():
            raise ValueError(
                "Batch inference expects all samples to share the same action token mask after left padding."
            )
        act_start = int(action_mask[0].to(torch.int64).argmax().item())
        act_end = act_start + self.config.chunk_size
        if not torch.all(action_mask[:, act_start:act_end]):
            raise ValueError("Action tokens must form a contiguous chunk of length chunk_size.")
        act_slice = slice(act_start, act_end)

        # 2. Encode the fixed prefix once and cache its KV state.
        batch_size = input_ids.shape[0]
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device)
        mm_token_type_ids = mm_token_type_ids.to(device)
        position_ids, _ = self.vlm_backbone.model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        position_ids = position_ids.to(device)

        outputs = self.vlm_backbone.model(
            input_ids=input_ids[:, :act_start],
            attention_mask=attention_mask[:, :act_start],
            position_ids=position_ids[..., :act_start],
            inputs_embeds=inputs_embeds[:, :act_start],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids[:, :act_start],
            use_cache=True,
            return_dict=True,
        )

        x_t = self.sample_noise(
            (batch_size, chunk_size, self.config.max_action_dim),
            device,
        ).to(dtype=self.action_in_proj.weight.dtype)
        past_key_values = outputs.past_key_values

        # 3. Denoise only the action chunk while keeping the prefix cache invariant.
        def denoise_fn(input_x_t, current_timestep):
            action_time_embs = self.embed_suffix(current_timestep, input_x_t)
            inputs_embeds[:, act_slice] = action_time_embs.to(inputs_embeds.dtype)

            # Keep the prefix KV cache invariant across denoising steps.
            past_key_values.crop(act_start)
            outputs = self.vlm_backbone.model(
                attention_mask=attention_mask[:, :act_end],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, act_slice],
                position_ids=position_ids[..., act_slice],
                use_cache=True,
                return_dict=True,
            )
            with self.flow_head_autocast_context():
                hidden_states = outputs.last_hidden_state[:, :chunk_size]
                hidden_states = hidden_states.to(dtype=self.action_out_proj.dtype)
                v_t = self.action_out_proj(hidden_states)
            return v_t.reshape(input_x_t.shape).to(input_x_t.dtype)

        x_t = euler_integrate(denoise_fn, x_t, self.config.num_denoise_steps)
        return x_t
