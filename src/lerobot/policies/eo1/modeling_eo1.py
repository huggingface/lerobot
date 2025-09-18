#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
"""
EO-1:

[Paper](https://huggingface.co/papers/2508.21112)

Install eo1 extra dependencies:
```bash
pip install -e ".[eo1]"
```

Example of finetuning a eo1. EO-1 is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=eo1 \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=32 \
--policy.push_to_hub=false \
--steps=200000
```
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoProcessor
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel as TransformersPreTrainedModel

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.eo1.configuration_eo1 import EO1Config, EO1VisionFlowMatchingConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.utils import get_safe_dtype

from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class EO1Policy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = EO1Config
    name = "eo1"

    def __init__(
        self,
        config: EO1Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.processor = AutoProcessor.from_pretrained(self.config.vla_model_name, trust_remote_code=True)
        self.model = EO1VisionFlowMatchingModel.from_pretrained(
            self.config.vla_model_name,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
        )
        self.model.set_requires_grad(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            freeze_state_proj=self.config.freeze_state_proj,
            freeze_lm_head=self.config.freeze_lm_head,
        )
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        state = self.prepare_state(batch)
        input_ids, attention_mask, pixel_values, image_grid_thw = self.prepare_image_language(batch)

        actions = self.model.sample_actions(
            input_ids, attention_mask, pixel_values, image_grid_thw, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        input_ids, attention_mask, pixel_values, image_grid_thw = self.prepare_image_language(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        # TODO: Add labels to support auto-regressive co-training @ DelinQu
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            states=state,
            labels=None,
            actions=actions,
            action_is_pad=actions_is_pad,
        )
        loss_dict["fm_loss"] = outputs.fm_loss
        loss_dict["ar_loss"] = outputs.ar_loss or 0
        return outputs.loss, loss_dict

    def prepare_image_language(self, batch):
        """prepare images and language for the processor"""
        images = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # 1. preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            images.append(img)

        # Create image features not present in the batch as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            images.append(img)

        image_tensors = []
        n_images, batch_size = len(images), images[0].shape[0]
        for i in range(batch_size):
            for j in range(n_images):
                image_tensors.append(images[j][i])

        # 2. prepare language
        languages = []
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        for task in tasks:
            img_replace = "<|vision_start|><|image_pad|><|vision_end|>" * n_images
            state_replace = "<|state_start|><|state_pad|><|state_end|>"
            action_replace = "<|action_start|><|action_pad|><|action_end|>"
            prompt = f"<|im_start|>system\nYou are a helpful physical assistant.<|im_end|>\n<|im_start|>user\n{img_replace}{state_replace}{task}<|vla|><|im_end|>\n<|im_start|>assistant\n{action_replace}<|im_end|>"
            languages.append(prompt)

        device = img.device
        inputs = self.processor(
            image_tensors,
            languages,
            return_tensors="pt",
            padding=True,
            min_pixels=self.config.image_min_pixels,
            max_pixels=self.config.image_max_pixels,
            noise_token_num=self.config.chunk_size,
        ).to(device)
        return inputs.input_ids, inputs.attention_mask, inputs.pixel_values, inputs.image_grid_thw

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


@dataclass
class EO1VisionFlowMatchingOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    fm_loss: torch.FloatTensor | None = None
    ar_loss: torch.FloatTensor | None = None

    actions: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None

    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


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


class EO1VisionFlowMatchingModel(TransformersPreTrainedModel, GenerationMixin):
    config_class = EO1VisionFlowMatchingConfig
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True

    def __init__(
        self,
        config: EO1VisionFlowMatchingConfig,
    ):
        super().__init__(config)

        hidden_size = self.config.text_config.hidden_size
        max_action_dim = self.config.max_action_dim
        self.vlm_backbone = Qwen2_5_VLForConditionalGeneration(self.config)
        self.state_proj = nn.Linear(max_action_dim, hidden_size)
        self.action_in_proj = nn.Linear(max_action_dim, hidden_size)
        self.action_out_proj = EO1VisionActionProjector(
            hidden_size,
            max_action_dim,
            self.config.num_action_layers,
            self.config.action_act,
        )
        self.action_time_mlp_in = nn.Linear(hidden_size * 2, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)

        self.post_init()
        self.to_bfloat16_vlm_backbone()

    def to_bfloat16_vlm_backbone(self):
        self.vlm_backbone = self.vlm_backbone.to(dtype=torch.bfloat16)

    def set_requires_grad(
        self,
        freeze_vision_encoder: bool = False,
        freeze_state_proj: bool = False,
        freeze_lm_head: bool = True,
    ):
        if freeze_vision_encoder:
            self.vlm_backbone.visual.eval()
            for params in self.vlm_backbone.visual.parameters():
                params.requires_grad = False

        for params in self.state_proj.parameters():
            params.requires_grad = not freeze_state_proj

        for params in self.vlm_backbone.lm_head.parameters():
            params.requires_grad = not freeze_lm_head

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def replace_special_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        special_features: torch.FloatTensor = None,
        special_token_ids: torch.LongTensor = None,
    ) -> torch.LongTensor:
        """Replace the special embeddings with the special features."""
        if special_features is not None and special_token_ids is not None:
            n_special_tokens = (input_ids == special_token_ids).sum().item()
            n_special_features = special_features.shape[0]
            assert n_special_tokens == n_special_features, (
                f"Special features and special tokens {special_token_ids} do not match: \
                tokens: {n_special_tokens}, features {n_special_features}"
            )
            mask = input_ids == special_token_ids
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            special_mask = mask_expanded.to(inputs_embeds.device)
            special_features = special_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_mask, special_features)
        return inputs_embeds, None

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
        """Embed the suffix: image, text, video, state"""
        if inputs_embeds is None:
            inputs_embeds = self.vlm_backbone.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.vlm_backbone.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.vlm_backbone.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.vlm_backbone.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.vlm_backbone.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if states is not None:
            states = states.type(self.state_proj.weight.dtype)
            state_embs = self.state_proj(states)
            inputs_embeds, _ = self.replace_special_embeddings(
                input_ids, inputs_embeds, state_embs, self.config.state_token_id
            )
        return inputs_embeds

    def embed_suffix(
        self,
        timestep: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.FloatTensor:
        """Embed the suffix: time and noisy actions"""
        time_embs = create_sinusoidal_pos_embedding(
            timestep,
            self.config.text_config.hidden_size,
            device=noisy_actions.device,
        )
        time_embs = time_embs.type(noisy_actions.dtype)
        noisy_actions = noisy_actions.type(self.action_in_proj.weight.dtype)
        action_embs = self.action_in_proj(noisy_actions)
        time_embs = time_embs[:, None, :].expand_as(action_embs)

        action_time_embs = torch.cat([action_embs, time_embs], dim=2)
        action_time_embs = self.action_time_mlp_in(action_time_embs)
        action_time_embs = F.silu(action_time_embs)
        action_time_embs = self.action_time_mlp_out(action_time_embs)
        return action_time_embs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        states: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        **kwargs,
    ) -> EO1VisionFlowMatchingOutputWithPast:
        """multi-modal forward pass, including image, video, state, action, and language."""

        inputs_embeds = self.embed_prefix(
            input_ids,
            inputs_embeds,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            states,
        )

        if actions is not None:
            noise_mask = input_ids == self.config.action_token_id
            pass_mask = input_ids == self.config.action_pass_id
            mask = noise_mask | pass_mask  # (b s)

            pass_mask_in_action = pass_mask[mask]  # (n, )
            pass_mask_in_action = pass_mask_in_action.reshape(*actions.shape[:2], 1)  # (b, h, 1)

            time = self.sample_time(actions.shape[0], inputs_embeds.device)  # (n,)
            time_expanded = time[:, None, None].repeat(1, actions.shape[1], 1)  # (b, h, 1)
            time_expanded[pass_mask_in_action] = 0.0

            noise = self.sample_noise(actions.shape, inputs_embeds.device)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            action_time_embs = self.embed_suffix(time, x_t)
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            action_mask = mask_expanded.to(inputs_embeds.device)

            action_time_embs = action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_time_embs)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            prefill_noncompiled_stage = (cache_position is not None and cache_position[0] == 0) or (
                past_key_values is None or past_key_values.get_seq_length() == 0
            )
            if prefill_noncompiled_stage or self.vlm_backbone.rope_deltas is None:
                position_ids, rope_deltas = self.vlm_backbone.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.vlm_backbone.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.vlm_backbone.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        # generation
        outputs = self.vlm_backbone.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        # only compute necessary logits, do not upcast to float if not computing loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.vlm_backbone.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        fm_loss = None
        v_t = None
        if actions is not None:
            action_time_embs = hidden_states[action_mask[..., 0]]
            action_time_embs = action_time_embs.type(self.action_out_proj.dtype)

            v_t = self.action_out_proj(action_time_embs)
            u_t = u_t.reshape(v_t.shape)
            v_t = v_t.type(u_t.dtype)

            losses = F.mse_loss(u_t, v_t, reduction="none")
            if action_is_pad is not None:
                in_episode_bound = (~action_is_pad).reshape(-1, 1)
                losses = losses * in_episode_bound

            in_denoise_bound = (~pass_mask_in_action).reshape(-1, 1)
            losses = losses * in_denoise_bound

            fm_loss = losses.mean()
            loss = fm_loss

        ar_loss = None
        if labels is not None:
            ar_loss = self.vlm_backbone.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
            loss = loss + ar_loss if loss is not None else ar_loss

        return EO1VisionFlowMatchingOutputWithPast(
            loss=loss,
            fm_loss=fm_loss,
            ar_loss=ar_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.vlm_backbone.rope_deltas,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> Tensor:
        """Sample actions from the model."""
        # prepare pos_ids and kv_cache
        seq_len = input_ids.shape[-1]
        pos_ids, _ = self.vlm_backbone.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        suffix_len = -2  # exclude <|action_end|><|im_end|>
        prefix_len = seq_len - self.config.action_chunk_size - 2

        # embed prefix
        inputs_embeds = self.embed_prefix(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            states=states,
        )

        # pass prefix, update kvcache
        outputs = self.vlm_backbone.model(
            position_ids=pos_ids[..., :prefix_len],
            attention_mask=attention_mask[:, :prefix_len],
            inputs_embeds=inputs_embeds[:, :prefix_len],
            use_cache=True,
        )

        # denoising
        device = states.device
        if noise is None:
            actions_shape = (states.shape[0], self.config.action_chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        x_t = noise.type(self.action_in_proj.weight.dtype)
        dt = torch.tensor(-1.0 / self.config.num_denoise_steps, device=device)
        time = torch.ones(inputs_embeds.shape[0], device=device)
        past_key_values = outputs.past_key_values

        action_mask = input_ids == self.config.action_token_id
        while time >= -dt / 2:
            action_time_embs = self.embed_suffix(time, x_t)
            inputs_embeds[action_mask] = action_time_embs.to(inputs_embeds.dtype)

            past_key_values.crop(prefix_len)
            outputs = self.vlm_backbone.model(
                position_ids=pos_ids[..., prefix_len:suffix_len],
                attention_mask=attention_mask[:, :suffix_len],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, prefix_len:suffix_len],
                use_cache=True,
            )
            action_time_embs = outputs.last_hidden_state
            action_time_embs = action_time_embs.type(self.action_out_proj.dtype)
            v_t = self.action_out_proj(action_time_embs)

            x_t += dt * v_t.reshape(x_t.shape)
            time += dt
        return x_t

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone.prepare_inputs_for_generation(*args, **kwargs)

    def _expand_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone._expand_inputs_for_generation(*args, **kwargs)
