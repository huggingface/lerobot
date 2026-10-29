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

"""Cosmos3 policy for LeRobot.

Cosmos3 jointly predicts future video frames and robot action chunks from robot
observations and a language instruction.
"""

import itertools
import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor

from lerobot.policies.chunk import ChunkPolicySpec
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.import_utils import _diffusers_available, _transformers_available, require_package

from .configuration_cosmos3 import (
    RESOLUTION_TIER_LIMITS,
    Cosmos3Config,
    Cosmos3EdgeConfig,
    Cosmos3NanoConfig,
    Nemotron3DenseVLMoTConfig,
    Nemotron3DenseVLMoTTextConfig,
    Nemotron3DenseVLProjectorConfig,
    Nemotron3DenseVLVisionConfig,
    Qwen3VLMoTConfig,
    Qwen3VLMoTTextConfig,
)

if TYPE_CHECKING or _transformers_available:
    from transformers.activations import ACT2FN
    from transformers.generation import GenerationMixin
    from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        BaseModelOutputWithDeepstackFeatures,
        Qwen3VLPreTrainedModel,
        Qwen3VLTextMLP,
        Qwen3VLTextRMSNorm,
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionModel,
        apply_rotary_pos_emb,
        eager_attention_forward,
        rotate_half,
    )
    from transformers.models.siglip2.modeling_siglip2 import (
        Siglip2Attention,
        Siglip2MLP,
        Siglip2VisionEmbeddings,
    )
    from transformers.utils import ModelOutput, auto_docstring, torch_compilable_check
    from transformers.utils.generic import is_flash_attention_requested
else:
    ACT2FN = None
    ALL_ATTENTION_FUNCTIONS = None
    BaseModelOutputWithDeepstackFeatures = None
    BaseModelOutputWithPooling = None
    GenerationMixin = object
    GradientCheckpointingLayer = nn.Module
    ModelOutput = object
    PreTrainedModel = nn.Module
    Qwen3VLPreTrainedModel = nn.Module
    Qwen3VLTextMLP = None
    Qwen3VLTextRMSNorm = None
    Qwen3VLTextRotaryEmbedding = None
    Qwen3VLVisionModel = None
    Siglip2Attention = nn.Module
    Siglip2MLP = None
    Siglip2VisionEmbeddings = nn.Module
    apply_rotary_pos_emb = None
    create_bidirectional_mask = None
    create_causal_mask = None
    eager_attention_forward = None
    is_flash_attention_requested = None
    rotate_half = None
    torch_compilable_check = None

    def auto_docstring(obj):
        return obj


if TYPE_CHECKING or _diffusers_available:
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.models.embeddings import TimestepEmbedding
else:
    AutoencoderKLWan = TimestepEmbedding = UniPCMultistepScheduler = None


class Cosmos3Policy(PreTrainedPolicy):
    """
    LeRobot policy of Cosmos3 around the flow matching model `OmniMoTModel`, shared by the policy types
    `cosmos3_nano` (`Cosmos3NanoPolicy`) and `cosmos3_edge` (`Cosmos3EdgePolicy`).

    The batch holds the outputs of the Cosmos3 processors (`processor_cosmos3.py`): the prompt of the
    understanding pathway (`input_ids`, `attention_mask`, `mm_token_type_ids`, and `pixel_values`,
    `image_grid_thw`, `pixel_values_videos`, `video_grid_thw` when it has images or videos), the video clip of
    the generation pathway (`pixel_values_videos_gen`, `video_grid_thw_gen`, `second_per_grid_ts_gen`), the
    negative prompt for inference (`negative_input_ids`, `negative_attention_mask`,
    `negative_mm_token_type_ids`), the current state `observation.state` with `use_state`, and the action
    chunk `action` for training.

    The state and the actions are zero padded to `max_action_dim`, the predicted actions are cropped to
    `original_action_dim`. With `save_predicted_video` the predicted video latents of every chunk go to
    `last_predicted_latents`, which lerobot-eval decodes with `decode_predicted_latents`.
    """

    config_class = Cosmos3Config
    name = "cosmos3"

    def __init__(self, config: Cosmos3Config, **kwargs):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="cosmos3")
        require_package("diffusers", extra="cosmos3")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = OmniMoTModel(config)
        if config.gradient_checkpointing:
            self.model.network.gradient_checkpointing_enable()
        if config.compile_model:
            self.model.network.compile_decoder_layers(config.compile_mode)
            self.model.network.compile_vae_encoder(config.compile_mode)
            if config.gradient_checkpointing:
                # Compiled checkpointed layers need Dynamo's DDP graph splitting off (process wide).
                torch._dynamo.config.optimize_ddp = False
                logging.info(
                    "Set torch._dynamo.config.optimize_ddp = False for compiled checkpointed layers."
                )

        # Video latents of the last chunk with `save_predicted_video`, read and cleared by lerobot-eval.
        self.last_predicted_latents: Tensor | None = None
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        """
        The trainable parameters in two groups: the action modules of the network
        (`Cosmos3VFMNetwork._action_modules`) at `optimizer_action_lr_multiplier` times the learning rate, the
        others at the learning rate.
        """
        action_modules = self.model.network._action_modules
        action_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(part in action_modules for part in name.split(".")):
                action_params.append(param)
            else:
                other_params.append(param)
        action_lr = self.config.optimizer_lr * self.config.optimizer_action_lr_multiplier
        return [{"params": other_params}, {"params": action_params, "lr": action_lr}]

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA on the attention projections of the generation pathway, the action modules train in full."""
        return {
            "target_modules": ["q_proj_gen", "k_proj_gen", "v_proj_gen", "o_proj_gen"],
            "modules_to_save": list(Cosmos3VFMNetwork._action_modules),
        }

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """
        Load the weights. Without `load_action_modules` the action modules keep their initialization (the
        native `keys_to_skip_loading`), then `load_action_modules` turns back to true so the checkpoints of
        the run load them.
        """
        if model.config.load_action_modules:
            return super()._load_as_safetensor(model, model_file, map_location, strict)
        action_modules = model.model.network._action_modules
        state_dict = {
            key: value
            for key, value in load_file(model_file, device=resolve_safetensors_device(map_location)).items()
            if not any(part in action_modules for part in key.split("."))
        }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        missing_keys = [
            key for key in missing_keys if not any(part in action_modules for part in key.split("."))
        ]
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Missing keys {missing_keys}, unexpected keys {unexpected_keys} in {model_file}."
            )
        log_model_loading_keys(missing_keys, unexpected_keys)
        model.config.load_action_modules = True
        logging.info(
            f"Kept the initialization of the action modules {action_modules} when loading {model_file}."
        )
        return model

    def reset(self):
        """Reset internal state, called when the environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.last_predicted_latents = None

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor | None:
        """The current state in the action layout zero padded to `max_action_dim`, None without `use_state`."""
        if not self.config.use_state:
            return None
        return pad_vector(batch[OBS_STATE], self.config.max_action_dim)

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """The action chunk in the action space of the model zero padded to `max_action_dim`."""
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Run the batch through the model and compute the joint video and action loss for training."""
        loss, losses = self.model(
            input_ids=batch["input_ids"],
            pixel_values_videos_gen=batch["pixel_values_videos_gen"],
            video_grid_thw_gen=batch["video_grid_thw_gen"],
            second_per_grid_ts_gen=batch["second_per_grid_ts_gen"],
            actions=self.prepare_action(batch),
            state=self.prepare_state(batch),
            attention_mask=batch.get("attention_mask"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
            pixel_values=batch.get("pixel_values"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
        )
        return loss, {name: value.item() for name, value in losses.items()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """
        Sample the action chunk `[batch, chunk_size, original_action_dim]` with the sampling settings of the
        config. With `save_predicted_video` the video latents of the clip go to `last_predicted_latents`.
        """
        self.eval()
        actions, video_latents = self.model.generate(
            input_ids=batch["input_ids"],
            pixel_values_videos_gen=batch["pixel_values_videos_gen"],
            video_grid_thw_gen=batch["video_grid_thw_gen"],
            second_per_grid_ts_gen=batch["second_per_grid_ts_gen"],
            state=self.prepare_state(batch),
            attention_mask=batch.get("attention_mask"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
            pixel_values=batch.get("pixel_values"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            negative_input_ids=batch.get("negative_input_ids"),
            negative_attention_mask=batch.get("negative_attention_mask"),
            negative_mm_token_type_ids=batch.get("negative_mm_token_type_ids"),
        )
        if self.config.save_predicted_video:
            self.last_predicted_latents = video_latents.cpu()
        return actions[:, :, : self.config.original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        """
        Chunk serving from the current observation: `chunk_size` actions predicted, `n_action_steps` executed.
        """
        if self.config.n_obs_steps != 1:
            raise ValueError("Cosmos3 chunk serving requires n_obs_steps=1.")
        return ChunkPolicySpec(
            prediction_steps=self.config.chunk_size, execution_steps=self.config.n_action_steps
        )

    @torch.no_grad()
    def decode_predicted_latents(self, latents: Tensor) -> Tensor:
        """
        Decode the predicted video latents that lerobot-eval concatenated over the chunks of a rollout,
        `[batch, z_dim, latent_frames, latent_height, latent_width]`, to the uint8 frames of the first sample,
        `[frames, height, width, 3]`. The chunks are decoded one by one and their frames concatenated.
        """
        vae = self.model.network.vae
        latent_frames_per_chunk = 1 + self.config.chunk_size // vae.config.scale_factor_temporal
        if latents.shape[2] % latent_frames_per_chunk != 0:
            raise ValueError(
                f"{latents.shape[2]} latent frames do not split into chunks of {latent_frames_per_chunk}."
            )
        device = next(vae.parameters()).device
        frames = []
        for chunk in latents[:1].split(latent_frames_per_chunk, dim=2):
            video = self.model.network.decode_video(chunk.to(device))[0].float()  # [3, frames, height, width]
            frames.append(((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0))
        return torch.cat(frames).cpu()


class Cosmos3NanoPolicy(Cosmos3Policy):
    """`Cosmos3Policy` of the `cosmos3_nano` policy type, Cosmos3 Nano."""

    config_class = Cosmos3NanoConfig
    name = "cosmos3_nano"
    # FSDP2 wrap units of the Qwen3-VL MoT backbone: the decoder layers and the vision blocks.
    _fsdp_wrap_modules = ["Qwen3VLMoTDecoderLayer", "Qwen3VLVisionBlock"]


class Cosmos3EdgePolicy(Cosmos3Policy):
    """`Cosmos3Policy` of the `cosmos3_edge` policy type, Cosmos3 Edge."""

    config_class = Cosmos3EdgeConfig
    name = "cosmos3_edge"
    # FSDP2 wrap units of the Nemotron 3 Dense VL MoT backbone: the decoder layers and the vision encoder layers.
    _fsdp_wrap_modules = ["Nemotron3DenseVLMoTDecoderLayer", "Nemotron3DenseVLVisionEncoderLayer"]


@dataclass
class FlowState:
    """
    The video latents and the actions as one pair of float32 tensors, a point of the flow matching trajectory or a
    velocity.

    Args:
        video_latents: Normalized video latents `[batch, z_dim, latent_frames, latent_height, latent_width]`.
        actions: The action sequence of the generation pathway `[batch, num_actions, max_action_dim]` (see
            `OmniMoTModel.pack_actions`), the state action first with `use_state`, zero padded after
            `original_action_dim`.
    """

    video_latents: Tensor
    actions: Tensor


@dataclass
class SequencePlan:
    """
    Which latent frames and actions of every sample are clean conditions, as tensors. See
    `OmniMoTModel.build_sequence_plan`.

    Args:
        video_condition_mask: `[batch, latent_frames]`, True on the clean condition frames.
        action_condition_mask: `[batch, num_actions]`, True on the clean condition actions.
        action_start_frame_offset: `[batch]`, the video frame the first action is aligned with on the M-RoPE
            time axis.
    """

    video_condition_mask: Tensor
    action_condition_mask: Tensor
    action_start_frame_offset: Tensor


# (x_t, timestep) -> v_t, the timestep a scalar tensor in [0, num_train_timesteps].
VelocityFn = Callable[[FlowState, Tensor], FlowState]


class OmniMoTModel(nn.Module):
    """
    Flow matching training and sampling of Cosmos3 around `Cosmos3VFMNetwork`, which owns every parameter.

    Training (`forward`) encodes the clean video latents, draws a noise level per sample, noises the latents
    and the actions outside their clean conditions and regresses the network velocity on `noise - x_0`.
    Sampling (`generate`) encodes the condition frames, starts from noise outside the conditions and
    integrates the flow ODE with UniPC, calling a `velocity_fn` closure over both classifier-free guidance
    branches.

    The understanding pathway reads the prompt with its images and videos. The generation pathway reads the
    `*_gen` inputs (the video clip, its latent grid and frame timing) and the action sequence of
    `pack_actions` (the current state and the actions, zero padded to `max_action_dim` by the policy). The
    sequence plan (`mode`, `use_state`, `n_obs_steps`) and the embodiment (`domain_id`) come from the config.

    The padded action channels after `original_action_dim` stay zero along the whole flow: no noise, no
    velocity and no loss.
    """

    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config
        self.network = Cosmos3VFMNetwork(config)

    def forward(
        self,
        input_ids: Tensor,
        pixel_values_videos_gen: Tensor,
        video_grid_thw_gen: Tensor,
        second_per_grid_ts_gen: Tensor,
        actions: Tensor,
        state: Tensor | None = None,
        attention_mask: Tensor | None = None,
        mm_token_type_ids: Tensor | None = None,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        noise: FlowState | None = None,
        video_sigmas: Tensor | None = None,
        action_sigmas: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Flow matching loss of a training batch.

        Args:
            input_ids: Prompt tokens of the understanding pathway `[batch, text_len]`, right padded.
            pixel_values_videos_gen: The whole video clip in `[-1, 1]`, `[batch, 3, frames, height, width]`,
                padded to the canvas by reflection on the right and at the bottom.
            video_grid_thw_gen: Latent grid of the content of the clip `[batch, 3]`, the same for every sample.
            second_per_grid_ts_gen: Seconds per latent frame `[batch]`, the temporal compression of the VAE
                divided by the frame rate of the clip.
            actions: The action chunk `[batch, chunk_size, max_action_dim]`, the actions of the frames after the
                first one, zero padded after `original_action_dim`.
            state: The current state in the action layout `[batch, max_action_dim]`, zero padded after
                `original_action_dim`. Needed with `use_state`.
            attention_mask: Padding mask of the prompt `[batch, text_len]`.
            mm_token_type_ids: Text (0), image (1) and video (2) tokens of the prompt `[batch, text_len]`. All
                text when None.
            pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw: Images and videos of the prompt
                for the vision tower of the backbone.
            noise: Noise of the interpolation, standard normal when None.
            video_sigmas: Noise level of the video of every sample `[batch]`, drawn with
                `sample_training_sigmas` when None (both or neither with `action_sigmas`).
            action_sigmas: Noise level of the actions of every sample `[batch]`.

        Returns:
            The weighted sum of the video and the action losses, and both losses.
        """
        packed_actions = self.pack_actions(actions, state)
        batch_size, num_actions = packed_actions.shape[:2]
        plan = self.build_sequence_plan(video_grid_thw_gen, num_actions)
        clean = self.encode_clean(pixel_values_videos_gen, video_grid_thw_gen, packed_actions, plan)
        if noise is None:
            noise = FlowState(torch.randn_like(clean.video_latents), torch.randn_like(clean.actions))
        if video_sigmas is None or action_sigmas is None:
            height, width = pixel_values_videos_gen.shape[-2:]
            video_sigmas, action_sigmas = self.sample_training_sigmas(
                batch_size, height, width, actions.device
            )
        noisy = self.add_noise(clean, noise, video_sigmas, action_sigmas, plan)

        position_ids = self.network.get_rope_index(
            input_ids,
            video_grid_thw_gen,
            second_per_grid_ts_gen,
            num_actions=num_actions,
            action_start_frame_offset=plan.action_start_frame_offset,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        velocity = self.predict_velocity(
            noisy,
            video_sigmas * self.config.num_train_timesteps,
            action_sigmas * self.config.num_train_timesteps,
            plan,
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        target = FlowState(noise.video_latents - clean.video_latents, noise.actions - clean.actions)
        video_loss, action_loss = self.flow_matching_loss(velocity, target, plan)
        loss = self.config.video_loss_weight * video_loss + self.config.action_loss_weight * action_loss
        return loss, {"video_loss": video_loss.detach(), "action_loss": action_loss.detach()}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        pixel_values_videos_gen: Tensor,
        video_grid_thw_gen: Tensor,
        second_per_grid_ts_gen: Tensor,
        state: Tensor | None = None,
        attention_mask: Tensor | None = None,
        mm_token_type_ids: Tensor | None = None,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        negative_input_ids: Tensor | None = None,
        negative_attention_mask: Tensor | None = None,
        negative_mm_token_type_ids: Tensor | None = None,
        noise: FlowState | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample the actions and the video latents with the sampling settings of the config. The arguments shared
        with `forward` mean the same, apart from the parts that get sampled.

        Args:
            pixel_values_videos_gen: At least the condition frames of the clip, the later frames are sampled.
            video_grid_thw_gen: Latent grid of the whole clip `[batch, 3]`, the frames to sample included: the
                `chunk_size + 1` frames make `1 + chunk_size / temporal_compression` latent frames.
            state: The current state in the action layout `[batch, max_action_dim]`. Needed with `use_state`.
            negative_input_ids: Text only negative prompt of the unconditional branch of classifier-free
                guidance `[batch, negative_text_len]`, right padded. Needed unless `guidance_scale` is 1.
            negative_attention_mask: Padding mask of the negative prompt.
            negative_mm_token_type_ids: Token types of the negative prompt, all text when None.
            noise: Initial noise of the trajectory `FlowState(video_latents, actions)`, the actions in the packed
                layout of `pack_actions`. Standard normal from `generator` when None.
            generator: Random generator of the initial noise.

        Returns:
            The sampled action chunk `[batch, chunk_size, max_action_dim]` without the state action, with the
            zero padded channels after `original_action_dim` for the policy to crop. The normalized video latents
            of the whole clip `[batch, z_dim, latent_frames, latent_height, latent_width]`, the condition frame
            included.
        """
        batch_size, device = video_grid_thw_gen.shape[0], pixel_values_videos_gen.device
        chunk_size = self.config.chunk_size
        num_frames = (int(video_grid_thw_gen[0, 0]) - 1) * self.network.vae.config.scale_factor_temporal + 1
        if num_frames != chunk_size + 1:
            raise ValueError(f"The clip has {num_frames} frames, chunk_size + 1 is {chunk_size + 1}.")
        # Placeholder actions, replaced by noise.
        actions = torch.zeros(batch_size, chunk_size, self.config.max_action_dim, device=device)
        packed_actions = self.pack_actions(actions, state)
        num_actions = packed_actions.shape[1]
        plan = self.build_sequence_plan(video_grid_thw_gen, num_actions)
        # M-RoPE positions of each classifier-free guidance branch, the generation stream follows its prompt.
        position_ids = self.network.get_rope_index(
            input_ids,
            video_grid_thw_gen,
            second_per_grid_ts_gen,
            num_actions=num_actions,
            action_start_frame_offset=plan.action_start_frame_offset,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        guidance_scale = self.config.guidance_scale
        # Timesteps strictly inside (low, high) get the guidance, all of them when guidance_interval is None.
        guidance_low, guidance_high = self.config.guidance_interval or (-math.inf, math.inf)
        negative_position_ids = None
        if guidance_scale != 1.0:
            if negative_input_ids is None:
                raise ValueError("Classifier-free guidance needs negative_input_ids (the negative prompt).")
            negative_position_ids = self.network.get_rope_index(
                negative_input_ids,
                video_grid_thw_gen,
                second_per_grid_ts_gen,
                num_actions=num_actions,
                action_start_frame_offset=plan.action_start_frame_offset,
                attention_mask=negative_attention_mask,
                mm_token_type_ids=negative_mm_token_type_ids,
            )

        clean = self.encode_clean(pixel_values_videos_gen, video_grid_thw_gen, packed_actions, plan)
        if noise is None:
            video_noise = torch.randn(clean.video_latents.shape, generator=generator, device=device)
            action_noise = torch.randn(clean.actions.shape, generator=generator, device=device)
            noise = FlowState(video_noise, action_noise)
        # Noise level 1 outside the conditions, the conditions keep their clean values.
        ones = clean.actions.new_ones(batch_size)
        noisy = self.add_noise(clean, noise, ones, ones, plan)

        def velocity_fn(x_t: FlowState, timestep: Tensor) -> FlowState:
            """The guided velocity at `timestep`, zero on the conditions and on the padded action channels."""
            timesteps = timestep.expand(batch_size)
            velocity = self.predict_velocity(
                x_t,
                timesteps,
                timesteps,
                plan,
                input_ids,
                position_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
            in_guidance_interval = guidance_low < timestep.item() < guidance_high
            if negative_input_ids is None or negative_position_ids is None or not in_guidance_interval:
                return self.mask_velocity(velocity, plan)

            uncond_velocity = self.predict_velocity(
                x_t,
                timesteps,
                timesteps,
                plan,
                negative_input_ids,
                negative_position_ids,
                attention_mask=negative_attention_mask,
            )
            # u + g * (c - u) in float32, the native classifier-free guidance.
            v_cond_video, v_uncond_video = velocity.video_latents, uncond_velocity.video_latents
            v_cond_action, v_uncond_action = velocity.actions, uncond_velocity.actions
            video_velocity = v_uncond_video + guidance_scale * (v_cond_video - v_uncond_video)
            action_velocity = v_uncond_action + guidance_scale * (v_cond_action - v_uncond_action)
            return self.mask_velocity(FlowState(video_velocity, action_velocity), plan)

        sampled = self.sample(velocity_fn, noisy)
        return self.unpack_actions(sampled.actions), sampled.video_latents

    def pack_actions(self, actions: Tensor, state: Tensor | None = None) -> Tensor:
        """
        The action sequence of the generation pathway: the current state in front of the action chunk with
        `use_state` (Case B of the sequence plan, a clean condition), the action chunk alone otherwise (Case A).

        Args:
            actions: The action chunk `[batch, chunk_size, max_action_dim]`.
            state: The current state in the action layout `[batch, max_action_dim]`.

        Returns:
            `[batch, num_actions, max_action_dim]` in float32, `num_actions = chunk_size + use_state`.
        """
        actions = actions.float()
        if not self.config.use_state:
            return actions
        if state is None:
            raise ValueError("use_state needs state, the current state in the action layout.")
        return torch.cat([state.float()[:, None], actions], dim=1)

    def unpack_actions(self, packed_actions: Tensor) -> Tensor:
        """The action chunk of a packed action sequence, the inverse of `pack_actions` (drops the state action)."""
        return packed_actions[:, 1:] if self.config.use_state else packed_actions

    def build_sequence_plan(self, video_grid_thw_gen: Tensor, num_actions: int) -> SequencePlan:
        """
        Which latent frames and actions are clean conditions and which video frame the first action lines up
        with on the M-RoPE time axis, the native `build_sequence_plan_from_mode` for the mode and the number of
        history actions of the config (the `n_obs_steps - 1` past states in front of the state action). Every
        sample of a batch shares the plan.

        The video conditions on its first latent frame ("inverse_dynamics" on every frame). The actions condition
        on the history actions in front of them ("forward_dynamics" on every action), and on the state action
        after the history when there is one. Leaving the history out, the actions line up with the video frames
        in one of three ways. Case A has one action per gap between frames (`num_frames - 1` actions, the first
        one on frame 1). Case B has the state action in front (`num_frames` actions, the state on frame 0).
        Case C has a denser action stream than the video, with a state action when the actions after it split
        evenly over the gaps between frames. The "policy" mode always lines up as Case A.

        Args:
            video_grid_thw_gen: Latent grid of the clip `[batch, 3]`, the same for every sample.
            num_actions: Actions of every sample, the history and state actions included.
        """
        mode, num_history_actions = self.config.mode, self.config.n_obs_steps - 1
        batch_size, num_latent_frames = video_grid_thw_gen.shape[0], int(video_grid_thw_gen[0, 0])
        temporal_compression = self.network.vae.config.scale_factor_temporal
        num_frames = (num_latent_frames - 1) * temporal_compression + 1  # pixel frames of the clip
        num_base_actions = num_actions - num_history_actions
        if mode == "policy" or num_base_actions == num_frames - 1:  # Case A
            has_state = False
        elif num_base_actions == num_frames:  # Case B
            has_state = True
        else:  # Case C
            has_state = num_frames > 1 and (num_base_actions - 1) % (num_frames - 1) == 0

        num_condition_frames = num_latent_frames if mode == "inverse_dynamics" else 1
        if mode == "forward_dynamics":
            num_condition_actions = num_actions
        else:
            num_condition_actions = num_history_actions + (1 if has_state else 0)
        # The first predicted action lines up with video frame 1, the state action with frame 0.
        action_start_frame_offset = (0 if has_state else 1) - num_history_actions

        device = video_grid_thw_gen.device
        video_condition_mask = torch.arange(num_latent_frames, device=device) < num_condition_frames
        action_condition_mask = torch.arange(num_actions, device=device) < num_condition_actions
        return SequencePlan(
            video_condition_mask.expand(batch_size, -1),
            action_condition_mask.expand(batch_size, -1),
            torch.full((batch_size,), action_start_frame_offset, device=device),
        )

    def encode_clean(
        self,
        pixel_values_videos_gen: Tensor,
        video_grid_thw_gen: Tensor,
        packed_actions: Tensor,
        plan: SequencePlan,
    ) -> FlowState:
        """
        The clean `x_0`: the normalized latents of `pixel_values_videos_gen` and the packed actions
        (`pack_actions`). For sampling only the condition frames are given, the latent frames after them are
        zero.
        """
        latents = self.network.encode_video(pixel_values_videos_gen, video_grid_thw_gen)
        num_latent_frames, num_encoded_frames = int(video_grid_thw_gen[0, 0]), latents.shape[2]
        if num_encoded_frames > num_latent_frames or plan.video_condition_mask[:, num_encoded_frames:].any():
            raise ValueError(
                f"The frames encode to {num_encoded_frames} latent frames, the clip has {num_latent_frames} "
                "and needs every condition frame."
            )
        latents = F.pad(latents, (0, 0, 0, 0, 0, num_latent_frames - num_encoded_frames))
        return FlowState(latents, packed_actions)

    def sample_training_sigmas(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        Noise levels in `[0, 1]` of the video and of the actions of every sample, `[batch]` each. The video level
        follows `video_time_distribution` with the shift of the resolution tier of the `height` x `width` canvas.
        The actions share it unless `independent_action_schedule`.
        """
        config = self.config
        shift = (
            config.training_shift[self.get_resolution_tier(height, width)]
            if isinstance(config.training_shift, dict)
            else config.training_shift
        )
        video_sigmas = self.sample_train_time(config.video_time_distribution, batch_size, shift)
        action_sigmas = video_sigmas
        if config.independent_action_schedule:
            action_shift = config.action_training_shift if config.action_training_shift is not None else shift
            action_sigmas = self.sample_train_time(config.action_time_distribution, batch_size, action_shift)
        return video_sigmas.to(device), action_sigmas.to(device)

    def sample_train_time(self, distribution: str, batch_size: int, shift: float) -> Tensor:
        """
        Training noise levels in `[0, 1]` of the native `TrainTimeSampler`, drawn on the CPU from the global random
        state: "uniform", "logitnormal" (sigmoid of a standard normal) or "waver" (`1 - u - 1.29 * (cos(pi * u /
        2)**2 - 1 + u)` of a uniform `u`, which favors high noise). Then shifted to `shift * t / (1 + (shift - 1) *
        t)`.
        """
        if distribution == "uniform":
            t = torch.rand((batch_size,))
        elif distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,)))
        elif distribution == "waver":
            u = torch.rand((batch_size,), dtype=torch.float32)
            t = 1.0 - u - 1.29 * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
        else:
            raise ValueError(f"Unknown noise level distribution {distribution!r}.")
        shifts = torch.full((batch_size,), shift, dtype=torch.float32)
        return shifts * t / (1 + (shifts - 1) * t)

    def get_resolution_tier(self, height: int, width: int) -> str:
        """Resolution tier of a canvas from its shorter side, the native `get_vision_data_resolution`."""
        shorter_side = min(height, width)
        for tier, limit in RESOLUTION_TIER_LIMITS:
            if shorter_side <= limit:
                return tier
        raise ValueError(f"Unsupported canvas size {height}x{width}.")

    def add_noise(
        self,
        clean: FlowState,
        noise: FlowState,
        video_sigmas: Tensor,
        action_sigmas: Tensor,
        plan: SequencePlan,
    ) -> FlowState:
        """
        `x_t = sigma * noise + (1 - sigma) * x_0` with the noise level of every sample `[batch]`, the native
        rectified flow interpolation. The condition frames and actions keep `x_0` (noise level 0), and the padded
        action channels after `original_action_dim` stay zero.
        """
        video_sigmas = video_sigmas[:, None] * ~plan.video_condition_mask  # [B,T]
        video_sigmas = video_sigmas[:, None, :, None, None]  # [B,1,T,1,1]
        video_latents = noise.video_latents * video_sigmas + clean.video_latents * (1 - video_sigmas)
        action_sigmas = (action_sigmas[:, None] * ~plan.action_condition_mask)[:, :, None]  # [B,T_act,1]
        actions = noise.actions * action_sigmas + clean.actions * (1 - action_sigmas)
        actions[..., self.config.original_action_dim :] = 0.0
        return FlowState(video_latents, actions)

    def flow_matching_loss(
        self, velocity: FlowState, target: FlowState, plan: SequencePlan
    ) -> tuple[Tensor, Tensor]:
        """
        Mean squared error of the velocity on the noisy frames and actions, averaged over the samples. As in the
        native loss, the video error is averaged over all latent elements of a sample (the condition frames count
        as zero error), and the action error over all actions and the real channels (`original_action_dim`).
        """
        noisy_frames = ~plan.video_condition_mask[:, None, :, None, None]  # [B,1,T,1,1]
        video_error = (velocity.video_latents - target.video_latents) ** 2 * noisy_frames
        video_loss = video_error.mean(dim=(1, 2, 3, 4)).mean()

        noisy_actions = ~plan.action_condition_mask[:, :, None]  # [B,T_act,1]
        action_error = (velocity.actions - target.actions)[..., : self.config.original_action_dim]
        action_loss = (action_error**2 * noisy_actions).mean(dim=(1, 2)).mean()
        return video_loss, action_loss

    def predict_velocity(
        self,
        x_t: FlowState,
        video_timesteps: Tensor,
        action_timesteps: Tensor,
        plan: SequencePlan,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
    ) -> FlowState:
        """
        One network call: the float32 velocity at `x_t` conditioned on one prompt, the conditional prompt with
        its images and videos or the text only negative prompt.

        Args:
            x_t: Noisy video latents and actions, the conditions hold their clean values.
            video_timesteps: Timestep of the video of every sample `[batch]`, in `[0, num_train_timesteps]`.
            action_timesteps: Timestep of the actions of every sample `[batch]`.
            plan: Condition frames and actions, which get no timestep embedding.
            input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw: The
                prompt of the branch, see `forward`.
            position_ids: M-RoPE positions of the branch from `Cosmos3VFMNetwork.get_rope_index`.
        """
        num_latent_frames = plan.video_condition_mask.shape[1]
        num_actions = plan.action_condition_mask.shape[1]
        output = self.network(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            video_latents=x_t.video_latents,
            video_timesteps=video_timesteps[:, None].expand(-1, num_latent_frames),
            video_condition_mask=plan.video_condition_mask,
            actions=x_t.actions,
            action_timesteps=action_timesteps[:, None].expand(-1, num_actions),
            action_condition_mask=plan.action_condition_mask,
        )
        if output.video_velocity is None or output.action_velocity is None:
            raise ValueError("The network output has no action velocity.")
        return FlowState(output.video_velocity.float(), output.action_velocity.float())

    def mask_velocity(self, velocity: FlowState, plan: SequencePlan) -> FlowState:
        """Zero the velocity on the condition frames and actions and on the padded action channels."""
        video_latents = velocity.video_latents.masked_fill(
            plan.video_condition_mask[:, None, :, None, None], 0.0
        )
        actions = velocity.actions.masked_fill(plan.action_condition_mask[:, :, None], 0.0)
        actions[..., self.config.original_action_dim :] = 0.0
        return FlowState(video_latents, actions)

    def sample(self, velocity_fn: VelocityFn, x_t: FlowState) -> FlowState:
        """
        Integrate the flow ODE from noise to the data with UniPC, calling only `velocity_fn`. The video and
        the actions get a scheduler each, which updates every element as one scheduler on their concatenation
        does (the native sampler).
        """
        device = x_t.video_latents.device
        video_scheduler = self.make_scheduler(device)
        action_scheduler = self.make_scheduler(device)
        for timestep in video_scheduler.timesteps:
            velocity = velocity_fn(x_t, timestep)
            x_t = FlowState(
                video_scheduler.step(velocity.video_latents, timestep, x_t.video_latents).prev_sample,
                action_scheduler.step(velocity.actions, timestep, x_t.actions).prev_sample,
            )
        return x_t

    def make_scheduler(self, device: torch.device | None = None) -> "UniPCMultistepScheduler":
        """
        A new UniPC scheduler for one trajectory with the native sigma schedule: `num_inference_steps` noise
        levels evenly spaced from `1 - 1 / num_train_timesteps` (rounded to float32 like the native sampler) down
        to 0, shifted to `shift * s / (1 + (shift - 1) * s)` with the `sampling_shift`.
        """
        num_inference_steps, shift = self.config.num_inference_steps, self.config.sampling_shift
        num_train_timesteps = self.config.num_train_timesteps
        scheduler = UniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=1.0,
        )
        sigma_max = float(np.float32(1.0 - 1.0 / num_train_timesteps))
        sigmas = np.linspace(sigma_max, 0.0, num_inference_steps + 1)[:-1]
        scheduler.set_timesteps(sigmas=shift * sigmas / (1.0 + (shift - 1.0) * sigmas), device=device)
        return scheduler


@dataclass
class Cosmos3VFMNetworkOutput(ModelOutput):
    """
    Output of `Cosmos3VFMNetwork`.

    Args:
        video_velocity: Velocity of the video latents, `[batch, z_dim, latent_frames, latent_height, latent_width]`.
            The values on the condition frames carry no meaning, `OmniMoTModel` sets them to zero.
        action_velocity: Velocity of the actions, `[batch, num_actions, max_action_dim]`, same convention. None when
            the network ran without actions.
        last_hidden_state: Final hidden states of the understanding pathway, `[batch, text_len, hidden_size]`.
        last_hidden_state_gen: Final hidden states of the generation pathway, `[batch, gen_len, hidden_size]`.
    """

    video_velocity: Tensor | None = None
    action_velocity: Tensor | None = None
    last_hidden_state: Tensor | None = None
    last_hidden_state_gen: Tensor | None = None


class Cosmos3VFMNetwork(nn.Module):
    """
    Generation network of Cosmos3, which owns every parameter of the policy: the MoT backbone, the projections
    between the backbone and the video latents and actions, the timestep embedding and the frozen Wan2.2 VAE.
    It also sets which parameters train and which stay in float32.

    `forward` maps the noisy video latents and actions at the given timesteps, conditioned on the prompt, to
    their velocities, without the VAE. `encode_video` and `decode_video` convert between pixels and normalized
    latents.

    The generation stream is laid out as `[video | actions]`: the patches of every latent frame in order, then
    every action. `merge_gen_stream`, `split_gen_stream` and `get_rope_index` share this layout.
    """

    # Modules kept in float32 whatever the policy dtype: the timestep embedding and the rotary embeddings.
    _keep_in_fp32_modules = ["time_proj", "time_embedder", "rotary_emb", "rotary_pos_emb"]
    # Action modules: trained at `optimizer_action_lr_multiplier` times the learning rate, not loaded with
    # `load_action_modules` false.
    _action_modules = ["action2llm", "llm2action", "action_modality_embed"]
    # Generation pathway modules of the backbone (the native "moe_gen" modules), the only backbone modules
    # that train with `train_gen_stream_only`. `k_norm_und_for_gen` of Cosmos3 Edge stays frozen.
    _gen_stream_modules = [
        "q_proj_gen",
        "k_proj_gen",
        "v_proj_gen",
        "o_proj_gen",
        "q_norm_gen",
        "k_norm_gen",
        "mlp_gen",
        "input_layernorm_gen",
        "post_attention_layernorm_gen",
        "norm_gen",
    ]

    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config

        self.mot_backbone = MOT_BACKBONE_MAPPING[type(config.mot_backbone)]._from_config(
            config.mot_backbone, attn_implementation=config.attn_implementation
        )
        self.vae = AutoencoderKLWan.from_config(asdict(config.vae))

        hidden_size = config.mot_backbone.get_text_config().hidden_size
        patch_dim = config.latent_patch_size**2 * self.vae.config.z_dim
        self.time_proj = SinusoidalTimesteps(config.timestep_embedding_dim)
        self.time_embedder = TimestepEmbedding(config.timestep_embedding_dim, hidden_size)
        self.vae2llm = nn.Linear(patch_dim, hidden_size)
        self.llm2vae = nn.Linear(hidden_size, patch_dim)
        self.action2llm = DomainAwareLinear(config.max_action_dim, hidden_size, config.num_embodiment_domains)
        self.llm2action = DomainAwareLinear(hidden_size, config.max_action_dim, config.num_embodiment_domains)
        self.action_modality_embed = ModalityEmbedding(hidden_size)

        self._init_weights()
        self._set_requires_grad()
        self._set_precision()

    def _init_weights(self):
        """
        Initialize the layers owned by the network like the native `Cosmos3VFMNetwork.init_weights`: truncated
        normal with a standard deviation of `1 / sqrt(fan_in)` and zero bias, the action modules last.
        """
        for linear in (self.time_embedder.linear_1, self.time_embedder.linear_2, self.vae2llm, self.llm2vae):
            std = linear.in_features**-0.5
            nn.init.trunc_normal_(linear.weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(linear.bias)
        self.action2llm._init_weights(self.action2llm.input_size**-0.5)
        self.llm2action._init_weights(self.llm2action.input_size**-0.5)
        std = self.action_modality_embed.weight.numel() ** -0.5
        nn.init.trunc_normal_(self.action_modality_embed.weight, std=std, a=-3 * std, b=3 * std)

    def _set_requires_grad(self):
        """Freeze the VAE, which never trains, and the parts the training switches of the config leave frozen."""
        self.vae.requires_grad_(False)
        # Freeze the unused language modeling head, unless it is tied to the input embedding.
        lm_head = self.mot_backbone.lm_head
        if lm_head.weight is not self.mot_backbone.get_input_embeddings().weight:
            lm_head.requires_grad_(False)
        if self.config.freeze_vision_encoder:
            self.mot_backbone.model.visual.requires_grad_(False)
        if self.config.train_gen_stream_only:
            for name, param in self.mot_backbone.named_parameters():
                if not any(part in self._gen_stream_modules for part in name.split(".")):
                    param.requires_grad_(False)

    def _set_precision(self):
        """
        Store the parameters and floating buffers in `config.dtype`, except the modules in
        `_keep_in_fp32_modules`. Loading a state dict afterwards keeps these dtypes.
        """
        dtype = self.config.dtype if self.config.dtype is not None else torch.get_default_dtype()
        for module_name, module in self.named_modules():
            keep_fp32 = any(part in self._keep_in_fp32_modules for part in module_name.split("."))
            module_dtype = torch.float32 if keep_fp32 else dtype
            for param in module.parameters(recurse=False):
                param.data = param.data.to(module_dtype)
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    setattr(module, buffer_name, buffer.to(module_dtype))

    def gradient_checkpointing_enable(self):
        """
        Non-reentrant transformers gradient checkpointing of the decoder layers and the vision blocks of the
        backbone. As in transformers, the token embeddings then require grad, so with `train_gen_stream_only`
        the backward also runs through the frozen understanding pathway.
        """
        self.mot_backbone.gradient_checkpointing_enable({"use_reentrant": False})

    def gradient_checkpointing_disable(self):
        """Keep the activations of the backbone for the backward pass again."""
        self.mot_backbone.gradient_checkpointing_disable()

    def compile_decoder_layers(self, mode: str = "default"):
        """
        `torch.compile` every decoder layer of the backbone in place: one graph shared by the layers, with
        dynamic prompt and generation lengths. Gradient checkpointing and the FSDP2 hooks run around the
        compiled layer. Turns off `use_duck_shape` process wide, as the native trainer does.
        """
        torch.fx.experimental._config.use_duck_shape = False
        for layer in self.mot_backbone.model.language_model.layers:
            layer.compile(mode=mode, fullgraph=True, dynamic=True)

    def compile_vae_encoder(self, mode: str = "default"):
        """
        `torch.compile` the VAE encoder in place, a few static graphs per canvas compiled on its first batch.
        """
        self.vae.encoder.compile(mode=mode)

    @torch.no_grad()
    def encode_video(self, pixel_values_videos_gen: Tensor, video_grid_thw_gen: Tensor) -> Tensor:
        """
        Encode frames into normalized latents.

        Args:
            pixel_values_videos_gen: Frames in `[-1, 1]`, `[batch, 3, frames, height, width]`, padded to the
                canvas by reflection on the right and at the bottom.
            video_grid_thw_gen: Latent grid of the content of every sample, `[batch, 3]`. The latents of the
                padding are cropped away.

        Returns:
            Float32 latents `[batch, z_dim, latent_frames, latent_height, latent_width]`, normalized with the
            channel statistics of the VAE.
        """
        latents = self.vae.encode(pixel_values_videos_gen.to(self.vae.dtype)).latent_dist.mode()
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / latents.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        latents = (latents - mean) * inv_std
        if not (video_grid_thw_gen == video_grid_thw_gen[0]).all():
            raise ValueError(
                f"All samples of a batch need the same latent grid, got {video_grid_thw_gen.tolist()}"
            )
        _, height, width = video_grid_thw_gen[0].tolist()
        return latents[..., :height, :width].float()

    @torch.no_grad()
    def decode_video(self, video_latents: Tensor) -> Tensor:
        """Decode normalized latents `[batch, z_dim, latent_frames, height, width]` to frames in `[-1, 1]`."""
        # Denormalize in the dtype of the VAE, as the native Wan VAE does.
        latents = video_latents.to(self.vae.dtype)
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / latents.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return self.vae.decode(latents / inv_std + mean).sample

    def get_rope_index(
        self,
        input_ids: Tensor,
        video_grid_thw_gen: Tensor,
        second_per_grid_ts_gen: Tensor,
        num_actions: int = 0,
        action_start_frame_offset: Tensor | None = None,
        attention_mask: Tensor | None = None,
        mm_token_type_ids: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
    ) -> Tensor:
        """
        M-RoPE positions of both pathways, the native unified 3D M-RoPE.

        The understanding pathway follows the transformers Qwen3-VL rule, see `MoTModelMixin.get_rope_index`.
        The generation pathway starts `temporal_modality_margin` after the largest text position of every
        sample. Latent frame `j` sits at `j * tokens_per_second * second_per_grid_ts_gen` on the temporal axis
        (the native FPS modulation, `tokens_per_second = base_fps / temporal_compression`, kept as float) and
        on its patch grid on the spatial axes. Action `i` shares the time axis of the video at `(i +
        action_start_frame_offset) * tokens_per_second * second_per_grid_ts_gen / temporal_compression`, with
        zero spatial positions.

        Args:
            input_ids: Text tokens `[batch, text_len]`, right padded.
            video_grid_thw_gen: Latent grid `[batch, 3]` of the video in the generation pathway.
            second_per_grid_ts_gen: Seconds per latent frame `[batch]`, the temporal compression of the VAE
                divided by the frame rate of the video.
            num_actions: Number of action tokens in the generation pathway, 0 without actions.
            action_start_frame_offset: Video frame of the first action `[batch]`, from the sequence plan. Zero
                when None.
            attention_mask: Text padding mask `[batch, text_len]`.
            mm_token_type_ids: Text (0), image (1) and video (2) tokens of the understanding pathway `[batch,
                text_len]`. All text when None.
            image_grid_thw: Patch grid of every image of the understanding pathway, `[num_images, 3]`.
            video_grid_thw: Patch grid of every video of the understanding pathway, `[num_videos, 3]`.

        Returns:
            Float positions `[3, batch, text_len + gen_len]`.
        """
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        text_position_ids, _ = self.mot_backbone.model.get_rope_index(
            input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask
        )
        text_position_ids = text_position_ids.float()  # [3,B,T]
        batch_size, device = input_ids.shape[0], input_ids.device

        gen_start = text_position_ids.amax(dim=(0, 2)) + 1 + self.config.temporal_modality_margin  # [B]
        temporal_compression = self.vae.config.scale_factor_temporal
        tokens_per_second = self.config.base_fps / temporal_compression
        frame_step = tokens_per_second * second_per_grid_ts_gen.to(device=device, dtype=torch.float32)  # [B]

        num_latent_frames, height, width = video_grid_thw_gen[0].tolist()
        patch_size = self.config.latent_patch_size
        grid_h, grid_w = -(-height // patch_size), -(-width // patch_size)
        frames = torch.arange(num_latent_frames, device=device, dtype=torch.float32)
        video_t = gen_start[:, None] + frames * frame_step[:, None]  # [B,T_lat]
        video_t = video_t[:, :, None].expand(-1, -1, grid_h * grid_w).flatten(1)  # [B,T_lat*h*w]
        video_h = (
            torch.arange(grid_h, device=device).repeat_interleave(grid_w).repeat(num_latent_frames)
        )  # [T_lat*h*w]
        video_w = torch.arange(grid_w, device=device).repeat(grid_h * num_latent_frames)  # [T_lat*h*w]
        video_position_ids = torch.stack(
            [video_t, video_h.float().expand(batch_size, -1), video_w.float().expand(batch_size, -1)]
        )  # [3,B,N_video]

        if action_start_frame_offset is None:
            action_start_frame_offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        action_frames = torch.arange(num_actions, device=device, dtype=torch.float32)
        action_frames = action_frames[None, :] + action_start_frame_offset.to(device)[:, None]  # [B,T_act]
        action_t = (
            gen_start[:, None] + action_frames * (frame_step / temporal_compression)[:, None]
        )  # [B,T_act]
        action_position_ids = torch.stack([action_t, torch.zeros_like(action_t), torch.zeros_like(action_t)])

        return torch.cat([text_position_ids, video_position_ids, action_position_ids], dim=2)

    def merge_gen_stream(self, video_embeds: Tensor, action_embeds: Tensor | None) -> Tensor:
        """Lay out the generation stream as `[video | actions]`."""
        if action_embeds is None:
            return video_embeds
        return torch.cat([video_embeds, action_embeds], dim=1)

    def split_gen_stream(
        self, hidden_states_gen: Tensor, num_video_tokens: int
    ) -> tuple[Tensor, Tensor | None]:
        """Inverse of `merge_gen_stream`: the video and action parts of the generation stream."""
        num_action_tokens = hidden_states_gen.shape[1] - num_video_tokens
        video_hidden_states, action_hidden_states = hidden_states_gen.split(
            [num_video_tokens, num_action_tokens], dim=1
        )
        return video_hidden_states, action_hidden_states if num_action_tokens > 0 else None

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        video_latents: Tensor,
        video_timesteps: Tensor,
        video_condition_mask: Tensor,
        attention_mask: Tensor | None = None,
        actions: Tensor | None = None,
        action_timesteps: Tensor | None = None,
        action_condition_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        **kwargs,
    ) -> Cosmos3VFMNetworkOutput:
        """
        Predict the velocity of the video latents and actions at the current noise level.

        Args:
            input_ids: Text tokens `[batch, text_len]`, right padded.
            position_ids: M-RoPE positions `[3, batch, text_len + gen_len]` of both pathways, from
                `get_rope_index`.
            video_latents: Video latents at the current noise level `[batch, z_dim, frames, height, width]`.
                The condition frames hold their clean latents.
            video_timesteps: Timestep of every latent frame `[batch, frames]`, in `[0, num_train_timesteps]`.
            video_condition_mask: `[batch, frames]`, True on the clean condition frames, which get no timestep
                embedding.
            attention_mask: Text padding mask `[batch, text_len]`.
            actions: Actions at the current noise level `[batch, num_actions, max_action_dim]`, zero padded,
                projected with the embodiment `config.domain_id`. The network runs without action tokens when
                None.
            action_timesteps: Timestep of every action `[batch, num_actions]`.
            action_condition_mask: `[batch, num_actions]`, True on the clean condition actions (the state and
                history actions).
            pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw: Images and videos of the
                understanding pathway for the vision tower, passed to the backbone unchanged.
        """
        # Every sample uses the embodiment of the config.
        domain_id = torch.full(
            (video_latents.shape[0],), self.config.domain_id, dtype=torch.long, device=video_latents.device
        )
        video_embeds = self.embed_video(video_latents, video_timesteps, video_condition_mask)  # [B,N_video,D]
        action_embeds = None
        if actions is not None:
            if action_timesteps is None or action_condition_mask is None:
                raise ValueError("actions need action_timesteps and action_condition_mask")
            action_embeds = self.embed_actions(actions, action_timesteps, action_condition_mask, domain_id)
        inputs_embeds_gen = self.merge_gen_stream(video_embeds, action_embeds)  # [B,G,D]

        outputs = self.mot_backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds_gen=inputs_embeds_gen,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        video_hidden_states, action_hidden_states = self.split_gen_stream(
            outputs.last_hidden_state_gen, num_video_tokens=video_embeds.shape[1]
        )
        video_velocity = self.unpatchify(self.llm2vae(video_hidden_states), video_latents.shape)
        action_velocity = None
        if action_hidden_states is not None:
            action_velocity = self.llm2action(action_hidden_states, domain_id)  # [B,T_act,A]

        return Cosmos3VFMNetworkOutput(
            video_velocity=video_velocity,
            action_velocity=action_velocity,
            last_hidden_state=outputs.last_hidden_state,
            last_hidden_state_gen=outputs.last_hidden_state_gen,
        )

    def embed_timesteps(self, timesteps: Tensor) -> Tensor:
        """Embed timesteps of any shape, `[...]` to `[..., hidden_size]`, under a float32 autocast on CUDA."""
        with torch.autocast(timesteps.device.type, dtype=torch.float32, enabled=timesteps.is_cuda):
            timestep_embeds = self.time_embedder(
                self.time_proj(timesteps.flatten().float() * self.config.timestep_scale)
            )
        return timestep_embeds.view(*timesteps.shape, -1)

    def embed_video(
        self, video_latents: Tensor, video_timesteps: Tensor, video_condition_mask: Tensor
    ) -> Tensor:
        """Patchify and project the video latents, and add the timestep embedding to the noisy frames."""
        patches = self.patchify(video_latents)  # [B,T,h*w,p*p*C]
        video_embeds = self.vae2llm(patches.to(self.vae2llm.weight.dtype))  # [B,T,h*w,D]
        timestep_embeds = self.embed_timesteps(video_timesteps).to(video_embeds.dtype)  # [B,T,D]
        timestep_embeds = timestep_embeds.masked_fill(video_condition_mask.unsqueeze(-1), 0.0)
        return (video_embeds + timestep_embeds.unsqueeze(2)).flatten(1, 2)  # [B,T*h*w,D]

    def embed_actions(
        self, actions: Tensor, action_timesteps: Tensor, action_condition_mask: Tensor, domain_id: Tensor
    ) -> Tensor:
        """
        Project the actions per embodiment and add the modality embedding and the timestep embedding of the
        noisy actions.
        """
        action_embeds = self.action2llm(actions.to(self.action2llm.fc.weight.dtype), domain_id)  # [B,T_act,D]
        action_embeds = self.action_modality_embed(action_embeds)
        timestep_embeds = self.embed_timesteps(action_timesteps).to(action_embeds.dtype)  # [B,T_act,D]
        return action_embeds + timestep_embeds.masked_fill(action_condition_mask.unsqueeze(-1), 0.0)

    def patchify(self, video_latents: Tensor) -> Tensor:
        """`[B,C,T,H,W]` to `[B,T,h*w,p*p*C]`, zero padding H and W at the bottom and right to multiples of p."""
        patch_size = self.config.latent_patch_size
        height, width = video_latents.shape[-2:]
        video_latents = F.pad(video_latents, (0, -width % patch_size, 0, -height % patch_size))
        batch_size, channels, frames, height, width = video_latents.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = video_latents.view(batch_size, channels, frames, grid_h, patch_size, grid_w, patch_size)
        patches = patches.permute(0, 2, 3, 5, 4, 6, 1)  # [B,T,h,w,p,p,C], the native "cthpwq->thwpqc"
        return patches.reshape(batch_size, frames, grid_h * grid_w, patch_size * patch_size * channels)

    def unpatchify(self, patches: Tensor, latent_shape: torch.Size) -> Tensor:
        """Inverse of `patchify`: `[B,T*h*w,p*p*C]` to `[B,C,T,H,W]`, cropping the padding away."""
        patch_size = self.config.latent_patch_size
        batch_size, channels, frames, height, width = latent_shape
        grid_h, grid_w = -(-height // patch_size), -(-width // patch_size)
        video = patches.view(batch_size, frames, grid_h, grid_w, patch_size, patch_size, channels)
        video = video.permute(0, 6, 1, 2, 4, 3, 5)  # [B,C,T,h,p,w,p]
        return video.reshape(batch_size, channels, frames, grid_h * patch_size, grid_w * patch_size)[
            ..., :height, :width
        ]


class SinusoidalTimesteps(nn.Module):
    """Cos and sin features of timesteps `[N]` to `[N,D]` from the native frequency table built on the CPU."""

    def __init__(self, embedding_dim: int, max_period: int = 10000) -> None:
        super().__init__()
        half_dim = embedding_dim // 2
        self.register_buffer(
            "freq",
            torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim),
            persistent=False,
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        freq = timesteps[:, None] * self.freq[None]  # [N,D/2]
        return torch.cat([torch.cos(freq), torch.sin(freq)], dim=-1)  # [N,D]


class ModalityEmbedding(nn.Module):
    """Learned embedding of one modality, added to every token of that modality."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, embeds: Tensor) -> Tensor:
        return embeds + self.weight


class DomainAwareLinear(nn.Module):
    """
    Linear layer with its own weight and bias per embodiment domain, the native Cosmos3 action projection
    (based on X-VLA, https://github.com/2toinf/X-VLA).

    The weights of all domains are stored in embeddings and selected by the domain id of every sample.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 50) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, output_size * input_size)  # [N,O*I]
        self.bias = nn.Embedding(num_domains, output_size)  # [N,O]
        self._init_weights(1.0 / math.sqrt(self.input_size))

    def _init_weights(self, std: float) -> None:
        """Truncated normal weights and zero biases, the native initialization of the action projections."""
        nn.init.trunc_normal_(self.fc.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: Tensor, domain_id: Tensor) -> Tensor:
        """
        Args:
            x: Inputs `[batch, input_size]` or `[batch, tokens, input_size]`.
            domain_id: Domain of every sample `[batch]`.

        Returns:
            Outputs `[batch, output_size]` or `[batch, tokens, output_size]`.
        """
        if x.numel() == 0:
            # Preserve the input and parameter autograd edges for an empty batch.
            weight = self.fc.weight[0].view(self.input_size, self.output_size)
            return x @ weight + self.bias.weight[0]

        # Grouped matrix multiplication on PyTorch 2.10 and newer, bmm before.
        if hasattr(F, "grouped_mm") and x.is_cuda and x.dtype == torch.bfloat16:
            return self._grouped_linear(x, domain_id)

        batch_size = domain_id.shape[0]
        weight = self.fc(domain_id).view(batch_size, self.input_size, self.output_size)
        bias = self.bias(domain_id).view(batch_size, self.output_size)
        if x.dim() == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        return torch.bmm(x, weight) + bias.unsqueeze(1)

    def _grouped_linear(self, x: Tensor, domain_id: Tensor) -> Tensor:
        """Apply each domain projection with one public grouped matrix multiplication."""
        flat_x = x.flatten(0, -2)
        tokens_per_sample = flat_x.shape[0] // domain_id.shape[0]
        flat_domain_id = domain_id.view(-1, 1).expand(-1, tokens_per_sample).reshape(-1)
        # grouped_mm needs the rows of every domain contiguous: gather them by domain, scatter back.
        permutation = torch.argsort(flat_domain_id)
        sorted_x = torch.gather(flat_x, 0, permutation[:, None].expand(-1, self.input_size))

        counts = torch.histc(
            flat_domain_id.to(torch.int32), bins=self.num_domains, min=0, max=self.num_domains - 1
        ).to(torch.int32)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        weight = self.fc.weight.view(self.num_domains, self.input_size, self.output_size)
        sorted_output = F.grouped_mm(sorted_x, weight, offs=offsets)
        output = torch.scatter(
            torch.empty_like(sorted_output),
            0,
            permutation[:, None].expand(-1, self.output_size),
            sorted_output,
        )
        output = output + self.bias.weight.index_select(0, flat_domain_id)
        return output.view(*x.shape[:-1], self.output_size)


class MoTAttention(nn.Module):
    """
    Attention of a MoT decoder layer over the understanding and generation pathways, with separate projections
    per pathway. Understanding tokens attend causally to the understanding tokens of their sample, generation
    tokens attend to every token of their sample. Each backbone subclass sets its QK norms after
    `super().__init__`.
    """

    # Set by the backbone subclasses. Annotations only, a class attribute would hide the submodule.
    q_norm: nn.Module
    k_norm: nn.Module
    q_norm_gen: nn.Module
    k_norm_gen: nn.Module
    # Norm on the understanding keys that the generation queries attend to, None when unused.
    k_norm_und_for_gen: nn.Module | None

    def __init__(self, config: Qwen3VLMoTTextConfig | Nemotron3DenseVLMoTTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Understanding pathway projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Generation pathway projections
        self.q_proj_gen = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_gen = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        hidden_states_gen: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_embeddings_gen: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        attention_mask_gen: Tensor | None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states: Understanding hidden states, `[batch, text_len, hidden_size]`, text right padded.
            hidden_states_gen: Generation hidden states, `[batch, gen_len, hidden_size]`.
            position_embeddings: `(cos, sin)` of the understanding tokens, each `[batch, text_len, head_dim]`.
            position_embeddings_gen: `(cos, sin)` of the generation tokens, each `[batch, gen_len, head_dim]`.
            attention_mask: Mask of the understanding pass in the format of `config._attn_implementation`.
            attention_mask_gen: Mask of the generation pass over the `[text | gen]` keys, same format.

        Returns:
            The understanding and generation outputs, `[batch, text_len, hidden_size]` and
            `[batch, gen_len, hidden_size]`.
        """
        input_shape = hidden_states.shape[:-1]
        input_shape_gen = hidden_states_gen.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        hidden_shape_gen = (*input_shape_gen, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states_gen = self.q_norm_gen(self.q_proj_gen(hidden_states_gen).view(hidden_shape_gen))
        query_states_gen = query_states_gen.transpose(1, 2)
        key_states_gen = self.k_norm_gen(self.k_proj_gen(hidden_states_gen).view(hidden_shape_gen))
        key_states_gen = key_states_gen.transpose(1, 2)
        value_states_gen = self.v_proj_gen(hidden_states_gen).view(hidden_shape_gen).transpose(1, 2)

        cos, sin = position_embeddings
        key_states_und_for_gen = None
        if self.k_norm_und_for_gen is not None:
            # Understanding keys for the generation queries: their own norm, then RoPE on the keys only.
            key_states_und = self.k_norm_und_for_gen(key_states)
            key_states_und_for_gen = (
                key_states_und * cos[:, None] + rotate_half(key_states_und) * sin[:, None]
            )

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        cos_gen, sin_gen = position_embeddings_gen
        query_states_gen, key_states_gen = apply_rotary_pos_emb(
            query_states_gen, key_states_gen, cos_gen, sin_gen
        )

        attn_output, attn_output_gen = self.dispatch_attention(
            query_states,
            key_states,
            value_states,
            query_states_gen,
            key_states_gen,
            value_states_gen,
            attention_mask,
            attention_mask_gen,
            key_und_for_gen=key_states_und_for_gen,
            **kwargs,
        )

        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
        attn_output_gen = self.o_proj_gen(attn_output_gen.reshape(*input_shape_gen, -1).contiguous())
        return attn_output, attn_output_gen

    def dispatch_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_gen: Tensor,
        key_gen: Tensor,
        value_gen: Tensor,
        attention_mask: Tensor | None,
        attention_mask_gen: Tensor | None,
        key_und_for_gen: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Run the joint attention selected by `config.joint_attn_implementation`."""
        if self.config.joint_attn_implementation != "two_way":
            raise NotImplementedError(
                f"joint_attn_implementation={self.config.joint_attn_implementation!r} is not supported, "
                "only 'two_way'."
            )
        return self.two_way_attention(
            query,
            key,
            value,
            query_gen,
            key_gen,
            value_gen,
            attention_mask,
            attention_mask_gen,
            key_und_for_gen=key_und_for_gen,
            **kwargs,
        )

    def two_way_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_gen: Tensor,
        key_gen: Tensor,
        value_gen: Tensor,
        attention_mask: Tensor | None,
        attention_mask_gen: Tensor | None,
        key_und_for_gen: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Two attention calls through the interface of `config._attn_implementation`: the understanding queries
        attend causally to the understanding keys, the generation queries attend to the `[text | gen]` keys of
        their sample.

        Query, key and value tensors are `[batch, num_heads, seq_len, head_dim]`, the outputs `[batch,
        seq_len, num_heads, head_dim]`. When `key_und_for_gen` is given, the generation queries attend to it
        instead of `key` for the understanding keys.
        """
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        dropout = 0.0 if not self.training else self.attention_dropout

        attn_output, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=self.scaling,
            is_causal=True,
            **kwargs,
        )

        key_all = torch.cat([key if key_und_for_gen is None else key_und_for_gen, key_gen], dim=2)
        value_all = torch.cat([value, value_gen], dim=2)
        attn_output_gen, _ = attention_interface(
            self,
            query_gen,
            key_all,
            value_all,
            attention_mask_gen,
            dropout=dropout,
            scaling=self.scaling,
            is_causal=False,
            **kwargs,
        )
        return attn_output, attn_output_gen


class MoTDecoderLayer(GradientCheckpointingLayer):
    """
    MoT (Mixture of Transformers) decoder layer: the joint attention, with a norm and an MLP per pathway. A
    backbone subclass names the attention, MLP and norm classes (`attention_cls`, `mlp_cls`, `norm_cls`).
    """

    attention_cls: type[MoTAttention]
    mlp_cls: type[nn.Module]
    norm_cls: type[nn.Module]

    def __init__(self, config: Qwen3VLMoTTextConfig | Nemotron3DenseVLMoTTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = self.attention_cls(config, layer_idx)

        self.mlp = self.mlp_cls(config)
        self.mlp_gen = self.mlp_cls(config)

        self.input_layernorm = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_gen = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_gen = self.norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        hidden_states_gen: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_embeddings_gen: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        attention_mask_gen: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Run both pathways through their own norms and MLPs around the joint attention. Pass both hidden states
        positionally, reentrant checkpointing only propagates gradients of positional inputs. See
        `MoTAttention.forward` for the arguments.
        """
        residual, residual_gen = hidden_states, hidden_states_gen
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_gen = self.input_layernorm_gen(hidden_states_gen)
        hidden_states, hidden_states_gen = self.self_attn(
            hidden_states,
            hidden_states_gen,
            position_embeddings,
            position_embeddings_gen,
            attention_mask,
            attention_mask_gen,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        hidden_states_gen = residual_gen + hidden_states_gen

        residual, residual_gen = hidden_states, hidden_states_gen
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states_gen = self.post_attention_layernorm_gen(hidden_states_gen)
        hidden_states = residual + self.mlp(hidden_states)
        hidden_states_gen = residual_gen + self.mlp_gen(hidden_states_gen)
        return hidden_states, hidden_states_gen


@dataclass
class MoTModelOutput(ModelOutput):
    """
    Output of the MoT text models.

    Args:
        last_hidden_state: Final hidden states of the understanding pathway, `[batch, text_len, hidden_size]`.
        last_hidden_state_gen: Final hidden states of the generation pathway, `[batch, gen_len, hidden_size]`.
    """

    last_hidden_state: Tensor | None = None
    last_hidden_state_gen: Tensor | None = None


class MoTTextModelMixin:
    """
    Forward shared by the MoT text models of both backbones, which build the modules annotated below in
    `__init__`.
    """

    # Set in `__init__` of each backbone. Annotations only, a class attribute would hide the module.
    config: Qwen3VLMoTTextConfig | Nemotron3DenseVLMoTTextConfig
    embed_tokens: nn.Embedding
    layers: nn.ModuleList
    norm: nn.Module
    norm_gen: nn.Module
    rotary_emb: nn.Module

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        inputs_embeds_gen: Tensor | None = None,
        attention_mask_gen: Tensor | None = None,
        # args for deepstack
        visual_pos_masks: Tensor | None = None,
        deepstack_visual_embeds: list[Tensor] | None = None,
        **kwargs,
    ) -> MoTModelOutput:
        """
        Args:
            input_ids: Text token ids, `[batch, text_len]`, right padded.
            attention_mask: Text padding mask, `[batch, text_len]`, 1 for real tokens.
            position_ids: Float M-RoPE positions of the `[text | gen]` tokens, `[3, batch, text_len +
                gen_len]`, from `Cosmos3VFMNetwork.get_rope_index`.
            inputs_embeds: Text embeddings, `[batch, text_len, hidden_size]`, instead of `input_ids`.
            inputs_embeds_gen: Generation embeddings (projected vision and action latents), `[batch, gen_len,
                hidden_size]`.
            attention_mask_gen: Generation padding mask, `[batch, gen_len]`. All real when None.
            visual_pos_masks: Positions of the image and video tokens in the understanding pathway, `[batch,
                text_len]`.
            deepstack_visual_embeds: DeepStack features of the vision tower, one `[num_visual_tokens,
                hidden_size]` tensor per layer. The i-th tensor is added to the text hidden states at
                `visual_pos_masks` after the i-th layer. Only the Qwen3-VL vision tower has DeepStack layers.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds_gen is None:
            raise ValueError("inputs_embeds_gen is required, the MoT text model always runs both pathways")
        if position_ids is None:
            raise ValueError(
                "position_ids [3, batch, text_len + gen_len] are required, see Cosmos3VFMNetwork.get_rope_index"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, text_len = inputs_embeds.shape[:2]
        gen_len = inputs_embeds_gen.shape[1]
        if position_ids.shape != (3, batch_size, text_len + gen_len):
            raise ValueError(
                f"position_ids must be [3, {batch_size}, {text_len + gen_len}], got {list(position_ids.shape)}"
            )
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, text_len, dtype=torch.long, device=inputs_embeds.device)
        if attention_mask_gen is None:
            attention_mask_gen = attention_mask.new_ones(batch_size, gen_len)

        # Masks in the format of `config._attn_implementation`, built without position_ids (no packing).
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        bidirectional_mask_gen = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds_gen,
            attention_mask=torch.cat([attention_mask, attention_mask_gen], dim=1),
            # Only read for the key length, dtype and device of the `[text | gen]` keys.
            encoder_hidden_states=inputs_embeds_gen.new_zeros(batch_size, text_len + gen_len, 1),
        )

        # One rotary table for `[text | gen]`, split per pathway.
        cos, sin = self.rotary_emb(inputs_embeds_gen, position_ids)
        position_embeddings = (cos[:, :text_len], sin[:, :text_len])
        position_embeddings_gen = (cos[:, text_len:], sin[:, text_len:])

        hidden_states, hidden_states_gen = inputs_embeds, inputs_embeds_gen
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states, hidden_states_gen = decoder_layer(
                hidden_states,
                hidden_states_gen,
                position_embeddings,
                position_embeddings_gen,
                attention_mask=causal_mask,
                attention_mask_gen=bidirectional_mask_gen,
                **kwargs,
            )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        return MoTModelOutput(
            last_hidden_state=self.norm(hidden_states),
            last_hidden_state_gen=self.norm_gen(hidden_states_gen),
        )

    def _deepstack_process(
        self, hidden_states: Tensor, visual_pos_masks: Tensor, visual_embeds: Tensor
    ) -> Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class MoTModelMixin:
    """
    Image handling and forward shared by the MoT models of both backbones. Each backbone builds `visual` and
    `language_model` in its `__init__` and implements `get_image_features`. The Qwen3-VL vision tower also
    returns DeepStack features, which the text model adds to the hidden states of its first layers, the
    Nemotron one returns none.
    """

    # Set in `__init__` of each backbone. Annotations only, a class attribute would hide the module.
    config: Qwen3VLMoTConfig | Nemotron3DenseVLMoTConfig
    visual: nn.Module
    language_model: nn.Module

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self, pixel_values: Tensor, image_grid_thw: Tensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        """
        Encode the images. `pooler_output` of the result holds one `[num_tokens, text_hidden_size]` tensor per
        image, in the hidden size of the language model. A vision tower with DeepStack layers also returns
        `deepstack_features`, one `[num_tokens_all_images, text_hidden_size]` tensor per DeepStack layer.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_image_features")

    def get_video_features(
        self, pixel_values_videos: Tensor, video_grid_thw: Tensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        """Encode the videos. Video frames go through the same vision path as images."""
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_placeholder_mask(
        self,
        input_ids: Tensor | None,
        inputs_embeds: Tensor,
        image_features: Tensor | None = None,
        video_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder
        token count is equal to the length of multimodal features. If the lengths are different, an error is
        raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                f"features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = (
            special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, "
                f"features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """
        Compute 3D positional indices for vision tokens derived from a single image or video input.

        The positions are generated from the input grid defined by temporal (T), height (H), and
        width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
        merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

        Args:
            start_position (`int`):
                Offset added to all computed positional indices.
            grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
                The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
            temp_merge_size (`int`, *optional*):
                Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
                by this value. Defaults to 1.
            spatial_merge_size (`int`, *optional*):
                Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
                by this value. Defaults to 1.
            time_interval (`int`, *optional*):
                Spacing factor applied between consecutive temporal position indices.Defaults to 1.
            device (`str` or `torch.device`, *optional*):
                Device on which the resulting tensor is allocated. If `None`, uses the current default device.

        Returns:
            torch.LongTensor of shape (3, sequence_length):
                Positional indices for temporal, height, and width dimensions,
                flattened into sequence form and offset by `start_position`.
        """
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(
            start_position, start_position + llm_grid_h, device=device
        ).repeat_interleave(llm_grid_w * llm_grid_t)
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        M-RoPE positions of the understanding pathway, the transformers `Qwen3VLModel.get_rope_index` (also
        for the Nemotron vision tower, with its own `spatial_merge_size`). `Cosmos3VFMNetwork.get_rope_index`
        continues them over the generation pathway.

        Difference from Qwen2VL/Qwen2.5VL's get_rope_index:
        - Since Qwen3.5 use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split too.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        # Separate video grid thw into multiple grids because timestamps are used to separate videos.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        vision_config = self.config.vision_config
        if vision_config is None or isinstance(vision_config, dict):
            raise ValueError("get_rope_index needs the built vision_config of the backbone config")
        spatial_merge_size = vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group_items = list(group)
                start_index = group_items[0][0]
                end_index = group_items[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                # image == 1, video == 2
                else:
                    grid_iter = grid_iters[modality_type]
                    if grid_iter is None:
                        raise ValueError(f"mm_token_type_ids has tokens of type {modality_type} but no grid")
                    grid_thw = next(grid_iter)
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(
                    position_ids.device
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        inputs_embeds_gen: Tensor | None = None,
        attention_mask_gen: Tensor | None = None,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        **kwargs,
    ) -> MoTModelOutput:
        """
        Write the image and video features into their placeholder tokens of the understanding pathway, collect
        the DeepStack features of the vision tower, then run the MoT text model. See
        `MoTTextModelMixin.forward` for the text and generation arguments, `position_ids` covers both
        pathways.

        Args:
            pixel_values: Patches of every image, packed as the backbone vision tower expects.
            pixel_values_videos: Patches of every video, packed the same way.
            image_grid_thw: Temporal, height and width patch grid of every image, `[num_images, 3]`.
            video_grid_thw: Temporal, height and width patch grid of every video, `[num_videos, 3]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_outputs.pooler_output
            # A vision tower without DeepStack features has zero DeepStack layers.
            deepstack_image_embeds = getattr(image_outputs, "deepstack_features", [])
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw is required with pixel_values_videos")
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = getattr(video_outputs, "deepstack_features", [])
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=True):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(
                    img_embed.device
                )
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            inputs_embeds_gen=inputs_embeds_gen,
            attention_mask_gen=attention_mask_gen,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )


@dataclass
class MoTCausalLMOutput(ModelOutput):
    """
    Output of the MoT models with a language modeling head.

    Args:
        loss: Language modeling loss (next token prediction) of the understanding pathway, returned when
            `labels` is given.
        logits: Language modeling head scores of the understanding pathway, `[batch, num_kept_tokens,
            vocab_size]`.
        last_hidden_state_gen: Final hidden states of the generation pathway, `[batch, gen_len, hidden_size]`.
    """

    loss: Tensor | None = None
    logits: Tensor | None = None
    last_hidden_state_gen: Tensor | None = None


class MoTForConditionalGenerationMixin:
    """
    Language modeling head and forward shared by the MoT backbones. Each backbone builds `model` and `lm_head`
    in its `__init__`. The head reads the understanding pathway, the generation pathway is returned as is.
    """

    # Set in `__init__` of each backbone. Annotations only, a class attribute would hide the module.
    config: Qwen3VLMoTConfig | Nemotron3DenseVLMoTConfig
    model: nn.Module
    lm_head: nn.Linear
    # Provided by `PreTrainedModel`, the loss of `config.loss_type` (causal LM loss by default).
    loss_function: Callable[..., Tensor]

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(
        self, pixel_values_videos: Tensor, video_grid_thw: Tensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        """Encode the videos with the vision tower of the backbone, see `MoTModelMixin.get_video_features`."""
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs
        )

    def get_image_features(
        self, pixel_values: Tensor, image_grid_thw: Tensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        """Encode the images with the vision tower of the backbone, see `MoTModelMixin.get_image_features`."""
        return self.model.get_image_features(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        inputs_embeds_gen: Tensor | None = None,
        attention_mask_gen: Tensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs,
    ) -> MoTCausalLMOutput:
        """
        Run the MoT model and the language modeling head on the understanding pathway. See
        `MoTModelMixin.forward` for the other arguments.

        Args:
            labels: Labels of the text tokens for the language modeling loss, `[batch, text_len]`. Indices are
                in `[0, vocab_size)` or -100 for ignored tokens (padding, image and video placeholders). The
                loss function shifts them by one token.
            logits_to_keep: Compute the logits of the last `logits_to_keep` text tokens only (all when 0), or
                of the text positions in a 1D tensor of indices.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            inputs_embeds_gen=inputs_embeds_gen,
            attention_mask_gen=attention_mask_gen,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return MoTCausalLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state_gen=outputs.last_hidden_state_gen,
        )


class Qwen3VLMoTAttention(MoTAttention):
    """MoT attention for the Qwen3-VL backbone (Cosmos3 Nano and Super)."""

    def __init__(self, config: Qwen3VLMoTTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        eps = config.rms_norm_eps

        # Qwen3-VL keeps its pretrained QK norm in the understanding pathway.
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)

        self.q_norm_gen = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm_gen = Qwen3VLTextRMSNorm(self.head_dim, eps=eps)

        self.k_norm_und_for_gen = None


class Qwen3VLMoTDecoderLayer(MoTDecoderLayer):
    """MoT decoder layer for the Qwen3-VL backbone."""

    attention_cls = Qwen3VLMoTAttention
    mlp_cls = Qwen3VLTextMLP
    norm_cls = Qwen3VLTextRMSNorm


@auto_docstring
class Qwen3VLMoTPreTrainedModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLMoTConfig
    _no_split_modules = ["Qwen3VLMoTDecoderLayer", "Qwen3VLVisionBlock"]


class Qwen3VLMoTTextModel(MoTTextModelMixin, Qwen3VLMoTPreTrainedModel):
    """Qwen3-VL MoT text model with dense MLPs."""

    config: Qwen3VLMoTTextConfig
    _no_split_modules = ["Qwen3VLMoTDecoderLayer"]

    def __init__(self, config: Qwen3VLMoTTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLMoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm of the understanding pathway.
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Final norm of the generation pathway.
        self.norm_gen = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embedding of both pathways.
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

        self.post_init()


class Qwen3VLMoTModel(MoTModelMixin, Qwen3VLMoTPreTrainedModel):
    """Qwen3-VL MoT backbone: vision tower and MoT text model."""

    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)

        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLMoTTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_image_features(
        self, pixel_values: Tensor, image_grid_thw: Tensor, **kwargs
    ) -> BaseModelOutputWithDeepstackFeatures:
        """
        Encode the images with the Qwen3-VL vision tower. `pooler_output` holds the merged features, one tensor
        per image, and `deepstack_features` the merged features of the DeepStack layers for all images.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(vision_output.pooler_output, split_sizes)
        return vision_output


class Qwen3VLMoTForConditionalGeneration(
    MoTForConditionalGenerationMixin, Qwen3VLMoTPreTrainedModel, GenerationMixin
):
    """Qwen3-VL MoT backbone with a language modeling head on the understanding pathway."""

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLMoTModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


class Nemotron3DenseVLMoTAttention(MoTAttention):
    """MoT attention for the Nemotron 3 Dense VL backbone (Cosmos3 Edge)."""

    def __init__(self, config: Nemotron3DenseVLMoTTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        eps = config.rms_norm_eps

        # Nemotron has no QK norm in the understanding pathway.
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

        self.q_norm_gen = Nemotron3DenseVLTextRMSNorm(self.head_dim, eps=eps)
        self.k_norm_gen = Nemotron3DenseVLTextRMSNorm(self.head_dim, eps=eps)

        self.k_norm_und_for_gen = Nemotron3DenseVLTextRMSNorm(self.head_dim, eps=eps)


class Nemotron3DenseVLTextMLP(nn.Module):
    """Non-gated MLP of the Nemotron 3 Dense VL backbone (ReLU squared by default)."""

    def __init__(self, config: Nemotron3DenseVLMoTTextConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Nemotron3DenseVLTextRMSNorm(nn.Module):
    """
    RMSNorm of the Nemotron 3 Dense VL backbone, which applies the weight in float32 before casting back.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Nemotron3DenseVLMoTDecoderLayer(MoTDecoderLayer):
    """MoT decoder layer for the Nemotron 3 Dense VL backbone."""

    attention_cls = Nemotron3DenseVLMoTAttention
    mlp_cls = Nemotron3DenseVLTextMLP
    norm_cls = Nemotron3DenseVLTextRMSNorm


class Nemotron3DenseVLVisionEmbeddings(Siglip2VisionEmbeddings):
    """SigLIP2 patch and learned position embeddings for packed Cosmos3 Edge vision inputs."""

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: Tensor, grid_thw: Tensor, spatial_merge_size: int
    ) -> Tensor:
        """
        Resize the learned square position grid for every packed image or video frame, packed in the
        block-major order of the input patches.

        Args:
            positional_embeddings: Learned position grid of shape ``(height, width, hidden_size)``.
            grid_thw: Temporal, height and width patch grid of every packed image or video.
            spatial_merge_size: Side of the patch blocks that the projector merges.
        """
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        source_dtype = positional_embeddings.dtype
        # Antialiased interpolation does not support bfloat16 or float16 on CPU.
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.float()

        position_chunks = []
        for temporal, height, width in grid_thw.tolist():
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized_embeddings = resized_embeddings.squeeze(0).permute(1, 2, 0).to(source_dtype)
            resized_embeddings = resized_embeddings.reshape(
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            # Row-major grid to block-major order, matching the packed patches.
            resized_embeddings = resized_embeddings.transpose(1, 2).reshape(height * width, -1)
            position_chunks.append(resized_embeddings.repeat(temporal, 1))

        return torch.cat(position_chunks, dim=0)

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        # Patch embeddings of the already patchified pixel values.
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        position_embeddings = self.resize_positional_embeddings(
            positional_embeddings, grid_thw, self.config.spatial_merge_size
        )
        torch_compilable_check(
            patch_embeds.shape[0] == position_embeddings.shape[0],
            "The packed visual patch count does not match `grid_thw`.",
        )
        return patch_embeds + position_embeddings


class Nemotron3DenseVLVisionAttention(Siglip2Attention):
    """SigLIP2 attention over packed Cosmos3 Edge vision inputs, one sequence per image or video frame."""

    def __init__(self, config: Nemotron3DenseVLVisionConfig):
        super().__init__(config)
        # Read by the attention interfaces, which expect the grouped query attention field.
        self.num_key_value_groups = 1

    def forward(
        self, hidden_states: Tensor, cu_seqlens: Tensor, max_seqlen: int | None = None, **kwargs
    ) -> Tensor:
        """Attend within every image or video frame of the packed sequence, delimited by `cu_seqlens`."""
        sequence_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).reshape(sequence_length, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(sequence_length, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(sequence_length, -1, self.head_dim)

        # [num_patches, num_heads, head_dim] -> [1, num_heads, num_patches, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            if max_seqlen is None:
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Without a varlen kernel, every frame is attended separately.
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_output = torch.cat(
                [
                    attention_interface(
                        self,
                        query,
                        key,
                        value,
                        attention_mask=None,
                        scaling=self.scale,
                        dropout=0.0 if not self.training else self.dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for query, key, value in zip(*splits, strict=True)
                ],
                dim=1,
            )

        return self.out_proj(attn_output.reshape(sequence_length, -1).contiguous())


class Nemotron3DenseVLVisionEncoderLayer(GradientCheckpointingLayer):
    """SigLIP2 encoder layer whose attention runs over packed Cosmos3 Edge vision inputs."""

    def __init__(self, config: Nemotron3DenseVLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Nemotron3DenseVLVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(self, hidden_states: Tensor, cu_seqlens: Tensor, **kwargs) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Nemotron3DenseVLVisionEncoder(nn.Module):
    """Stack of SigLIP2 encoder layers over packed Cosmos3 Edge vision inputs."""

    def __init__(self, config: Nemotron3DenseVLVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Nemotron3DenseVLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: Tensor, grid_thw: Tensor, **kwargs) -> Tensor:
        # Every image or video frame is its own attention sequence.
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = None
        if is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **kwargs
            )

        return hidden_states


@auto_docstring
class Nemotron3DenseVLMoTPreTrainedModel(PreTrainedModel):
    config: Nemotron3DenseVLMoTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Nemotron3DenseVLMoTDecoderLayer", "Nemotron3DenseVLVisionEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True


class Nemotron3DenseVLMoTTextModel(MoTTextModelMixin, Nemotron3DenseVLMoTPreTrainedModel):
    """Nemotron 3 Dense VL MoT text model (Cosmos3 Edge)."""

    config: Nemotron3DenseVLMoTTextConfig
    _no_split_modules = ["Nemotron3DenseVLMoTDecoderLayer"]

    def __init__(self, config: Nemotron3DenseVLMoTTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Nemotron3DenseVLMoTDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final norm of the understanding pathway.
        self.norm = Nemotron3DenseVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Final norm of the generation pathway.
        self.norm_gen = Nemotron3DenseVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # The interleaved M-RoPE of Qwen3-VL, with the rope_theta of Edge.
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

        self.post_init()


class Nemotron3DenseVLVisionModel(Nemotron3DenseVLMoTPreTrainedModel):
    """
    SigLIP2 vision tower of the Nemotron 3 Dense VL backbone (Cosmos3 Edge), without the attention pooling
    head. It takes the patches of all images packed into one sequence with their `grid_thw`, in the
    block-major order described in `processor_cosmos3.py`.
    """

    config: Nemotron3DenseVLVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Nemotron3DenseVLVisionEncoderLayer"]

    def __init__(self, config: Nemotron3DenseVLVisionConfig):
        super().__init__(config)
        self.embeddings = Nemotron3DenseVLVisionEmbeddings(config)
        self.encoder = Nemotron3DenseVLVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def forward(self, pixel_values: Tensor, grid_thw: Tensor, **kwargs) -> BaseModelOutputWithPooling:
        """
        Args:
            pixel_values: Patches of every image or video frame packed into one sequence in block-major
                order, of shape ``(num_patches, num_channels * patch_size * patch_size)``.
            grid_thw: Temporal, height and width patch grid of every packed image or video, of shape
                ``(num_images_or_videos, 3)``.
        """
        hidden_states = self.embeddings(pixel_values, grid_thw)
        hidden_states = self.encoder(hidden_states, grid_thw, **kwargs)
        last_hidden_state = self.post_layernorm(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, pooler_output=last_hidden_state
        )


class Nemotron3DenseVLPatchMerger(nn.Module):
    """
    Projector that merges spatially adjacent SigLIP2 patches into language model tokens, its hidden width
    `merger_intermediate_size`.
    """

    def __init__(self, config: Nemotron3DenseVLProjectorConfig) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.input_hidden_size = config.input_hidden_size
        self.hidden_size = config.input_hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = config.use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if self.use_postshuffle_norm else self.input_hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, config.merger_intermediate_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.merger_intermediate_size, config.out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        # Every run of ``spatial_merge_size**2`` consecutive tokens is one block of adjacent patches.
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Nemotron3DenseVLMoTModel(MoTModelMixin, Nemotron3DenseVLMoTPreTrainedModel):
    """Nemotron 3 Dense VL MoT backbone (Cosmos3 Edge): vision tower, projector and MoT text model."""

    base_model_prefix = "model"
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)

        self.visual = Nemotron3DenseVLVisionModel._from_config(config.vision_config)
        self.projector = Nemotron3DenseVLPatchMerger(config.projector_config)
        self.language_model = Nemotron3DenseVLMoTTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_image_features(
        self, pixel_values: Tensor, image_grid_thw: Tensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        """
        Encode the images with the SigLIP2 vision tower and the projector. `pooler_output` holds the projected
        features, one tensor per image.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = self.projector(vision_output.last_hidden_state)
        split_sizes = (image_grid_thw.prod(-1) // self.projector.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_output


class Nemotron3DenseVLMoTForConditionalGeneration(
    MoTForConditionalGenerationMixin, Nemotron3DenseVLMoTPreTrainedModel, GenerationMixin
):
    """
    Nemotron 3 Dense VL MoT backbone (Cosmos3 Edge) with a language modeling head on the understanding
    pathway.
    """

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Nemotron3DenseVLMoTModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


# Backbone class of every backbone config, built by `Cosmos3VFMNetwork`.
MOT_BACKBONE_MAPPING: dict[type, type[PreTrainedModel]] = {
    Qwen3VLMoTConfig: Qwen3VLMoTForConditionalGeneration,
    Nemotron3DenseVLMoTConfig: Nemotron3DenseVLMoTForConditionalGeneration,
}
