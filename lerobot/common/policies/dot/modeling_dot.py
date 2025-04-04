#!/usr/bin/env python

# Copyright 2025 Ilia Larchenko and The HuggingFace Inc. team. All rights reserved.
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

"""The implementation of the Decoder-Only Transformer (DOT) policy.

More details here: https://github.com/IliaLarchenko/dot_policy
"""

import math

import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import InterpolationMode

from lerobot.common.policies.dot.configuration_dot import DOTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class DOT(nn.Module):
    """The underlying neural network for DOT
    Note: Unlike ACT, DOT has no encoder, no VAE, and no cross-attention. All inputs are directly projected
    to the model dimension and passed as memory to a Transformer decoder.

    - Inputs (images, state, env_state) are linearly projected and concatenated.
    - A trainable prefix token and positional embeddings are added.
    - The Transformer decoder predicts a sequence of future actions autoregressively.

                              DOT Transformer
                          Used for autoregressive action prediction
                          (no encoder, no VAE)

          ┌──────────────────────────────────────────────────────┐
          │              image emb.  state emb.  env_state emb.  │
          │                  │          │             │          │
          │          ┌───────┘          │             │          │
          │          │         ┌────────┘             │          │
          │          ▼         ▼                      ▼          │
          │      ┌──────────────────────────────────────────┐    │
          │      │    Concatenate + Add Positional Emb.     │    │
          │      └──────────────────────────────────────────┘    │
          │                            │                         │
          │                            ▼                         │
          │         ┌───────────────────────────────────┐        │
          │         │     Transformer Decoder (L layers)│        │
          │         └───────────────────────────────────┘        │
          │                            │                         │
          │                            ▼                         │
          │                  Linear projection to action space   │
          │                            │                         │
          │                            ▼                         │
          │                         Outputs                      │
          └──────────────────────────────────────────────────────┘
    """

    def __init__(self, config: DOTConfig):
        super().__init__()
        self.config = config

        self.projections = nn.ModuleDict()
        self.n_features = 0

        self.image_names = sorted(config.image_features.keys())

        # Set up a shared visual backbone (e.g., ResNet18) for all cameras.
        # The final layer is replaced with a linear projection to match model_dim.
        if len(self.image_names) > 0:
            backbone = getattr(torchvision.models, self.config.vision_backbone)(
                weights=self.config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            backbone.fc = nn.Linear(backbone.fc.in_features, self.config.dim_model)

            self.projections["images"] = add_lora_to_backbone(backbone, rank=config.lora_rank)
            self.n_features += len(self.image_names) * self.config.n_obs_steps

        if self.config.robot_state_feature:
            self.projections["state"] = nn.Linear(
                self.config.robot_state_feature.shape[0], self.config.dim_model
            )
            self.n_features += self.config.n_obs_steps

        if self.config.env_state_feature:
            self.projections["environment_state"] = nn.Linear(
                self.config.env_state_feature.shape[0], self.config.dim_model
            )
            self.n_features += self.config.n_obs_steps

        self.projections_names = sorted(self.projections.keys())
        obs_mapping = {
            "images": "observation.images",
            "state": "observation.state",
            "environment_state": "observation.environment_state",
        }
        self.obs_mapping = {k: v for k, v in obs_mapping.items() if k in self.projections_names}

        # Optional trainable prefix token added to the input sequence (can be used for task conditioning or extra context)
        self.prefix_input = nn.Parameter(torch.randn(1, 1, config.dim_model))

        # Setup transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.config.dim_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=self.config.pre_norm,
        )

        decoder_norm = nn.LayerNorm(self.config.dim_model)
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=self.config.n_decoder_layers, norm=decoder_norm
        )

        # Sinusoidal positional encodings for the decoder input tokens (fixed, not trainable)
        decoder_pos = create_sinusoidal_pos_embedding(
            config.train_horizon + config.lookback_obs_steps, config.dim_model
        )
        decoder_pos = torch.cat(
            [
                decoder_pos[:1],
                decoder_pos[-config.train_horizon - config.n_obs_steps + 2 :],
            ],
            dim=0,
        )
        self.register_buffer("decoder_pos", decoder_pos)

        # Extend positional encodings for inference (when inference_horizon > train_horizon)
        decoder_pos_inf = self.decoder_pos[
            : self.decoder_pos.shape[0] + self.config.inference_horizon - self.config.train_horizon
        ]
        self.register_buffer("decoder_pos_inf", decoder_pos_inf)
        # Causal mask for decoder: prevent attending to future positions
        mask = torch.zeros(len(decoder_pos), len(decoder_pos), dtype=torch.bool)
        mask[
            : len(decoder_pos) + config.inference_horizon - config.train_horizon,
            len(decoder_pos) + config.inference_horizon - config.train_horizon :,
        ] = True
        self.register_buffer("mask", mask)

        # Learnable positional embeddings for input tokens (state/image/env projections)
        self.inputs_pos_emb = nn.Parameter(torch.empty(1, self.n_features, self.config.dim_model))
        nn.init.uniform_(
            self.inputs_pos_emb,
            -((1 / self.config.dim_model) ** 0.5),
            (1 / self.config.dim_model) ** 0.5,
        )

        # The output actions are generated by a linear layer
        self.action_head = nn.Linear(self.config.dim_model, self.config.action_feature.shape[0])

    def _process_inputs(self, batch):
        # Project all inputs to the model dimension and concatenate them
        inputs_projections_list = []

        for state in self.projections_names:
            batch_state = self.obs_mapping[state]
            if batch_state in batch:
                batch_size, n_obs, *obs_shape = batch[batch_state].shape
                enc = self.projections[state](batch[batch_state].view(batch_size * n_obs, *obs_shape)).view(
                    batch_size, n_obs, -1
                )
                inputs_projections_list.append(enc)

        return torch.cat(inputs_projections_list, dim=1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """
        A forward pass through the Decision Transformer (DOT).

        The model uses a transformer decoder to predict a sequence of future actions from projected
        and positionally-embedded image, state, and environment features.

        Args:
            batch (dict): A dictionary containing the following keys (if available):
                - "observation.images": (B, T, C, H, W) tensor of camera frames.
                - "observation.state": (B, T, D) tensor of proprioceptive robot states.
                - "observation.environment_state": (B, T, D) tensor of environment states.

        Returns:
            Tensor: A tensor of shape (B, horizon, action_dim) containing predicted future actions.
        """
        # Project image/state/env_state inputs to the model dimension and concatenate along the time axis.
        inputs_projections = self._process_inputs(batch)  # (B, T, D)
        batch_size = inputs_projections.shape[0]

        # Add learnable positional embeddings to each projected input token.
        inputs_projections += self.inputs_pos_emb.expand(batch_size, -1, -1)

        # Prepend a trainable prefix token to the input sequence
        inputs_projections = torch.cat(
            [self.prefix_input.expand(batch_size, -1, -1), inputs_projections], dim=1
        )  # (B, T+1, D)

        # Use different positional encodings and masks for training vs. inference.
        if self.training:
            decoder_out = self.decoder(
                self.decoder_pos.expand(batch_size, -1, -1), inputs_projections, self.mask
            )
        else:
            decoder_out = self.decoder(self.decoder_pos_inf.expand(batch_size, -1, -1), inputs_projections)
        return self.action_head(decoder_out)


class DOTPolicy(PreTrainedPolicy):
    """
    Decision Transformer (DOT) Policy. (github: https://github.com/IliaLarchenko/dot_policy)

    A minimal transformer decoder-based policy for autoregressive action prediction in robot control.
    This is a simplified alternative to ACT: no encoder, no VAE, and no cross-attention, making it efficient
    for deployment in low-dimensional environments with visual and proprioceptive inputs.
    """

    name = "dot"
    config_class = DOTConfig

    def __init__(
        self,
        config: DOTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config (DOTConfig): Configuration for the DOT model and policy behavior.
            dataset_stats (optional): Dataset statistics used for normalizing inputs/outputs.
                If not provided, stats should be set later via `load_state_dict()` before inference.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.image_names = sorted(config.image_features.keys())

        if config.override_dataset_stats:
            if dataset_stats is None:
                dataset_stats = {}
            for k, v in config.new_dataset_stats.items():
                if k not in dataset_stats:
                    dataset_stats[k] = {}
                for k1, v1 in v.items():
                    dataset_stats[k][k1] = torch.tensor(v1)

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = DOT(self.config)

        self.state_noise = self.config.state_noise
        self.crop_scale = self.config.crop_scale
        self.alpha = self.config.alpha
        self.inference_horizon = self.config.inference_horizon
        self.return_every_n = self.config.return_every_n
        self.predict_every_n = self.config.predict_every_n

        # Inference action chunking and observation queues
        self._old_predictions = None
        self._input_buffers = {}

        # Weights used for chunking
        action_weights = self.alpha ** torch.arange(self.inference_horizon).float()
        action_weights /= action_weights.sum()
        action_weights = action_weights.view(1, -1, 1)
        self.register_buffer("action_weights", action_weights)

        # Weights for the loss computations
        # Actions that are further in the future are weighted less
        loss_weights = torch.ones(self.config.train_horizon + self.config.n_obs_steps - 1)
        loss_weights[-self.config.train_horizon :] = (
            self.config.train_alpha ** torch.arange(self.config.train_horizon).float()
        )
        loss_weights /= loss_weights.mean()
        loss_weights = loss_weights.view(1, -1, 1)
        self.register_buffer("loss_weights", loss_weights)

        # TODO(jadechoghari): Move augmentations to dataloader (__getitem__) for CPU-side processing.
        # Nearest interpolation is required for PushT but may be not the best in general
        self.resize_transform = transforms.Resize(
            config.rescale_shape, interpolation=InterpolationMode.NEAREST
        )

        self.step = 0
        self.last_action = None

    def reset(self):
        self._old_predictions = None
        self._input_buffers = {}
        self.last_action = None
        self.step = 0

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def _update_observation_buffers(self, buffer_name: str, observation: Tensor) -> Tensor:
        # Maintain a rolling buffer of lookback_obs_steps + 1;
        # shift left and append new observation each step
        if buffer_name not in self._input_buffers:
            self._input_buffers[buffer_name] = observation.unsqueeze(1).repeat(
                1,
                self.config.lookback_obs_steps + 1,
                *torch.ones(len(observation.shape[1:])).int(),
            )
        else:
            self._input_buffers[buffer_name] = self._input_buffers[buffer_name].roll(shifts=-1, dims=1)
            self._input_buffers[buffer_name][:, -1] = observation

        return torch.cat(
            [
                self._input_buffers[buffer_name][:, :1],
                self._input_buffers[buffer_name][:, -(self.config.n_obs_steps - 1) :],
            ],
            dim=1,
        )

    def _prepare_batch_for_inference(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.normalize_inputs(batch)

        # Resize and stack all images
        if len(self.image_names) > 0:
            batch["observation.images"] = torch.stack(
                [self.resize_transform(batch[k]) for k in self.image_names],
                dim=1,
            )  # batch_size, n_cam, c, h, w

        # Update observation queues for all inputs and stack the last n_obs_steps
        for name, batch_name in self.model.obs_mapping.items():
            batch[batch_name] = self._update_observation_buffers(name, batch[batch_name])

        # Reshape images tensor to keep the same order as during training
        if "observation.images" in batch:
            batch["observation.images"] = batch["observation.images"].flatten(1, 2)
            # batch_size, n_obs * n_cam, c, h, w

        return batch

    def _chunk_actions(self, actions: Tensor) -> Tensor:
        # Store the previous action predictions in a buffer
        # Compute the weighted average of the inference horizon action predictions
        if self._old_predictions is not None:
            self._old_predictions[:, 0] = actions
        else:
            self._old_predictions = actions.unsqueeze(1).repeat(1, self.config.inference_horizon, 1, 1)

        action = (self._old_predictions[:, :, 0] * self.action_weights).sum(dim=1)
        self._old_predictions = self._old_predictions.roll(shifts=(1, -1), dims=(1, 2))

        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select an action given current environment observations.

        This function handles autoregressive rollout during inference using a fixed prediction horizon.
        The model predicts every `predict_every_n` steps, and returns actions every `return_every_n` steps.
        Between predictions, previously predicted actions are reused by shifting and repeating the last step.
        """
        self.eval()

        batch = self._prepare_batch_for_inference(batch)

        # Only run model prediction every predict_every_n steps
        if self.step % self.predict_every_n == 0:
            actions_pred = self.model(batch)[:, -self.config.inference_horizon :]
            self.last_action = self.unnormalize_outputs({"action": actions_pred})["action"]
        else:
            # Otherwise shift previous predictions and repeat last action
            self.last_action = self.last_action.roll(-1, dims=1)
            self.last_action[:, -1] = self.last_action[:, -2]

        self.step += 1

        # Return chunked actions for return_every_n steps
        action = self._chunk_actions(self.last_action)
        for _ in range(self.return_every_n - 1):
            self.last_action = self.last_action.roll(-1, dims=1)
            self.last_action[:, -1] = self.last_action[:, -2]
            action = self._chunk_actions(self.last_action)

        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        lookback_ind = torch.randint(0, 2 * self.config.lookback_aug + 1, (1,)).item()
        for k in list(self.model.obs_mapping.values()) + list(self.image_names) + ["action", "action_is_pad"]:
            if k != "observation.images":
                batch[k] = torch.cat(
                    [
                        batch[k][:, lookback_ind : lookback_ind + 1],
                        batch[k][:, 2 * self.config.lookback_aug + 1 :],
                    ],
                    1,
                )
        batch = self.normalize_targets(self.normalize_inputs(batch))

        if len(self.config.image_features) > 0:
            scale = 1 - torch.rand(1) * (1 - self.crop_scale)
            new_shape = (
                int(self.config.rescale_shape[0] * scale),
                int(self.config.rescale_shape[1] * scale),
            )
            crop_transform = transforms.RandomCrop(new_shape)

            for k in self.image_names:
                batch_size, n_obs, c, h, w = batch[k].shape
                batch[k] = batch[k].view(batch_size * n_obs, c, h, w)
                batch[k] = crop_transform(self.resize_transform(batch[k]))
                batch[k] = batch[k].view(batch_size, n_obs, c, *batch[k].shape[-2:])
            batch["observation.images"] = torch.stack([batch[k] for k in self.image_names], dim=2).flatten(
                1, 2
            )  # batch_size, n_obs * n_cam, c, h, w

        # Add random noise to states during training
        # TODO(jadechoghari): better to move this to the dataloader
        if self.state_noise is not None:
            for k in self.model.obs_mapping.values():
                if k != "observation.images":
                    batch[k] += (torch.rand_like(batch[k]) * 2 - 1) * self.state_noise

        actions_hat = self.model(batch)

        l1_loss = nn.functional.l1_loss(batch["action"], actions_hat, reduction="none")
        rev_padding = (~batch["action_is_pad"]).unsqueeze(-1)

        # Apply padding, weights and decay to the loss
        l1_loss = (l1_loss * rev_padding * self.loss_weights).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        # Reduce the aggressiveness of augmentations
        self.state_noise *= self.config.noise_decay
        self.crop_scale = 1 - (1 - self.crop_scale) * self.config.noise_decay

        return loss, loss_dict

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Load model from pretrained checkpoint and merge LoRA after loading"""
        policy = super().from_pretrained(pretrained_name_or_path, *args, **kwargs)

        if getattr(policy.config, "merge_lora", False):
            print("Merging LoRA after loading pretrained model...")
            policy.model = merge_lora_weights(policy.model)

        return policy


class LoRAConv2d(nn.Module):
    """
    Applies Low-Rank Adaptation (LoRA) to a Conv2D layer.

    LoRA adds trainable low-rank matrices (A and B) to adapt pretrained weights without full fine-tuning.
    The adaptation is merged into the base conv weights via `merge_lora()` after training.

    Args:
        base_conv (nn.Conv2d): The original convolutional layer to be adapted.
        rank (int): The rank of the low-rank approximation (default: 4).
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.base_conv = base_conv

        # Flatten the original conv weight
        out_channels, in_channels, kh, kw = base_conv.weight.shape
        self.weight_shape = (out_channels, in_channels, kh, kw)
        fan_in = in_channels * kh * kw

        # Low-rank trainable matrices A and B
        self.lora_A = nn.Parameter(torch.normal(0, 0.02, (out_channels, rank)))
        self.lora_B = nn.Parameter(torch.normal(0, 0.02, (rank, fan_in)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)

        return nn.functional.conv2d(
            x,
            self.base_conv.weight + lora_update,
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )

    def merge_lora(self) -> nn.Conv2d:
        """Merge LoRA weights into the base convolution and return a standard Conv2d layer"""
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        self.base_conv.weight.copy_(self.base_conv.weight + lora_update)

        return self.base_conv


def replace_conv2d_with_lora(module: nn.Module, rank: int = 4) -> nn.Module:
    """Recursively replace Conv2d layers with LoRAConv2d in the module"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank))
        else:
            replace_conv2d_with_lora(child, rank)
    return module


def merge_lora_weights(module: nn.Module) -> nn.Module:
    """Recursively merge LoRA weights in the module"""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(module, name, child.merge_lora())
        else:
            merge_lora_weights(child)
    return module


def add_lora_to_backbone(backbone: nn.Module, rank: int = 4) -> nn.Module:
    """
    Adds LoRA to a convolutional backbone by replacing Conv2d layers
    and freezing all other weights except LoRA layers and the final classifier.
    """
    replace_conv2d_with_lora(backbone, rank)

    for name, param in backbone.named_parameters():
        if "lora_" in name or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return backbone


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """Generates sinusoidal positional embeddings like in the original Transformer paper."""
    position = torch.arange(num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float) * (-math.log(10000.0) / dimension))
    pe = torch.zeros(num_positions, dimension)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
