#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Humanoid Imitation Transformer Policy

As per HumanPlus: Humanoid Shadowing and Imitation from Humans (http://arxiv.org/abs/2406.10454).
This code is directly modified from the `act` policy in the current repository.
"""

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.act.modeling_act import (
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    get_activation_fn,
)
from lerobot.common.policies.hit.configuration_hit import HITConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize


class HITPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "hit"],
):
    """
    Humanoid Imitation Transformer Policy as per HumanPlus: Humanoid Shadowing and Imitation from Humans
    (paper: http://arxiv.org/abs/2406.10454, code: https://github.com/MarkFzp/humanplus)
    """

    name = "hit"

    def __init__(
        self,
        config: HITConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = HITConfig()
        self.config: HITConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = HIT(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.use_future_loss = config.feature_loss_weight > 0
        self.feature_loss_weight = config.feature_loss_weight

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
            batch["observation.images_is_pad"] = torch.stack(
                [batch[k + "_is_pad"] for k in self.expected_image_keys], dim=-1
            )
        batch = self.normalize_targets(batch)
        actions_hat, (feat_future, pred_feat_future) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        loss_dict = {"l1_loss": l1_loss.item()}
        loss_dict["loss"] = l1_loss

        if self.use_future_loss:
            pad_mask = torch.any(batch["observation.images_is_pad"], dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
            feature_loss = (F.mse_loss(feat_future, pred_feat_future, reduction="none") * ~pad_mask).mean()
            loss_dict["feature_loss"] = feature_loss.item()
            loss_dict["loss"] = loss_dict["loss"] + self.feature_loss_weight * feature_loss

        return loss_dict


class HIT(nn.Module):
    """Humanoid Imitation Transformer: The underlying neural network for HITPolicy.

    Note: In this code we use the terms `encoder` because it is a BERT-style, encoder-only transformer.
    However, in the original paper and code, the authors refer to it as a "decoder".
    """

    def __init__(self, config: HITConfig):
        super().__init__()
        self.config = config
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes
        self.use_future_loss = config.feature_loss_weight > 0

        # Backbone for image feature extraction.
        if self.use_images:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer encoder.
        self.encoder = HITEncoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [(actions), (robot_state), (env_state), (image_feature_map_pixels)].
        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
        if self.use_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.input_shapes["observation.environment_state"][0], config.dim_model
            )
        if self.use_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = config.chunk_size  # For action positional embeddings.
        if self.use_robot_state:
            n_1d_tokens += 1
        if self.use_env_state:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Humanoid Imitation Transformer.

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        # Prepare transformer encoder inputs.
        # Placeholder for positional embedding that generate action tokens.
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        encoder_in_tokens = list(
            torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=self.encoder_1d_feature_pos_embed.weight.dtype,
                device=self.encoder_1d_feature_pos_embed.weight.device,
            )
        )
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.use_robot_state:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        # Environment state token.
        if self.use_env_state:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        use_future_loss = False
        # Camera observation features and positional embeddings.
        if self.use_images:
            is_training = any(k.endswith("is_pad") for k in batch)
            use_future_loss = self.use_future_loss and is_training
            if not is_training:
                batch["observation.images"] = batch["observation.images"].unsqueeze(1)
            all_cam_features = []
            all_cam_features_future = []
            all_cam_pos_embeds = []

            for cam_index in range(batch["observation.images"].shape[-4]):
                cam_features = self.backbone(batch["observation.images"][:, 0, cam_index])["feature_map"]
                # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
                # buffer
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)
                # compute futre image feature
                if use_future_loss:
                    cam_features_future = self.backbone(batch["observation.images"][:, 1, cam_index])[
                        "feature_map"
                    ]
                    cam_features_future = self.encoder_img_feat_input_proj(cam_features_future)
                    all_cam_features_future.append(cam_features_future)
            # Concatenate camera observation feature maps and positional embeddings along the width dimension,
            # and move to (sequence, batch, dim).
            all_cam_features = torch.cat(all_cam_features, axis=-1)
            encoder_in_tokens.extend(einops.rearrange(all_cam_features, "b c h w -> (h w) b c"))
            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
            encoder_in_pos_embed.extend(einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c"))
            if use_future_loss:
                all_cam_features_future = torch.cat(all_cam_features_future, axis=-1)
                all_cam_features_future = einops.rearrange(all_cam_features_future, "b c h w -> (h w) b c")
                len_img_feat = all_cam_features_future.shape[0]

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # Change back to (batch, sequence, dim) format.
        action_tokens = encoder_out[: self.config.chunk_size].transpose(0, 1)
        actions = self.action_head(action_tokens)

        feat_future = all_cam_features_future.transpose(0, 1) if use_future_loss else None
        pred_feat_future = encoder_out[-len_img_feat:].transpose(0, 1) if use_future_loss else None

        return actions, (feat_future, pred_feat_future)


class HITEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization.
    Same as `ACTEncoder`."""

    def __init__(self, config: HITConfig):
        super().__init__()
        num_layers = config.n_layers
        self.layers = nn.ModuleList([HITEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class HITEncoderLayer(nn.Module):
    """Same as `ACTEncoderLayer`."""

    def __init__(self, config: HITConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x
