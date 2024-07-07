#!/usr/bin/env python

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

from collections import deque
from typing import Callable

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.mlpbc.configuration_mlpbc import MLPBCConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize


class MLPBCPolicy(nn.Module, PyTorchModelHubMixin):
    """
    MLP-BC policy.
    """

    name = "act"

    def __init__(
        self,
        config: MLPBCConfig | None = None,
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
            config = MLPBCConfig()
        self.config: MLPBCConfig = config
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = MLPBC(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_momentum is not None:
            self._ensembled_actions = None
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
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # If we are doing temporal ensembling, keep track of the exponential moving average (EMA), and return
        # the first action.
        if self.config.temporal_ensemble_momentum is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            if self._ensembled_actions is None:
                # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
                # time step of the episode.
                self._ensembled_actions = actions.clone()
            else:
                # self._ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
                # the EMA update for those entries.
                alpha = self.config.temporal_ensemble_momentum
                self._ensembled_actions = alpha * self._ensembled_actions + (1 - alpha) * actions[:, :-1]
                # The last action, which has no prior moving average, needs to get concatenated onto the end.
                self._ensembled_actions = torch.cat([self._ensembled_actions, actions[:, -1:]], dim=1)
            # "Consume" the first action.
            action, self._ensembled_actions = self._ensembled_actions[:, 0], self._ensembled_actions[:, 1:]
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
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        actions_hat = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss_dict["loss"] = l1_loss

        return loss_dict


class MLPBC(nn.Module):
    """
    MLP-BC model, practically an MLP model with an MSE loss predicting the next action(s) in a sequence.
    """

    def __init__(self, config: MLPBCConfig):
        super().__init__()
        self.config = config
        self.use_input_state = "observation.state" in config.input_shapes

        # Backbone for image feature extraction.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final feature
        # map).
        # Note: The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # calculate model input shapes.
        policy_input_shape = (
            config.dim_model if self.use_input_state else 0  # robot state
        ) + config.dim_model * len(expected_image_keys)  # image feature maps

        # Policy, which is a BC-MLP.
        self.policy = torchvision.ops.MLP(
            in_channels=policy_input_shape,
            hidden_channels=[config.dim_model] * config.num_hidden_layers,
            dropout=config.dropout,
        )

        if self.use_input_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, config.dim_model, kernel_size=1
        )

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0] * config.chunk_size)
        self.chunk_size = config.chunk_size

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """A forward pass through the MLP-BC model.

        `batch` should have the following structure:

        {
            "observation.state": (B, state_dim) batch of robot states.
            "observation.images": (B, n_cameras, C, H, W) batch of images.
            "action": (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        # Raise exception if neither image nor states were found.
        if not self.use_input_state and "observation.images" not in batch:
            raise ValueError("No input data found for the policy.")

        batch_size = (
            batch["observation.images"].shape[0]
            if "observation.images" in batch
            else batch["observation.state"].shape[0]
        )

        # Prepare all other transformer encoder inputs.
        if "observation.images" in batch:
            all_cam_features = []
            images = batch["observation.images"]

            for cam_index in range(images.shape[-4]):
                cam_features = self.backbone(images[:, cam_index])["feature_map"]
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                # Now flatten the last two dimensions by averaging.
                cam_features = einops.reduce(cam_features, "b c h w -> b c", reduction="mean")
                all_cam_features.append(cam_features)
            # Concatenate camera observation feature maps and positional embeddings along the width dimension.
            image_encoder_out = torch.cat(all_cam_features, axis=-1)
        else:
            image_encoder_out = torch.zeros(batch_size, 0, device=batch["observation.state"].device)

        # Get positional embeddings for robot state and latent.
        if self.use_input_state:
            robot_state_embed = self.encoder_robot_state_input_proj(batch["observation.state"])  # (B, C)
        else:
            robot_state_embed = torch.zeros(batch_size, 0, device=batch["observation.state"].device)

        # Stack encoder input and state input to (B, C).
        policy_in = torch.cat(
            [
                robot_state_embed,
                image_encoder_out,
            ],
            dim=-1,
        )

        policy_out = self.policy(policy_in)  # (B, (d_model))
        actions = self.action_head(policy_out)  # (B, chunk_size * action_dim)
        actions = einops.rearrange(
            actions, "b (chunk_size action_dim) -> b chunk_size action_dim", chunk_size=self.chunk_size
        )
        return actions


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
