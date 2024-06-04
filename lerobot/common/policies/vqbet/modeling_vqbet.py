from collections import deque
from typing import Callable, List
from functools import partial
from random import randrange
import math
from math import ceil
import warnings

import einops
from einops import rearrange, repeat, reduce, pack, unpack
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn, einsum
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from huggingface_hub import PyTorchModelHubMixin

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues

class VQBeTPolicy(nn.Module, PyTorchModelHubMixin):
    """
    VQ-BeT Policy as per "Behavior Generation with Latent Actions"
    """

    name = "vqbet"
    def __init__(
        self,
        config: VQBeTConfig | None = None,
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
            config = VQBeTConfig()
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )


        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.vqbet = VQBeTModel(config)

        self.reset()

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        """
        self._queues = {
            "observation.image": deque(maxlen=self.config.n_obs_steps),
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_pred_chunk),
        }

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        self.eval()

        batch = self.normalize_inputs(batch)
        self._queues = populate_queues(self._queues, batch)

        assert self.vqbet.action_head.vqvae_model.check_discretized(), "To evaluate in the environment, your VQ-BeT model should contain a pretrained Residual VQ."
        assert "observation.image" in batch
        assert "observation.state" in batch

        if len(self._queues["action"]) == 0:

            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}
            actions = self.vqbet(batch, rollout=True)[:, : self.config.n_action_pred_chunk]

            # the dimension of returned action is (batch_size, n_action_pred_chunk, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            # since the data in the action queue's dimension is (n_action_pred_chunk, batch_size, action_dim), we transpose the action and fill the queue
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        # VQ-BeT discretizes action using VQ-VAE before training BeT (please refer to section 3.2 in the VQ-BeT paper https://arxiv.org/pdf/2403.03181)
        if not self.vqbet.action_head.vqvae_model.check_discretized():
            loss, n_different_codes, n_different_combinations = self.vqbet.discretize(self.config.discretize_step, batch['action'])
            return {"loss": loss, "n_different_codes": n_different_codes, "n_different_combinations": n_different_combinations}
        # if Residual VQ is already trained, VQ-BeT trains its GPT and bin ped header / offset header parts.
        _, loss_dict = self.vqbet(batch, rollout=False)

        return loss_dict

class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints

class VQBeTModel(nn.Module):
    """VQ-BeT: The underlying neural network for VQ-BeT

    Note: In this code we use the terms `rgb_encoder`, 'policy', `action_head`. The meanings are as follows.
        - The `rgb_encoder` process rgb-style image observations to one-dimensional embedding vectors
        - A `policy` is a minGPT architecture, that takes observation sequences and action query tokens to generate `features`.
        - These `features` pass through the action head, which passes through the code prediction, offset prediction head, 
        and finally generates a prediction for the action chunks.

        -------------------------------** legend **-------------------------------
        │   n = n_obs_steps, p = n_action_pred_token, c = n_action_pred_chunk)   │
        │   o_{t} : visual observation at timestep {t}                           │
        │   s_{t} : state observation at timestep {t}                            │
        │   a_{t} : action at timestep {t}                                       │
        │   A_Q : action_query_token                                             │
        --------------------------------------------------------------------------

        
        Phase 1. Discretize action using Residual VQ (for config.discretize_step steps)


        ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
        │                 │            │                 │            │                 │
        │   RVQ encoder   │    ─►      │     Residual    │    ─►      │   RVQ Decoder   │
        │ (a_{t}~a_{t+p}) │            │  Code Quantizer │            │                 │
        │                 │            │                 │            │                 │
        └─────────────────┘            └─────────────────┘            └─────────────────┘

        Phase 2.
        
          timestep {t-n+1}   timestep {t-n+2}                timestep {t}
            ┌─────┴─────┐     ┌─────┴─────┐                 ┌─────┴─────┐

        o_{t-n+1}         o_{t-n+2}           ...         o_{t}
            │                 │                             │ 
            │ s_{t-n+1}       │ s_{t-n+2}         ...       │   s_{t}           p
            │     │           │     │                       │     │     ┌───────┴───────┐
            │     │    A_Q    │     │    A_Q          ...   │     │    A_Q     ...     A_Q
            │     │     │     │     │     │                 │     │     │               │
        ┌───▼─────▼─────▼─────▼─────▼─────▼─────────────────▼─────▼─────▼───────────────▼───┐
        │                                                                                   │
        │                                       GPT                                         │       =>    policy
        │                                                                                   │
        └───────────────▼─────────────────▼─────────────────────────────▼───────────────▼───┘
                        │                 │                             │               │
                    ┌───┴───┐         ┌───┴───┐                     ┌───┴───┐       ┌───┴───┐
                  code    offset    code    offset                code    offset  code    offset
                    ▼       │         ▼       │                     ▼       │       ▼       │       =>    action_head
               RVQ Decoder  │    RVQ Decoder  │                RVQ Decoder  │  RVQ Decoder  │
                    └── + ──┘         └── + ──┘                     └── + ──┘       └── + ──┘
                        ▼                 ▼                             ▼               ▼
                   action chunk      action chunk                  action chunk     action chunk
                    a_{t_n+1} ~       a_{t_n+2} ~                   a_{t} ~     ...  a_{t+p-1} ~ 
                     a_{t_n+c}         a_{t_n+c+1}                   a_{t+c-1}        a_{t+p+c-1}

                                                                        ▼
                                                      ONLY this chunk is used in rollout!
    """
    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config

        self.rgb_encoder = VQBeTRgbEncoder(config)

        # This action query token is used as a prompt for querying action chunks. Please refer to "A_Q" in the image above.
        self._action_token = nn.Parameter(torch.randn(1, 1, self.config.gpt_input_dim))

        # To input state and observation features into GPT layers, we first project the features to fit the shape of input size of GPT.
        self.state_projector = MLP(
                config.output_shapes["action"][0], 
                hidden_channels=[self.config.gpt_input_dim]
            )
        self.obs_projector = MLP(
                self.rgb_encoder.feature_dim, 
                hidden_channels=[self.config.gpt_input_dim]
            )
        
        # GPT part of VQ-BeT
        self.policy = GPT(config)
        # bin prediction header / offset prediction header part of VQ-BeT
        self.action_head = VQBeTHead(config)

    def discretize(self, discretize_step, actions):
        return self.action_head.discretize(discretize_step, actions)

    def forward(self, batch: dict[str, Tensor], rollout: bool) -> Tensor:
        # Input validation.
        assert set(batch).issuperset({"observation.state", "observation.image"})
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Extract image feature (first combine batch and sequence dims).
        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        
        # image observation feature, state feature, and action query token are grouped together with the same timestpe to form a group, which is listed in order to be entered into GPT sequentially.
        observation_feature = torch.cat([
                torch.unsqueeze(self.obs_projector(img_features), dim=2), 
                torch.unsqueeze(self.state_projector(batch["observation.state"]), dim=2), 
                self._action_token.repeat(batch_size, n_obs_steps, 1, 1)
            ], dim=-2).view(batch_size, -1, self.config.gpt_input_dim)
        if img_features.shape[1] != n_obs_steps:
            raise NotImplementedError
        len_additional_action_token = self.config.n_action_pred_token-1
        action_token = self._action_token.repeat(batch_size, len_additional_action_token, 1)

        # add additional action query tokens for predicting future action chunks
        observation_feature = torch.cat([observation_feature, action_token], dim=1)

        
        # get action features (pass through GPT)
        features = self.policy(observation_feature)
        historical_act_pred_index = np.arange(0, n_obs_steps) * (self.config.gpt_num_obs_mode+1) + self.config.gpt_num_obs_mode

        # only extract the output tokens at the position of action query
        features = torch.cat([
            features[:, historical_act_pred_index],
            features[:, -len_additional_action_token:]
        ], dim=1)
        # pass through action head
        pred_action = self.action_head(
            features,
        )
        # if rollout, VQ-BeT don't calculate loss
        if rollout:
            return pred_action["predicted_action"][:, n_obs_steps-1, :].reshape(batch_size, self.config.n_action_pred_chunk, -1)
        # else, it calculate overall loss (bin prediction loss, and offset loss)
        else:
            action = batch["action"]
            n, total_w, act_dim = action.shape
            act_w = self.config.n_action_pred_chunk
            num_token = total_w + 1 - act_w
            output_shape = (n, num_token, act_w, act_dim)
            output = torch.empty(output_shape).to(action.device)
            for i in range(num_token):
                output[:, i, :, :] = action[:, i : i + act_w, :]
            action = output

            loss_dict = self.action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
            )
            return pred_action, loss_dict


class VQBeTHead(nn.Module):
    def __init__(self, config: VQBeTConfig):
        """
        VQBeTHead takes output of GPT layers, and pass the feature through bin prediction head (`self.map_to_cbet_preds_bin`), and offset prediction head (`self.map_to_cbet_preds_offset`)

        self.map_to_cbet_preds_bin: outputs probability of each code (for each layer).
            The input dimension of `self.map_to_cbet_preds_bin` is same with the output of GPT, 
            and the output dimension of `self.map_to_cbet_preds_bin` is `self.config.vqvae_groups * self.config.vqvae_n_embed`, where 
                `self.config.vqvae_groups` is number of RVQ layers, and
                `self.config.vqvae_n_embed` is codebook size of RVQ.

        self.map_to_cbet_preds_offset: output the predicted offsets for all the codes in all the layers. 
            The input dimension of ` self.map_to_cbet_preds_offset` is same with the output of GPT, 
            and the output dimension of ` self.map_to_cbet_preds_offset` is `self.config.vqvae_groups * self.config.vqvae_n_embed * config.n_action_pred_chunk * config.output_shapes["action"][0]`, where
                `self.config.vqvae_groups` is number of RVQ layers,
                `self.config.vqvae_n_embed` is codebook size of RVQ,
                `config.n_action_pred_chunk is action chunk size of each token, and
                `config.output_shapes["action"][0]` is the dimension of action 
        """

        super().__init__()
        self.config = config



        self.map_to_cbet_preds_bin = MLP(
            in_channels=config.gpt_output_dim,
            hidden_channels=[self.config.vqvae_groups * self.config.vqvae_n_embed],
        )
        self.map_to_cbet_preds_offset = MLP(
            in_channels=config.gpt_output_dim,
            hidden_channels=[
                self.config.vqvae_groups * self.config.vqvae_n_embed * config.n_action_pred_chunk * config.output_shapes["action"][0],
            ],
        )
        # init vqvae
        self.vqvae_model = VqVae(config)
        # loss
        self._criterion = FocalLoss(gamma=2.0)

    def discretize(self, discretize_step, actions):
        if next(self.vqvae_model.encoder.parameters()).device != get_device_from_parameters(self):
            self.vqvae_model.encoder.to(get_device_from_parameters(self))
            self.vqvae_model.vq_layer.to(get_device_from_parameters(self))
            self.vqvae_model.decoder.to(get_device_from_parameters(self))
            self.vqvae_model.device = get_device_from_parameters(self)

        loss, n_different_codes, n_different_combinations = pretrain_vqvae(self.vqvae_model, discretize_step, actions)
        # if we updated RVQ more than `discretize_step` steps,
        if self.vqvae_model.check_discretized():
            print("Finished discretizing action data!")
            self.vqvae_model.eval()
            for param in self.vqvae_model.vq_layer.parameters():
                param.requires_grad = False
        return loss, n_different_codes, n_different_combinations

    def forward(self, x, **kwargs):
        # N is the batch size, and T is number of action query tokens, which are process through same GPT
        N, T, _ = x.shape
        # we calculate N and T side parallely. Thus, the dimensions would be 
        # (batch size * number of action query tokens, action chunk size, action dimension)
        x = einops.rearrange(x, "N T WA -> (N T) WA")

        cbet_logits = self.map_to_cbet_preds_bin(x)
        cbet_offsets = self.map_to_cbet_preds_offset(x)
        cbet_logits = einops.rearrange(
            cbet_logits, "(NT) (G C) -> (NT) G C", G=self.config.vqvae_groups
        )
        cbet_offsets = einops.rearrange(
            cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self.config.vqvae_groups, C=self.config.vqvae_n_embed
        )
        cbet_probs = torch.softmax(cbet_logits / self.config.bet_softmax_temperature, dim=-1)
        NT, G, choices = cbet_probs.shape
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
            "(NT G) 1 -> NT G",
            NT=NT,
        )

        indices = (
            torch.arange(NT).unsqueeze(1).cuda(),
            torch.arange(self.config.vqvae_groups).unsqueeze(0).cuda(),
            sampled_centers,
        )
        # Use advanced indexing to sample the values
        sampled_offsets = cbet_offsets[indices]
        # Extract the only offsets corresponding to the sampled codes.
        sampled_offsets = sampled_offsets.sum(dim=1)
        # Get the centroids of each layer to pass it through RVQ decoder
        centers = self.vqvae_model.draw_code_forward(sampled_centers).view(
            NT, -1, self.config.vqvae_embedding_dim
        )
        return_decoder_input = einops.rearrange(
            centers.clone().detach(), "NT 1 D -> NT D"
        )
        # pass the centroids through decoder to get actions.
        decoded_action = (
            self.vqvae_model.get_action_from_latent(return_decoder_input)
            .clone()
            .detach()
        )
        # reshaped extracted offset to match with decoded centroids
        sampled_offsets = einops.rearrange(
            sampled_offsets, "NT (W A) -> NT W A", W=self.config.n_action_pred_chunk
        )
        # add offset and decoded centroids
        predicted_action = decoded_action + sampled_offsets
        predicted_action = einops.rearrange(
            predicted_action,
            "(N T) W A -> N T (W A)",
            N=N,
            T=T,
            W=self.config.n_action_pred_chunk,
        )

        return {
            "cbet_logits": cbet_logits if "cbet_logits" in locals() else None,
            "predicted_action": predicted_action,
            "sampled_centers": sampled_centers,
            "decoded_action": decoded_action, 
        }

    def loss_fn(self, pred, target, **kwargs):
        """
        for given ground truth action values (target), and prediction (pred) this function calculates the overall loss.

        predicted_action: predicted action chunk (offset + decoded centroids)
        sampled_centers: sampled centroids (code of RVQ)
        decoded_action: decoded action, which is produced by passing sampled_centers through RVQ decoder
        NT: batch size * T
        T: number of action query tokens, which are process through same GPT
        cbet_logits: probability of all codes in each layer
        """
        action_seq = target
        predicted_action = pred["predicted_action"]
        sampled_centers = pred["sampled_centers"]
        decoded_action = pred["decoded_action"]
        NT = predicted_action.shape[0] * predicted_action.shape[1]
        T = predicted_action.shape[1]
        cbet_logits = pred["cbet_logits"]

        predicted_action = einops.rearrange(
            predicted_action, "N T (W A) -> (N T) W A", W=self.config.n_action_pred_chunk
        )

        action_seq = einops.rearrange(action_seq, "N T W A -> (N T) W A")
        # Figure out the loss for the actions.
        # First, we need to find the closest cluster center for each ground truth action.
        state_vq, action_bins = self.vqvae_model.get_code(
            action_seq
        )  # action_bins: NT, G

        # Now we can compute the loss.
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)

        # offset loss is L1 distance between the predicted action and ground truth action
        offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)

        # calculate primary code prediction loss
        cbet_loss1 = self._criterion(  # F.cross_entropy
            cbet_logits[:, 0, :],
            action_bins[:, 0],
        )
        # calculate secondary code prediction loss
        cbet_loss2 = self._criterion(  # F.cross_entropy
            cbet_logits[:, 1, :],
            action_bins[:, 1],
        )
        # add all the prediction loss
        cbet_loss = cbet_loss1 * self.config.primary_code_loss_weight + cbet_loss2 * self.config.secondary_code_loss_weight

        equal_primary_code_rate = torch.sum(
            (action_bins[:, 0] == sampled_centers[:, 0]).int()
        ) / (NT)
        equal_secondary_code_rate = torch.sum(
            (action_bins[:, 1] == sampled_centers[:, 1]).int()
        ) / (NT)

        action_mse_error = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=T),
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=T),
        )
        vq_action_error = (abs(einops.rearrange(action_seq, "(N T) W A -> N T W A", T=T) - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=T))).mean()
        offset_action_error = (abs(einops.rearrange(action_seq, "(N T) W A -> N T W A", T=T) - einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=T))).mean()
        action_error_max = (abs(einops.rearrange(action_seq, "(N T) W A -> N T W A", T=T) - einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=T))).max()

        loss = cbet_loss + self.config.offset_loss_weight * offset_loss

        loss_dict = {
            "loss": loss,
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "equal_primary_code_rate": equal_primary_code_rate.detach().cpu().item(),
            "equal_secondary_code_rate": equal_secondary_code_rate.detach().cpu().item(),
            "vq_action_error": vq_action_error.detach().cpu().item(),
            "offset_action_error": offset_action_error.detach().cpu().item(),
            "action_error_max": action_error_max.detach().cpu().item(),
            "action_mse_error": action_mse_error.detach().cpu().item(),

        }
        return loss_dict

class VQBeTOptimizer:
    def __init__(self, policy, cfg):
        self.discretize_step = cfg.training.discretize_step
        self.offline_steps = cfg.training.offline_steps
        self.optimizing_step = 0


        vqvae_params = (
            list(policy.vqbet.action_head.vqvae_model.encoder.parameters())
            + list(policy.vqbet.action_head.vqvae_model.decoder.parameters())
            + list(policy.vqbet.action_head.vqvae_model.vq_layer.parameters())
        )
        self.vqvae_optimizer = torch.optim.Adam(
            vqvae_params, lr=cfg.training.vqvae_lr, weight_decay=0.0001
        )

        self.encoder_optimizer = torch.optim.Adam(
            policy.vqbet.rgb_encoder.parameters(),
            cfg.training.lr,
            cfg.training.adam_betas,
            cfg.training.adam_eps,
            cfg.training.adam_weight_decay,
        )

        self.bet_optimizer1 = policy.vqbet.policy.configure_optimizers(
            weight_decay=cfg.training.bet_weight_decay,
            learning_rate=cfg.training.bet_learning_rate,
            betas=cfg.training.bet_betas,
        )

        self.bet_optimizer1.add_param_group(
                {"params": policy.vqbet._action_token}
            )
        self.bet_optimizer1.add_param_group(
                {"params": policy.vqbet.state_projector.parameters()}
            )
        self.bet_optimizer1.add_param_group(
                {"params": policy.vqbet.obs_projector.parameters()}
            )

        self.bet_optimizer2 = torch.optim.AdamW(
            policy.vqbet.action_head.map_to_cbet_preds_bin.parameters(),
            lr=cfg.training.bet_learning_rate,
            weight_decay=cfg.training.bet_weight_decay,
            betas=cfg.training.bet_betas,
        )
        
        self.bet_optimizer3 = torch.optim.AdamW(
            policy.vqbet.action_head.map_to_cbet_preds_offset.parameters(),
            lr=cfg.training.bet_learning_rate,
            weight_decay=cfg.training.bet_weight_decay,
            betas=cfg.training.bet_betas,
        )

        self.param_groups = self.encoder_optimizer.param_groups

    def step(self):
        self.optimizing_step +=1
        # pretraining VQ-VAE (phase 1)
        if self.optimizing_step < self.discretize_step:
            self.vqvae_optimizer.step()
        # training BeT (phase 2)
        else:
            self.encoder_optimizer.step()
            self.bet_optimizer1.step()
            self.bet_optimizer2.step()
            self.bet_optimizer3.step()

    def zero_grad(self):
        # pretraining VQ-VAE (phase 1)
        if self.optimizing_step < self.discretize_step:
            self.vqvae_optimizer.zero_grad()
        # training BeT (phase 2)
        else:
            self.encoder_optimizer.zero_grad()
            self.bet_optimizer1.zero_grad()
            self.bet_optimizer2.zero_grad()
            self.bet_optimizer3.zero_grad()

class VQBeTScheduler:
    def __init__(self, optimizer, cfg):
        # VQ-BeT use scheduler only for rgb encoder. Since we took rgb encoder part from diffusion policy, we also follow the same scheduler from it.
        from diffusers.optimization import get_scheduler
        self.discretize_step = cfg.training.discretize_step
        self.optimizing_step = 0

        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer.encoder_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )


    def step(self):
        self.optimizing_step +=1
        if self.optimizing_step >= self.discretize_step:
            self.lr_scheduler.step()

class VQBeTRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.

    Same with DiffusionRgbEncoder from modeling_diffusion.py
    """

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        with torch.inference_mode():
            feat_map_shape = tuple(
                self.backbone(torch.zeros(size=(1, *config.input_shapes["observation.image"]))).shape[1:]
            )
        self.pool = SpatialSoftmax(feat_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module




class VqVae(nn.Module):
    def __init__(
        self, config: VQBeTConfig,
    ):
        """
        VQ-VAE is composed of three parts: encoder, vq_layer, and decoder.
        Encoder and decoder are MLPs consisting of an input, output layer, and hidden layer, respectively.
        The vq_layer uses residual VQs.
        """

        super(VqVae, self).__init__()
        self.config = config

        self.register_buffer('discretized', torch.tensor(False))
        self.optimized_steps = 0

        self.vq_layer = ResidualVQ(
            dim=config.vqvae_embedding_dim,
            num_quantizers=config.vqvae_groups,
            codebook_size=config.vqvae_n_embed,
        )

        self.encoder = MLP(
            in_channels=self.config.output_shapes["action"][0] * self.config.n_action_pred_chunk,
            hidden_channels=[config.vqvae_enc_hidden_dim, config.vqvae_enc_hidden_dim, config.vqvae_embedding_dim],
        )
        self.decoder = MLP(
            in_channels=config.vqvae_embedding_dim,
            hidden_channels=[config.vqvae_enc_hidden_dim, config.vqvae_enc_hidden_dim, self.config.output_shapes["action"][0] * self.config.n_action_pred_chunk],
        )

        self.train()

    def toggle_discretized(self, state=True):
        self.discretized = torch.tensor(state)

    def check_discretized(self):
        return self.discretized.item()

    def eval(self):
        self.training = False
        self.vq_layer.eval()
        self.encoder.eval()
        self.decoder.eval()

    def train(self, mode=True):
        """
        This function forces the RVQ to no longer update when action discretization is complete.
        Since VQs are partly updated via the EMA method, simply passing data through them can cause unintended modifications.
        Therefore, we use function overriding to prevent RVQs from being updated during the training of VQ-BeT after discretization completes.
        """
        if mode:
            if self.check_discretized():
                pass
            else:
                self.training = True
                self.vq_layer.train()
                self.decoder.train()
                self.encoder.train()
        else:
            self.eval()

    def draw_logits_forward(self, encoding_logits):
        z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent):
        output = self.decoder(latent)
        if self.config.n_action_pred_chunk == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.config.output_shapes["action"][0])
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.config.output_shapes["action"][0])

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.copy())
        if self.config.n_action_pred_chunk == 1:
            state = state.squeeze(-2)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state

    def get_code(self, state, required_recon=False):
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            if required_recon:
                recon_state = self.decoder(state_vq)
                recon_state_ae = self.decoder(state_rep)
                if self.config.n_action_pred_chunk == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                else:
                    return (
                        state_vq,
                        vq_code,
                        torch.swapaxes(recon_state, -2, -1),
                        torch.swapaxes(recon_state_ae, -2, -1),
                    )
            else:
                return state_vq, vq_code

    def vqvae_forward(self, state):
        state = self.preprocess(state)
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        vq_loss_state = torch.sum(vq_loss_state)

        dec_out = self.decoder(state_vq)
        encoder_loss = (state - dec_out).abs().mean()

        rep_loss = encoder_loss + vq_loss_state * 5

        metric = (
            encoder_loss.clone().detach(),
            vq_loss_state.clone().detach(),
            vq_code,
            rep_loss.item(),
        )
        return rep_loss, metric


    def load_state_dict(self, *args, **kwargs):
        super(VqVae, self).state_dict(self, *args, **kwargs)
        self.eval()
        self.toggle_discretized(True)





def pretrain_vqvae(vqvae_model, discretize_step, actions):
    if vqvae_model.config.n_action_pred_chunk == 1:
        # not using action chunk
        actions = actions.reshape(-1, 1, actions.shape[-1])
    else:
        # using action chunk
        slices = []
        slices.extend([actions[:, j:j+vqvae_model.config.n_action_pred_chunk, :] for j in range(actions.shape[1]+1-vqvae_model.config.n_action_pred_chunk)])
        actions = torch.cat(slices, dim=0)


    actions = actions.to(get_device_from_parameters(vqvae_model))

    loss, metric = vqvae_model.vqvae_forward(
        actions
    )
    n_different_codes = len(torch.unique(metric[2]))
    n_different_combinations = len(torch.unique(metric[2], dim=0))
    vqvae_model.optimized_steps += 1
    if vqvae_model.optimized_steps >= discretize_step:
        vqvae_model.toggle_discretized(True)
    return loss, n_different_codes, n_different_combinations


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


class ResidualVQ(nn.Module):
    """
    MIT License

    Copyright (c) 2020 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim=None,
        shared_codebook=False,
        heads=1,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        accept_image_fmap=False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, "residual vq is not compatible with multi-headed codes"
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        )

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=codebook_dim,
                    codebook_dim=codebook_dim,
                    accept_image_fmap=accept_image_fmap,
                    **kwargs
                )
                for _ in range(num_quantizers)
            ]
        )

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        codebooks = rearrange(codebooks, "q 1 c d -> q c d")
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], "b * q")

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, "q c d -> q b c d", b=batch)
        gather_indices = repeat(indices, "b n q -> q b n d", d=codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(
            mask, 0
        )  # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(2, gather_indices)  # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.0)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        (all_codes,) = unpack(all_codes, ps, "q b * d")

        return all_codes

    def draw_logits_forward(self, encoding_logits):
        # encoding_indices : dim1 = batch_size  dim2 = 4 (number of groups) dim3 = vq dict size (header)
        encoding_logits = encoding_logits
        bs = encoding_logits.shape[0]
        quantized = torch.zeros((bs, self.codebooks.shape[-1]))
        for q in range(encoding_logits.shape[1]):
            quantized += torch.matmul(encoding_logits[:, q], self.codebooks[q])
        return quantized

    def forward(
        self, x, indices=None, return_all_codes=False, sample_codebook_temp=None
    ):
        num_quant, quant_dropout_multiple_of, return_loss, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            exists(indices),
            x.device,
        )

        x = self.project_in(x)

        assert not (self.accept_image_fmap and exists(indices))

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(
                indices == -1
            ), "some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss"
            ce_losses = []

        should_quantize_dropout = (
            self.training and self.quantize_dropout and not return_loss
        )

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:
            rand_quantize_dropout_index = randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

            null_indices_shape = (
                (x.shape[0], *x.shape[-2:])
                if self.accept_image_fmap
                else tuple(x.shape[:2])
            )
            null_indices = torch.full(
                null_indices_shape, -1.0, device=device, dtype=torch.long
            )
            null_loss = torch.full((1,), 0.0, device=device, dtype=x.dtype)

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):
            if (
                should_quantize_dropout
                and quantizer_index > rand_quantize_dropout_index
            ):
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_losses, all_indices = map(
            partial(torch.stack, dim=-1), (all_losses, all_indices)
        )

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret




class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=False,
        decay=0.8,
        eps=1e-5,
        freeze_codebook=False,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=1.0,
        commitment_use_cross_entropy_loss=False,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        stochastic_sample_codes=False,
        sample_codebook_temp=1.0,
        straight_through=False,
        reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook=None,
        sync_affine_param=False,
        ema_update=True,
        learnable_codebook=False,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
        sync_update_v=0.0,  # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        )

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss  # whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (
            ema_update and learnable_codebook
        ), "learnable codebook not compatible with EMA update"

        assert 0 <= sync_update_v <= 1.0
        assert not (
            sync_update_v > 0.0 and not learnable_codebook
        ), "learnable codebook must be turned on"

        self.sync_update_v = sync_update_v


        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic=stochastic_sample_codes,
            reinmax=reinmax,
            straight_through=straight_through,
        )

        if not exists(sync_codebook):
            sync_codebook = (
                distributed.is_initialized() and distributed.get_world_size() > 1
            )

        codebook_kwargs = dict(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp=sample_codebook_temp,
            gumbel_sample=gumbel_sample_fn,
            ema_update=ema_update,
        )

        if affine_param:
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param=True,
                sync_affine_param=sync_affine_param,
                affine_param_batch_decay=affine_param_batch_decay,
                affine_param_codebook_decay=affine_param_codebook_decay,
            )

        self._codebook = EuclideanCodebook(**codebook_kwargs)

        self.in_place_codebook_optimizer = (
            in_place_codebook_optimizer(self._codebook.parameters())
            if exists(in_place_codebook_optimizer)
            else None
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, "1 ... -> ...")

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, "... -> 1 ...")

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack_one(indices, "b * h")
        indices = rearrange(indices, "b n h -> b h n")

        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        codes = unpack_one(codes, ps, "b * d")
        return codes

    def forward(
        self,
        x,
        indices=None,
        mask=None,
        sample_codebook_temp=None,
        freeze_codebook=False,
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, "b d -> b 1 d")

        shape, device, heads, is_multiheaded, codebook_size, return_loss = (
            x.shape,
            x.device,
            self.heads,
            self.heads > 1,
            self.codebook_size,
            exists(indices),
        )

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c")

        if need_transpose:
            x = rearrange(x, "b d n -> b n d")

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp=sample_codebook_temp,
            mask=mask,
            freeze_codebook=freeze_codebook,
        )

        # quantize

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:
            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction="none")

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(
                        mask,
                        "b n -> c (b h) n",
                        c=loss.shape[0],
                        h=loss.shape[1] // mask.shape[0],
                    )

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # quantize again

            quantize, embed_ind, distances = self._codebook(
                x, **codebook_forward_kwargs
            )

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = (
                torch.detach
                if not self.learnable_codebook or freeze_codebook
                else identity
            )

            commit_quantize = maybe_detach(quantize)

            # straight through

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.0:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (
                    quantize - quantize.detach()
                )

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = "1 b n l -> b l n"
            elif self.separate_codebook_per_head:
                dist_einops_eq = "c b n l -> b l n c"
            else:
                dist_einops_eq = "1 (b h) n l -> b l n h"

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b=shape[0]), codes, ignore_index=-1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(
                embed_ind, "b (h w) ... -> b h w ...", h=height, w=width
            )

        if only_one:
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        # aggregate loss

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, "b n -> b n h", h=heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction="none")

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(
                                loss_mask,
                                "b n -> c (b h) n",
                                c=commit_loss.shape[0],
                                h=commit_loss.shape[1] // mask.shape[0],
                            )

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (
                        is_multiheaded and self.separate_codebook_per_head
                    ), "orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet"
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if (
                    exists(self.orthogonal_reg_max_codes)
                    and num_codes > self.orthogonal_reg_max_codes
                ):
                    rand_ids = torch.randperm(num_codes, device=device)[
                        : self.orthogonal_reg_max_codes
                    ]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, "b n d -> b d n")

        if self.accept_image_fmap:
            quantize = rearrange(quantize, "b (h w) c -> b c h w", h=height, w=width)

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")

        # if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = torch.where(
                rearrange(mask, "... -> ... 1"), quantize, orig_input
            )

        return quantize, embed_ind, loss

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def identity(t):
    return t


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).sqrt()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    reinmax=False,
    dim=-1,
    training=True,
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (
        reinmax and not straight_through
    ), "reinmax can only be turned on if using straight through gumbel softmax"

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack(
        [sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0
    )


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(
            num, all_num_samples / all_num_samples.sum()
        )
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    sample_fn=batched_sample_vectors,
    all_reduce_fn=noop,
):
    num_codebooks, dim, dtype, device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


# regularization losses


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


# distance types


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        affine_param=False,
        sync_affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
            use_ddp and num_codebooks > 1 and kmeans_init
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = (
            sample_vectors_distributed
            if use_ddp and sync_kmeans
            else batched_sample_vectors
        )
        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and sync_kmeans else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer("batch_mean", None)
        self.register_buffer("batch_variance", None)

        self.register_buffer("codebook_mean_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_mean", torch.empty(num_codebooks, 1, dim))
        self.register_buffer("codebook_variance_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_variance", torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * rearrange(cluster_size, "... -> ... 1")

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask=None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased=False)

        # calculate codebook mean and variance

        embed = rearrange(embed, "h ... d -> h (...) d")

        if self.training:
            self.update_with_decay(
                "codebook_mean",
                reduce(embed, "h n d -> h 1 d", "mean"),
                self.affine_param_codebook_decay,
            )
            self.update_with_decay(
                "codebook_variance",
                reduce(embed, "h n d -> h 1 d", var_fn),
                self.affine_param_codebook_decay,
            )

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, "h ... d -> h (...) d")

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.update_with_decay(
                "batch_mean",
                reduce(data, "h n d -> h 1 d", "mean"),
                self.affine_param_batch_decay,
            )
            self.update_with_decay(
                "batch_variance",
                reduce(data, "h n d -> h 1 d", var_fn),
                self.affine_param_batch_decay,
            )
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator

        num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)
        distributed.all_reduce(num_vectors)

        # calculate distributed mean

        batch_sum = reduce(data, "h n d -> h 1 d", "sum")
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay("batch_mean", batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, "h n d -> h 1 d", "sum")
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay(
            "batch_variance", batch_variance, self.affine_param_batch_decay
        )

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(
            zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))
        ):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(
                rearrange(samples, "... -> 1 ..."), mask.sum().item()
            )
            sampled = rearrange(sampled, "1 ... -> ...")

            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        dtype = x.dtype
        flatten, ps = pack_one(x, "h * d")

        if exists(mask):
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )

        self.init_embed_(flatten, mask=mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (
                batch_std / codebook_std
            ) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (
                    codebook_std / batch_std
                ) + self.codebook_mean

            if exists(mask):
                embed_onehot[~mask] = 0.0

            cluster_size = embed_onehot.sum(dim=1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps
            ) * self.cluster_size.sum(dim=-1, keepdim=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(
                lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind)
            )

        dist = unpack_one(dist, ps, "h * d")

        return quantize, embed_ind, dist



class FocalLoss(nn.Module):
    """
    From https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if len(input.shape) == 3:
            N, T, _ = input.shape
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(N, T, 1)).view(N, T)
        elif len(input.shape) == 2:
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class MLP(torch.nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
    ):

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1]))

        super().__init__(*layers)



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.gpt_hidden_dim % config.gpt_n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.gpt_hidden_dim, 3 * config.gpt_hidden_dim)
        # output projection
        self.c_proj = nn.Linear(config.gpt_hidden_dim, config.gpt_hidden_dim)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.gpt_block_size, config.gpt_block_size)).view(
                1, 1, config.gpt_block_size, config.gpt_block_size
            ),
        )
        self.gpt_n_head = config.gpt_n_head
        self.gpt_hidden_dim = config.gpt_hidden_dim

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (gpt_hidden_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.gpt_hidden_dim, dim=2)
        k = k.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.gpt_hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.gpt_hidden_dim)
        self.mlp = nn.Sequential(
                    nn.Linear(config.gpt_hidden_dim, 4 * config.gpt_hidden_dim),
                    nn.GELU(),
                    nn.Linear(4 * config.gpt_hidden_dim, config.gpt_hidden_dim),
                    nn.Dropout(config.dropout)
                )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




class GPT(nn.Module):
    """
    An adaptation of Andrej Karpathy's nanoGPT implementation in PyTorch.
    Original source: https://github.com/karpathy/nanoGPT

    Original License:
    MIT License

    Copyright (c) 2022 Andrej Karpathy

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Original comments:
    Full definition of a GPT Language Model, all of it in this single file.
    References:
    1) the official GPT-2 TensorFlow implementation released by OpenAI:
    https://github.com/openai/gpt-2/blob/master/src/model.py
    2) huggingface/transformers PyTorch implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """


    def __init__(self, config: VQBeTConfig):
        super().__init__()
        assert config.gpt_output_dim is not None
        assert config.gpt_block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.gpt_input_dim, config.gpt_hidden_dim),
                wpe=nn.Embedding(config.gpt_block_size, config.gpt_hidden_dim),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.gpt_n_layer)]),
                ln_f=nn.LayerNorm(config.gpt_hidden_dim),
            )
        )
        self.lm_head = nn.Linear(config.gpt_hidden_dim, config.gpt_output_dim, bias=False)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.gpt_n_layer)
                )

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, input, targets=None):
        device = input.device
        b, t, d = input.size()
        assert (
            t <= self.config.gpt_block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.gpt_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            input
        )  # token embeddings of shape (b, t, gpt_hidden_dim)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, gpt_hidden_dim)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, gpt_block_size):
        assert gpt_block_size <= self.config.gpt_block_size
        self.config.gpt_block_size = gpt_block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:gpt_block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :gpt_block_size, :gpt_block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, optimizer="Adamw", eps=None):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        if optimizer=="Adamw":
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        elif optimizer=="Adam":
            optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, eps=eps)
        else:
            raise NotImplementedError
        return optimizer
