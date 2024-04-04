"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
"""

import logging
import math
import time
from itertools import chain
from typing import Callable, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
import torchvision.transforms as transforms
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.abstract import AbstractPolicy
from lerobot.common.utils import get_safe_torch_device


class ActionChunkingTransformerPolicy(AbstractPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (https://arxiv.org/abs/2304.13705).
    """

    name = "act"

    def __init__(self, cfg, device, n_action_steps=1):
        """
        Args:
            vae: Whether to use the variational objective. TODO(now): Give more details.
            temporal_agg: Whether to do temporal aggregation. For each timestep during rollout, the action
                returned as an exponential moving average of previously generated actions for that timestep.
            n_obs_steps: Number of time steps worth of observation to use as input.
            horizon: The number of actions to generate in one forward pass.
            kl_weight: Weight for KL divergence. Defaults to None. Only applicable when using the variational
                objective.
            batch_size: Training batch size.
            grad_clip_norm: Optionally clip the gradients to have this value as the norm at most. Defaults to
                None meaning gradient clipping is not applied.
            lr: Learning rate.
        """
        super().__init__(n_action_steps)
        self.cfg = cfg
        self.n_action_steps = n_action_steps
        self.device = get_safe_torch_device(device)

        self.model = ActionChunkingTransformer(
            cfg,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            camera_names=cfg.camera_names,
            use_vae=cfg.vae,
        )

        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not n.startswith("backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n.startswith("backbone") and p.requires_grad
                ],
                "lr": cfg.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.kl_weight = self.cfg.kl_weight
        logging.info(f"KL Weight {self.kl_weight}")
        self.to(self.device)

    def update(self, replay_buffer, step):
        del step

        self.train()

        num_slices = self.cfg.batch_size
        batch_size = self.cfg.horizon * num_slices

        assert batch_size % self.cfg.horizon == 0
        assert batch_size % num_slices == 0

        def process_batch(batch, horizon, num_slices):
            # trajectory t = 64, horizon h = 16
            # (t h) ... -> t h ...
            batch = batch.reshape(num_slices, horizon)

            image = batch["observation", "image", "top"]
            image = image[:, 0]  # first observation t=0
            # batch, num_cam, channel, height, width
            image = image.unsqueeze(1)
            assert image.ndim == 5
            image = image.float()

            state = batch["observation", "state"]
            state = state[:, 0]  # first observation t=0
            # batch, qpos_dim
            assert state.ndim == 2

            action = batch["action"]
            # batch, seq, action_dim
            assert action.ndim == 3
            assert action.shape[1] == horizon

            if self.cfg.n_obs_steps > 1:
                raise NotImplementedError()
                # # keep first n observations of the slice corresponding to t=[-1,0]
                # image = image[:, : self.cfg.n_obs_steps]
                # state = state[:, : self.cfg.n_obs_steps]

            out = {
                "obs": {
                    "image": image.to(self.device, non_blocking=True),
                    "agent_pos": state.to(self.device, non_blocking=True),
                },
                "action": action.to(self.device, non_blocking=True),
            }
            return out

        start_time = time.time()

        batch = replay_buffer.sample(batch_size)
        batch = process_batch(batch, self.cfg.horizon, num_slices)

        data_s = time.time() - start_time
        print(data_s)

        loss = self.compute_loss(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )

        self.optimizer.step()
        self.optimizer.zero_grad()

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.cfg.lr,
            "data_s": data_s,
            "update_s": time.time() - start_time,
        }

        return info

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)

    def compute_loss(self, batch):
        loss_dict = self._forward(
            qpos=batch["obs"]["agent_pos"],
            image=batch["obs"]["image"],
            actions=batch["action"],
        )
        loss = loss_dict["loss"]
        return loss

    @torch.no_grad()
    def select_actions(self, observation, step_count):
        # TODO(rcadene): remove unused step_count
        del step_count

        self.eval()

        # TODO(rcadene): remove hack
        # add 1 camera dimension
        observation["image", "top"] = observation["image", "top"].unsqueeze(1)

        obs_dict = {
            "image": observation["image", "top"],
            "agent_pos": observation["state"],
        }
        # qpos = obs_dict["agent_pos"]
        # img = obs_dict["image"]
        # qpos_ = torch.load('/tmp/qpos.pth')
        # img_ = torch.load('/tmp/curr_image.pth')
        # out_ = torch.load('/tmp/out.pth')
        # import cv2, numpy as np
        # cv2.imwrite("ours.png", (obs_dict["image"][0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite("theirs.png", (img_[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        # out = self._forward(qpos_, img_)
        # breakpoint()
        action = self._forward(qpos=obs_dict["agent_pos"] * 0.182, image=obs_dict["image"])

        if self.cfg.temporal_agg:
            # TODO(rcadene): implement temporal aggregation
            raise NotImplementedError()
            # all_time_actions[[t], t:t+num_queries] = action
            # actions_for_curr_step = all_time_actions[:, t]
            # actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            # actions_for_curr_step = actions_for_curr_step[actions_populated]
            # k = 0.01
            # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            # exp_weights = exp_weights / exp_weights.sum()
            # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

        # take first predicted action or n first actions
        action = action[: self.n_action_steps]
        return action

    def _forward(self, qpos, image, actions=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        is_training = actions is not None
        if is_training:  # training time
            actions = actions[:, : self.model.horizon]

            a_hat, (mu, log_sigma_x2) = self.model(qpos, image, actions)

            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = all_l1.mean()

            loss_dict = {}
            loss_dict["l1"] = l1
            if self.cfg.vae:
                # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
                # each dimension independently, we sum over the latent dimension to get the total
                # KL-divergence per batch element, then take the mean over the batch.
                # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
                mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - (log_sigma_x2).exp())).sum(-1).mean()
                loss_dict["kl"] = mean_kld
                loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            else:
                loss_dict["loss"] = loss_dict["l1"]
            return loss_dict
        else:
            action, _ = self.model(qpos, image)  # no action, sample from prior
            return action


def create_sinusoidal_position_embedding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# TODO(alexander-soare) move all this code into the policy when we have the policy API established.
class ActionChunkingTransformer(nn.Module):
    """
    Action Chunking Transformer as per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
    (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └───▲───┘             │
                  │       │     │     │                 │
                inputs    └─────┼─────┘                 │
                                │                       │
                                └───────────────────────┘
    """

    def __init__(self, args, state_dim, action_dim, horizon, camera_names, use_vae):
        """Initializes the model.
        Parameters:
            state_dim: robot state dimension of the environment
            horizon: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.

        Args:
            state_dim: Robot positional state dimension.
            action_dim: Action dimension.
            horizon: The number of actions to generate in one forward pass.
            use_vae: Whether to use the variational objective. TODO(now): Give more details.
        """
        super().__init__()

        self.camera_names = camera_names
        self.use_vae = use_vae
        self.horizon = horizon
        self.hidden_dim = args.hidden_dim

        transformer_common_kwargs = dict(  # noqa: C408
            d_model=self.hidden_dim,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation,
            normalize_before=args.pre_norm,
        )

        # BERT style VAE encoder with input [cls, *joint_space_configuration, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        if use_vae:
            # TODO(now): args.enc_layers shouldn't be shared with the transformer decoder
            self.vae_encoder = TransformerEncoder(num_layers=args.enc_layers, **transformer_common_kwargs)
            self.cls_embed = nn.Embedding(1, self.hidden_dim)
            # Projection layer for joint-space configuration to hidden dimension.
            self.vae_encoder_robot_state_input_proj = nn.Linear(state_dim, self.hidden_dim)
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(state_dim, self.hidden_dim)
            # Final size of latent z. TODO(now): Add to hyperparams.
            self.latent_dim = 32
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(self.hidden_dim, self.latent_dim * 2)
            # Fixed sinusoidal positional embedding the whole input to the VAE encoder.
            self.register_buffer(
                "vae_encoder_pos_enc", create_sinusoidal_position_embedding(1 + 1 + horizon, self.hidden_dim)
            )

        # Backbone for image feature extraction.
        self.backbone_position_embedding = SinusoidalPositionEmbedding2D(self.hidden_dim // 2)
        backbone_model = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation],
            pretrained=True,  # TODO(now): Add pretrained option
            norm_layer=FrozenBatchNorm2d,
        )
        # Note: The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = TransformerEncoder(num_layers=args.enc_layers, **transformer_common_kwargs)
        self.decoder = TransformerDecoder(num_layers=args.dec_layers, **transformer_common_kwargs)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, robot_state, image_feature_map_pixels].
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, self.hidden_dim, kernel_size=1
        )
        self.encoder_robot_state_input_proj = nn.Linear(state_dim, self.hidden_dim)
        self.encoder_latent_input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        # TODO(now): Fix this nonsense. One positional embedding is needed. We should extract the image
        # feature dimension with a dry run.
        self.additional_pos_embed = nn.Embedding(
            2, self.hidden_dim
        )  # learned position embedding for proprio and latent

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed_embed = nn.Embedding(horizon, self.hidden_dim)
        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(self.hidden_dim, action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, robot_state, image, actions=None):
        """
        Args:
            robot_state: (B, J) batch of robot joint configurations.
            image: (B, N, C, H, W) batch of N camera frames.
            actions: (B, S, A) batch of actions from the target dataset which must be provided if the
                VAE is enabled and the model is in training mode.
        """
        if self.use_vae and self.training:
            assert (
                actions is not None
            ), "actions must be provided when using the variational objective in training mode."

        batch_size, _ = robot_state.shape

        # Prepare the latent for input to the transformer.
        if self.use_vae and actions is not None:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(self.cls_embed.weight, "1 d -> b 1 d", b=batch_size)  # (B, 1, D)
            robot_state_embed = self.vae_encoder_robot_state_input_proj(robot_state).unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(actions)  # (B, S, D)
            vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)  # (B, S+2, D)
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            # Prepare fixed positional embedding.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)
            # Forward pass through VAE encoder and sample the latent with the reparameterization trick.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2), pos=pos_embed.permute(1, 0, 2)
            )[0]  # (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]
            # Use reparameterization trick to sample from the latent's PDF.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.latent_dim], dtype=torch.float32).to(
                robot_state.device
            )

        # Prepare all other transformer inputs.
        # Image observation features and position embeddings.
        all_cam_features = []
        all_cam_pos = []
        for cam_id, _ in enumerate(self.camera_names):
            cam_features = self.backbone(image[:, cam_id])["feature_map"]
            pos = self.backbone_position_embedding(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
            all_cam_features.append(cam_features)
            all_cam_pos.append(pos)
        # Concatenate image observation feature maps along the width dimension.
        encoder_in = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        robot_state_embed = self.encoder_robot_state_input_proj(robot_state)
        latent_embed = self.encoder_latent_input_proj(latent_sample)

        # TODO(now): Explain all of this madness.
        encoder_in = torch.cat(
            [
                torch.stack([latent_embed, robot_state_embed], axis=0),
                encoder_in.flatten(2).permute(2, 0, 1),
            ]
        )
        pos_embed = torch.cat(
            [self.additional_pos_embed.weight.unsqueeze(1), pos.flatten(2).permute(2, 0, 1)], axis=0
        )

        encoder_out = self.encoder(encoder_in, pos=pos_embed)
        decoder_in = torch.zeros(
            (self.horizon, batch_size, self.hidden_dim), dtype=pos_embed.dtype, device=pos_embed.device
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=pos_embed,
            decoder_pos_embed=self.decoder_pos_embed_embed.weight.unsqueeze(1),
        ).transpose(0, 1)  # back to (B, S, C)

        actions = self.action_head(decoder_out)
        return actions, [mu, log_sigma_x2]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, normalize_before
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if normalize_before else nn.Identity()

    def forward(self, x, pos: Optional[Tensor] = None):
        for layer in self.layers:
            x = layer(x, pos=pos)
        x = self.norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, x, pos_embed: Optional[Tensor] = None):
        skip = x
        if self.normalize_before:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.normalize_before:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, normalize_before
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x, encoder_out, decoder_pos_embed: Tensor | None = None, encoder_pos_embed: Tensor | None = None
    ):
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.normalize_before:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.normalize_before:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.normalize_before:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.normalize_before:
            x = self.norm3(x)
        return x


class SinusoidalPositionEmbedding2D(nn.Module):
    """Sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, [0]])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def _get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
