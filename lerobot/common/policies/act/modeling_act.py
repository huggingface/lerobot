"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
import time
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.act.configuration_act import ActionChunkingTransformerConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize


class ActionChunkingTransformerPolicy(nn.Module):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)

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

    name = "act"

    def __init__(self, cfg: ActionChunkingTransformerConfig | None = None, dataset_stats=None):
        """
        Args:
            cfg: Policy configuration class instance or None, in which case the default instantiation of the
                 configuration class is used.
        """
        super().__init__()
        if cfg is None:
            cfg = ActionChunkingTransformerConfig()
        self.cfg = cfg
        self.normalize_inputs = Normalize(cfg.input_shapes, cfg.input_normalization_modes, dataset_stats)
        self.normalize_targets = Normalize(cfg.output_shapes, cfg.output_normalization_modes, dataset_stats)
        self.unnormalize_outputs = Unnormalize(
            cfg.output_shapes, cfg.output_normalization_modes, dataset_stats
        )

        # BERT style VAE encoder with input [cls, *joint_space_configuration, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        if self.cfg.use_vae:
            self.vae_encoder = _TransformerEncoder(cfg)
            self.vae_encoder_cls_embed = nn.Embedding(1, cfg.d_model)
            # Projection layer for joint-space configuration to hidden dimension.
            self.vae_encoder_robot_state_input_proj = nn.Linear(
                cfg.input_shapes["observation.state"][0], cfg.d_model
            )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                cfg.input_shapes["observation.state"][0], cfg.d_model
            )
            self.latent_dim = cfg.latent_dim
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(cfg.d_model, self.latent_dim * 2)
            # Fixed sinusoidal positional embedding the whole input to the VAE encoder. Unsqueeze for batch
            # dimension.
            self.register_buffer(
                "vae_encoder_pos_enc",
                _create_sinusoidal_position_embedding(1 + 1 + cfg.chunk_size, cfg.d_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        backbone_model = getattr(torchvision.models, cfg.vision_backbone)(
            replace_stride_with_dilation=[False, False, cfg.replace_final_stride_with_dilation],
            weights=cfg.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final feature
        # map).
        # Note: The forward method of this returns a dict: {"feature_map": output}.
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = _TransformerEncoder(cfg)
        self.decoder = _TransformerDecoder(cfg)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, robot_state, image_feature_map_pixels].
        self.encoder_robot_state_input_proj = nn.Linear(cfg.input_shapes["observation.state"][0], cfg.d_model)
        self.encoder_latent_input_proj = nn.Linear(self.latent_dim, cfg.d_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, cfg.d_model, kernel_size=1
        )
        # Transformer encoder positional embeddings.
        self.encoder_robot_and_latent_pos_embed = nn.Embedding(2, cfg.d_model)
        self.encoder_cam_feat_pos_embed = _SinusoidalPositionEmbedding2D(cfg.d_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(cfg.chunk_size, cfg.d_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(cfg.d_model, cfg.output_shapes["action"][0])

        self._reset_parameters()
        self._create_optimizer()

    def _create_optimizer(self):
        optimizer_params_dicts = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith("backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n.startswith("backbone") and p.requires_grad
                ],
                "lr": self.cfg.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_params_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.cfg.n_action_steps is not None:
            self._action_queue = deque([], maxlen=self.cfg.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], **_) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            # `_forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue effectively
            # has shape (n_action_steps, batch_size, *), hence the transpose.
            actions = self._forward(batch)[0][: self.cfg.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch, **_) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        actions_hat, (mu_hat, log_sigma_x2_hat) = self._forward(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss}
        if self.cfg.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld
            loss_dict["loss"] = l1_loss + mean_kld * self.cfg.kl_weight
        else:
            loss_dict["loss"] = l1_loss

        return loss_dict

    def update(self, batch, **_) -> dict:
        """Run the model in train mode, compute the loss, and do an optimization step."""
        start_time = time.time()
        self.train()

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        loss_dict = self.forward(batch)
        # TODO(rcadene): self.unnormalize_outputs(out_dict)
        loss = loss_dict["loss"]
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )

        self.optimizer.step()
        self.optimizer.zero_grad()

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.cfg.lr,
            "update_s": time.time() - start_time,
        }

        return info

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Stacks all the images in a batch and puts them in a new key: "observation.images".

        This function expects `batch` to have (at least):
        {
            "observation.state": (B, state_dim) batch of robot states.
            "observation.images.{name}": (B, C, H, W) tensor of images.
        }
        """
        # Stack images in the order dictated by input_shapes.
        batch["observation.images"] = torch.stack(
            [batch[k] for k in self.cfg.input_shapes if k.startswith("observation.images.")],
            dim=-4,
        )

    def _forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:

        {
            "observation.state": (B, state_dim) batch of robot states.
            "observation.images": (B, n_cameras, C, H, W) batch of images.
            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.cfg.use_vae and self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."

        self._stack_images(batch)

        batch_size = batch["observation.state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.cfg.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"]).unsqueeze(
                1
            )  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)
            vae_encoder_input = torch.cat([cls_embed, robot_state_embed, action_embed], axis=1)  # (B, S+2, D)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2)
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # Prepare all other transformer encoder inputs.
        # Camera observation features and positional embeddings.
        all_cam_features = []
        all_cam_pos_embeds = []
        images = batch["observation.images"]
        for cam_index in range(images.shape[-4]):
            cam_features = self.backbone(images[:, cam_index])["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
            all_cam_features.append(cam_features)
            all_cam_pos_embeds.append(cam_pos_embed)
        # Concatenate camera observation feature maps and positional embeddings along the width dimension.
        encoder_in = torch.cat(all_cam_features, axis=3)
        cam_pos_embed = torch.cat(all_cam_pos_embeds, axis=3)

        # Get positional embeddings for robot state and latent.
        robot_state_embed = self.encoder_robot_state_input_proj(batch["observation.state"])
        latent_embed = self.encoder_latent_input_proj(latent_sample)

        # Stack encoder input and positional embeddings moving to (S, B, C).
        encoder_in = torch.cat(
            [
                torch.stack([latent_embed, robot_state_embed], axis=0),
                encoder_in.flatten(2).permute(2, 0, 1),
            ]
        )
        pos_embed = torch.cat(
            [
                self.encoder_robot_and_latent_pos_embed.weight.unsqueeze(1),
                cam_pos_embed.flatten(2).permute(2, 0, 1),
            ],
            axis=0,
        )

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in, pos_embed=pos_embed)
        decoder_in = torch.zeros(
            (self.cfg.chunk_size, batch_size, self.cfg.d_model),
            dtype=pos_embed.dtype,
            device=pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp)
        self.load_state_dict(d)


class _TransformerEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, cfg: ActionChunkingTransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([_TransformerEncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model) if cfg.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        x = self.norm(x)
        return x


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg: ActionChunkingTransformerConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.activation = _get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
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


class _TransformerDecoder(nn.Module):
    def __init__(self, cfg: ActionChunkingTransformerConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([_TransformerDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, cfg: ActionChunkingTransformerConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.multihead_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

        self.activation = _get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

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
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def _create_sinusoidal_position_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class _SinusoidalPositionEmbedding2D(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

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
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
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
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def _get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
