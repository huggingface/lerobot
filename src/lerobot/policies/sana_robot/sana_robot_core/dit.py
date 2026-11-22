# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/backbone/dit.py. Import paths are rewritten to the LeRobot package;
# the token builder of SANA-Robot-base is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lerobot.policies.sana_robot.sana_robot_core.attnres import BlockAttnRes
from lerobot.policies.sana_robot.sana_robot_core.block import softmax_layer_indices
from lerobot.policies.sana_robot.sana_robot_core.embedders import (
    CaptionEmbedder,
    FinalLayer,
    TimestepEmbedder,
    VideoPatchEmbed,
)
from lerobot.policies.sana_robot.sana_robot_core.multiview import (
    SPATIAL_LAYOUT_2X2,
    SPATIAL_LAYOUT_T_SHAPE,
    batch_view_slot_ids,
    t_shape_origins,
    tile_2x2_origins,
    view_position_ids,
)
from lerobot.policies.sana_robot.sana_robot_core.norms import RMSNorm
from lerobot.policies.sana_robot.sana_robot_core.robot import ROBOT_DIM, ActionFinalLayer, MaskedProjector
from lerobot.policies.sana_robot.sana_robot_core.rope import PhysicalTimeRotary


class SanaRobotBackbone(nn.Module):
    """Hybrid Gated DeltaNet / softmax diffusion transformer over the tokens of video latents, the robot state and the actions. It owns the embedders, the layer stack and the output heads; ``SanaRobotBase`` and ``SanaRobotAR`` define the token layout and the forward pass."""

    def __init__(
        self,
        hidden_size: int = 2560,
        depth: int = 32,
        cross_attention_heads: int = 20,
        in_channels: int = 128,
        caption_channels: int = 2304,
        linear_head_dim: int = 128,
        softmax_head_dim: int = 256,
        softmax_ratio: float = 0.25,
        attn_res_block_size: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: tuple = (1, 1, 1),
        vae_temporal_stride: int = 8,
        spatial_layout: str = SPATIAL_LAYOUT_2X2,
    ) -> None:
        """Build the embedders, the layer stack and the output heads.

        Args:
            hidden_size: Token width.
            depth: Number of layers.
            cross_attention_heads: Heads of the text cross-attention.
            in_channels: Channels of the video latents.
            caption_channels: Width of the text encoder's embeddings.
            linear_head_dim: Head width of the Gated DeltaNet layers.
            softmax_head_dim: Head width of the softmax attention layers.
            softmax_ratio: Share of the layers that use softmax attention, spread evenly over the depth.
            attn_res_block_size: Layers per AttnRes block.
            mlp_ratio: Hidden width of the MLP relative to ``hidden_size``.
            patch_size: Latent patch (frames, height, width) of one video token.
            vae_temporal_stride: Video frames per latent frame of the video VAE.
            spatial_layout: Where the views of a multi-view input sit on the spatial rotary positions: ``"2x2"``
                puts every view on its tile of a 2x2 grid, ``"t_shape"`` puts the head view on the left and the
                wrist views stacked in one column to its right.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_heads = cross_attention_heads
        self.out_channels = in_channels
        self.linear_head_dim = linear_head_dim
        self.softmax_head_dim = softmax_head_dim
        self.mlp_ratio = mlp_ratio
        self.patch_size = tuple(patch_size)
        self.attn_res_block_size = attn_res_block_size
        self.vae_temporal_stride = int(vae_temporal_stride)
        self.spatial_layout = spatial_layout
        self.softmax_layer_indices = softmax_layer_indices(depth, softmax_ratio)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size)
        self.attention_y_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-05)
        self.x_embedder = VideoPatchEmbed(self.patch_size, in_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [self.build_block(index in self.softmax_layer_indices) for index in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, math.prod(self.patch_size) * self.out_channels)
        self.attn_res = BlockAttnRes(hidden_size)
        self.initialize()
        self.rope_linear = PhysicalTimeRotary(linear_head_dim)
        self.rope_softmax = PhysicalTimeRotary(softmax_head_dim)
        self.state_embed = MaskedProjector(ROBOT_DIM, hidden_size)
        self.action_embed = MaskedProjector(ROBOT_DIM, hidden_size)
        self.action_head = ActionFinalLayer(hidden_size)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def initialize(self) -> None:
        """Initialize the weights: Xavier-uniform linear layers, small normal embedders, zero AttnRes projections, and the GDN recurrence at its starting values."""

        def initialize_linear(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize_linear)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        nn.init.zeros_(self.attn_res.attn_proj.weight)
        nn.init.zeros_(self.attn_res.mlp_proj.weight)
        nn.init.zeros_(self.attn_res.final_proj.weight)
        for index, block in enumerate(self.blocks):
            if index not in self.softmax_layer_indices:
                block.attn.reset_recurrence_parameters()

    def set_latent_grid(self, video_latents: torch.Tensor) -> None:
        """Record the latent frames, height and width of ``video_latents`` in patches, for the helpers of this forward."""
        (self.f, self.h, self.w) = (
            video_latents.shape[-3] // self.patch_size[0],
            video_latents.shape[-2] // self.patch_size[1],
            video_latents.shape[-1] // self.patch_size[2],
        )

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the latent video [B, C, F, H, W] rebuilt from the video tokens of the recorded latent grid."""
        channels = self.out_channels
        (patch_f, patch_h, patch_w) = self.patch_size
        tokens = tokens.reshape(tokens.shape[0], self.f, self.h, self.w, patch_f, patch_h, patch_w, channels)
        tokens = torch.einsum("nfhwopqc->ncfohpwq", tokens)
        return tokens.reshape(tokens.shape[0], channels, self.f * patch_f, self.h * patch_h, self.w * patch_w)

    def embed_text(self, text: torch.Tensor, text_mask: torch.Tensor | None):
        """Project and normalize the grouped text embeddings [B, groups, 1, L, C]; returns ([B, groups, L, hidden], mask [B, groups, L] or None)."""
        (batch, groups, _, length, channels) = text.shape
        text = (
            self.y_embedder(text.reshape(batch * groups, 1, length, channels))
            .reshape(batch, groups, 1, length, self.hidden_size)
            .squeeze(2)
        )
        text = self.attention_y_norm(text)
        if text_mask is None:
            return (text, None)
        return (text, text_mask.to(torch.int16).reshape(batch, groups, -1))

    def video_token_timesteps(self, timestep: torch.Tensor, video_tokens: int) -> torch.Tensor:
        """Expand a per-batch or per-latent-frame video timestep to one value per video token, [B, N]."""
        timestep = timestep.reshape(timestep.shape[0], -1)
        if timestep.shape[1] == 1:
            return timestep.expand(-1, video_tokens)
        if timestep.shape[1] == self.f:
            return timestep.repeat_interleave(self.h * self.w, dim=1)
        return timestep

    def action_row_timesteps(self, data_info: dict, batch: int, action_rows: int, device) -> torch.Tensor:
        """Return ``data_info["action_timestep"]`` as one value per action row, [B, A]."""
        value = data_info.get("action_timestep")
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(device=device).reshape(batch, -1)
        if value.shape[1] == 1:
            value = value.expand(-1, action_rows)
        return value.long().float()

    def embed_timesteps(self, token_timestep: torch.Tensor) -> torch.Tensor:
        """Return the time embedding [B, 1, N, hidden] of per-token timesteps [B, 1, N]."""
        return self.t_embedder(token_timestep.flatten()).unflatten(0, token_timestep.shape)

    @staticmethod
    def batch_fps(model_fps, batch: int, device: torch.device) -> torch.Tensor:
        """Return ``model_fps`` as a float64 [B] tensor."""
        fps = model_fps.detach() if isinstance(model_fps, torch.Tensor) else torch.as_tensor(model_fps)
        if fps.ndim == 0 or fps.numel() == 1:
            fps = fps.reshape(1).expand(batch)
        return fps.to(device=device, dtype=torch.float64)

    @staticmethod
    def video_frame_stride(data_info: dict) -> int:
        """Return how many source frames one video frame of the batch spans; 1 when ``data_info`` does not say."""
        value = data_info.get("video_frame_stride")
        if value is None:
            return 1
        return int(torch.as_tensor(value).reshape(-1)[0].item())

    def video_position_ids(self, rope, data_info: dict, view_shapes: tuple, video_fps, frames: int, device):
        """Return the (time, y, x) position ids [B, N, 3] of the video tokens over ``frames`` latent frames: one view on its own grid, several views placed by ``spatial_layout``."""
        if len(view_shapes) == 1:
            ((height, width),) = view_shapes
            return rope.grid_position_ids(frames, height, width, video_fps, device)
        view_slot_ids = batch_view_slot_ids(data_info, len(view_shapes))
        if self.spatial_layout == SPATIAL_LAYOUT_T_SHAPE:
            origins = t_shape_origins(view_shapes, view_slot_ids)
        else:
            origins = tile_2x2_origins(view_shapes, view_slot_ids)
        return view_position_ids(
            frames=frames,
            view_shapes=view_shapes,
            origins=origins,
            fps=video_fps,
            base_fps=rope.base_fps,
            device=device,
        )

    def predict_actions(self, action_hidden, action_condition, data_info: dict) -> torch.Tensor:
        """Return the action prediction [B, A, 80] with the unused action dimensions zeroed."""
        action_mask = data_info["action_mask"].to(action_hidden.device)
        return self.action_head(action_hidden, action_condition).masked_fill(~action_mask, 0)
