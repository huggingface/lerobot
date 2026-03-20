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
"""ACT Simple + World Model Head

Combines the act_simple encoder-decoder (non-autoregressive, continuous L1 loss)
with the world model decoder from AWM. The action decoder is identical to act_simple;
the WM decoder takes *continuous* action embeddings (projected via a linear layer)
instead of discrete token embeddings, and predicts future encoder representations.

Key differences from AWM:
  - Action decoder: non-autoregressive (parallel chunk prediction), L1 loss on continuous actions.
  - WM action conditioning: continuous action chunks projected through a linear layer,
    not discrete token embeddings from a tokenizer vocabulary.
  - No action tokenizer needed.
"""

from collections import deque
from copy import copy, deepcopy
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act_simple.modeling_act_simple import (
    ACTDecoder,
    ACTEncoder,
    ACTLearnedPositionEmbedding2d,
    get_activation_fn,
)
from lerobot.policies.act_simple_with_awm_head.configuration_act_simple_with_awm_head import (
    ACTSimpleWithAWMHeadConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Helpers (shared with AWM)
# ---------------------------------------------------------------------------

class ResBlock2d(nn.Module):
    """Conv2d residual block: two 3×3 convs with a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.block(x))


class WMImageDecoder(nn.Module):
    """Debug image decoder: (S_img, B, dim_model) -> (B, C, H, W)."""

    def __init__(self, dim_model: int, image_shape: tuple[int, int, int], replace_final_stride_with_dilation: bool = False):
        super().__init__()
        C, H, W = image_shape
        stride = 16 if replace_final_stride_with_dilation else 32
        h0, w0 = H // stride, W // stride
        base_ch = 128

        self.h0 = h0
        self.w0 = w0
        self.chan_proj = nn.Conv2d(dim_model, base_ch, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 64, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(16),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            ResBlock2d(8),
            nn.ConvTranspose2d(8, C, 4, stride=2, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        S_img, B, D = z.shape
        x = z.permute(1, 2, 0).view(B, D, self.h0, self.w0)
        x = self.chan_proj(x)
        return self.decoder(x)


def _n_encoder_tokens(config: ACTSimpleWithAWMHeadConfig) -> int:
    """Compute the total number of encoder output tokens S from config."""
    n = sum([bool(config.robot_state_feature), bool(config.env_state_feature)])
    if config.image_features:
        for feat in config.image_features.values():
            C, H, W = feat.shape
            stride = 16 if config.replace_final_stride_with_dilation else 32
            n += (H // stride) * (W // stride)
    return n


def _slice_obs_batch(batch: dict[str, Tensor], idx: int) -> dict[str, Tensor]:
    """Return a batch dict with observation tensors sliced to a single temporal index."""
    result = {}
    for key, val in batch.items():
        if key.startswith("observation.") and isinstance(val, Tensor) and val.ndim >= 2:
            result[key] = val[:, idx]
        else:
            result[key] = val
    return result


def _compute_wm_loss(z_pred: Tensor, z_target: Tensor, valid_wm: Tensor) -> Tensor:
    """Cosine similarity world-model loss, masked at episode boundaries."""
    valid_wm_f = valid_wm.to(dtype=z_pred.dtype)
    valid_count = valid_wm_f.sum()
    cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean(dim=0)  # (B,)
    wm_loss = 1 - (cos_sim * valid_wm_f).sum() / valid_count.clamp(min=1.0)
    return wm_loss


def _compute_image_reconstruction_metrics(
    pred: Tensor, target: Tensor, prefix: str, valid_mask: Tensor | None = None,
) -> dict[str, float]:
    if valid_mask is not None:
        if not valid_mask.any():
            return {}
        pred = pred[valid_mask]
        target = target[valid_mask]
    mse = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
    return {f"{prefix}/mse": float(mse.item()), f"{prefix}/psnr": float(psnr.item())}


# ---------------------------------------------------------------------------
# WM Decoder (non-causal, bidirectional — reused from AWM)
# ---------------------------------------------------------------------------

class WMDecoder(nn.Module):
    """Stack of WMDecoderLayer modules. Non-causal (bidirectional self-attention)."""

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        self.layers = nn.ModuleList([WMDecoderLayer(config) for _ in range(config.n_wm_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, cross_kv, cross_pos)
        return self.norm(x)


class WMDecoderLayer(nn.Module):
    """Single WM decoder layer: bidirectional self-attention + cross-attention + FFN."""

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout,
        )

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def _add_pos(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
    ) -> Tensor:
        # Bidirectional self-attention (no causal mask).
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x
        x = self.self_attn(q, k, value=x, need_weights=False)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # Cross-attention on encoder tokens.
        x = self.multihead_attn(
            query=x,
            key=self._add_pos(cross_kv, cross_pos),
            value=cross_kv,
            need_weights=False,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        # Feed-forward.
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)

        return x


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class ACTSimpleWithAWMHeadPolicy(PreTrainedPolicy):
    """ACT Simple policy with a world model head for future state prediction.

    Action decoder: identical to act_simple (non-autoregressive, L1 loss on continuous actions).
    World model: predicts future encoder representations from current encoder + action chunk.
    Training loss: L1(action) + wm_weight * cosine_wm_loss.
    """

    config_class = ACTSimpleWithAWMHeadConfig
    name = "act_simple_with_awm_head"

    def __init__(self, config: ACTSimpleWithAWMHeadConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTSimpleWithAWMHead(config)
        self._train_step = 0
        self._ema_step = 0
        self._pending_ema_momentum = None

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def update(self):
        if self._pending_ema_momentum is not None:
            self.model.update_ema(self._pending_ema_momentum)
            self._ema_step += 1
            self._pending_ema_momentum = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        return self.model.predict(batch)

    @torch.no_grad()
    def visualize(self, batch: dict[str, Tensor], n_pairs: int = 12) -> dict[str, Tensor] | None:
        """Generate WM image reconstruction pairs for debugging.

        Returns a dict with keys "curr" and "next", each (N, C, H, 2W) float in [0, 1]:
          curr: GT (left) | decoded current encoder tokens (right)
          next: GT (left) | decoded WM future prediction (right)

        Returns None when no image features are configured.
        """
        if not self.config.image_features or not hasattr(self.model, "wm_image_decoder"):
            return None

        was_training = self.training
        self.eval()

        n = min(n_pairs, batch[ACTION].shape[0])

        def _prep(raw_batch: dict, idx: int) -> dict:
            sliced = _slice_obs_batch(raw_batch, idx)
            d = {k: v[:n] if isinstance(v, Tensor) else v for k, v in sliced.items()}
            d = dict(d)
            d[OBS_IMAGES] = [d[k][:n] for k in self.config.image_features]
            return d

        curr_batch = _prep(batch, 0)
        next_batch = _prep(batch, 1)

        n_1d = self.model.n_1d_tokens
        s_img = self.model.img_tokens_per_cam

        # Encode current obs.
        batch_size, encoder_out, encoder_pos, curr_encoder_in = self.model._encode(curr_batch)

        # Current observation: decode from pre-transformer encoder input tokens.
        curr_img_z = curr_encoder_in[n_1d : n_1d + s_img]
        if self.config.normalize_wm_representations:
            curr_img_z = F.normalize(curr_img_z, dim=-1)
        decoded_curr = self.model.wm_image_decoder(curr_img_z)
        gt_curr = curr_batch[OBS_IMAGES][0]

        # Run WM decoder to get future state prediction.
        actions = batch[ACTION][:n]
        T = actions.shape[1]
        action_embeds = self.model.wm_action_proj(actions).transpose(0, 1)
        wm_action_pos = self.model.wm_action_pos_embed.weight[:T].unsqueeze(1)
        S = self.model.n_encoder_tokens
        query_pos = self.model.wm_query_pos_embed.weight.unsqueeze(1)
        queries = (self.model.wm_query_tokens + query_pos).expand(-1, batch_size, -1)
        wm_in = torch.cat([queries, action_embeds + wm_action_pos], dim=0)
        wm_encoder_in = curr_encoder_in.detach() if self.config.detach_encoder_from_wm else curr_encoder_in
        wm_cross_kv = self.model.wm_cross_attn_proj(wm_encoder_in)
        wm_cross_pos = encoder_pos
        wm_out = self.model.wm_decoder(wm_in, wm_cross_kv, wm_cross_pos)
        z_pred = self.model.wm_proj_head(wm_out[:S])

        # Future observation: decode from WM predicted tokens.
        next_img_z = z_pred[n_1d : n_1d + s_img]
        if self.config.normalize_wm_representations:
            next_img_z = F.normalize(next_img_z, dim=-1)
        decoded_next = self.model.wm_image_decoder(next_img_z)
        gt_next = _prep(batch, 1)[OBS_IMAGES][0]

        def _to_01(t: Tensor) -> Tensor:
            B = t.shape[0]
            t_flat = t.view(B, -1)
            lo = t_flat.min(dim=1).values.view(B, 1, 1, 1)
            hi = t_flat.max(dim=1).values.view(B, 1, 1, 1)
            return ((t - lo) / (hi - lo + 1e-8)).clamp(0, 1)

        if was_training:
            self.train()

        curr = torch.cat([_to_01(gt_curr), _to_01(decoded_curr)], dim=3)
        next_ = torch.cat([_to_01(gt_next), _to_01(decoded_next)], dim=3)
        return {"curr": curr.cpu(), "next": next_.cpu()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass: BC loss + WM loss."""
        # Split temporally-stacked obs (B, 2, ...) into current (t) and next (t+H).
        curr_batch = _slice_obs_batch(batch, 0)
        next_batch = _slice_obs_batch(batch, 1)

        if self.config.image_features:
            curr_batch = dict(curr_batch)
            curr_batch[OBS_IMAGES] = [curr_batch[key] for key in self.config.image_features]
            next_batch = dict(next_batch)
            next_batch[OBS_IMAGES] = [next_batch[key] for key in self.config.image_features]

        # Episode-boundary mask: True where t+H is beyond the episode end.
        next_obs_is_pad = batch.get(
            "observation.state_is_pad",
            batch.get("observation.environment_state_is_pad"),
        )

        actions_hat, wm_tensors = self.model(curr_batch, next_batch)

        # BC loss: L1 on continuous actions, masked by action padding.
        action_loss = (
            F.l1_loss(curr_batch[ACTION], actions_hat, reduction="none")
            * ~curr_batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        # WM loss.
        z_pred, z_target, decoded_curr, gt_curr_img = wm_tensors
        valid_wm = ~next_obs_is_pad[:, 1]  # (B,)
        wm_loss = _compute_wm_loss(z_pred, z_target, valid_wm)

        if self.config.wm_warmup_steps > 0 and self.training:
            warmup_frac = min(self._train_step / self.config.wm_warmup_steps, 1.0)
            effective_wm_weight = self.config.wm_loss_weight * warmup_frac
        else:
            effective_wm_weight = self.config.wm_loss_weight

        loss = action_loss + effective_wm_weight * wm_loss
        info = {
            "action_loss": action_loss.item(),
            "wm_loss": wm_loss.item(),
            "effective_wm_loss_weight": effective_wm_weight,
            "z_target_norm": z_target.norm(dim=-1).mean().item(),
            "z_pred_norm": z_pred.norm(dim=-1).mean().item(),
            "z_pred_batch_std": z_pred.std(dim=1).mean().item(),
            "z_target_batch_std": z_target.std(dim=1).mean().item(),
        }

        with torch.no_grad():
            info["wm_cosine_sim"] = F.cosine_similarity(z_pred, z_target, dim=-1).mean().item()
            info["z_pred_target_norm_ratio"] = (
                z_pred.norm(dim=-1).mean() / z_target.norm(dim=-1).mean().clamp(min=1e-8)
            ).item()

        # Image reconstruction loss on current obs.
        if decoded_curr is not None and gt_curr_img is not None:
            decoder_loss = F.mse_loss(decoded_curr, gt_curr_img)
            loss = loss + self.config.decoder_loss_weight * decoder_loss
            info["decoder_loss"] = decoder_loss.item()

            with torch.no_grad():
                info.update(
                    _compute_image_reconstruction_metrics(
                        decoded_curr.detach(), gt_curr_img.detach(), prefix="wm_curr",
                    )
                )

                next_img_z = z_pred[
                    self.model.n_1d_tokens : self.model.n_1d_tokens + self.model.img_tokens_per_cam
                ]
                decoded_next = self.model.wm_image_decoder(next_img_z.detach())
                gt_next_img = next_batch[OBS_IMAGES][0].detach()
                info.update(
                    _compute_image_reconstruction_metrics(
                        decoded_next, gt_next_img, prefix="wm_next", valid_mask=valid_wm,
                    )
                )

        if self.config.use_ema_target and self.training:
            t = min(self._ema_step / max(self.config.ema_anneal_steps, 1), 1.0)
            momentum = self.config.ema_momentum + t * (
                self.config.ema_momentum_end - self.config.ema_momentum
            )
            self._pending_ema_momentum = momentum
            info["ema_momentum"] = momentum

        if self.training:
            self._train_step += 1

        info["loss"] = loss.item()
        return loss, info


# ---------------------------------------------------------------------------
# Core network
# ---------------------------------------------------------------------------

class ACTSimpleWithAWMHead(nn.Module):
    """Core network: ACT Simple encoder-decoder + WM decoder head.

    Encoder: identical to ACT Simple (ResNet backbone + transformer encoder).
    Action decoder: identical to ACT Simple (non-autoregressive, DETR-style queries).
    WM decoder: non-causal transformer that takes [S query tokens, T action tokens]
        and predicts future encoder representations.
    """

    def __init__(self, config: ACTSimpleWithAWMHeadConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Vision backbone
        # ------------------------------------------------------------------
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ------------------------------------------------------------------
        # Transformer encoder (shared between action decoder and WM)
        # ------------------------------------------------------------------
        self.encoder = ACTEncoder(config)

        # ------------------------------------------------------------------
        # Action decoder (identical to act_simple)
        # ------------------------------------------------------------------
        self.action_decoder = ACTDecoder(config)

        # ------------------------------------------------------------------
        # Encoder input projections
        # ------------------------------------------------------------------
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        if config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # ------------------------------------------------------------------
        # Encoder positional embeddings
        # ------------------------------------------------------------------
        n_1d_tokens = sum([bool(config.robot_state_feature), bool(config.env_state_feature)])
        self.n_1d_tokens = n_1d_tokens
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if config.image_features:
            C, H, W = config.image_features["observation.image"].shape
            self.encoder_cam_feat_pos_embed = ACTLearnedPositionEmbedding2d(H, W, config.dim_model)

        # ------------------------------------------------------------------
        # Action decoder positional embeddings + head (act_simple style)
        # ------------------------------------------------------------------
        self.action_decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        action_dim = config.action_feature.shape[0]
        self.action_head = nn.Linear(config.dim_model, action_dim)

        # ------------------------------------------------------------------
        # World model decoder
        # ------------------------------------------------------------------
        self.wm_decoder = WMDecoder(config)

        # WM action conditioning: project continuous action chunks to dim_model.
        self.wm_action_proj = nn.Linear(action_dim, config.dim_model)

        # Positional embeddings for WM action tokens.
        self.wm_action_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # S learnable query tokens — one per encoder output token.
        n_enc = _n_encoder_tokens(config)
        self.n_encoder_tokens = n_enc
        self.wm_query_tokens = nn.Parameter(torch.zeros(n_enc, 1, config.dim_model))
        nn.init.trunc_normal_(self.wm_query_tokens, std=0.02)
        self.wm_query_pos_embed = nn.Embedding(n_enc, config.dim_model)

        if config.image_features:
            stride = 16 if config.replace_final_stride_with_dilation else 32
            C, H, W = next(iter(config.image_features.values())).shape
            self.img_tokens_per_cam = (H // stride) * (W // stride)

        # WM projection head: maps query outputs to predicted next-state latent tokens.
        self.wm_proj_head = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # WM cross-attention projection for encoder input tokens.
        self.wm_cross_attn_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # ------------------------------------------------------------------
        # Image decoder (debug only)
        # ------------------------------------------------------------------
        if config.image_features:
            first_feat = next(iter(config.image_features.values()))
            self.wm_image_decoder = WMImageDecoder(
                config.dim_model, tuple(first_feat.shape), config.replace_final_stride_with_dilation
            )

        self._reset_parameters()

        if config.use_ema_target:
            self._build_ema_encoder()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.use_ema_target:
            self._set_ema_eval_mode()
        return self

    def _reset_parameters(self):
        modules = [
            self.encoder.parameters(),
            self.action_decoder.parameters(),
            self.wm_decoder.parameters(),
            self.wm_proj_head.parameters(),
            self.wm_cross_attn_proj.parameters(),
            self.wm_query_pos_embed.parameters(),
        ]
        if hasattr(self, "wm_image_decoder"):
            modules.append(self.wm_image_decoder.parameters())
        for p in chain(*modules):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_ema_encoder(self):
        if hasattr(self, "backbone"):
            self.ema_backbone = deepcopy(self.backbone)
        self.ema_encoder = deepcopy(self.encoder)
        if hasattr(self, "encoder_robot_state_input_proj"):
            self.ema_encoder_robot_state_input_proj = deepcopy(self.encoder_robot_state_input_proj)
        if hasattr(self, "encoder_env_state_input_proj"):
            self.ema_encoder_env_state_input_proj = deepcopy(self.encoder_env_state_input_proj)
        if hasattr(self, "encoder_img_feat_input_proj"):
            self.ema_encoder_img_feat_input_proj = deepcopy(self.encoder_img_feat_input_proj)
        self.ema_encoder_1d_feature_pos_embed = deepcopy(self.encoder_1d_feature_pos_embed)
        if hasattr(self, "encoder_cam_feat_pos_embed"):
            self.ema_encoder_cam_feat_pos_embed = deepcopy(self.encoder_cam_feat_pos_embed)

        for name, param in self.named_parameters():
            if name.startswith("ema_"):
                param.requires_grad = False
        self._set_ema_eval_mode()

    def _set_ema_eval_mode(self):
        for name, module in self.named_children():
            if name.startswith("ema_"):
                module.eval()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, batch: dict[str, Tensor]) -> tuple[int, Tensor, Tensor, Tensor]:
        """Run the encoder.

        Returns:
            batch_size, encoder_out (S, B, dim_model), encoder_pos (S, 1, dim_model),
            encoder_in (S, B, dim_model) — pre-transformer input tokens.
        """
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        else:
            batch_size = batch[OBS_STATE].shape[0]

        encoder_in_tokens: list[Tensor] = []
        encoder_in_pos_embed: list[Tensor] = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        return batch_size, encoder_out, encoder_in_pos_embed, encoder_in_tokens

    @torch.no_grad()
    def _encode_ema(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations with EMA modules and return pre-transformer encoder tokens."""
        encoder_in_tokens: list[Tensor] = []
        encoder_in_pos_embed: list[Tensor] = list(self.ema_encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.ema_encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.ema_encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.ema_backbone(img)["feature_map"]
                cam_pos_embed = self.ema_encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.ema_encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        _ = self.ema_encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_in_tokens

    @torch.no_grad()
    def update_ema(self, momentum: float):
        if not self.config.use_ema_target:
            return

        ema_pairs = []
        if hasattr(self, "ema_backbone"):
            ema_pairs.extend(zip(self.backbone.parameters(), self.ema_backbone.parameters()))
        ema_pairs.extend(zip(self.encoder.parameters(), self.ema_encoder.parameters()))
        if hasattr(self, "ema_encoder_robot_state_input_proj"):
            ema_pairs.extend(
                zip(self.encoder_robot_state_input_proj.parameters(),
                    self.ema_encoder_robot_state_input_proj.parameters())
            )
        if hasattr(self, "ema_encoder_env_state_input_proj"):
            ema_pairs.extend(
                zip(self.encoder_env_state_input_proj.parameters(),
                    self.ema_encoder_env_state_input_proj.parameters())
            )
        if hasattr(self, "ema_encoder_img_feat_input_proj"):
            ema_pairs.extend(
                zip(self.encoder_img_feat_input_proj.parameters(),
                    self.ema_encoder_img_feat_input_proj.parameters())
            )
        ema_pairs.extend(
            zip(self.encoder_1d_feature_pos_embed.parameters(),
                self.ema_encoder_1d_feature_pos_embed.parameters())
        )
        if hasattr(self, "ema_encoder_cam_feat_pos_embed"):
            ema_pairs.extend(
                zip(self.encoder_cam_feat_pos_embed.parameters(),
                    self.ema_encoder_cam_feat_pos_embed.parameters())
            )

        for online_p, ema_p in ema_pairs:
            ema_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        next_batch: dict[str, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor | None, Tensor | None]]:
        """Training forward: action prediction + world model.

        Returns:
            actions_hat: (B, T, action_dim) — predicted continuous actions.
            wm_tensors:  (z_pred, z_target, decoded_curr, gt_curr_img)
        """
        batch_size, encoder_out, encoder_pos, encoder_in = self._encode(batch)

        # === Action decoder (act_simple style) ===
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_pos.dtype,
            device=encoder_pos.device,
        )
        decoder_out = self.action_decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos,
            decoder_pos_embed=self.action_decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, T, dim_model)
        actions_hat = self.action_head(decoder_out)  # (B, T, action_dim)

        # === World model decoder ===
        # Target: encoder input tokens of the next observation (pre-transformer, stop-gradient).
        if self.config.use_ema_target:
            z_target = self._encode_ema(next_batch)
        else:
            _, _, _, next_encoder_in = self._encode(next_batch)
            z_target = next_encoder_in.detach()  # (S, B, dim_model)
        if self.config.normalize_wm_representations:
            z_target = F.normalize(z_target, dim=-1)

        # WM input: [S query tokens, T continuous action tokens].
        actions = batch[ACTION]  # (B, T, action_dim)
        T = actions.shape[1]
        action_embeds = self.wm_action_proj(actions).transpose(0, 1)  # (T, B, dim_model)
        wm_action_pos = self.wm_action_pos_embed.weight[:T].unsqueeze(1)  # (T, 1, dim_model)
        query_pos = self.wm_query_pos_embed.weight.unsqueeze(1)  # (S, 1, dim_model)
        queries = (self.wm_query_tokens + query_pos).expand(-1, batch_size, -1)  # (S, B, dim_model)
        wm_in = torch.cat([queries, action_embeds + wm_action_pos], dim=0)  # (S+T, B, dim_model)

        # WM cross-attends to encoder INPUT tokens (pre-transformer) — same space as target.
        S = self.n_encoder_tokens
        wm_encoder_in = encoder_in.detach() if self.config.detach_encoder_from_wm else encoder_in
        wm_cross_kv = self.wm_cross_attn_proj(wm_encoder_in)  # (S, B, dim_model)
        wm_cross_pos = encoder_pos  # (S, 1, dim_model)
        wm_out = self.wm_decoder(wm_in, wm_cross_kv, wm_cross_pos)  # (S+T, B, dim_model)
        z_pred = self.wm_proj_head(wm_out[:S])  # (S, B, dim_model)
        if self.config.normalize_wm_representations:
            z_pred = F.normalize(z_pred, dim=-1)

        # Image decoder.
        decoded_curr, gt_curr_img = None, None
        if hasattr(self, "wm_image_decoder") and OBS_IMAGES in batch:
            curr_img_z = encoder_in[self.n_1d_tokens : self.n_1d_tokens + self.img_tokens_per_cam]
            if self.config.normalize_wm_representations:
                curr_img_z = F.normalize(curr_img_z, dim=-1)
            decoded_curr = self.wm_image_decoder(curr_img_z.detach())
            gt_curr_img = batch[OBS_IMAGES][0].detach()

        wm_tensors = (z_pred, z_target, decoded_curr, gt_curr_img)
        return actions_hat, wm_tensors

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        """Inference: predict action chunk (no WM needed)."""
        batch_size, encoder_out, encoder_pos, _ = self._encode(batch)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_pos.dtype,
            device=encoder_pos.device,
        )
        decoder_out = self.action_decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos,
            decoder_pos_embed=self.action_decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        return self.action_head(decoder_out)
