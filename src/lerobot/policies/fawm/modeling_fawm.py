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
"""FAWM Policy — Flow-Matching Action Head with World-Model Auxiliary Loss

Replaces the discrete autoregressive action decoder in AWM with a continuous
flow-matching action head. The world model head, encoder, and backbone are
identical to AWM.

Key differences from AWM:
  1. Continuous flow-matching action head: denoises a full action chunk in
     parallel via bidirectional self-attention (no causal mask).
  2. No tokenizer: actions remain continuous throughout.
  3. Training loss: MSE on velocity field prediction (conditional OT).
  4. Inference: Euler integration from noise to actions.
"""

import math
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
    ACTEncoder,
    ACTLearnedPositionEmbedding2d,
    get_activation_fn,
)
from lerobot.policies.fawm.configuration_fawm import FAWMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class WMImageDecoder(nn.Module):
    """Lightweight debug image decoder: (S_img, B, dim_model) → (B, C, H, W)."""

    def __init__(self, dim_model: int, image_shape: tuple[int, int, int], replace_final_stride_with_dilation: bool = False):
        super().__init__()
        C, H, W = image_shape
        stride = 16 if replace_final_stride_with_dilation else 32
        h0, w0 = H // stride, W // stride
        base_ch = 32

        self.h0 = h0
        self.w0 = w0
        self.base_ch = base_ch
        self.chan_proj = nn.Conv2d(dim_model, base_ch, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,       8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d( 8,       4, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d( 4,       2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d( 2,       C, 4, stride=2, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        S_img, B, D = z.shape
        x = z.permute(1, 2, 0).view(B, D, self.h0, self.w0)
        x = self.chan_proj(x)
        return self.decoder(x)


def _n_encoder_tokens(config: FAWMConfig) -> int:
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


def _compute_wm_loss(
    z_pred: Tensor,
    z_target: Tensor,
    valid_wm: Tensor,
    config: FAWMConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the configured world-model loss and auxiliary metrics."""
    valid_wm_f = valid_wm.to(dtype=z_pred.dtype)
    valid_count = valid_wm_f.sum()

    if config.use_normalized_mse_wm_loss:
        z_pred_norm = F.normalize(z_pred, dim=-1)
        z_target_norm = F.normalize(z_target, dim=-1)

        mse_per_batch = F.mse_loss(z_pred_norm, z_target_norm, reduction="none").mean(dim=(0, 2))
        wm_reconstruction_loss = (mse_per_batch * valid_wm_f).sum() / valid_count.clamp(min=1.0)

        if valid_wm.sum() > 1:
            std_pred = z_pred_norm[:, valid_wm, :].std(dim=1, correction=0)
            wm_variance_loss = F.relu(1.0 - std_pred).mean()
        else:
            wm_variance_loss = z_pred.new_zeros(())

        wm_loss = wm_reconstruction_loss + config.wm_variance_loss_weight * wm_variance_loss
        metrics = {
            "wm_reconstruction_loss": wm_reconstruction_loss,
            "wm_variance_loss": wm_variance_loss,
        }
        return wm_loss, metrics

    cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean(dim=0)
    wm_loss = 1 - (cos_sim * valid_wm_f).sum() / valid_count.clamp(min=1.0)
    return wm_loss, {}


def _compute_image_reconstruction_metrics(
    pred: Tensor,
    target: Tensor,
    prefix: str,
    valid_mask: Tensor | None = None,
) -> dict[str, float]:
    """Compute scalar image reconstruction metrics in normalized pixel space."""
    if valid_mask is not None:
        if not valid_mask.any():
            return {}
        pred = pred[valid_mask]
        target = target[valid_mask]

    mse = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
    return {
        f"{prefix}/mse": float(mse.item()),
        f"{prefix}/psnr": float(psnr.item()),
    }


def sinusoidal_timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal timestep embedding.

    Args:
        t: (B, 1, 1) or (B,) scalar timesteps in [0, 1].
        dim: Embedding dimension.

    Returns:
        (B, dim) embedding.
    """
    t = t.reshape(-1) * 1000.0  # scale [0,1] → [0,1000] for proper frequency coverage
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim)
    args = t[:, None] * freqs[None, :]  # (B, half_dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim) or (B, dim-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class FAWMPolicy(PreTrainedPolicy):
    """FAWM: Flow-matching Action Chunking Transformer with world-model auxiliary loss.

    At training time, the decoder denoises the full action chunk in parallel with
    flow-matching (MSE on velocity field). At inference time, Euler integration
    from noise to actions.
    """

    config_class = FAWMConfig
    name = "fawm"

    def __init__(self, config: FAWMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = FAWM(config)
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
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions via flow-matching Euler integration."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        return self.model.predict_flow(batch, num_steps=self.config.num_inference_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass; returns combined action (flow) + world model loss."""
        curr_batch = _slice_obs_batch(batch, 0)
        next_batch = _slice_obs_batch(batch, 1)

        if self.config.image_features:
            curr_batch = dict(curr_batch)
            curr_batch[OBS_IMAGES] = [curr_batch[key] for key in self.config.image_features]
            next_batch = dict(next_batch)
            next_batch[OBS_IMAGES] = [next_batch[key] for key in self.config.image_features]

        next_obs_is_pad = batch.get(
            "observation.state_is_pad",
            batch.get("observation.environment_state_is_pad"),
        )

        action_loss, v_pred, v_target, t_sampled, wm_tensors = self.model(curr_batch, next_batch)

        # World model loss — masked at episode boundaries.
        z_pred, z_target, decoded_curr, gt_curr_img = wm_tensors
        valid_wm = ~next_obs_is_pad[:, 1]  # (B,)
        wm_loss, wm_metrics = _compute_wm_loss(z_pred, z_target, valid_wm, self.config)

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
            "wm_variance_loss": F.relu(1.0 - z_pred.std(dim=1, correction=0)).mean().item(),
            "flow_t_mean": t_sampled.mean().item(),
        }
        info.update({key: value.item() for key, value in wm_metrics.items()})

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
                        decoded_curr.detach(),
                        gt_curr_img.detach(),
                        prefix="wm_curr",
                    )
                )

                next_img_z = z_pred[
                    self.model.n_1d_tokens : self.model.n_1d_tokens + self.model.img_tokens_per_cam
                ]
                decoded_next = self.model.wm_image_decoder(next_img_z.detach())
                gt_next_img = next_batch[OBS_IMAGES][0].detach()
                info.update(
                    _compute_image_reconstruction_metrics(
                        decoded_next,
                        gt_next_img,
                        prefix="wm_next",
                        valid_mask=valid_wm,
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

    @torch.no_grad()
    def visualize(self, batch: dict[str, Tensor], n_pairs: int = 12) -> dict[str, Tensor] | None:
        """Generate WM image reconstruction pairs for debugging."""
        if not self.config.image_features or not hasattr(self.model, "wm_image_decoder"):
            return None

        was_training = self.training
        self.eval()

        n = min(n_pairs, batch["action"].shape[0])

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

        batch_size, cross_kv, cross_pos, curr_encoder_in = self.model._encode(curr_batch)

        curr_img_z = curr_encoder_in[n_1d : n_1d + s_img]
        if self.config.normalize_wm_representations:
            curr_img_z = F.normalize(curr_img_z, dim=-1)
        decoded_curr = self.model.wm_image_decoder(curr_img_z)
        gt_curr = curr_batch[OBS_IMAGES][0]

        # Run WM decoder with ground-truth continuous actions.
        actions = batch[ACTION][:n]
        T = actions.shape[1]
        action_embeds = self.model.wm_action_proj(actions).transpose(0, 1)
        wm_pos = self.model.wm_decoder_pos_embed.weight[:T].unsqueeze(1)
        S = self.model.n_encoder_tokens
        query_pos = self.model.wm_query_pos_embed.weight.unsqueeze(1)
        queries = (self.model.wm_query_tokens + query_pos).expand(-1, batch_size, -1)
        wm_in = torch.cat([queries, action_embeds + wm_pos], dim=0)
        wm_encoder_in = curr_encoder_in.detach() if self.config.detach_encoder_from_wm else curr_encoder_in
        wm_cross_kv = self.model.wm_cross_attn_proj(wm_encoder_in)
        wm_out = self.model.wm_decoder(wm_in, wm_cross_kv, cross_pos, None)
        z_pred = self.model.wm_proj_head(wm_out[:S])
        if self.config.normalize_wm_representations:
            z_pred = F.normalize(z_pred, dim=-1)

        if self.config.use_ema_target:
            _ = self.model._encode_ema(next_batch)
        else:
            _, _, _, _ = self.model._encode(next_batch)

        next_img_z = z_pred[n_1d : n_1d + s_img]
        decoded_next = self.model.wm_image_decoder(next_img_z)
        gt_next = next_batch[OBS_IMAGES][0]

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


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class FAWM(nn.Module):
    """Core network for FAWMPolicy.

    Encoder: identical to ACT/AWM (ResNet backbone + transformer encoder).
    Decoder: flow-matching action head with bidirectional self-attention +
             cross-attention on (optionally compressed) encoder outputs.
    World model: identical to AWM but conditioned on continuous actions.
    """

    def __init__(self, config: FAWMConfig):
        super().__init__()
        self.config = config

        action_dim = config.action_feature.shape[0]
        self.action_dim = action_dim

        # ------------------------------------------------------------------
        # Vision backbone (optional)
        # ------------------------------------------------------------------
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ------------------------------------------------------------------
        # Transformer encoder and decoder
        # ------------------------------------------------------------------
        self.encoder = ACTEncoder(config)
        self.decoder = FAWMDecoder(config)

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
        # Cross-attention dimension reduction
        # ------------------------------------------------------------------
        self.cross_attn_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.cross_attn_dim),
            nn.ReLU(),
            nn.Linear(config.cross_attn_dim, config.cross_attn_dim),
        )
        self.cross_attn_pos_proj = nn.Linear(config.dim_model, config.cross_attn_dim)
        self.wm_cross_attn_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.cross_attn_dim),
            nn.ReLU(),
            nn.Linear(config.cross_attn_dim, config.cross_attn_dim),
        )

        # ------------------------------------------------------------------
        # Flow-matching action head
        # ------------------------------------------------------------------
        self.action_input_proj = nn.Linear(action_dim, config.dim_model)
        self.action_output_proj = nn.Linear(config.dim_model, action_dim)

        # Timestep conditioning: sinusoidal embedding → 2-layer MLP → dim_model
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.SiLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # Decoder positional embeddings for action positions.
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # ------------------------------------------------------------------
        # World model — conditioned on continuous actions via wm_action_proj
        # ------------------------------------------------------------------
        self.wm_action_proj = nn.Linear(action_dim, config.dim_model)

        wm_cfg = copy(config)
        wm_cfg.n_decoder_layers = config.n_wm_decoder_layers
        self.wm_decoder = FAWMDecoder(wm_cfg)

        self.wm_decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        n_enc = _n_encoder_tokens(config)
        self.n_encoder_tokens = n_enc
        self.wm_query_tokens = nn.Parameter(torch.zeros(n_enc, 1, config.dim_model))
        nn.init.trunc_normal_(self.wm_query_tokens, std=0.02)
        self.wm_query_pos_embed = nn.Embedding(n_enc, config.dim_model)
        if config.image_features:
            stride = 16 if config.replace_final_stride_with_dilation else 32
            C, H, W = next(iter(config.image_features.values())).shape
            self.img_tokens_per_cam = (H // stride) * (W // stride)

        self.wm_proj_head = nn.Sequential(
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
        """Xavier-uniform initialisation for transformer and projection weights."""
        modules = [
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.wm_decoder.parameters(),
            self.wm_proj_head.parameters(),
            self.cross_attn_proj.parameters(),
            self.cross_attn_pos_proj.parameters(),
            self.wm_cross_attn_proj.parameters(),
            self.wm_query_pos_embed.parameters(),
        ]
        if hasattr(self, "wm_image_decoder"):
            modules.append(self.wm_image_decoder.parameters())
        for p in chain(*modules):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_ema_encoder(self):
        """Create EMA copies of encoder-side modules used to build WM targets."""
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, batch: dict[str, Tensor]) -> tuple[int, Tensor, Tensor, Tensor]:
        """Run the encoder and project its output for decoder cross-attention.

        Returns:
            batch_size, cross_kv, cross_pos, encoder_in
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

        cross_kv = self.cross_attn_proj(encoder_out)
        cross_pos = self.cross_attn_pos_proj(encoder_in_pos_embed)

        return batch_size, cross_kv, cross_pos, encoder_in_tokens

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
        """Update EMA encoder parameters from the online encoder."""
        if not self.config.use_ema_target:
            return

        ema_pairs = []
        if hasattr(self, "ema_backbone"):
            ema_pairs.extend(zip(self.backbone.parameters(), self.ema_backbone.parameters()))
        ema_pairs.extend(zip(self.encoder.parameters(), self.ema_encoder.parameters()))
        if hasattr(self, "ema_encoder_robot_state_input_proj"):
            ema_pairs.extend(
                zip(
                    self.encoder_robot_state_input_proj.parameters(),
                    self.ema_encoder_robot_state_input_proj.parameters(),
                )
            )
        if hasattr(self, "ema_encoder_env_state_input_proj"):
            ema_pairs.extend(
                zip(
                    self.encoder_env_state_input_proj.parameters(),
                    self.ema_encoder_env_state_input_proj.parameters(),
                )
            )
        if hasattr(self, "ema_encoder_img_feat_input_proj"):
            ema_pairs.extend(
                zip(
                    self.encoder_img_feat_input_proj.parameters(),
                    self.ema_encoder_img_feat_input_proj.parameters(),
                )
            )
        ema_pairs.extend(
            zip(
                self.encoder_1d_feature_pos_embed.parameters(),
                self.ema_encoder_1d_feature_pos_embed.parameters(),
            )
        )
        if hasattr(self, "ema_encoder_cam_feat_pos_embed"):
            ema_pairs.extend(
                zip(
                    self.encoder_cam_feat_pos_embed.parameters(),
                    self.ema_encoder_cam_feat_pos_embed.parameters(),
                )
            )

        for online_p, ema_p in ema_pairs:
            ema_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        next_batch: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple]:
        """Flow-matching training forward pass.

        Returns:
            action_loss: MSE on velocity prediction.
            v_pred: predicted velocity field.
            v_target: target velocity field.
            t: sampled timesteps.
            wm_tensors: (z_pred, z_target, decoded_curr, gt_curr_img).
        """
        batch_size, cross_kv, cross_pos, encoder_in = self._encode(batch)

        actions = batch[ACTION]  # (B, T, action_dim)
        T = actions.shape[1]

        # Flow matching: linear interpolation + velocity prediction.
        t = torch.rand(batch_size, 1, 1, device=actions.device)  # uniform [0, 1]
        x_0 = torch.randn_like(actions)  # source distribution (noise)
        x_t = (1 - t) * x_0 + t * actions  # linear interpolation
        v_target = actions - x_0  # conditional OT velocity target

        # Project to decoder space, add timestep embedding.
        decoder_in = self.action_input_proj(x_t)  # (B, T, dim_model)
        t_embed = self.timestep_mlp(
            sinusoidal_timestep_embedding(t.squeeze(-1).squeeze(-1), self.config.dim_model)
        )  # (B, dim_model)
        t_embed = t_embed.unsqueeze(1)  # (B, 1, dim_model)
        decoder_in = decoder_in + t_embed  # timestep only; pos added inside decoder layers
        decoder_in = decoder_in.transpose(0, 1)  # (T, B, dim_model)

        # Bidirectional self-attn + cross-attn to encoder (causal_mask=None).
        decoder_pos_embed = self.decoder_pos_embed.weight[:T].unsqueeze(1)  # (T, 1, dim_model)
        decoder_out = self.decoder(
            decoder_in, cross_kv, cross_pos, causal_mask=None,
            decoder_pos_embed=decoder_pos_embed,
        )

        v_pred = self.action_output_proj(decoder_out.transpose(0, 1))  # (B, T, action_dim)

        # Loss: MSE on velocity, masked by valid actions.
        valid = ~batch["action_is_pad"].reshape(-1)
        action_loss = F.mse_loss(
            v_pred.reshape(-1, self.action_dim)[valid],
            v_target.reshape(-1, self.action_dim)[valid],
        )

        # ------------------------------------------------------------------
        # World model forward
        # ------------------------------------------------------------------
        if self.config.use_ema_target:
            z_target = self._encode_ema(next_batch)
        else:
            _, _, _, next_encoder_in = self._encode(next_batch)
            z_target = next_encoder_in.detach()
        if self.config.normalize_wm_representations:
            z_target = F.normalize(z_target, dim=-1)

        # WM decoder input: [S query tokens, T action tokens].
        action_embeds = self.wm_action_proj(actions).transpose(0, 1)  # (T, B, dim_model)
        wm_pos = self.wm_decoder_pos_embed.weight[:T].unsqueeze(1)
        query_pos = self.wm_query_pos_embed.weight.unsqueeze(1)
        queries = (self.wm_query_tokens + query_pos).expand(-1, batch_size, -1)
        wm_in = torch.cat([queries, action_embeds + wm_pos], dim=0)

        S = self.n_encoder_tokens
        wm_encoder_in = encoder_in.detach() if self.config.detach_encoder_from_wm else encoder_in
        wm_cross_kv = self.wm_cross_attn_proj(wm_encoder_in)
        wm_out = self.wm_decoder(wm_in, wm_cross_kv, cross_pos, None)
        z_pred = self.wm_proj_head(wm_out[:S])
        if self.config.normalize_wm_representations:
            z_pred = F.normalize(z_pred, dim=-1)

        # Image decoder — trained on current encoder tokens.
        decoded_curr, gt_curr_img = None, None
        if hasattr(self, "wm_image_decoder") and OBS_IMAGES in batch:
            curr_img_z = encoder_in[self.n_1d_tokens : self.n_1d_tokens + self.img_tokens_per_cam]
            if self.config.normalize_wm_representations:
                curr_img_z = F.normalize(curr_img_z, dim=-1)
            decoded_curr = self.wm_image_decoder(curr_img_z.detach())
            gt_curr_img = batch[OBS_IMAGES][0].detach()

        wm_tensors = (z_pred, z_target, decoded_curr, gt_curr_img)

        return action_loss, v_pred, v_target, t, wm_tensors

    def predict_flow(self, batch: dict[str, Tensor], num_steps: int = 5) -> Tensor:
        """Euler integration from noise to actions.

        Args:
            batch: Observation batch.
            num_steps: Number of Euler steps.

        Returns:
            (B, chunk_size, action_dim) continuous actions.
        """
        batch_size, cross_kv, cross_pos, _ = self._encode(batch)

        x = torch.randn(
            batch_size, self.config.chunk_size, self.action_dim,
            device=cross_kv.device,
        )
        dt = 1.0 / num_steps

        T = self.config.chunk_size
        decoder_pos_embed = self.decoder_pos_embed.weight[:T].unsqueeze(1)

        for i in range(num_steps):
            t = torch.full((batch_size, 1, 1), i * dt, device=x.device)
            v = self._predict_velocity(x, t, cross_kv, cross_pos, decoder_pos_embed)
            x = x + v * dt

        return x

    def _predict_velocity(
        self,
        x: Tensor,
        t: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
        decoder_pos_embed: Tensor,
    ) -> Tensor:
        """Single velocity prediction for Euler integration."""
        decoder_in = self.action_input_proj(x)
        t_embed = self.timestep_mlp(
            sinusoidal_timestep_embedding(t.squeeze(-1).squeeze(-1), self.config.dim_model)
        )
        t_embed = t_embed.unsqueeze(1)
        decoder_in = decoder_in + t_embed  # timestep only; pos added inside decoder layers
        decoder_in = decoder_in.transpose(0, 1)

        decoder_out = self.decoder(
            decoder_in, cross_kv, cross_pos, causal_mask=None,
            decoder_pos_embed=decoder_pos_embed,
        )
        return self.action_output_proj(decoder_out.transpose(0, 1))


# ---------------------------------------------------------------------------
# Decoder modules
# ---------------------------------------------------------------------------


class FAWMDecoder(nn.Module):
    """Stack of FAWMDecoderLayer modules followed by optional layer norm."""

    def __init__(self, config: FAWMConfig):
        super().__init__()
        self.layers = nn.ModuleList([FAWMDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        cross_kv: Tensor,
        cross_pos: Tensor,
        causal_mask: Tensor | None,
        decoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, cross_kv, cross_pos, causal_mask, decoder_pos_embed=decoder_pos_embed)
        return self.norm(x)


class FAWMDecoderLayer(nn.Module):
    """Single FAWM decoder layer: self-attention + compressed cross-attention + FFN."""

    def __init__(self, config: FAWMConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model,
            config.n_heads,
            dropout=config.dropout,
            kdim=config.cross_attn_dim,
            vdim=config.cross_attn_dim,
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
        causal_mask: Tensor | None,
        decoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # --- Self-attention ---
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self._add_pos(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x, attn_mask=causal_mask, need_weights=False)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # --- Cross-attention ---
        x = self.multihead_attn(
            query=self._add_pos(x, decoder_pos_embed),
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

        # --- Feed-forward ---
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)

        return x
