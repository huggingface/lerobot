#!/usr/bin/env python

# Copyright 2025 Qianzhong Chen, Justin Yu, Mac Schwager, Pieter Abbeel, Yide Shentu, Philipp Wu
# and The HuggingFace Inc. team. All rights reserved.
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

"""
SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation.

Paper: https://arxiv.org/abs/2509.25358

- StageTransformer: Predicts stage classification (sparse/dense)
- SubtaskTransformer: Predicts within-stage progress (tau) conditioned on stage
"""

import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.sarm.configuration_sarm import SARMConfig
from lerobot.rewards.sarm.sarm_utils import (
    normalize_stage_tau,
    pad_state_to_max_dim,
)
from lerobot.utils.constants import OBS_STR


class StageTransformer(nn.Module):
    """
    Stage classification transformer for SARM.

    Predicts which stage/subtask the current frame belongs to.
    Supports both sparse (high-level) and dense (fine-grained) annotation schemes.

    Input streams: [vis_proj, lang_proj, state_proj] concatenated -> (B, N+2, T, D)
    Output: stage logits (B, T, num_classes)
    """

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
        num_classes_sparse: int = 4,
        num_classes_dense: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # Projections
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # Positional bias on first visual frame
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # Shared fusion MLP
        fused_in = d_model * (num_cameras + 2)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        # Scheme-specific heads
        self.heads = nn.ModuleDict(
            {
                "sparse": nn.Linear(d_model, num_classes_sparse),
                "dense": nn.Linear(d_model, num_classes_dense),
            }
        )

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:  # noqa: N803
        """Prepare language embeddings for fusion."""
        if lang_emb.dim() == 3:
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1)
        else:
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)
        return lang_proj

    def forward(
        self,
        img_seq: torch.Tensor,
        lang_emb: torch.Tensor,
        state: torch.Tensor,
        lengths: torch.Tensor,
        scheme: str = "sparse",
    ) -> torch.Tensor:
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, B, T, D)

        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        x_tokens = x.view(B, (N + 2) * T, D)
        L = x_tokens.size(1)  # noqa: N806

        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 2, T).reshape(B, (N + 2) * T)

        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)

        h = h.view(B, N + 2, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 2) * D)
        fused = self.fusion_backbone(h)

        logits = self.heads[scheme](fused)
        return logits


class SubtaskTransformer(nn.Module):
    """
    Subtask progress regression transformer for SARM.

    Predicts within-stage normalized progress (tau) conditioned on stage prior.
    """

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)

        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        fused_in = d_model * (num_cameras + 3)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict(
            {
                "sparse": nn.Linear(d_model, 1),
                "dense": nn.Linear(d_model, 1),
            }
        )

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:  # noqa: N803
        if lang_emb.dim() == 3:
            return self.lang_proj(lang_emb).unsqueeze(1)
        else:
            return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)

    def _stage_to_dmodel(self, stage_prior: torch.Tensor) -> torch.Tensor:
        B, one, T, C = stage_prior.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        if D == C:
            return stage_prior
        elif D > C:
            pad = torch.zeros(B, one, T, D - C, device=stage_prior.device, dtype=stage_prior.dtype)
            return torch.cat([stage_prior, pad], dim=-1)
        else:
            return stage_prior[..., :D]

    def forward(
        self,
        img_seq: torch.Tensor,
        lang_emb: torch.Tensor,
        state: torch.Tensor,
        lengths: torch.Tensor,
        stage_prior: torch.Tensor,
        scheme: str = "sparse",
    ) -> torch.Tensor:
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        device = img_seq.device

        vis_proj = self.visual_proj(img_seq)
        state_proj = self.state_proj(state).unsqueeze(1)
        lang_proj = self._prep_lang(lang_emb, B, T, D)
        stage_emb = self._stage_to_dmodel(stage_prior)

        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        x_tokens = x.view(B, (N + 3) * T, D)
        L = x_tokens.size(1)  # noqa: N806

        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)

        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)

        h = h.view(B, N + 3, T, D)
        h_flat = h.permute(0, 2, 1, 3).reshape(B, T, (N + 3) * D)
        fused = self.fusion_backbone(h_flat)

        r = torch.sigmoid(self.heads[scheme](fused)).squeeze(-1)
        return r


def gen_stage_emb(num_classes: int, targets: torch.Tensor) -> torch.Tensor:
    """Generate one-hot stage embeddings from targets."""
    idx = targets.long().clamp(min=0, max=num_classes - 1)
    C = num_classes  # noqa: N806
    stage_onehot = torch.eye(C, device=targets.device)[idx]
    stage_onehot = stage_onehot.unsqueeze(1)
    return stage_onehot


class SARMRewardModel(PreTrainedRewardModel):
    """
    SARM Reward Model for stage-aware task completion rewards.

    Uses two separate transformer models:
    - StageTransformer: Classifies which stage/subtask
    - SubtaskTransformer: Predicts within-stage progress (tau)

    Training uses 75%/25% GT/predicted stage conditioning (teacher forcing).
    """

    name = "sarm"
    config_class = SARMConfig

    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None, dataset_meta=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = torch.device(
            config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load temporal proportions based on annotation_mode
        if config.annotation_mode == "single_stage":
            logging.info(f"Using single_stage mode: sparse_subtask_names={config.sparse_subtask_names}")
        elif dataset_meta is not None:
            self._load_temporal_proportions(dataset_meta)

        # Create two separate models
        self.stage_model = StageTransformer(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,
            num_classes_sparse=config.num_sparse_stages,
            num_classes_dense=config.num_dense_stages or config.num_sparse_stages,
        )

        self.subtask_model = SubtaskTransformer(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,
        )

        self.stage_model.to(self.device)
        self.subtask_model.to(self.device)

        self.gt_stage_ratio = 0.75

        if config.uses_dual_heads:
            logging.info(
                f"SARM initialized with dual heads: {config.num_sparse_stages} sparse stages, "
                f"{config.num_dense_stages} dense stages"
            )
        else:
            logging.info(f"SARM initialized with sparse head only: {config.num_sparse_stages} stages")

        logging.info(f"SARM initialized on {self.device}")

    def _load_proportions_from_json(self, path, annotation_type: str) -> tuple[list[str], list[float]]:
        """Load temporal proportions from a JSON file (preserving order)."""
        if not path.exists():
            raise ValueError(
                f"{annotation_type.capitalize()} temporal proportions not found at {path}. "
                f"Run the subtask annotation tool with --{annotation_type}-subtasks to generate annotations."
            )
        with open(path) as f:
            proportions_dict = json.load(f)
        names = list(proportions_dict.keys())
        logging.info(f"Loaded {len(names)} {annotation_type} subtasks: {names}")
        logging.info(f"{annotation_type.capitalize()} temporal proportions: {proportions_dict}")
        return names, [proportions_dict[name] for name in names]

    def _load_temporal_proportions(self, dataset_meta) -> None:
        """Load temporal proportions based on annotation_mode."""
        meta_path = dataset_meta.root / "meta"

        if self.config.annotation_mode == "dual":
            names, props = self._load_proportions_from_json(
                meta_path / "temporal_proportions_sparse.json", "sparse"
            )
            (
                self.config.num_sparse_stages,
                self.config.sparse_subtask_names,
                self.config.sparse_temporal_proportions,
            ) = len(names), names, props

        if self.config.annotation_mode in ["dense_only", "dual"]:
            names, props = self._load_proportions_from_json(
                meta_path / "temporal_proportions_dense.json", "dense"
            )
            (
                self.config.num_dense_stages,
                self.config.dense_subtask_names,
                self.config.dense_temporal_proportions,
            ) = len(names), names, props
            if self.config.annotation_mode == "dense_only":
                logging.info(f"Using auto-generated sparse 'task' stage: {self.config.sparse_subtask_names}")

    def to(self, device):
        """Override to method to ensure all components move together."""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.stage_model.to(device)
        self.subtask_model.to(device)
        return self

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute dense progress reward in [0, 1] from batch.

        Expects batch to contain:
        - "observation_features" or video embeddings: (B, T, 512)
        - "language_embedding" or text embeddings: (B, 512)
        - optionally "observation.state": (B, T, state_dim)
        """
        text_emb = batch.get("language_embedding", batch.get("text_features"))
        video_emb = batch.get("observation_features", batch.get("video_features"))
        state = batch.get("observation.state", batch.get("state_features"))

        rewards = self.calculate_rewards(text_emb, video_emb, state)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        return rewards

    @torch.no_grad()
    def calculate_rewards(
        self,
        text_embeddings: np.ndarray | torch.Tensor,
        video_embeddings: np.ndarray | torch.Tensor,
        state_features: np.ndarray | torch.Tensor | None = None,
        lengths: np.ndarray | torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_stages: bool = False,
        return_confidence: bool = False,
        head_mode: str | None = "sparse",
        frame_index: int | None = None,
    ) -> np.ndarray | tuple:
        """
        Calculate rewards for given text, video, and state representations.

        This is the canonical method for SARM reward computation, used for:
        - Inference/visualization
        - RA-BC weight computation
        """
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = torch.tensor(state_features, dtype=torch.float32)

        if text_embeddings.dim() == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
            video_embeddings = video_embeddings.unsqueeze(0)
            if state_features is not None:
                state_features = state_features.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        batch_size = video_embeddings.shape[0]
        seq_len = video_embeddings.shape[1]

        scheme = head_mode

        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
        elif isinstance(lengths, np.ndarray):
            lengths = torch.tensor(lengths, dtype=torch.int32)

        img_seq = video_embeddings.unsqueeze(1).to(self.device)
        lang_emb = text_embeddings.to(self.device)
        state = (
            state_features.to(self.device)
            if state_features is not None
            else torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device)
        )
        lens = lengths.to(self.device)

        state = pad_state_to_max_dim(state, self.config.max_state_dim)

        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        stage_logits = self.stage_model(img_seq, lang_emb, state, lens, scheme=scheme)
        stage_probs = F.softmax(stage_logits, dim=-1)
        stage_idx = stage_probs.argmax(dim=-1)
        stage_conf = stage_probs.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)

        stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()
        stage_emb = stage_onehot.unsqueeze(1)

        tau_pred = self.subtask_model(img_seq, lang_emb, state, lens, stage_emb, scheme=scheme)

        raw_reward = stage_idx.float() + tau_pred

        if scheme == "sparse":
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=num_classes,
                temporal_proportions=self.config.sparse_temporal_proportions,
                subtask_names=self.config.sparse_subtask_names,
            )
        else:
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=num_classes,
                temporal_proportions=self.config.dense_temporal_proportions,
                subtask_names=self.config.dense_subtask_names,
            )

        if frame_index is None:
            frame_index = self.config.n_obs_steps

        if return_all_frames:
            rewards = normalized_reward.cpu().numpy()
        else:
            rewards = normalized_reward[:, frame_index].cpu().numpy()

        if single_sample:
            rewards = rewards[0] if not return_all_frames else rewards[0]

        outputs = [rewards]
        if return_stages:
            probs = stage_probs.cpu().numpy()
            if single_sample:
                probs = probs[0]
            outputs.append(probs)
        if return_confidence:
            conf = stage_conf.cpu().numpy()
            if single_sample:
                conf = conf[0]
            outputs.append(conf)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def train(self, mode: bool = True):
        """Set training mode for both models."""
        super().train(mode)
        self.stage_model.train(mode)
        self.subtask_model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for both models."""
        return self.train(False)

    def parameters(self):
        """Override to return trainable parameters from both models."""
        from itertools import chain

        return chain(self.stage_model.parameters(), self.subtask_model.parameters())

    def get_optim_params(self):
        """Override to return optimizer parameters from both models."""
        return self.parameters()

    def reset(self):
        pass

    def _train_step(
        self,
        img_emb: torch.Tensor,
        lang_emb: torch.Tensor,
        state: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        scheme: str,
    ) -> dict[str, torch.Tensor]:
        """Single training step for one annotation scheme."""
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        gt_stage = torch.floor(targets).long().clamp(0, num_classes - 1)
        gt_tau = torch.remainder(targets, 1.0)

        stage_pred = self.stage_model(img_emb, lang_emb, state, lengths, scheme=scheme)

        if random.random() < self.gt_stage_ratio:
            stage_emb = gen_stage_emb(num_classes, targets)
        else:
            stage_idx = stage_pred.argmax(dim=-1)
            stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()
            stage_emb = stage_onehot.unsqueeze(1)

        tau_pred = self.subtask_model(img_emb, lang_emb, state, lengths, stage_emb, scheme=scheme)

        stage_loss = F.cross_entropy(stage_pred.view(-1, num_classes), gt_stage.view(-1), reduction="mean")
        subtask_loss = F.mse_loss(tau_pred, gt_tau, reduction="mean")

        return {
            "stage_loss": stage_loss,
            "subtask_loss": subtask_loss,
            "total_loss": stage_loss + subtask_loss,
        }

    def forward(self, batch):
        """Forward pass for SARM reward model training."""
        observation = batch.get(OBS_STR, batch)

        video_features = observation["video_features"].to(self.device)
        text_features = observation["text_features"].to(self.device)
        state_features = observation.get("state_features")
        if state_features is not None:
            state_features = state_features.to(self.device)

        batch_size = video_features.shape[0]
        seq_len = video_features.shape[1]

        lengths = observation.get("lengths")
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=self.device)
        else:
            lengths = lengths.to(self.device)

        img_emb = video_features.unsqueeze(1)

        if state_features is None:
            state_features = torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device)
        else:
            state_features = pad_state_to_max_dim(state_features, self.config.max_state_dim)

        output_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        sparse_targets = observation.get("sparse_targets")
        if sparse_targets is None:
            sparse_targets = observation.get("targets")
        if sparse_targets is None:
            raise ValueError("sparse_targets (or targets) is required for SARM training")
        sparse_targets = sparse_targets.to(self.device)

        sparse_result = self._train_step(
            img_emb, text_features, state_features, lengths, sparse_targets, scheme="sparse"
        )
        output_dict["sparse_stage_loss"] = sparse_result["stage_loss"].item()
        output_dict["sparse_subtask_loss"] = sparse_result["subtask_loss"].item()
        total_loss = total_loss + sparse_result["total_loss"]

        if self.config.uses_dual_heads:
            dense_targets = observation.get("dense_targets")
            if dense_targets is not None:
                dense_targets = dense_targets.to(self.device)
                dense_result = self._train_step(
                    img_emb, text_features, state_features, lengths, dense_targets, scheme="dense"
                )
                output_dict["dense_stage_loss"] = dense_result["stage_loss"].item()
                output_dict["dense_subtask_loss"] = dense_result["subtask_loss"].item()
                total_loss = total_loss + dense_result["total_loss"]

        output_dict["total_loss"] = total_loss.item()
        return total_loss, output_dict


def compute_stage_loss(stage_logits: torch.Tensor, target_stages: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for stage classification."""
    _, _, num_stages = stage_logits.shape
    stage_logits_flat = stage_logits.reshape(-1, num_stages)
    target_stages_flat = target_stages.reshape(-1).clamp(0, num_stages - 1)
    return F.cross_entropy(stage_logits_flat, target_stages_flat)
