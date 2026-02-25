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

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.sarm_utils import (
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
        # Fuses (num_cameras + 2) streams: cameras + lang + state
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
        """
        Prepare language embeddings for fusion.

        Accepts lang_emb of shape:
          - (B, text_emb_dim) -> broadcast across time
          - (B, T, text_emb_dim) -> per-timestep (dense annotation mode)

        Returns: (B, 1, T, D)
        """
        if lang_emb.dim() == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1)
        else:
            # (B, E) -> (B, 1, 1, D) -> expand to (B, 1, T, D)
            lang_proj = self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)
        return lang_proj

    def forward(
        self,
        img_seq: torch.Tensor,  # (B, N, T, vis_emb_dim)
        lang_emb: torch.Tensor,  # (B, E) or (B, T, E)
        state: torch.Tensor,  # (B, T, state_dim)
        lengths: torch.Tensor,  # (B,) - valid sequence lengths
        scheme: str = "sparse",  # "sparse" or "dense"
    ) -> torch.Tensor:
        """
        Forward pass for stage classification.

        Args:
            img_seq: Image embeddings (B, N, T, vis_emb_dim) where N=num_cameras
            lang_emb: Language embeddings (B, E) or (B, T, E) for dense
            state: State features (B, T, state_dim)
            lengths: Valid sequence lengths (B,) for masking
            scheme: "sparse" or "dense" for head selection

        Returns:
            Stage logits (B, T, num_classes)
        """
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        device = img_seq.device

        # Project inputs
        vis_proj = self.visual_proj(img_seq)  # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)  # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)  # (B, 1, T, D)

        # Concatenate streams
        # cameras + lang + state -> (B, N+2, T, D)
        x = torch.cat([vis_proj, lang_proj, state_proj], dim=1)

        # Add positional bias to first visual frame
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        # Flatten to tokens for Transformer
        x_tokens = x.view(B, (N + 2) * T, D)
        L = x_tokens.size(1)  # noqa: N806

        # Create padding mask
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        mask = base_mask.unsqueeze(1).expand(B, N + 2, T).reshape(B, (N + 2) * T)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        # Encode
        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)

        # Reshape and fuse
        h = h.view(B, N + 2, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 2) * D)
        fused = self.fusion_backbone(h)  # (B, T, D)

        # Scheme-specific logits
        logits = self.heads[scheme](fused)  # (B, T, num_classes)
        return logits


class SubtaskTransformer(nn.Module):
    """
    Subtask progress regression transformer for SARM.

    Predicts within-stage normalized progress (tau) conditioned on stage prior.
    The stage prior is a one-hot encoding passed from StageTransformer predictions.

    Input streams: [vis_proj, lang_proj, state_proj, stage_emb] -> (B, N+3, T, D)
    Output: tau predictions (B, T) in [0, 1]
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

        # Projections
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Encoder
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)

        # Learned bias on first visual frame
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # Shared fusion backbone
        # Fuses (num_cameras + 3) streams: cameras + lang + state + stage_emb
        fused_in = d_model * (num_cameras + 3)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        # Scheme-specific regression heads
        self.heads = nn.ModuleDict(
            {
                "sparse": nn.Linear(d_model, 1),
                "dense": nn.Linear(d_model, 1),
            }
        )

    def _prep_lang(self, lang_emb: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:  # noqa: N803
        """
        Prepare language embeddings for fusion.
        """
        if lang_emb.dim() == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            return self.lang_proj(lang_emb).unsqueeze(1)
        else:
            # (B, E) -> (B, 1, 1, D) -> (B, 1, T, D)
            return self.lang_proj(lang_emb).unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)

    def _stage_to_dmodel(self, stage_prior: torch.Tensor) -> torch.Tensor:
        """
        Deterministic projection of one-hot stage to d_model by pad/truncate.

        Args:
            stage_prior: One-hot stage embedding (B, 1, T, C)

        Returns:
            Projected stage embedding (B, 1, T, d_model)
        """
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
        img_seq: torch.Tensor,  # (B, N, T, vis_emb_dim)
        lang_emb: torch.Tensor,  # (B, E) or (B, T, E)
        state: torch.Tensor,  # (B, T, state_dim)
        lengths: torch.Tensor,  # (B,) - valid sequence lengths
        stage_prior: torch.Tensor,  # (B, 1, T, C) one-hot from gen_stage_emb
        scheme: str = "sparse",  # "sparse" or "dense"
    ) -> torch.Tensor:
        """
        Forward pass for subtask progress regression.

        Args:
            img_seq: Image embeddings (B, N, T, vis_emb_dim)
            lang_emb: Language embeddings (B, E) or (B, T, E)
            state: State features (B, T, state_dim)
            lengths: Valid sequence lengths (B,) for masking
            stage_prior: One-hot stage prior (B, 1, T, num_classes)
            scheme: "sparse" or "dense" for head selection

        Returns:
            Tau predictions (B, T) in [0, 1] via sigmoid
        """
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape  # noqa: N806
        D = self.d_model  # noqa: N806
        device = img_seq.device

        # Project inputs
        vis_proj = self.visual_proj(img_seq)  # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)  # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)  # (B, 1, T, D)
        stage_emb = self._stage_to_dmodel(stage_prior)  # (B, 1, T, D)

        # Concatenate all streams
        # cameras + lang + state + stage_emb -> (B, N+3, T, D)
        x = torch.cat([vis_proj, lang_proj, state_proj, stage_emb], dim=1)

        # Add positional bias to first visual frame
        x[:, :N, 0, :] = x[:, :N, 0, :] + self.first_pos

        # Flatten to tokens
        x_tokens = x.view(B, (N + 3) * T, D)
        L = x_tokens.size(1)  # noqa: N806

        # Create padding mask
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        # Encode
        h = self.transformer(x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True)

        # Reshape and fuse
        h = h.view(B, N + 3, T, D)
        h_flat = h.permute(0, 2, 1, 3).reshape(B, T, (N + 3) * D)
        fused = self.fusion_backbone(h_flat)  # (B, T, D)

        # Scheme-specific regression head -> sigmoid
        r = torch.sigmoid(self.heads[scheme](fused)).squeeze(-1)  # (B, T)
        return r


def gen_stage_emb(num_classes: int, targets: torch.Tensor) -> torch.Tensor:
    """
    Generate one-hot stage embeddings from targets.

    Args:
        num_classes: Number of stage classes
        targets: Target values (B, T) where integer part is stage index

    Returns:
        One-hot stage embedding (B, 1, T, num_classes)
    """
    # Integer part of float targets -> [0, C-1]
    idx = targets.long().clamp(min=0, max=num_classes - 1)  # (B, T)
    C = num_classes  # noqa: N806
    # Identity-lookup one-hot
    stage_onehot = torch.eye(C, device=targets.device)[idx]  # (B, T, C)
    stage_onehot = stage_onehot.unsqueeze(1)  # (B, 1, T, C)
    return stage_onehot


class SARMRewardModel(PreTrainedPolicy):
    """
    SARM Reward Model for stage-aware task completion rewards.

    Uses two separate transformer models:
    - StageTransformer: Classifies which stage/subtask
    - SubtaskTransformer: Predicts within-stage progress (tau)

    Training uses 75%/25% GT/predicted stage conditioning (teacher forcing).
    """

    name = "sarm"
    config_class = SARMConfig

    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__(config, dataset_stats)
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
            num_cameras=1,  # Single camera for now
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

        # GT/predicted stage ratio for teacher forcing
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

        Args:
            text_embeddings: Encoded text representations (batch_size, 512)
            video_embeddings: Encoded video representations (batch_size, num_frames, 512)
            state_features: Joint state features (batch_size, num_frames, state_dim)
            lengths: Valid sequence lengths (batch_size,)
            return_all_frames: If True, return rewards for all frames
            return_stages: If True, also return stage predictions
            return_confidence: If True, also return stage confidence
            head_mode: Which head to use ("sparse" or "dense")
            frame_index: Index of the target frame to extract (default: n_obs_steps).

        Returns:
            Rewards and optionally stage probs/confidence.
        """
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = torch.tensor(state_features, dtype=torch.float32)

        # Handle single sample case
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

        # Default lengths if not provided
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
        elif isinstance(lengths, np.ndarray):
            lengths = torch.tensor(lengths, dtype=torch.int32)

        # Reshape video to (B, N, T, D) for multi-camera format
        # Currently single camera: (B, T, D) -> (B, 1, T, D)
        img_seq = video_embeddings.unsqueeze(1).to(self.device)
        lang_emb = text_embeddings.to(self.device)
        state = (
            state_features.to(self.device)
            if state_features is not None
            else torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device)
        )
        lens = lengths.to(self.device)

        # Pad state to max_state_dim
        state = pad_state_to_max_dim(state, self.config.max_state_dim)

        # Get num_classes for this scheme
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        # Run stage model
        stage_logits = self.stage_model(img_seq, lang_emb, state, lens, scheme=scheme)
        stage_probs = F.softmax(stage_logits, dim=-1)  # (B, T, num_classes)
        stage_idx = stage_probs.argmax(dim=-1)  # (B, T)
        stage_conf = stage_probs.gather(-1, stage_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Create one-hot stage prior
        stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()  # (B, T, C)
        stage_emb = stage_onehot.unsqueeze(1)  # (B, 1, T, C)

        # Run subtask model
        tau_pred = self.subtask_model(img_seq, lang_emb, state, lens, stage_emb, scheme=scheme)

        # Compute final reward: stage + tau
        raw_reward = stage_idx.float() + tau_pred  # (B, T)

        # Normalize to [0, 1] using temporal proportions for proper weighting
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

        # Default frame index is n_obs_steps (last observation frame)
        if frame_index is None:
            frame_index = self.config.n_obs_steps

        # Prepare outputs (batch mode or no smoothing)
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
        """Required by PreTrainedPolicy but not used for reward models."""
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Required by PreTrainedPolicy but not used for reward models."""
        raise NotImplementedError("SARM model does not predict action chunks")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Required by PreTrainedPolicy but not used for SARM."""
        raise NotImplementedError("SARM model does not select actions")

    def _train_step(
        self,
        img_emb: torch.Tensor,  # (B, N, T, D)
        lang_emb: torch.Tensor,  # (B, E) or (B, T, E)
        state: torch.Tensor,  # (B, T, state_dim)
        lengths: torch.Tensor,  # (B,)
        targets: torch.Tensor,  # (B, T) - format: stage.tau
        scheme: str,
    ) -> dict[str, torch.Tensor]:
        """
        Single training step for one annotation scheme.

        Implements 75%/25% GT/predicted stage conditioning.

        Args:
            img_emb: Image embeddings (B, N, T, D)
            lang_emb: Language embeddings
            state: State features
            lengths: Valid sequence lengths
            targets: Target values where floor=stage, remainder=tau
            scheme: "sparse" or "dense"

        Returns:
            Dict with stage_loss, subtask_loss, total_loss
        """
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        # Ground truth: stage (integer) and tau (fractional)
        # Clamp stage indices to valid range [0, num_classes-1] to handle edge cases
        # where targets may exceed expected range (e.g., frames between subtasks)
        gt_stage = torch.floor(targets).long().clamp(0, num_classes - 1)  # (B, T)
        gt_tau = torch.remainder(targets, 1.0)  # (B, T)

        # Run stage model
        stage_pred = self.stage_model(img_emb, lang_emb, state, lengths, scheme=scheme)

        # 75%/25% GT/predicted stage conditioning
        if random.random() < self.gt_stage_ratio:
            # Mode 1: Use ground truth stage -> one-hot
            stage_emb = gen_stage_emb(num_classes, targets)  # (B, 1, T, C)
        else:
            # Mode 2: Use predicted stage argmax -> one-hot
            stage_idx = stage_pred.argmax(dim=-1)  # (B, T)
            stage_onehot = F.one_hot(stage_idx, num_classes=num_classes).float()  # (B, T, C)
            stage_emb = stage_onehot.unsqueeze(1)  # (B, 1, T, C)

        # Run subtask model with stage prior
        tau_pred = self.subtask_model(img_emb, lang_emb, state, lengths, stage_emb, scheme=scheme)

        # Compute losses
        stage_loss = F.cross_entropy(stage_pred.view(-1, num_classes), gt_stage.view(-1), reduction="mean")
        subtask_loss = F.mse_loss(tau_pred, gt_tau, reduction="mean")

        return {
            "stage_loss": stage_loss,
            "subtask_loss": subtask_loss,
            "total_loss": stage_loss + subtask_loss,
        }

    def forward(self, batch):
        """
        Forward pass for SARM reward model training.

        Uses stage+tau target format where:
        - Integer part = stage index
        - Fractional part = within-stage progress (tau)

        Training uses 75%/25% GT/predicted stage conditioning.

        Args:
            batch: Dictionary with 'observation' containing:
                - 'video_features': (B, T, 512) pre-encoded video features
                - 'text_features': (B, 512) or (B, T, 512) text features
                - 'state_features': (B, T, state_dim) joint state features
                - 'lengths': (B,) valid sequence lengths
                - 'sparse_targets': (B, T) sparse targets (stage.tau format)
                - 'dense_targets': (B, T) dense targets (optional, for dual mode)

        Returns:
            Tuple of (total_loss, output_dict with loss components)
        """
        observation = batch.get(OBS_STR, batch)

        # Extract features
        video_features = observation["video_features"].to(self.device)
        text_features = observation["text_features"].to(self.device)
        state_features = observation.get("state_features")
        if state_features is not None:
            state_features = state_features.to(self.device)

        batch_size = video_features.shape[0]
        seq_len = video_features.shape[1]

        # Get lengths (default to full sequence)
        lengths = observation.get("lengths")
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=self.device)
        else:
            lengths = lengths.to(self.device)

        # Reshape video to (B, N, T, D) - single camera
        img_emb = video_features.unsqueeze(1)

        # Pad state to max_state_dim
        if state_features is None:
            state_features = torch.zeros(batch_size, seq_len, self.config.max_state_dim, device=self.device)
        else:
            state_features = pad_state_to_max_dim(state_features, self.config.max_state_dim)

        output_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Sparse training (always)
        sparse_targets = observation.get("sparse_targets")
        if sparse_targets is None:
            # Try legacy format
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

        # Dense training (if dual mode)
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
    # Clamp target stage indices to valid range [0, num_stages-1]
    target_stages_flat = target_stages.reshape(-1).clamp(0, num_stages - 1)
    return F.cross_entropy(stage_logits_flat, target_stages_flat)
