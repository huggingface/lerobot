#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.sarm_utils import compute_cumulative_progress_batch, pad_state_to_max_dim


class SARMTransformer(nn.Module):
    """SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation (https://arxiv.org/pdf/2509.25358)."""

    def __init__(
        self,
        video_dim: int = 512,
        text_dim: int = 512,
        max_state_dim: int = 32,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        max_length: int = 9,
        dropout: float = 0.1,
        uses_dual_heads: bool = False,
        # Sparse (always required)
        num_sparse_stages: int = 5,
        sparse_temporal_proportions: list[float] | None = None,
        # Dense (only required when uses_dual_heads=True)
        num_dense_stages: int | None = None,
        dense_temporal_proportions: list[float] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.max_state_dim = max_state_dim
        self.uses_dual_heads = uses_dual_heads
        self.num_sparse_stages = num_sparse_stages
        self.num_dense_stages = num_dense_stages

        # Sparse proportions (always needed)
        if sparse_temporal_proportions is None:
            raise ValueError(
                "sparse_temporal_proportions is required for SARM. "
                "Provide subtask annotations in your dataset or set sparse_temporal_proportions in config."
            )
        sparse_alpha = torch.tensor(sparse_temporal_proportions, dtype=torch.float32)
        sparse_cumulative = torch.zeros(self.num_sparse_stages + 1, dtype=torch.float32)
        sparse_cumulative[1:] = torch.cumsum(sparse_alpha, dim=0)
        self.register_buffer("sparse_alpha", sparse_alpha)
        self.register_buffer("sparse_cumulative_prior", sparse_cumulative)

        if uses_dual_heads:
            # Dual mode: also need dense proportions
            if dense_temporal_proportions is None:
                raise ValueError("dense_temporal_proportions is required when uses_dual_heads=True")
            self.num_dense_stages = num_dense_stages or len(dense_temporal_proportions)

            # Dense proportions
            dense_alpha = torch.tensor(dense_temporal_proportions, dtype=torch.float32)
            dense_cumulative = torch.zeros(self.num_dense_stages + 1, dtype=torch.float32)
            dense_cumulative[1:] = torch.cumsum(dense_alpha, dim=0)
            self.register_buffer("dense_alpha", dense_alpha)
            self.register_buffer("dense_cumulative_prior", dense_cumulative)

        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.state_proj = nn.Linear(max_state_dim, hidden_dim)

        # Position embedding only for the first frame
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sparse heads
        self.sparse_stage_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.num_sparse_stages),
        )
        self.sparse_stage_embedding = nn.Embedding(self.num_sparse_stages, hidden_dim // 4)
        subtask_input_dim = hidden_dim + hidden_dim // 4
        self.sparse_subtask_head = nn.Sequential(
            nn.Linear(subtask_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if uses_dual_heads:
            # Dense heads
            self.dense_stage_head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, self.num_dense_stages),
            )
            self.dense_stage_embedding = nn.Embedding(self.num_dense_stages, hidden_dim // 4)
            self.dense_subtask_head = nn.Sequential(
                nn.Linear(subtask_input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

        # Attention mask
        self.register_buffer("attention_mask", None, persistent=False)

    def _get_attention_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve cached causal attention mask."""
        if self.attention_mask is None or self.attention_mask.shape[0] != seq_length:
            # Create causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
            self.attention_mask = mask
        return self.attention_mask

    def _compute_backbone_features(
        self,
        video_frames: torch.Tensor,
        text_embed: torch.Tensor,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute shared backbone features from inputs."""
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Pad state features to max_state_dim before projection
        state_features_padded = pad_state_to_max_dim(state_features, self.max_state_dim)
        state_embed = self.state_proj(state_features_padded)  # [batch_size, seq_len, hidden_dim]

        # Fuse video and state features
        video_embed = video_embed + state_embed

        # Add positional embedding to first video frame
        video_embed[:, 0] += self.first_pos_embed

        # Combine sequence: [text, video_frames]
        sequence = torch.cat([text_embed, video_embed], dim=1)

        # Get causal attention mask
        seq_length = sequence.shape[1]
        attention_mask = self._get_attention_mask(seq_length, sequence.device)

        # Pass through transformer with causal masking
        transformed = self.transformer(sequence, mask=attention_mask, is_causal=True)

        # Get frame features
        frame_features = transformed[:, 1:]  # [batch_size, seq_len, hidden_dim]
        return frame_features

    def forward(
        self,
        video_frames: torch.Tensor,
        text_embed: torch.Tensor,
        state_features: torch.Tensor | None = None,
        head_mode: str = "both",  # "sparse", "dense", or "both" (only for dual mode)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict:
        """
        Forward pass through the SARM transformer.

        Args:
            video_frames: Video frame embeddings (batch_size, seq_len, video_dim)
            text_embed: Text embeddings (batch_size, text_dim)
            state_features: Joint state features (batch_size, seq_len, state_dim)
            head_mode: Which head(s) to use in dual mode ("sparse", "dense", or "both")

        Returns:
            For single-head mode:
                Tuple of (stage_logits, stage_probs, progress_preds)
            For dual-head mode:
                Dict with keys "sparse" and/or "dense", each containing
                (stage_logits, stage_probs, progress_preds)
        """
        # Compute shared backbone features
        frame_features = self._compute_backbone_features(video_frames, text_embed, state_features)

        if not self.uses_dual_heads:
            # Single head mode: sparse only
            sparse_stage_logits = self.sparse_stage_head(frame_features)
            sparse_stage_probs = F.softmax(sparse_stage_logits, dim=-1)
            sparse_stage_indices = torch.argmax(sparse_stage_probs, dim=-1)
            sparse_stage_embeds = self.sparse_stage_embedding(sparse_stage_indices)
            conditioned_features = torch.cat([frame_features, sparse_stage_embeds], dim=-1)
            tau_preds = self.sparse_subtask_head(conditioned_features)
            progress_preds = compute_cumulative_progress_batch(
                tau_preds, sparse_stage_indices, self.sparse_alpha, self.sparse_cumulative_prior
            )
            return sparse_stage_logits, sparse_stage_probs, progress_preds

        # Dual head mode: compute outputs for requested heads
        results = {}

        if head_mode in ["sparse", "both"]:
            # Sparse head
            sparse_stage_logits = self.sparse_stage_head(frame_features)
            sparse_stage_probs = F.softmax(sparse_stage_logits, dim=-1)
            sparse_stage_indices = torch.argmax(sparse_stage_probs, dim=-1)
            sparse_stage_embeds = self.sparse_stage_embedding(sparse_stage_indices)
            sparse_conditioned = torch.cat([frame_features, sparse_stage_embeds], dim=-1)
            sparse_tau_preds = self.sparse_subtask_head(sparse_conditioned)
            sparse_progress_preds = compute_cumulative_progress_batch(
                sparse_tau_preds, sparse_stage_indices, self.sparse_alpha, self.sparse_cumulative_prior
            )
            results["sparse"] = (sparse_stage_logits, sparse_stage_probs, sparse_progress_preds)

        if head_mode in ["dense", "both"]:
            # Dense head
            dense_stage_logits = self.dense_stage_head(frame_features)
            dense_stage_probs = F.softmax(dense_stage_logits, dim=-1)
            dense_stage_indices = torch.argmax(dense_stage_probs, dim=-1)
            dense_stage_embeds = self.dense_stage_embedding(dense_stage_indices)
            dense_conditioned = torch.cat([frame_features, dense_stage_embeds], dim=-1)
            dense_tau_preds = self.dense_subtask_head(dense_conditioned)
            dense_progress_preds = compute_cumulative_progress_batch(
                dense_tau_preds, dense_stage_indices, self.dense_alpha, self.dense_cumulative_prior
            )
            results["dense"] = (dense_stage_logits, dense_stage_probs, dense_progress_preds)

        return results


class SARMRewardModel(PreTrainedPolicy):
    """SARM Reward Model for stage-aware task completion rewards."""

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

        if config.uses_dual_heads:
            self.sarm_transformer = SARMTransformer(
                video_dim=config.image_dim,
                text_dim=config.text_dim,
                max_state_dim=config.max_state_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                max_length=config.max_length,
                dropout=config.dropout,
                uses_dual_heads=True,
                num_sparse_stages=config.num_sparse_stages,
                sparse_temporal_proportions=config.sparse_temporal_proportions,
                num_dense_stages=config.num_dense_stages,
                dense_temporal_proportions=config.dense_temporal_proportions,
            )
            logging.info(
                f"SARM initialized with dual heads: {config.num_sparse_stages} sparse stages, {config.num_dense_stages} dense stages"
            )
        else:
            self.sarm_transformer = SARMTransformer(
                video_dim=config.image_dim,
                text_dim=config.text_dim,
                max_state_dim=config.max_state_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                max_length=config.max_length,
                dropout=config.dropout,
                uses_dual_heads=False,
                num_sparse_stages=config.num_sparse_stages,
                sparse_temporal_proportions=config.sparse_temporal_proportions,
            )
            logging.info(f"SARM initialized with sparse head only: {config.num_sparse_stages} stages")
        self.sarm_transformer.to(self.device)

        # Random word embedding pool for misalignment loss (language grounding)
        self.num_random_words = 100
        self.register_buffer(
            "random_word_embeddings",
            torch.randn(self.num_random_words, config.text_dim) * 0.5  # Scaled similar to CLIP embeddings
        )
        self.misalignment_loss_weight = getattr(config, "misalignment_loss_weight", 0.1)
        self.misalignment_prob = getattr(config, "misalignment_prob", 0.2)

        logging.info(f"SARM initialized on {self.device}")

    def _generate_random_text_embeddings(self, batch_size: int) -> torch.Tensor:
        """Generate random text embeddings by sampling and combining random word embeddings.
        
        This creates text embeddings that are unrelated to the actual task,
        used for training the model to output low progress for mismatched text-video pairs.
        """
        # Sample 3-5 random "words" per sample and average them (simulates random sentence)
        num_words = random.randint(3, 5)
        indices = torch.randint(0, self.num_random_words, (batch_size, num_words), device=self.device)
        # Gather and average
        sampled = self.random_word_embeddings[indices]  # (batch_size, num_words, text_dim)
        random_text = sampled.mean(dim=1)  # (batch_size, text_dim)
        # Normalize to unit length like CLIP embeddings
        random_text = F.normalize(random_text, dim=-1)
        return random_text

    def _load_proportions_from_json(self, path, annotation_type: str) -> tuple[list[str], list[float]]:
        """Load temporal proportions from a JSON file."""
        if not path.exists():
            raise ValueError(
                f"{annotation_type.capitalize()} temporal proportions not found at {path}. "
                f"Run the subtask annotation tool with --{annotation_type}-subtasks to generate annotations."
            )
        with open(path) as f:
            proportions_dict = json.load(f)
        names = sorted(proportions_dict.keys())
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
        self.sarm_transformer.to(device)
        return self

    @torch.no_grad()
    def calculate_rewards(
        self,
        text_embeddings: np.ndarray | torch.Tensor,
        video_embeddings: np.ndarray | torch.Tensor,
        state_features: np.ndarray | torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_stages: bool = False,
        head_mode: str | None = None,
    ) -> np.ndarray | tuple:
        """
        Calculate rewards for given text, video, and state representations.

        Args:
            text_embeddings: Encoded text representations (batch_size, 512)
            video_embeddings: Encoded video representations (batch_size, num_frames, 512)
            state_features: Joint state features (batch_size, num_frames, state_dim)
            return_all_frames: If True, return rewards for all frames
            return_stages: If True, also return stage predictions
            head_mode: Which head to use for dual-head models ("sparse" or "dense").
                       If None, uses config.dual_inference_mode. Ignored for single-head models.

        Returns:
            If return_stages=False:
                Reward values (batch_size,) or (batch_size, num_frames)
            If return_stages=True:
                Tuple of (rewards, stage_probs)
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

        # Determine which head to use for dual-head models
        if head_mode is None:
            head_mode = self.config.dual_inference_mode

        # Process in batches
        all_rewards = []
        all_stage_probs = []

        for i in range(0, len(video_embeddings), self.config.batch_size):
            batch_texts = text_embeddings[i : i + self.config.batch_size].to(self.device)
            batch_videos = video_embeddings[i : i + self.config.batch_size].to(self.device)
            batch_states = None
            if state_features is not None:
                batch_states = state_features[i : i + self.config.batch_size].to(self.device)

            # Get predictions
            if self.config.uses_dual_heads:
                # Dual-head model: select which head to use
                results = self.sarm_transformer(
                    batch_videos.float(),
                    batch_texts.float(),
                    batch_states.float() if batch_states is not None else None,
                    head_mode=head_mode,
                )
                # Use the requested head (default to sparse if "both" was requested)
                selected_head = "sparse" if head_mode == "both" else head_mode
                stage_logits, stage_probs, progress_preds = results[selected_head]
            else:
                # Single-head model: returns tuple directly
                stage_logits, stage_probs, progress_preds = self.sarm_transformer(
                    batch_videos.float(),
                    batch_texts.float(),
                    batch_states.float() if batch_states is not None else None,
                )

            if return_all_frames:
                all_rewards.append(progress_preds.squeeze(-1).cpu())
            else:
                # Return only last frame reward
                all_rewards.append(progress_preds[:, -1, 0].cpu())

            if return_stages:
                all_stage_probs.append(stage_probs.cpu())

        rewards = torch.cat(all_rewards).numpy()

        if single_sample:
            rewards = rewards[0] if not return_all_frames else rewards[0]

        if return_stages:
            stage_probs = torch.cat(all_stage_probs).numpy()
            if single_sample:
                stage_probs = stage_probs[0]
            return rewards, stage_probs

        return rewards

    def train(self, mode: bool = True):
        """Set training mode for the SARM transformer."""
        super().train(mode)
        self.sarm_transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for the SARM transformer."""
        return self.train(False)

    def parameters(self):
        """Override to return trainable parameters (only SARM transformer, not CLIP encoder)."""
        return self.sarm_transformer.parameters()

    def get_optim_params(self):
        """Override to return optimizer parameters (only SARM transformer, not CLIP encoder)."""
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

    def _apply_temporal_augmentation(
        self,
        video: torch.Tensor,
        progress: torch.Tensor,
        state: torch.Tensor | None,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply temporal augmentation by appending reversed frames (SARM paper A.4).

        Simulates rewinding from a stopping point by going backwards through
        previously seen frames. Keeps frames up to a cut point, then appends
        frames going backwards from just before that point.

        Example: [1,2,3,4,5,6] with n=2 → [1,2,3,4,3,2]
        (progress to 4, then rewind: 4→3→2)
        """
        seq_len = video.shape[0]
        num_reverse = random.randint(1, 4)

        # Cut point
        cut_idx = seq_len - num_reverse

        # Rewind: go backwards from (cut_idx - 1) for num_reverse steps
        # e.g., cut_idx=4, num_reverse=2 → indices 2,1 → values 3,2
        rewind_start = cut_idx - num_reverse - 1
        rewind_end = cut_idx - 1

        keep_video = video[:cut_idx]
        rewind_video = video[rewind_start:rewind_end].flip(0)
        video = torch.cat([keep_video, rewind_video], dim=0)

        keep_progress = progress[:cut_idx]
        rewind_progress = progress[rewind_start:rewind_end].flip(0)
        progress = torch.cat([keep_progress, rewind_progress], dim=0)

        if state is not None:
            keep_state = state[:cut_idx]
            rewind_state = state[rewind_start:rewind_end].flip(0)
            state = torch.cat([keep_state, rewind_state], dim=0)

        return video, progress, state

    def _ensure_sequence_length(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad or trim tensor to target length."""
        current_len = tensor.shape[0]
        if current_len == target_len:
            return tensor
        if current_len < target_len:
            padding = target_len - current_len
            return torch.cat([tensor, tensor[-1:].expand(padding, *tensor.shape[1:])])
        return tensor[:target_len]

    def _prepare_progress_tensor(self, progress: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Prepare progress tensor with correct dimensions."""
        progress = progress.to(self.device)
        if progress.dim() == 2:
            progress = progress.unsqueeze(-1)
        if progress.shape[0] == 1:
            progress = progress.expand(batch_size, -1, -1)
        return progress

    def _process_batch_with_augmentation(
        self,
        video_features: torch.Tensor,
        state_features: torch.Tensor | None,
        progress_tensors: list[torch.Tensor],
        batch_size: int,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Process samples with temporal augmentation for one or more progress tensors."""
        processed_videos, processed_states = [], []
        processed_progress = [[] for _ in progress_tensors]

        for i in range(batch_size):
            video = video_features[i]
            state = state_features[i] if state_features is not None else None
            progs = [p[i].squeeze(-1) for p in progress_tensors]

            # Apply temporal REWIND augmentation with 50% probability
            if random.random() < 0.5:
                video, progs[0], state = self._apply_temporal_augmentation(video, progs[0], state, max_length)
                for j in range(1, len(progs)):
                    progs[j] = self._ensure_sequence_length(progs[j].unsqueeze(-1), max_length).squeeze(-1)

            # Ensure correct sequence length
            video = self._ensure_sequence_length(video, max_length)
            for j in range(len(progs)):
                progs[j] = self._ensure_sequence_length(progs[j].unsqueeze(-1), max_length).squeeze(-1)
            if state is not None:
                state = self._ensure_sequence_length(state, max_length)

            processed_videos.append(video)
            for j, prog in enumerate(progs):
                processed_progress[j].append(prog)
            if state is not None:
                processed_states.append(state)

        return (
            torch.stack(processed_videos),
            torch.stack(processed_states) if processed_states else None,
            [torch.stack(p).unsqueeze(-1) for p in processed_progress],
        )

    def _compute_stage_loss_with_labels(
        self, stage_logits: torch.Tensor, labels: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Compute stage loss if labels are available."""
        if labels is None:
            return None
        labels = labels.to(self.device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0).expand(batch_size, -1)
        return compute_stage_loss(stage_logits, labels)

    def forward(self, batch):
        """
        Forward pass for SARM reward model training.

        Uses annotation-based progress targets following SARM paper Eq. 2:
        yt = Pk-1 + α̅k × τt
        where:
        - τt = (t - sk) / (ek - sk) is within-subtask normalized time
        - Pk-1 is cumulative prior (sum of previous subtask proportions)
        - α̅k is the temporal proportion for subtask k

        Args:
            batch: Dictionary with 'observation' containing:
                - 'video_features': (B, T, 512) pre-encoded video features
                - 'text_features': (B, 512) pre-encoded text features (CLIP)
                - 'state_features': (B, T, state_dim) joint state features
                - For single mode (annotation_mode="single_stage", sparse only):
                    - 'sparse_stage_labels': (B, T) sparse stage labels from annotations
                    - 'sparse_progress_targets': (B, T, 1) sparse progress targets from annotations
                    (also accepts legacy 'stage_labels' and 'progress_targets' for backward compat)
                - For dual mode (annotation_mode in ["dense_only", "dual"]):
                    - 'sparse_stage_labels': (B, T) sparse stage labels
                    - 'sparse_progress_targets': (B, T, 1) sparse progress targets
                    - 'dense_stage_labels': (B, T) dense stage labels
                    - 'dense_progress_targets': (B, T, 1) dense progress targets

        Returns:
            Tuple of (total_loss, output_dict with loss components)
        """
        observation = batch.get("observation", batch)

        # Extract required features
        video_features = observation["video_features"].to(self.device)
        text_features = observation["text_features"].to(self.device)
        state_features = observation.get("state_features").to(self.device)

        batch_size = video_features.shape[0]
        max_length = self.config.num_frames

        # Ensure 3D video features (B, T, D)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1).expand(-1, max_length, -1)
        if state_features is not None and state_features.dim() == 2:
            state_features = state_features.unsqueeze(1).expand(-1, max_length, -1)

        if self.config.uses_dual_heads:
            return self._forward_dual(
                observation, video_features, text_features, state_features, batch_size, max_length
            )
        else:
            return self._forward_single(
                observation, video_features, text_features, state_features, batch_size, max_length
            )

    def _forward_single(
        self, observation, video_features, text_features, state_features, batch_size, max_length
    ):
        """Forward pass for single-head mode (sparse only)."""
        progress = observation.get("sparse_progress_targets")
        if progress is None:
            progress = observation.get("progress_targets")
        if progress is None:
            raise ValueError("sparse_progress_targets (or progress_targets) is required for SARM training")
        progress = self._prepare_progress_tensor(progress, batch_size)

        processed_videos, processed_states, [progress_targets] = self._process_batch_with_augmentation(
            video_features, state_features, [progress], batch_size, max_length
        )

        stage_logits, _, progress_preds = self.sarm_transformer(
            processed_videos, text_features, processed_states
        )

        output_dict = {"sparse_progress_loss": F.mse_loss(progress_preds, progress_targets).item()}
        total_loss = F.mse_loss(progress_preds, progress_targets)

        stage_labels = observation.get("sparse_stage_labels")
        if stage_labels is None:
            stage_labels = observation.get("stage_labels")
        if stage_labels is None:
            raise ValueError("sparse_stage_labels (or stage_labels) is required for SARM training")
        stage_loss = self._compute_stage_loss_with_labels(stage_logits, stage_labels, batch_size)
        total_loss = total_loss + self.config.stage_loss_weight * stage_loss
        output_dict["sparse_stage_loss"] = stage_loss.item()

        # Misalignment loss: train model to output low progress for random/unrelated text
        # This encourages language grounding - the model should understand task descriptions
        if random.random() < self.misalignment_prob:
            random_text = self._generate_random_text_embeddings(batch_size)
            _, _, misaligned_progress = self.sarm_transformer(
                processed_videos, random_text, processed_states
            )
            # Target: zero progress for misaligned text-video pairs
            misaligned_loss = F.mse_loss(misaligned_progress, torch.zeros_like(misaligned_progress))
            total_loss = total_loss + self.misalignment_loss_weight * misaligned_loss
            output_dict["misalignment_loss"] = misaligned_loss.item()

        output_dict["total_loss"] = total_loss.item()
        return total_loss, output_dict

    def _forward_dual(
        self, observation, video_features, text_features, state_features, batch_size, max_length
    ):
        """Forward pass for dual-head mode (sparse and dense annotations)."""
        sparse_progress = observation.get("sparse_progress_targets")
        dense_progress = observation.get("dense_progress_targets")
        if sparse_progress is None or dense_progress is None:
            raise ValueError(
                "Both sparse_progress_targets and dense_progress_targets are required for dual mode training"
            )

        sparse_progress = self._prepare_progress_tensor(sparse_progress, batch_size)
        dense_progress = self._prepare_progress_tensor(dense_progress, batch_size)

        processed_videos, processed_states, [sparse_targets, dense_targets] = (
            self._process_batch_with_augmentation(
                video_features, state_features, [sparse_progress, dense_progress], batch_size, max_length
            )
        )

        results = self.sarm_transformer(processed_videos, text_features, processed_states, head_mode="both")
        sparse_logits, _, sparse_preds = results["sparse"]
        dense_logits, _, dense_preds = results["dense"]

        output_dict = {}
        total_loss = F.mse_loss(sparse_preds, sparse_targets) + F.mse_loss(dense_preds, dense_targets)
        output_dict["sparse_progress_loss"] = F.mse_loss(sparse_preds, sparse_targets).item()
        output_dict["dense_progress_loss"] = F.mse_loss(dense_preds, dense_targets).item()

        for prefix, logits, labels in [
            ("sparse", sparse_logits, observation.get("sparse_stage_labels")),
            ("dense", dense_logits, observation.get("dense_stage_labels")),
        ]:
            stage_loss = self._compute_stage_loss_with_labels(logits, labels, batch_size)
            if stage_loss is not None:
                total_loss = total_loss + self.config.stage_loss_weight * stage_loss
                output_dict[f"{prefix}_stage_loss"] = stage_loss.item()

        # Misalignment loss: train model to output low progress for random/unrelated text
        if random.random() < self.misalignment_prob:
            random_text = self._generate_random_text_embeddings(batch_size)
            misaligned = self.sarm_transformer(
                processed_videos,
                random_text,
                processed_states,
                head_mode="both",
            )
            misaligned_loss = F.mse_loss(misaligned["sparse"][2], torch.zeros_like(misaligned["sparse"][2]))
            misaligned_loss += F.mse_loss(misaligned["dense"][2], torch.zeros_like(misaligned["dense"][2]))
            total_loss = total_loss + self.misalignment_loss_weight * misaligned_loss
            output_dict["misalignment_loss"] = misaligned_loss.item()

        output_dict["total_loss"] = total_loss.item()
        return total_loss, output_dict


def compute_stage_loss(stage_logits: torch.Tensor, target_stages: torch.Tensor) -> torch.Tensor:
    _, _, num_stages = stage_logits.shape
    stage_logits_flat = stage_logits.reshape(-1, num_stages)
    target_stages_flat = target_stages.reshape(-1)

    loss = F.cross_entropy(stage_logits_flat, target_stages_flat)
    return loss
