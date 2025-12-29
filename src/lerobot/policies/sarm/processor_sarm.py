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

"""SARM Processor for encoding images/text and generating stage+tau targets."""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from faker import Faker
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.sarm.sarm_utils import (
    apply_rewind_augmentation,
    compute_absolute_indices,
    find_stage_and_tau,
    pad_state_to_max_dim,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (
    from_tensor_to_numpy,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class SARMEncodingProcessorStep(ProcessorStep):
    """ProcessorStep that encodes images and text with CLIP and generates stage and progress labels for SARM."""

    def __init__(
        self,
        config: SARMConfig,
        image_key: str | None = None,
        dataset_meta=None,
        dataset_stats: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.image_key = image_key or config.image_key
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats
        self.annotation_mode = config.annotation_mode

        # Helper to create temporal proportions dict
        def make_props_dict(names, props):
            return dict(zip(names, props, strict=True)) if names and props else None

        # Sparse annotations (always needed)
        self.sparse_temporal_proportions = make_props_dict(
            config.sparse_subtask_names, config.sparse_temporal_proportions
        )
        self.sparse_subtask_names = config.sparse_subtask_names

        # Dense annotations (only for dual mode)
        self.dense_subtask_names = config.dense_subtask_names if config.uses_dual_heads else None
        self.dense_temporal_proportions = (
            make_props_dict(config.dense_subtask_names, config.dense_temporal_proportions)
            if config.uses_dual_heads
            else None
        )

        self.device = torch.device(
            self.config.device if self.config.device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()

        self.verbs = ["move", "grasp", "rotate", "push", "pull", "slide", "lift", "place"]
        self.fake = Faker()

    def _find_episode_for_frame(self, frame_idx: int) -> int:
        """Find the episode index for a given frame index."""
        for ep_idx in range(len(self.dataset_meta.episodes)):
            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            if ep_start <= frame_idx < ep_end:
                return ep_idx
        return 0

    def _get_episode_indices(self, frame_indices: np.ndarray, episode_index) -> np.ndarray:
        """Get episode indices for each frame index."""
        if episode_index is None:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        episode_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(episode_index)))

        # If single episode but multiple frames, compute episode for each frame
        if len(episode_indices) == 1 and len(frame_indices) > 1:
            return np.array([self._find_episode_for_frame(int(f)) for f in frame_indices])

        return episode_indices

    def _generate_perturbed_task(self) -> str:
        """Generate a random perturbed task string for language perturbation."""
        num_words = random.randint(1, 5)
        verb = random.choice(self.verbs)
        phrase = " ".join([verb] + self.fake.words(nb=num_words))
        return phrase

    def _get_annotation_config(self, annotation_type: str) -> tuple[list[str], dict[str, float] | None]:
        """Get global subtask names and temporal proportions for an annotation type."""
        if annotation_type == "dense":
            return self.dense_subtask_names, self.dense_temporal_proportions
        return self.sparse_subtask_names, self.sparse_temporal_proportions

    def _load_episode_annotations(
        self,
        ep_idx: int,
        episodes_df: pd.DataFrame | None,
        annotation_type: str,
        global_names: list[str],
    ) -> tuple[list | None, list | None, list | None]:
        """Load subtask annotations for an episode from DataFrame."""
        # Single-stage mode: (linear progress 0→1)
        if episodes_df is None or len(global_names) == 1:
            return None, None, None

        # Resolve column name with fallback
        def col(suffix):
            prefixed = f"{annotation_type}_{suffix}"
            return prefixed if prefixed in episodes_df.columns else suffix

        col_names = col("subtask_names")
        if col_names not in episodes_df.columns or ep_idx >= len(episodes_df):
            return None, None, None

        subtask_names = episodes_df.loc[ep_idx, col_names]
        if subtask_names is None or (isinstance(subtask_names, float) and pd.isna(subtask_names)):
            return None, None, None

        return (
            subtask_names,
            episodes_df.loc[ep_idx, col("subtask_start_frames")],
            episodes_df.loc[ep_idx, col("subtask_end_frames")],
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Encode images, text, and normalize states in the transition.

        Implements SARM training data preparation:
        - Applies language perturbation (20% probability)
        - Applies rewind augmentation (80% probability)
        - Generates stage+tau targets for all frames
        - Outputs lengths tensor for valid sequence masking
        """
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        frame_index = comp_data.get("index")
        episode_index = comp_data.get("episode_index")

        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")

        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = self._get_episode_indices(frame_indices, episode_index)

        image = observation.get(self.image_key)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # If 4D (T, C, H, W) from delta_timestamps, add batch dim
        # If 3D (C, H, W) single frame, add batch and time dims
        if image.ndim == 4:
            image = image[np.newaxis, ...]  # (T, C, H, W) -> (1, T, C, H, W)
        elif image.ndim == 3:
            image = image[np.newaxis, np.newaxis, ...]  # (C, H, W) -> (1, 1, C, H, W)

        batch_size = image.shape[0]
        total_frames = image.shape[1]  # Should be 13: 9 obs + 4 rewind placeholders
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        n_obs_frames = 1 + n_obs_steps  # 9 observation frames (including current)

        # Rewind augmentation
        rewind_steps = torch.zeros(batch_size, dtype=torch.int32)
        apply_rewind = self.training and random.random() < self.config.rewind_probability

        if apply_rewind and self.dataset_meta is not None:
            for b_idx, (ep_idx, frame_idx) in enumerate(
                zip(episode_indices.tolist(), frame_indices.tolist(), strict=True)
            ):
                ep_idx, frame_idx = int(ep_idx), int(frame_idx)
                ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]

                rewind_step, _ = apply_rewind_augmentation(
                    frame_idx, ep_start, n_obs_steps, max_rewind_steps, frame_gap=self.config.frame_gap
                )
                rewind_steps[b_idx] = rewind_step

        # Compute valid lengths: n_obs_frames + rewind_steps
        lengths = n_obs_frames + rewind_steps  # (B,)

        # Apply rewind masking to images
        # For frames beyond valid length, we mask with zeros (or copy last valid frame)
        for b_idx in range(batch_size):
            valid_len = lengths[b_idx].item()
            if valid_len < total_frames:
                image[b_idx, valid_len:] = 0  # Zero out frames beyond valid length

        # Encode images with CLIP
        video_features = self._encode_images_batch(image)
        observation["video_features"] = video_features

        state_key = self.config.state_key
        state_data = observation.get(state_key)

        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)  # (T, D) -> (1, T, D)
        elif state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (D,) -> (1, 1, D)

        # Apply same rewind masking to state
        for b_idx in range(batch_size):
            valid_len = lengths[b_idx].item()
            if valid_len < state_tensor.shape[1]:
                state_tensor[b_idx, valid_len:] = 0  # Zero out frames beyond valid length

        observation["state_features"] = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

        task = comp_data.get("task")
        if isinstance(task, list):
            task = task[0] if task else ""

        # Apply language perturbation during training (20% probability)
        # When perturbed, targets will be zeroed to train model to output low values for irrelevant text
        apply_perturbation = self.training and random.random() < self.config.language_perturbation_probability
        if apply_perturbation:
            task = self._generate_perturbed_task()

        # Encode text with CLIP
        observation["text_features"] = self._encode_text_clip(task, batch_size)

        # Store lengths for model
        observation["lengths"] = lengths

        # When language is perturbed, targets are zero so perturbed samples don't contribute to progress loss
        if self.dataset_meta is not None:
            episodes_df = None
            if self.sparse_subtask_names != ["task"]:
                episodes_df = self.dataset_meta.episodes.to_pandas()

            # Generate sparse targets
            if self.sparse_temporal_proportions is not None:
                if apply_perturbation:
                    # Zero targets when language is perturbed
                    sparse_targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)
                else:
                    sparse_targets = self._compute_batch_targets(
                        frame_indices, episode_indices, lengths, rewind_steps, episodes_df, "sparse"
                    )
                observation["sparse_targets"] = sparse_targets

            # Generate dense targets (for dual mode)
            if self.config.uses_dual_heads and self.dense_temporal_proportions is not None:
                if apply_perturbation:
                    # Zero targets when language is perturbed
                    dense_targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)
                else:
                    dense_targets = self._compute_batch_targets(
                        frame_indices, episode_indices, lengths, rewind_steps, episodes_df, "dense"
                    )
                observation["dense_targets"] = dense_targets

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def _compute_batch_targets(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
        lengths: torch.Tensor,
        rewind_steps: torch.Tensor,
        episodes_df: pd.DataFrame | None,
        annotation_type: str,
    ) -> torch.Tensor:
        """Compute stage+tau targets for a batch of samples."""
        batch_size = len(frame_indices)
        n_obs_steps = self.config.n_obs_steps
        max_rewind_steps = self.config.max_rewind_steps
        total_frames = 1 + n_obs_steps + max_rewind_steps
        frame_gap = self.config.frame_gap

        global_names, temporal_props = self._get_annotation_config(annotation_type)
        targets = torch.zeros(batch_size, total_frames, dtype=torch.float32)

        for b_idx in range(batch_size):
            ep_idx = int(episode_indices[b_idx])
            frame_idx = int(frame_indices[b_idx])

            ep_start = self.dataset_meta.episodes[ep_idx]["dataset_from_index"]
            ep_end = self.dataset_meta.episodes[ep_idx]["dataset_to_index"]
            ep_length = ep_end - ep_start

            subtask_names, subtask_start_frames, subtask_end_frames = self._load_episode_annotations(
                ep_idx, episodes_df, annotation_type, global_names
            )

            # Compute observation frame indices
            obs_indices, _ = compute_absolute_indices(
                frame_idx, ep_start, ep_end, n_obs_steps, frame_gap=frame_gap
            )
            obs_indices = obs_indices.tolist()

            # Compute targets for observation frames
            for t_idx, abs_idx in enumerate(obs_indices):
                rel_frame = abs_idx - ep_start
                targets[b_idx, t_idx] = find_stage_and_tau(
                    rel_frame,
                    ep_length,
                    subtask_names,
                    subtask_start_frames,
                    subtask_end_frames,
                    global_names,
                    temporal_props,
                    return_combined=True,
                )

            # Compute targets for rewind frames (if any)
            rewind_step = rewind_steps[b_idx].item()
            if rewind_step > 0:
                _, rewind_indices = apply_rewind_augmentation(
                    frame_idx,
                    ep_start,
                    n_obs_steps,
                    max_rewind_steps,
                    frame_gap=frame_gap,
                    rewind_step=rewind_step,
                )

                for r_idx, abs_idx in enumerate(rewind_indices[:rewind_step]):
                    rel_frame = max(0, abs_idx - ep_start)
                    targets[b_idx, n_obs_steps + 1 + r_idx] = find_stage_and_tau(
                        rel_frame,
                        ep_length,
                        subtask_names,
                        subtask_start_frames,
                        subtask_end_frames,
                        global_names,
                        temporal_props,
                        return_combined=True,
                    )

        return targets

    @property
    def training(self) -> bool:
        return getattr(self, "_training_mode", True)

    def train(self, mode: bool = True):
        """Set training mode for augmentation decisions."""
        self._training_mode = mode
        return self

    def eval(self):
        """Set evaluation mode (disable augmentations)."""
        return self.train(False)

    @torch.no_grad()
    def _encode_images_batch(self, images: np.ndarray) -> torch.Tensor:
        """Encode a batch of images using CLIP.

        Args:
            images: Batched images with shape: (B, T, C, H, W)

        Returns:
            Encoded feature vectors with shape (B, T, 512)
        """

        batch_size, seq_length = images.shape[0], images.shape[1]
        images = images.reshape(batch_size * seq_length, *images.shape[2:])

        num_frames = images.shape[0]
        images_list = []
        for i in range(num_frames):
            img = images[i]
            if img.shape[0] in [1, 3]:  # Channel first (C, H, W)
                img = img.transpose(1, 2, 0)

            # Handle single channel
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            images_list.append(Image.fromarray(img))

        all_embeddings = []
        for i in range(0, num_frames, self.config.clip_batch_size):
            batch_imgs = images_list[i : i + self.config.clip_batch_size]

            inputs = self.clip_processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get image embeddings
            embeddings = self.clip_model.get_image_features(**inputs).detach().cpu()

            # Handle single frame case
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)

            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings)  # (B*T, 512)
        all_embeddings = all_embeddings.reshape(batch_size, seq_length, -1)  # (B, T, 512)

        return all_embeddings

    @torch.no_grad()
    def _encode_text_clip(self, text: str, batch_size: int) -> torch.Tensor:
        """Encode text using CLIP text encoder (per SARM paper A.4).

        Args:
            text: Task description text to encode
            batch_size: Batch size to replicate for

        Returns:
            Encoded text features with shape (B, 512)
        """
        inputs = self.clip_processor.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        text_embedding = self.clip_model.get_text_features(**inputs).detach().cpu()
        text_embedding = text_embedding.expand(batch_size, -1)

        return text_embedding

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add encoded features to the observation features."""
        features[PipelineFeatureType.OBSERVATION]["video_features"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(self.config.num_frames, self.config.image_dim)
        )
        features[PipelineFeatureType.OBSERVATION]["text_features"] = PolicyFeature(
            type=FeatureType.LANGUAGE, shape=(self.config.text_dim,)
        )
        features[PipelineFeatureType.OBSERVATION]["state_features"] = PolicyFeature(
            type=FeatureType.STATE, shape=(self.config.num_frames, self.config.max_state_dim)
        )
        return features


def make_sarm_pre_post_processors(
    config: SARMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create pre-processor and post-processor pipelines for SARM."""
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[
                AddBatchDimensionProcessorStep(),
                RenameObservationsProcessorStep(rename_map={}),
                NormalizerProcessorStep(
                    features={**config.input_features, **config.output_features},
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                SARMEncodingProcessorStep(
                    config=config, dataset_meta=dataset_meta, dataset_stats=dataset_stats
                ),
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
