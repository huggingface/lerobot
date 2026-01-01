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

import pytest

pytest.importorskip("faker")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.processor.core import TransitionKey


class MockDatasetMeta:
    """Mock dataset metadata for testing processor."""

    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    @property
    def episodes(self):
        """Return episodes as a mock object with to_pandas() method."""
        mock = MagicMock()
        mock.__len__ = lambda s: len(self._episodes)
        mock.__getitem__ = lambda s, idx: self._episodes[idx]
        mock.to_pandas = lambda: pd.DataFrame(self._episodes)
        return mock


class MockConfig:
    """Mock SARMConfig for testing processor methods."""

    def __init__(
        self,
        n_obs_steps: int = 8,
        max_rewind_steps: int = 4,
        frame_gap: int = 30,
        sparse_subtask_names: list = None,
        sparse_temporal_proportions: list = None,
        dense_subtask_names: list = None,
        dense_temporal_proportions: list = None,
        image_key: str = "observation.images.top",
        state_key: str = "observation.state",
        max_state_dim: int = 32,
        device: str = None,
        rewind_probability: float = 0.8,
        language_perturbation_probability: float = 0.2,
        annotation_mode: str = "dual",
        clip_batch_size: int = 64,
        text_dim: int = 512,
    ):
        self.n_obs_steps = n_obs_steps
        self.max_rewind_steps = max_rewind_steps
        self.frame_gap = frame_gap
        self.sparse_subtask_names = sparse_subtask_names or ["task"]
        self.sparse_temporal_proportions = sparse_temporal_proportions or [1.0]
        self.dense_subtask_names = dense_subtask_names
        self.dense_temporal_proportions = dense_temporal_proportions
        self.uses_dual_heads = annotation_mode in ["dense_only", "dual"]
        self.image_key = image_key
        self.state_key = state_key
        self.max_state_dim = max_state_dim
        self.device = device
        self.rewind_probability = rewind_probability
        self.language_perturbation_probability = language_perturbation_probability
        self.annotation_mode = annotation_mode
        self.clip_batch_size = clip_batch_size
        self.text_dim = text_dim

        # Compute observation delta indices (same as config: bidirectional)
        half_steps = self.n_obs_steps // 2
        past_deltas = [-self.frame_gap * i for i in range(half_steps, 0, -1)]
        future_deltas = [self.frame_gap * i for i in range(1, half_steps + 1)]
        obs_deltas = past_deltas + [0] + future_deltas
        rewind_deltas = [-self.frame_gap * (i + 1) for i in range(self.max_rewind_steps)]
        self.observation_delta_indices = obs_deltas + rewind_deltas

    @property
    def num_frames(self) -> int:
        return 1 + self.n_obs_steps + self.max_rewind_steps


class TestSARMEncodingProcessorStepEndToEnd:
    """End-to-end test for SARMEncodingProcessorStep with dummy batch data."""

    @pytest.fixture
    def mock_clip_model(self):
        """Mock CLIP model to avoid loading real weights."""
        with (
            patch("lerobot.policies.sarm.processor_sarm.CLIPModel") as mock_model_cls,
            patch("lerobot.policies.sarm.processor_sarm.CLIPProcessor") as mock_processor_cls,
        ):
            # Mock the CLIP model - return embeddings based on input batch size
            mock_model = MagicMock()

            def get_image_features_side_effect(**kwargs):
                pixel_values = kwargs.get("pixel_values")
                batch_size = pixel_values.shape[0] if pixel_values is not None else 1
                return torch.randn(batch_size, 512)

            mock_model.get_image_features.side_effect = get_image_features_side_effect
            mock_model.get_text_features.return_value = torch.randn(1, 512)
            mock_model.to.return_value = mock_model
            mock_model_cls.from_pretrained.return_value = mock_model

            # Mock the CLIP processor - return tensors based on input images
            mock_processor = MagicMock()

            def processor_side_effect(images=None, **kwargs):
                num_images = len(images) if images is not None else 1
                return {
                    "pixel_values": torch.randn(num_images, 3, 224, 224),
                }

            mock_processor.side_effect = processor_side_effect
            # Mock tokenizer for text encoding
            mock_processor.tokenizer.return_value = {
                "input_ids": torch.ones(1, 77, dtype=torch.long),
                "attention_mask": torch.ones(1, 77, dtype=torch.long),
            }
            mock_processor_cls.from_pretrained.return_value = mock_processor

            yield mock_model, mock_processor

    @pytest.fixture
    def processor_with_mocks(self, mock_clip_model):
        """Create a processor with mocked CLIP and dataset metadata for dual mode."""
        from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep

        # Dual mode config with both sparse and dense annotations
        config = MockConfig(
            n_obs_steps=8,
            max_rewind_steps=4,
            frame_gap=30,
            rewind_probability=0.0,  # Disable for deterministic test
            language_perturbation_probability=0.0,  # Disable for deterministic test
            annotation_mode="dual",
            sparse_subtask_names=["reach", "grasp", "lift"],
            sparse_temporal_proportions=[0.3, 0.4, 0.3],
            dense_subtask_names=["approach", "contact", "close_gripper", "lift_up"],
            dense_temporal_proportions=[0.25, 0.25, 0.25, 0.25],
        )

        # Create mock dataset metadata with one episode of 300 frames
        # Include annotation columns for dual mode
        episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 300,
                "task": "pick up the cube",
                "sparse_subtask_names": ["reach", "grasp", "lift"],
                "sparse_subtask_start_frames": [0, 90, 210],
                "sparse_subtask_end_frames": [90, 210, 300],
                "dense_subtask_names": ["approach", "contact", "close_gripper", "lift_up"],
                "dense_subtask_start_frames": [0, 75, 150, 225],
                "dense_subtask_end_frames": [75, 150, 225, 300],
            }
        ]
        dataset_meta = MockDatasetMeta(episodes)

        processor = SARMEncodingProcessorStep(
            config=config,
            dataset_meta=dataset_meta,
        )
        processor.train(True)  # Use train() method, not direct assignment

        return processor, config

    def test_call_with_single_frame_batch(self, processor_with_mocks):
        """Test processor __call__ with a single-frame batch."""
        processor, config = processor_with_mocks

        # Create dummy input transition
        batch_size = 1
        num_frames = config.num_frames  # 13 frames (9 obs + 4 rewind)

        # Image: (T, C, H, W) format as expected by processor
        dummy_image = np.random.rand(num_frames, 3, 224, 224).astype(np.float32)

        # State: (T, D) format
        dummy_state = np.random.rand(num_frames, 6).astype(np.float32)

        transition = {
            TransitionKey.OBSERVATION: {
                config.image_key: dummy_image,
                config.state_key: dummy_state,
            },
            TransitionKey.COMPLEMENTARY_DATA: {
                "index": 150,  # Middle of episode
                "episode_index": 0,
                "task": "pick up the cube",
            },
        }

        # Run processor
        result = processor(transition)

        # Verify output structure
        obs = result[TransitionKey.OBSERVATION]

        # Check video features exist and have correct shape
        assert "video_features" in obs
        video_features = obs["video_features"]
        assert video_features.shape[0] == batch_size
        assert video_features.shape[1] == num_frames
        assert video_features.shape[2] == 512  # CLIP embedding dim

        # Check state features exist and have correct shape
        assert "state_features" in obs
        state_features = obs["state_features"]
        assert state_features.shape[0] == batch_size
        assert state_features.shape[1] == num_frames
        assert state_features.shape[2] == config.max_state_dim  # Padded to max_state_dim

        # Check text features exist and have correct shape
        assert "text_features" in obs
        text_features = obs["text_features"]
        assert text_features.shape[0] == batch_size
        assert text_features.shape[1] == 512  # CLIP embedding dim

        # Check lengths tensor
        assert "lengths" in obs
        lengths = obs["lengths"]
        assert lengths.shape[0] == batch_size
        assert lengths.dtype == torch.int32

        # Check sparse_targets exist
        assert "sparse_targets" in obs
        sparse_targets = obs["sparse_targets"]
        assert sparse_targets.shape == (batch_size, num_frames)
        # All targets should be in [0, max_stages] range (stage.tau format)
        assert (sparse_targets >= 0).all()

        # Check dense_targets exist (for dual mode)
        assert "dense_targets" in obs
        dense_targets = obs["dense_targets"]
        assert dense_targets.shape == (batch_size, num_frames)
        assert (dense_targets >= 0).all()

    def test_call_with_batched_input(self, mock_clip_model):
        """Test processor __call__ with a batched input (multiple frames) in dual mode."""
        from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep

        config = MockConfig(
            n_obs_steps=8,
            max_rewind_steps=4,
            frame_gap=30,
            rewind_probability=0.0,
            language_perturbation_probability=0.0,
            annotation_mode="dual",
            sparse_subtask_names=["reach", "grasp"],
            sparse_temporal_proportions=[0.5, 0.5],
            dense_subtask_names=["step1", "step2", "step3"],
            dense_temporal_proportions=[0.33, 0.34, 0.33],
        )

        # Two episodes with different lengths, each with sparse+dense annotations
        episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 200,
                "task": "task A",
                "sparse_subtask_names": ["reach", "grasp"],
                "sparse_subtask_start_frames": [0, 100],
                "sparse_subtask_end_frames": [100, 200],
                "dense_subtask_names": ["step1", "step2", "step3"],
                "dense_subtask_start_frames": [0, 66, 133],
                "dense_subtask_end_frames": [66, 133, 200],
            },
            {
                "dataset_from_index": 200,
                "dataset_to_index": 500,
                "task": "task B",
                "sparse_subtask_names": ["reach", "grasp"],
                "sparse_subtask_start_frames": [200, 350],
                "sparse_subtask_end_frames": [350, 500],
                "dense_subtask_names": ["step1", "step2", "step3"],
                "dense_subtask_start_frames": [200, 300, 400],
                "dense_subtask_end_frames": [300, 400, 500],
            },
        ]
        dataset_meta = MockDatasetMeta(episodes)

        processor = SARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
        processor.train(True)

        batch_size = 2
        num_frames = config.num_frames

        # Image: (B, T, C, H, W) format
        dummy_image = np.random.rand(batch_size, num_frames, 3, 224, 224).astype(np.float32)
        dummy_state = np.random.rand(batch_size, num_frames, 6).astype(np.float32)

        transition = {
            TransitionKey.OBSERVATION: {
                config.image_key: dummy_image,
                config.state_key: dummy_state,
            },
            TransitionKey.COMPLEMENTARY_DATA: {
                "index": np.array([100, 350]),  # One frame from each episode
                "episode_index": np.array([0, 1]),
                "task": ["task A", "task B"],
            },
        }

        result = processor(transition)
        obs = result[TransitionKey.OBSERVATION]

        # Verify batch dimension is preserved for all outputs
        assert obs["video_features"].shape[0] == batch_size
        assert obs["state_features"].shape[0] == batch_size
        assert obs["lengths"].shape[0] == batch_size
        assert obs["sparse_targets"].shape[0] == batch_size
        assert obs["dense_targets"].shape[0] == batch_size  # Dual mode has dense targets

    def test_targets_increase_with_progress(self, mock_clip_model):
        """Test that both sparse and dense targets increase as frame index progresses."""
        from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep

        config = MockConfig(
            n_obs_steps=8,
            max_rewind_steps=4,
            frame_gap=30,
            rewind_probability=0.0,
            language_perturbation_probability=0.0,
            annotation_mode="dual",
            sparse_subtask_names=["phase1", "phase2"],
            sparse_temporal_proportions=[0.5, 0.5],
            dense_subtask_names=["a", "b", "c", "d"],
            dense_temporal_proportions=[0.25, 0.25, 0.25, 0.25],
        )

        episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 300,
                "task": "test task",
                "sparse_subtask_names": ["phase1", "phase2"],
                "sparse_subtask_start_frames": [0, 150],
                "sparse_subtask_end_frames": [150, 300],
                "dense_subtask_names": ["a", "b", "c", "d"],
                "dense_subtask_start_frames": [0, 75, 150, 225],
                "dense_subtask_end_frames": [75, 150, 225, 300],
            }
        ]
        dataset_meta = MockDatasetMeta(episodes)

        processor = SARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
        processor.train(True)

        num_frames = config.num_frames

        # Test at early, middle, and late points in episode
        frame_indices = [30, 150, 270]
        sparse_center_targets = []
        dense_center_targets = []

        for frame_idx in frame_indices:
            dummy_image = np.random.rand(num_frames, 3, 224, 224).astype(np.float32)
            dummy_state = np.random.rand(num_frames, 6).astype(np.float32)

            transition = {
                TransitionKey.OBSERVATION: {
                    config.image_key: dummy_image,
                    config.state_key: dummy_state,
                },
                TransitionKey.COMPLEMENTARY_DATA: {
                    "index": frame_idx,
                    "episode_index": 0,
                    "task": "test task",
                },
            }

            result = processor(transition)
            obs = result[TransitionKey.OBSERVATION]
            # Get target at center frame (index 4 in 9-frame observation window)
            sparse_center_targets.append(obs["sparse_targets"][0, 4].item())
            dense_center_targets.append(obs["dense_targets"][0, 4].item())

        # Both sparse and dense targets should increase with frame index
        assert sparse_center_targets[0] < sparse_center_targets[2], (
            f"Early sparse target ({sparse_center_targets[0]}) should be < late ({sparse_center_targets[2]})"
        )
        assert dense_center_targets[0] < dense_center_targets[2], (
            f"Early dense target ({dense_center_targets[0]}) should be < late ({dense_center_targets[2]})"
        )

    def test_progress_labels_exact_values(self, mock_clip_model):
        """Test that progress labels (stage.tau) are computed correctly for known positions."""
        from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep

        # Simple setup: 2 sparse stages, 4 dense stages, 100 frame episode
        config = MockConfig(
            n_obs_steps=8,
            max_rewind_steps=4,
            frame_gap=10,  # Smaller gap for easier calculation
            rewind_probability=0.0,
            language_perturbation_probability=0.0,
            annotation_mode="dual",
            sparse_subtask_names=["A", "B"],
            sparse_temporal_proportions=[0.5, 0.5],
            dense_subtask_names=["d1", "d2", "d3", "d4"],
            dense_temporal_proportions=[0.25, 0.25, 0.25, 0.25],
        )

        # Episode: frames 0-99, sparse stages at [0-49], [50-99]
        # Dense stages at [0-24], [25-49], [50-74], [75-99]
        episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 100,
                "task": "test",
                "sparse_subtask_names": ["A", "B"],
                "sparse_subtask_start_frames": [0, 50],
                "sparse_subtask_end_frames": [50, 100],
                "dense_subtask_names": ["d1", "d2", "d3", "d4"],
                "dense_subtask_start_frames": [0, 25, 50, 75],
                "dense_subtask_end_frames": [25, 50, 75, 100],
            }
        ]
        dataset_meta = MockDatasetMeta(episodes)

        processor = SARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
        processor.train(True)

        num_frames = config.num_frames

        # Test at frame 50 (center of episode)
        # With frame_gap=10, n_obs_steps=8:
        # obs indices around frame 50: [10, 20, 30, 40, 50, 60, 70, 80, 90] (9 frames)
        dummy_image = np.random.rand(num_frames, 3, 224, 224).astype(np.float32)
        dummy_state = np.random.rand(num_frames, 6).astype(np.float32)

        transition = {
            TransitionKey.OBSERVATION: {
                config.image_key: dummy_image,
                config.state_key: dummy_state,
            },
            TransitionKey.COMPLEMENTARY_DATA: {
                "index": 50,
                "episode_index": 0,
                "task": "test",
            },
        }

        result = processor(transition)
        obs = result[TransitionKey.OBSERVATION]
        sparse_targets = obs["sparse_targets"][0]  # (13,)
        dense_targets = obs["dense_targets"][0]  # (13,)

        # First 9 frames are observation frames, last 4 are rewind placeholders (zeros when no rewind)
        # Check that obs frames have non-zero targets
        obs_sparse = sparse_targets[:9]
        obs_dense = dense_targets[:9]

        # Verify targets are monotonically increasing for observation frames
        for i in range(1, 9):
            assert obs_sparse[i] >= obs_sparse[i - 1], (
                f"Sparse targets should be monotonic: {obs_sparse[i - 1].item():.3f} -> {obs_sparse[i].item():.3f}"
            )
            assert obs_dense[i] >= obs_dense[i - 1], (
                f"Dense targets should be monotonic: {obs_dense[i - 1].item():.3f} -> {obs_dense[i].item():.3f}"
            )

        # Rewind slots should be zero when rewind is disabled
        rewind_targets = sparse_targets[9:]
        assert (rewind_targets == 0).all(), "Rewind slots should be zero when rewind is disabled"

        # Check stage transitions: frame 50 is at boundary of sparse stage A->B
        # Center frame (index 4) corresponds to actual frame 50
        center_sparse = obs_sparse[4].item()
        # At frame 50, sparse stage B starts, so target should be ~1.0 (stage 1 + tau 0)
        assert 0.9 <= center_sparse <= 1.1, (
            f"At sparse boundary, target should be ~1.0, got {center_sparse:.3f}"
        )

    def test_rewind_augmentation_applied(self, mock_clip_model):
        """Test that rewind augmentation correctly extends sequence and generates targets."""
        import random

        from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep

        config = MockConfig(
            n_obs_steps=8,
            max_rewind_steps=4,
            frame_gap=10,
            rewind_probability=1.0,  # Always apply rewind
            language_perturbation_probability=0.0,
            annotation_mode="dual",
            sparse_subtask_names=["A", "B"],
            sparse_temporal_proportions=[0.5, 0.5],
            dense_subtask_names=["d1", "d2"],
            dense_temporal_proportions=[0.5, 0.5],
        )

        episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 200,
                "task": "test",
                "sparse_subtask_names": ["A", "B"],
                "sparse_subtask_start_frames": [0, 100],
                "sparse_subtask_end_frames": [100, 200],
                "dense_subtask_names": ["d1", "d2"],
                "dense_subtask_start_frames": [0, 100],
                "dense_subtask_end_frames": [100, 200],
            }
        ]
        dataset_meta = MockDatasetMeta(episodes)

        processor = SARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
        processor.train(True)

        num_frames = config.num_frames  # 13

        # Test at frame 150 (center of bidirectional window)
        # With n_obs_steps=8, half_steps=4, frame_gap=10:
        # - Earliest obs frame = 150 - 4*10 = 110
        # - Rewind can go back from 110 to frames like 100, 90, 80, 70
        # - History available = 110 - 0 = 110, so max rewind = 110/10 = 11 (capped at 4)
        dummy_image = np.random.rand(num_frames, 3, 224, 224).astype(np.float32)
        dummy_state = np.random.rand(num_frames, 6).astype(np.float32)

        transition = {
            TransitionKey.OBSERVATION: {
                config.image_key: dummy_image,
                config.state_key: dummy_state,
            },
            TransitionKey.COMPLEMENTARY_DATA: {
                "index": 150,
                "episode_index": 0,
                "task": "test",
            },
        }

        # Seed random for reproducibility
        random.seed(42)
        result = processor(transition)
        obs = result[TransitionKey.OBSERVATION]

        lengths = obs["lengths"][0].item()
        sparse_targets = obs["sparse_targets"][0]

        # With rewind_probability=1.0 and enough history, lengths should be > 9 (9 obs + some rewind)
        assert lengths > 9, f"With rewind enabled, lengths should be > 9, got {lengths}"
        assert lengths <= num_frames, f"Lengths should not exceed total frames {num_frames}, got {lengths}"

        # Rewind targets should be non-zero for frames within valid length
        n_obs_frames = 9
        rewind_count = lengths - n_obs_frames

        if rewind_count > 0:
            # Check that rewind frames have targets
            rewind_targets = sparse_targets[n_obs_frames : n_obs_frames + rewind_count]
            # Rewind frames are from BEFORE the earliest obs frame (110)
            # These frames (100, 90, 80, 70) are earlier in the episode
            earliest_obs_target = sparse_targets[0].item()  # Frame 110

            # Rewind targets should be less than earliest obs (they're from earlier frames)
            for i, rt in enumerate(rewind_targets):
                assert rt.item() < earliest_obs_target, (
                    f"Rewind target {i} ({rt.item():.3f}) should be < earliest obs ({earliest_obs_target:.3f})"
                )

            # Rewind targets should be decreasing (going further back in time)
            for i in range(1, len(rewind_targets)):
                assert rewind_targets[i] <= rewind_targets[i - 1], (
                    f"Rewind targets should decrease: {rewind_targets[i - 1].item():.3f} -> {rewind_targets[i].item():.3f}"
                )

    def test_full_sequence_target_consistency(self, mock_clip_model):
        """Test that the full sequence of targets is consistent with frame positions."""
        from lerobot.policies.sarm.processor_sarm import SARMEncodingProcessorStep
        from lerobot.policies.sarm.sarm_utils import find_stage_and_tau

        config = MockConfig(
            n_obs_steps=8,
            max_rewind_steps=4,
            frame_gap=10,
            rewind_probability=0.0,
            language_perturbation_probability=0.0,
            annotation_mode="dual",
            sparse_subtask_names=["s1", "s2", "s3"],
            sparse_temporal_proportions=[0.33, 0.34, 0.33],
            dense_subtask_names=["d1", "d2"],
            dense_temporal_proportions=[0.5, 0.5],
        )

        # 3 sparse stages: [0-33), [33-66), [66-99]
        # 2 dense stages: [0-50), [50-100)
        episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 100,
                "task": "test",
                "sparse_subtask_names": ["s1", "s2", "s3"],
                "sparse_subtask_start_frames": [0, 33, 66],
                "sparse_subtask_end_frames": [33, 66, 100],
                "dense_subtask_names": ["d1", "d2"],
                "dense_subtask_start_frames": [0, 50],
                "dense_subtask_end_frames": [50, 100],
            }
        ]
        dataset_meta = MockDatasetMeta(episodes)

        processor = SARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
        processor.train(True)

        num_frames = config.num_frames

        # Test at frame 50 (middle of episode)
        dummy_image = np.random.rand(num_frames, 3, 224, 224).astype(np.float32)
        dummy_state = np.random.rand(num_frames, 6).astype(np.float32)

        transition = {
            TransitionKey.OBSERVATION: {
                config.image_key: dummy_image,
                config.state_key: dummy_state,
            },
            TransitionKey.COMPLEMENTARY_DATA: {
                "index": 50,
                "episode_index": 0,
                "task": "test",
            },
        }

        result = processor(transition)
        obs = result[TransitionKey.OBSERVATION]
        sparse_targets = obs["sparse_targets"][0]
        dense_targets = obs["dense_targets"][0]

        # Manually compute expected targets for observation frames
        # With frame_gap=10, n_obs_steps=8, center at 50:
        # obs frames: [10, 20, 30, 40, 50, 60, 70, 80, 90]
        expected_obs_frames = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        sparse_names = ["s1", "s2", "s3"]
        sparse_starts = [0, 33, 66]
        sparse_ends = [33, 66, 100]
        sparse_props = {"s1": 0.33, "s2": 0.34, "s3": 0.33}

        dense_names = ["d1", "d2"]
        dense_starts = [0, 50]
        dense_ends = [50, 100]
        dense_props = {"d1": 0.5, "d2": 0.5}

        for i, frame in enumerate(expected_obs_frames):
            expected_sparse = find_stage_and_tau(
                frame,
                100,
                sparse_names,
                sparse_starts,
                sparse_ends,
                sparse_names,
                sparse_props,
                return_combined=True,
            )
            expected_dense = find_stage_and_tau(
                frame,
                100,
                dense_names,
                dense_starts,
                dense_ends,
                dense_names,
                dense_props,
                return_combined=True,
            )

            actual_sparse = sparse_targets[i].item()
            actual_dense = dense_targets[i].item()

            assert abs(actual_sparse - expected_sparse) < 0.01, (
                f"Frame {frame}: sparse mismatch {actual_sparse:.3f} vs expected {expected_sparse:.3f}"
            )
            assert abs(actual_dense - expected_dense) < 0.01, (
                f"Frame {frame}: dense mismatch {actual_dense:.3f} vs expected {expected_dense:.3f}"
            )
