#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Regression tests for VQBeT modeling internals."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTHead  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def _make_minimal_config(n_vqvae_training_steps: int = 3) -> VQBeTConfig:
    """Return a small VQBeTConfig suitable for CPU unit tests."""
    config = VQBeTConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    # Use tiny sizes so the test is fast on CPU.
    config.n_vqvae_training_steps = n_vqvae_training_steps
    config.vqvae_n_embed = 8
    config.vqvae_embedding_dim = 32
    config.vqvae_enc_hidden_dim = 32
    config.action_chunk_size = 2
    config.crop_shape = (84, 84)
    config.device = "cpu"
    return config


def test_discretize_updates_buffers_in_place():
    """Regression test: discretize() must update registered buffers in-place.

    Replacing them with `self.vqvae_model.discretized = torch.tensor(True)` creates
    a new CPU tensor and breaks DDP buffer synchronisation with NCCL when the model
    is on GPU (RuntimeError: No backend type associated with device type cpu).

    This test verifies that after discretization the underlying tensor storage
    (data_ptr) is unchanged, i.e. no new tensor was allocated.
    """
    config = _make_minimal_config(n_vqvae_training_steps=3)
    head = VQBeTHead(config)
    head.eval()

    vqvae = head.vqvae_model
    vq_layer = vqvae.vq_layer

    # Capture storage addresses of the two registered buffers before discretization.
    discretized_ptr_before = vqvae.discretized.data_ptr()
    freeze_codebook_ptr_before = vq_layer.freeze_codebook.data_ptr()

    # Both flags should be False at init.
    assert not vqvae.discretized.item(), "discretized should be False before training"
    assert not vq_layer.freeze_codebook.item(), "freeze_codebook should be False before training"

    # Run discretize() until the threshold is crossed (n_vqvae_training_steps calls).
    batch_size = 4
    seq_len = config.action_chunk_size  # minimum sequence length that produces at least one chunk
    action_dim = config.action_feature.shape[0]
    dummy_actions = torch.randn(batch_size, seq_len, action_dim)

    n_steps = config.n_vqvae_training_steps
    for _ in range(n_steps):
        head.discretize(n_steps, dummy_actions)

    # After discretization both flags must be True.
    assert vqvae.discretized.item(), "discretized should be True after training"
    assert vq_layer.freeze_codebook.item(), "freeze_codebook should be True after training"

    # Most importantly: the storage must be the *same* tensor objects (in-place update).
    # If a new tensor was created the data_ptr would differ, and DDP would try to
    # NCCL-broadcast a CPU tensor when the model lives on GPU.
    assert vqvae.discretized.data_ptr() == discretized_ptr_before, (
        "vqvae_model.discretized was replaced with a new tensor instead of being updated in-place. "
        "This breaks DDP GPU buffer synchronisation."
    )
    assert vq_layer.freeze_codebook.data_ptr() == freeze_codebook_ptr_before, (
        "vq_layer.freeze_codebook was replaced with a new tensor instead of being updated in-place. "
        "This breaks DDP GPU buffer synchronisation."
    )

    # The buffers must still appear in the module's named_buffers() after update.
    buffer_names = {name for name, _ in vqvae.named_buffers()}
    assert "discretized" in buffer_names, (
        "vqvae_model.discretized is no longer a registered buffer after discretize(). "
        "Use .fill_() instead of direct assignment to preserve buffer registration."
    )
    freeze_buffer_names = {name for name, _ in vq_layer.named_buffers()}
    assert "freeze_codebook" in freeze_buffer_names, (
        "vq_layer.freeze_codebook is no longer a registered buffer after discretize(). "
        "Use .fill_() instead of direct assignment to preserve buffer registration."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_discretize_keeps_buffers_on_gpu():
    """Regression test: discretize() must not move registered buffers from GPU to CPU.

    This is the exact failure mode that caused the DDP crash:
        RuntimeError: No backend type associated with device type cpu

    When `self.vqvae_model.discretized = torch.tensor(True)` is used (wrong),
    torch.tensor() creates a CPU tensor by default, silently moving the buffer
    off the GPU. DDP's _sync_buffers() then tries to NCCL-broadcast a CPU tensor,
    which NCCL does not support.

    This test places the model on GPU and verifies that after discretize() both
    buffers remain on CUDA, preventing the above regression.
    """
    config = _make_minimal_config(n_vqvae_training_steps=3)
    head = VQBeTHead(config)
    device = torch.device("cuda:0")
    head = head.to(device)

    vqvae = head.vqvae_model
    vq_layer = vqvae.vq_layer

    # Confirm buffers start on GPU.
    assert vqvae.discretized.device.type == "cuda", "discretized should start on CUDA"
    assert vq_layer.freeze_codebook.device.type == "cuda", "freeze_codebook should start on CUDA"

    # Run discretize() until the threshold is crossed.
    batch_size = 4
    seq_len = config.action_chunk_size
    action_dim = config.action_feature.shape[0]
    dummy_actions = torch.randn(batch_size, seq_len, action_dim, device=device)

    n_steps = config.n_vqvae_training_steps
    for _ in range(n_steps):
        head.discretize(n_steps, dummy_actions)

    # Flags must be True.
    assert vqvae.discretized.item(), "discretized should be True after training"
    assert vq_layer.freeze_codebook.item(), "freeze_codebook should be True after training"

    # Core assertion: buffers must still live on GPU after discretize().
    # A direct-assignment `= torch.tensor(True)` creates a CPU tensor and fails here.
    assert vqvae.discretized.device.type == "cuda", (
        "vqvae_model.discretized was moved to CPU during discretize(). "
        "This would cause 'RuntimeError: No backend type associated with device type cpu' "
        "in DDP._sync_buffers(). Use .fill_(True) instead of = torch.tensor(True)."
    )
    assert vq_layer.freeze_codebook.device.type == "cuda", (
        "vq_layer.freeze_codebook was moved to CPU during discretize(). "
        "This would cause 'RuntimeError: No backend type associated with device type cpu' "
        "in DDP._sync_buffers(). Use .fill_(True) instead of = torch.tensor(True)."
    )
