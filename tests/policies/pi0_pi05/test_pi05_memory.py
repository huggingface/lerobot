# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.memory import (
    encode_video_with_mem,
    sample_observation_history,
    temporal_sinusoidal_embedding,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi052.modeling_pi052 import PI052Policy


def test_pi05_memory_delta_indices_are_modality_specific():
    baseline = PI05Config()
    memory = PI05Config(
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=6,
        memory_stride=10,
    )
    assert baseline.image_observation_delta_indices is None
    assert baseline.state_observation_delta_indices is None
    assert memory.image_observation_delta_indices == [-50, -40, -30, -20, -10, 0]
    assert memory.state_observation_delta_indices == [-50, -40, -30, -20, -10, 0]


def test_delta_timestamps_respect_raw_robomme_rename_map():
    metadata = SimpleNamespace(
        fps=10,
        features={"image": {}, "wrist_image": {}, "state": {}, "actions": {}},
    )
    config = PI05Config(
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=3,
        memory_stride=5,
    )
    rename_map = {
        "image": "observation.images.camera1",
        "wrist_image": "observation.images.camera2",
        "state": "observation.state",
        "actions": "action",
    }
    deltas = resolve_delta_timestamps(config, metadata, rename_map)
    assert deltas["image"] == [-1.0, -0.5, 0.0]
    assert deltas["wrist_image"] == [-1.0, -0.5, 0.0]
    assert deltas["state"] == [-1.0, -0.5, 0.0]
    assert deltas["actions"] == [index / 10 for index in range(50)]


@pytest.mark.parametrize(
    "field",
    ["memory_frames", "memory_stride", "memory_temporal_attention_every"],
)
def test_pi05_memory_config_rejects_non_positive_values(field):
    with pytest.raises(ValueError, match=field):
        PI05Config(**{field: 0})


def test_inference_history_order_and_padding():
    history = [torch.full((2, 1), value) for value in range(11)]
    values, padding = sample_observation_history(history, num_frames=3, stride=5, steps_seen=1)
    assert values[:, :, 0].tolist() == [[0, 5, 10], [0, 5, 10]]
    assert padding.tolist() == [[True, True, False], [True, True, False]]


def test_current_temporal_position_is_exactly_zero():
    embedding = temporal_sinusoidal_embedding(4, 16, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(embedding[-1], torch.zeros(16))
    assert torch.count_nonzero(embedding[:-1]) > 0


def _tiny_siglip():
    transformers = pytest.importorskip("transformers")
    config = transformers.SiglipVisionConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        vision_use_head=False,
    )
    from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

    return SiglipVisionTransformer(config).eval()


def test_single_frame_mem_matches_original_siglip():
    model = _tiny_siglip()
    image = torch.randn(2, 3, 16, 16)
    expected = model(image).last_hidden_state
    actual = encode_video_with_mem(
        model,
        image[:, None],
        torch.ones(2, 1, dtype=torch.bool),
        temporal_attention_every=4,
    )
    torch.testing.assert_close(actual, expected)


def test_mem_video_encoder_compresses_time_and_backpropagates():
    model = _tiny_siglip()
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    video = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
    output = encode_video_with_mem(
        model,
        video,
        torch.ones(2, 3, dtype=torch.bool),
        temporal_attention_every=4,
    )
    output.sum().backward()
    assert output.shape == (2, 4, 16)
    assert {id(parameter) for parameter in model.parameters()} == parameter_ids
    assert video.grad is not None


def test_masked_history_cannot_change_current_embedding():
    model = _tiny_siglip()
    first = torch.randn(1, 3, 3, 16, 16)
    second = first.clone()
    second[:, :2] = torch.randn_like(second[:, :2]) * 100
    frame_mask = torch.tensor([[False, False, True]])
    first_output = encode_video_with_mem(model, first, frame_mask, temporal_attention_every=4)
    second_output = encode_video_with_mem(model, second, frame_mask, temporal_attention_every=4)
    torch.testing.assert_close(first_output, second_output)


class _EmbeddingStub(nn.Module):
    def embed_image(self, image, **kwargs):
        return torch.zeros(image.shape[0], 2, 8)

    def embed_language_tokens(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 8)


def test_proprioceptive_history_adds_one_masked_token_per_frame():
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(memory_temporal_attention_every=4)
    model.checkpoint_vision_embeddings = False
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _EmbeddingStub()
    model.proprio_history_proj = nn.Linear(4, 8)
    images = [torch.zeros(2, 3, 16, 16)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    states = torch.randn(2, 3, 4)
    state_masks = torch.tensor([[False, True, True], [True, True, True]])
    tokens = torch.ones(2, 5, dtype=torch.long)
    token_masks = torch.ones(2, 5, dtype=torch.bool)
    embeddings, padding, _ = model.embed_prefix(images, image_masks, tokens, token_masks, states, state_masks)
    assert embeddings.shape == (2, 10, 8)
    torch.testing.assert_close(padding[:, 2:5], state_masks)


def test_pi052_plain_flow_bypasses_subtask_generation():
    policy = PI052Policy.__new__(PI052Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(recipe_path=None)
    batch = {"sentinel": torch.tensor(1)}

    assert policy._prepare_action_batch(batch) is batch


def test_pi052_base_checkpoint_keeps_fresh_proprio_memory_projection():
    policy = PI052Policy.__new__(PI052Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_proprioceptive_memory=True)
    policy.model = nn.Module()
    policy.model.proprio_history_proj = nn.Linear(4, 8)

    state_dict = policy._prepare_pretrained_state_dict({})

    assert "model.proprio_history_proj.weight" in state_dict
    assert "model.proprio_history_proj.bias" in state_dict
