# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.visual_memory import (
    causal_temporal_mask,
    encode_video_with_mem,
    sample_visual_history,
    temporal_sinusoidal_embedding,
)


def test_visual_memory_observation_delta_indices():
    baseline = SmolVLAConfig()
    memory = SmolVLAConfig(use_visual_memory=True, visual_memory_frames=6, visual_memory_stride=10)

    assert baseline.observation_delta_indices == [0]
    assert memory.observation_delta_indices == [-50, -40, -30, -20, -10, 0]


def test_delta_timestamps_respect_raw_dataset_rename_map():
    class RawMetadata:
        fps = 10
        features = {"image": {}, "state": {}, "actions": {}}

    config = SmolVLAConfig(use_visual_memory=True, visual_memory_frames=3, visual_memory_stride=5)
    delta_timestamps = resolve_delta_timestamps(
        config,
        RawMetadata(),
        {
            "image": "observation.images.camera1",
            "state": "observation.state",
            "actions": "action",
        },
    )

    assert delta_timestamps["image"] == [-1.0, -0.5, 0.0]
    assert delta_timestamps["state"] == [-1.0, -0.5, 0.0]
    assert delta_timestamps["actions"] == [index / 10 for index in range(50)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visual_memory_frames", 0),
        ("visual_memory_stride", 0),
        ("visual_memory_temporal_attention_every", 0),
    ],
)
def test_visual_memory_config_rejects_non_positive_values(field, value):
    with pytest.raises(ValueError, match=field):
        SmolVLAConfig(**{field: value})


def test_current_temporal_position_is_exactly_zero():
    embedding = temporal_sinusoidal_embedding(4, 16, device=torch.device("cpu"), dtype=torch.float32)

    torch.testing.assert_close(embedding[-1], torch.zeros(16))
    assert torch.count_nonzero(embedding[:-1]) > 0


def test_causal_temporal_mask_combines_causality_and_padding():
    frame_mask = torch.tensor([[False, True, True]])
    mask = causal_temporal_mask(frame_mask, dtype=torch.float32, num_patches=2)

    assert mask.shape == (2, 1, 3, 3)
    assert mask[0, 0, 1, 0] < -1e30
    assert mask[0, 0, 1, 1] == 0
    assert mask[0, 0, 1, 2] < -1e30
    assert mask[0, 0, 2, 1] == 0


def test_inference_history_matches_training_order_and_padding():
    history = [torch.full((2, 1), value) for value in range(11)]

    initial_video, initial_padding = sample_visual_history(history, num_frames=3, stride=5, steps_seen=1)
    full_video, full_padding = sample_visual_history(history, num_frames=3, stride=5, steps_seen=11)

    assert initial_video[:, :, 0].tolist() == [[0, 5, 10], [0, 5, 10]]
    assert initial_padding.tolist() == [[True, True, False], [True, True, False]]
    torch.testing.assert_close(full_video[:, :, 0], torch.tensor([[0, 5, 10], [0, 5, 10]]))
    assert not full_padding.any()


def test_single_frame_mem_matches_original_siglip_encoder():
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

    model = SiglipVisionTransformer(config).eval()
    image = torch.randn(2, 3, 16, 16)

    expected = model(image).last_hidden_state
    actual = encode_video_with_mem(
        model,
        image[:, None],
        torch.ones(2, 1, dtype=torch.bool),
        temporal_attention_every=4,
    )

    torch.testing.assert_close(actual, expected)


def test_mem_video_encoder_compresses_time_without_new_parameters():
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

    model = SiglipVisionTransformer(config)
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


def test_mem_video_encoder_supports_smolvlm_vision_tower():
    transformers = pytest.importorskip("transformers")
    config = transformers.SmolVLMVisionConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer

    model = SmolVLMVisionTransformer(config).eval()
    video = torch.randn(2, 3, 3, 16, 16)

    output = encode_video_with_mem(
        model,
        video,
        torch.ones(2, 3, dtype=torch.bool),
        temporal_attention_every=4,
    )
    single_frame = encode_video_with_mem(
        model,
        video[:, -1:],
        torch.ones(2, 1, dtype=torch.bool),
        temporal_attention_every=4,
    )

    assert output.shape == (2, 4, 16)
    torch.testing.assert_close(single_frame, model(video[:, -1]).last_hidden_state)


def test_masked_history_cannot_change_current_embedding():
    transformers = pytest.importorskip("transformers")
    config = transformers.SmolVLMVisionConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer

    model = SmolVLMVisionTransformer(config).eval()
    first_video = torch.randn(1, 3, 3, 16, 16)
    second_video = first_video.clone()
    second_video[:, :2] = torch.randn_like(second_video[:, :2]) * 100
    frame_mask = torch.tensor([[False, False, True]])

    first_output = encode_video_with_mem(model, first_video, frame_mask, temporal_attention_every=4)
    second_output = encode_video_with_mem(model, second_video, frame_mask, temporal_attention_every=4)

    torch.testing.assert_close(first_output, second_output)
