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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.memory import (
    causal_temporal_mask,
    encode_video_with_mem,
    sample_observation_history,
    space_time_attention,
    temporal_sinusoidal_embedding,
)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch


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
    pytest.importorskip("datasets")
    from lerobot.datasets.factory import resolve_delta_timestamps

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


def test_inference_history_anchors_on_the_newest_frame():
    """Ages are relative to the newest frame, so an over-long queue still ends at it."""
    history = [torch.full((1, 1), value) for value in range(14)]
    values, _ = sample_observation_history(history, num_frames=3, stride=5, steps_seen=99)
    assert values[:, :, 0].tolist() == [[3, 8, 13]]


def test_inference_history_rejects_a_too_short_queue():
    history = [torch.zeros(1, 1) for _ in range(10)]
    with pytest.raises(ValueError, match="need at least 11"):
        sample_observation_history(history, num_frames=3, stride=5, steps_seen=99)


def test_visual_memory_requires_a_matching_image_key():
    pytest.importorskip("datasets")
    from lerobot.datasets.factory import resolve_delta_timestamps

    config = PI05Config(use_visual_memory=True, memory_frames=3, memory_stride=5)

    # Singular `observation.image` is a valid LeRobot convention and must get history.
    singular = SimpleNamespace(fps=10, features={"observation.image": {}, "action": {}})
    assert resolve_delta_timestamps(config, singular)["observation.image"] == [-1.0, -0.5, 0.0]

    # A dataset with no image key at all must fail instead of training single-frame.
    stateless = SimpleNamespace(fps=10, features={"observation.state": {}, "action": {}})
    with pytest.raises(ValueError, match="no dataset feature maps to an image key"):
        resolve_delta_timestamps(config, stateless)


def test_proprioceptive_memory_removes_the_state_from_the_prompt():
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.utils.constants import OBS_STATE

    def prompt_for(*, include_state: bool) -> str:
        transition = {
            TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 2, 4)},
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick the cube"]},
        }
        step = Pi05PrepareStateTokenizerProcessorStep(include_state_in_prompt=include_state)
        return step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt_for(include_state=True) == "Task: pick the cube, State: 128 128 128 128;\nAction: "
    assert prompt_for(include_state=False) == "Task: pick the cube;\nAction: "


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
    model.config._attn_implementation = "sdpa"  # noqa: SLF001
    image = torch.randn(2, 3, 16, 16)
    expected = model(image).last_hidden_state
    actual = encode_video_with_mem(
        model,
        image[:, None],
        torch.ones(2, 1, dtype=torch.bool),
        temporal_attention_every=4,
    )
    torch.testing.assert_close(actual, expected)
    # MEM composes attention weights itself, so it must not downgrade the tower.
    assert model.config._attn_implementation == "sdpa"  # noqa: SLF001


def test_composed_space_time_attention_is_identity_over_one_frame():
    """A softmax over a single timestep is the identity, so eq. 3 reduces to SigLIP."""
    model = _tiny_siglip()
    model.config._attn_implementation = "sdpa"  # noqa: SLF001
    attention = model.encoder.layers[0].self_attn
    hidden = torch.randn(2, 1, 4, 16)
    mask = causal_temporal_mask(
        torch.ones(2, 1, dtype=torch.bool), dtype=hidden.dtype, num_patches=hidden.shape[2]
    )

    composed = space_time_attention(attention, hidden, mask)
    spatial_only = attention(hidden_states=hidden[:, 0], attention_mask=None)[0]

    torch.testing.assert_close(composed[:, 0], spatial_only)


def test_mem_drops_past_frames_after_the_last_temporal_layer():
    model = _tiny_siglip()  # 4 layers; temporal_attention_every=3 -> temporal at index 2 only
    batch_size, num_frames = 2, 3
    # Spatial-only layers are invoked as whole layers with a (batch * frames) leading
    # dim; temporal layers drive the sub-modules directly and so never fire this hook.
    spatial_batches: list[int] = []
    for layer in model.encoder.layers:
        layer.register_forward_pre_hook(
            lambda _module, args, sink=spatial_batches: sink.append(args[0].shape[0])
        )

    encode_video_with_mem(
        model,
        torch.randn(batch_size, num_frames, 3, 16, 16),
        torch.ones(batch_size, num_frames, dtype=torch.bool),
        temporal_attention_every=3,
    )

    # Layers 0 and 1 still see every frame; layer 3 sits above the last temporal layer,
    # so it must only ever see the current frame.
    assert spatial_batches == [
        batch_size * num_frames,
        batch_size * num_frames,
        batch_size,
    ]


def test_mem_normalizes_tuple_encoder_layer_output(monkeypatch):
    model = _tiny_siglip()
    layer = model.encoder.layers[0]
    original_forward = layer.forward

    def tuple_forward(*args, **kwargs):
        return (original_forward(*args, **kwargs),)

    monkeypatch.setattr(layer, "forward", tuple_forward)
    video = torch.randn(1, 1, 3, 16, 16)

    output = encode_video_with_mem(
        model,
        video,
        torch.ones(1, 1, dtype=torch.bool),
        temporal_attention_every=4,
    )

    assert output.shape == (1, 4, 16)


def test_mem_rejects_incompatible_siglip_layer():
    model = _tiny_siglip()
    model.encoder.layers[0] = nn.Identity()

    with pytest.raises(TypeError, match="layer 0 is missing"):
        encode_video_with_mem(
            model,
            torch.randn(1, 1, 3, 16, 16),
            torch.ones(1, 1, dtype=torch.bool),
            temporal_attention_every=4,
        )


def test_mem_rejects_temporal_interval_larger_than_vision_depth():
    model = _tiny_siglip()

    with pytest.raises(ValueError, match=r"temporal_attention_every \(5\).*layers \(4\)"):
        encode_video_with_mem(
            model,
            torch.randn(1, 3, 3, 16, 16),
            torch.ones(1, 3, dtype=torch.bool),
            temporal_attention_every=5,
        )


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


def test_pi05_base_checkpoint_keeps_fresh_proprio_memory_projection():
    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_proprioceptive_memory=True)
    policy.model = nn.Module()
    policy.model.proprio_history_proj = nn.Linear(4, 8)

    state_dict = policy._prepare_pretrained_state_dict({})

    assert "model.proprio_history_proj.weight" in state_dict
    assert "model.proprio_history_proj.bias" in state_dict


def _make_inference_memory_policy() -> PI05Policy:
    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        n_action_steps=1,
        use_visual_memory=True,
        use_proprioceptive_memory=False,
        memory_frames=3,
        memory_stride=1,
        image_features=["camera"],
    )
    policy.reset()
    return policy


def test_inference_memory_snapshots_observations_instead_of_storing_references():
    policy = _make_inference_memory_policy()
    observation = torch.ones(2, 1)

    policy._stack_inference_memory({"camera": observation})
    observation.fill_(9)
    history = policy._stack_inference_memory({"camera": observation})["camera"]

    assert history[:, :, 0].tolist() == [[1, 1, 9], [1, 1, 9]]


def test_inference_memory_requires_reset_before_batch_size_changes():
    policy = _make_inference_memory_policy()
    policy._stack_inference_memory({"camera": torch.ones(2, 1)})

    with pytest.raises(ValueError, match="without policy.reset"):
        policy._stack_inference_memory({"camera": torch.ones(1, 1)})

    policy.reset()
    history = policy._stack_inference_memory({"camera": torch.ones(1, 1)})["camera"]
    assert history.shape == (1, 3, 1)
