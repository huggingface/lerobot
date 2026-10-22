#!/usr/bin/env python

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

"""CPU checks for the bounded IMPRINT policy.

These tests cover registration, the named memory rules, a tiny training step,
reset and rollout state, save and load, and the default processor. They do not
measure the paper's robot success rates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.imprint.configuration_imprint import ImprintConfig
from lerobot.policies.imprint.memory import (
    EVENT_ORDER,
    FullAttentionMemory,
    MemoryModulation,
    RecurrentMemory,
    causal_action_stream,
    chunk_type_ids,
    gated_delta_step,
    language_condition_queries,
    memory_pretraining_loss,
)
from lerobot.policies.imprint.modeling_imprint import ImprintPolicy
from lerobot.policies.imprint.processor_imprint import ImprintEpisodeProgressStep
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAM = f"{OBS_IMAGES}.cam"


def _features(state_dim: int = 4, action_dim: int = 4, image: int = 8) -> dict[str, dict[str, PolicyFeature]]:
    return {
        "input_features": {
            CAM: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image, image)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        "output_features": {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
    }


def tiny_config(**overrides: object) -> ImprintConfig:
    params: dict[str, object] = {
        "device": "cpu",
        "history_length": 4,
        "chunk_size": 4,
        "n_action_steps": 1,
        "chunk_frames": 2,
        "tokens_per_view": 1,
        "memory_dim": 32,
        "num_latents": 4,
        "num_heads": 4,
        "num_cross_blocks": 1,
        "delta_num_layers": 1,
        "delta_num_heads": 4,
        "delta_head_dim": 8,
        "delta_conv_kernel": 4,
        "delta_ffn_inner": 32,
        "recent_fusion_blocks": 1,
        "max_chunks": 8,
        "visual_dim": 16,
        "chunk_feature_dim": 16,
        "recent_feature_dim": 16,
        "lang_dim": 8,
        "tokenizer_hidden": 16,
        "num_decoder_blocks": 1,
        "align_dim": 16,
        "num_query_samples": 2,
        "max_query_steps": 32,
        "decoder_ffn_inner": 32,
        "expert_layers": 1,
        "num_inference_steps": 2,
    }
    params.update(_features())
    params.update(overrides)
    return ImprintConfig(**params)


def _visual(batch: int, time: int, image: int = 8) -> torch.Tensor:
    return torch.rand(batch, time, 3, image, image)


def _state(batch: int, time: int, dim: int = 4) -> torch.Tensor:
    return torch.randn(batch, time, dim)


def _action(batch: int, time: int, dim: int = 4) -> torch.Tensor:
    return torch.randn(batch, time, dim)


def _assert_grad(parameter: nn.Parameter) -> None:
    grad = parameter.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_policy_is_registered() -> None:
    config = make_policy_config("imprint", device="cpu")
    assert isinstance(config, ImprintConfig)
    assert config.type == "imprint"
    assert get_policy_class("imprint") is ImprintPolicy


def test_default_joint_step_matches_the_user_example() -> None:
    """Same shapes as the README example, on CPU, with the default widths."""
    torch.manual_seed(0)
    state_dim = 6
    action_dim = 6
    image = 16
    config = ImprintConfig(device="cpu", **_features(state_dim, action_dim, image))
    policy = ImprintPolicy(config)
    policy.train()
    batch = {
        CAM: _visual(1, config.history_length, image),
        OBS_STATE: _state(1, config.history_length, state_dim),
        ACTION: _action(1, config.history_length + config.chunk_size, action_dim),
    }
    loss, logs = policy(batch)
    assert torch.isfinite(loss)
    assert "loss_action" in logs
    loss.backward()
    assert isinstance(policy.memory, RecurrentMemory)
    _assert_grad(policy.memory.proj_chunk.weight)


def test_event_order_language_and_causal_shift() -> None:
    assert EVENT_ORDER == "action_proprio_then_finalized_visual_v1"
    ids = chunk_type_ids(chunk_frames=2, tokens_per_chunk=1)
    assert torch.equal(ids, torch.tensor([0, 1, 0, 1, 2]))

    queries = torch.randn(2, 4, 32)
    film = nn.Linear(8, 64)
    nn.init.zeros_(film.bias)
    nn.init.ones_(film.weight)
    assert torch.equal(language_condition_queries(queries, None, film), queries)
    conditioned = language_condition_queries(queries, torch.ones(2, 8), film)
    assert not torch.allclose(conditioned, queries)

    executed = torch.tensor([[[1.0], [2.0], [3.0]]])
    assert torch.equal(causal_action_stream(executed), torch.tensor([[[0.0], [1.0], [2.0]]]))


def test_tokenizer_and_full_attention_shape() -> None:
    config = tiny_config()
    memory = RecurrentMemory(config, tokens_per_chunk=1)
    tokens = memory.tok_action(torch.randn(2, 3, 4))
    assert tokens.shape == (2, 3, config.memory_dim)

    full = FullAttentionMemory(config)
    out = full(
        torch.randn(2, 2, config.chunk_feature_dim),
        torch.randn(2, 1, config.recent_feature_dim),
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 4),
        None,
    )
    assert out.shape == (2, config.num_latents, config.memory_dim)
    changed = full(
        torch.randn(2, 2, config.chunk_feature_dim),
        torch.randn(2, 1, config.recent_feature_dim),
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 4),
        torch.randn(2, config.lang_dim),
    )
    assert changed.shape == out.shape


def test_incremental_matches_full_scan() -> None:
    torch.manual_seed(1)
    config = tiny_config(delta_num_layers=2)
    memory = RecurrentMemory(config, tokens_per_chunk=1)
    memory.eval()
    batch, chunks, frames = 2, 2, config.chunk_frames
    chunk_visual = torch.randn(batch, chunks, 1, config.chunk_feature_dim)
    chunk_action = torch.randn(batch, chunks, frames, 4)
    chunk_proprio = torch.randn(batch, chunks, frames, 4)
    recent_visual = torch.randn(batch, 0, config.recent_feature_dim)
    recent_action = torch.randn(batch, 0, 4)
    recent_proprio = torch.randn(batch, 0, 4)
    full, _state = memory.forward_sequence(
        chunk_visual, chunk_action, chunk_proprio, recent_visual, recent_action, recent_proprio, None
    )
    state = memory.init_state(batch, chunk_visual.device, chunk_visual.dtype)
    for index in range(chunks):
        state = memory.update_incremental(
            state, chunk_visual[:, index], chunk_action[:, index], chunk_proprio[:, index]
        )
    incremental = memory.read(state, recent_visual, recent_action, recent_proprio, None)
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
    assert torch.equal(state.chunk_count, torch.tensor([chunks, chunks]))


def test_invalid_gated_delta_keeps_state() -> None:
    state = torch.randn(1, 2, 4, 4)
    valid = torch.zeros(1, dtype=torch.bool)
    next_state, output = gated_delta_step(
        state,
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 4),
        torch.randn(1, 2, 4),
        torch.rand(1, 2),
        torch.rand(1, 2),
        valid,
    )
    assert torch.equal(next_state, state)
    assert torch.equal(output, torch.zeros_like(output))


def test_stage1_loss_weights_and_progress_range() -> None:
    prediction = {
        "align": torch.zeros(2, 1, 1, 1),
        "action": torch.zeros(2, 1, 1),
        "proprio": torch.zeros(2, 1, 1),
        "progress": torch.zeros(2),
    }
    targets = {key: torch.ones_like(value) for key, value in prediction.items()}
    loss, logs = memory_pretraining_loss(prediction, targets)
    assert loss.item() == torch.tensor(2.5).item()
    assert logs["loss_visual"] == 1.0
    assert logs["loss_action"] == 1.0
    assert logs["loss_proprio"] == 1.0
    assert logs["loss_progress"] == 1.0

    config = tiny_config(training_stage="memory")
    policy = ImprintPolicy(config)
    assert policy.decoder is not None
    steps = torch.zeros(2, 1, dtype=torch.long)
    progress = policy.decoder(torch.randn(2, config.num_latents, config.memory_dim), steps)["progress"]
    assert progress.shape == (2,)
    assert torch.all((progress > 0) & (progress < 1))


def test_disabled_modulation_is_identity() -> None:
    modulation = MemoryModulation(32, 4)
    features = torch.randn(2, 3, 32)
    output = modulation(features, torch.randn(2, 2, 32), torch.ones(2, 2, dtype=torch.bool), enabled=False)
    assert torch.equal(output, features)


def test_memory_stage_backward() -> None:
    torch.manual_seed(2)
    config = tiny_config(training_stage="memory")
    assert config.action_delta_indices == list(range(-config.history_length, 0))
    policy = ImprintPolicy(config)
    policy.train()
    history = config.history_length
    batch = {
        CAM: _visual(2, history),
        OBS_STATE: _state(2, history),
        ACTION: _action(2, history),
        "episode_progress": torch.tensor([0.25, 0.75]),
        "query_steps": torch.tensor([[0, 1], [1, 2]]),
    }
    causal = policy._causal_from_batch(batch, history, 2, torch.device("cpu"), torch.float32)
    assert torch.equal(causal, batch[ACTION])
    loss, logs = policy(batch)
    assert torch.isfinite(loss)
    assert "loss_progress" in logs
    loss.backward()
    assert policy.decoder is not None
    _assert_grad(policy.decoder.head_reward.weight)
    per_sample, _logs = policy(batch, reduction="none")
    assert per_sample.shape == (2,)


def test_joint_stage_backward_recurrent_and_full() -> None:
    for encoder in ("recurrent", "full_attention"):
        torch.manual_seed(3)
        config = tiny_config(encoder=encoder)
        history = config.history_length
        assert config.action_delta_indices[0] == -history
        assert config.action_delta_indices[-1] == config.chunk_size - 1
        assert config.observation_delta_indices[-1] == 0
        assert len(config.observation_delta_indices) == history
        policy = ImprintPolicy(config)
        policy.train()
        action = _action(2, history + config.chunk_size)
        batch = {
            CAM: _visual(2, history),
            OBS_STATE: _state(2, history),
            ACTION: action,
        }
        target, causal = policy._split_joint_action(batch, history)
        assert torch.equal(causal, action[:, :history])
        assert torch.equal(target, action[:, history:])
        loss, logs = policy(batch)
        assert torch.isfinite(loss)
        assert "loss_action" in logs
        loss.backward()
        assert policy.memory is not None
        _assert_grad(policy.memory.proj_chunk.weight)


def test_joint_without_memory_still_trains() -> None:
    torch.manual_seed(4)
    config = tiny_config(use_memory=False)
    policy = ImprintPolicy(config)
    assert policy.memory is None
    history = config.history_length
    loss, _logs = policy(
        {
            CAM: _visual(2, history),
            OBS_STATE: _state(2, history),
            ACTION: _action(2, history + config.chunk_size),
        }
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.expert is not None
    _assert_grad(policy.expert.out_proj.weight)


def test_predict_does_not_advance_memory_and_select_action_does() -> None:
    torch.manual_seed(5)
    config = tiny_config()
    policy = ImprintPolicy(config)
    step = {CAM: _visual(1, 1), OBS_STATE: _state(1, 1)}
    assert policy.finalized_chunk_counts().numel() == 0
    chunk = policy.predict_action_chunk(step)
    assert chunk.shape == (1, config.chunk_size, 4)
    assert policy.finalized_chunk_counts().numel() == 0

    first = policy.select_action(step)
    assert first.shape == (1, 4)
    assert int(policy.finalized_chunk_counts()[0]) == 0
    second = policy.select_action({CAM: _visual(1, 1), OBS_STATE: _state(1, 1)})
    assert second.shape == (1, 4)
    assert int(policy.finalized_chunk_counts()[0]) == 1
    again = policy.predict_action_chunk(step)
    assert again.shape == chunk.shape
    assert int(policy.finalized_chunk_counts()[0]) == 1

    policy.reset()
    assert policy.finalized_chunk_counts().numel() == 0


def test_full_attention_rollout_has_no_chunk_counter() -> None:
    torch.manual_seed(6)
    policy = ImprintPolicy(tiny_config(encoder="full_attention"))
    step = {CAM: _visual(1, 1), OBS_STATE: _state(1, 1)}
    policy.select_action(step)
    policy.select_action(step)
    assert policy.finalized_chunk_counts().numel() == 0
    policy.reset()
    assert policy.finalized_chunk_counts().numel() == 0


def test_save_load_roundtrip(tmp_path) -> None:
    torch.manual_seed(7)
    config = tiny_config()
    policy = ImprintPolicy(config)
    policy.save_pretrained(tmp_path)
    loaded = ImprintPolicy.from_pretrained(tmp_path, config=config)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    step = {CAM: _visual(1, 1), OBS_STATE: _state(1, 1)}
    torch.manual_seed(8)
    original = policy.predict_action_chunk(step)
    torch.manual_seed(8)
    restored = loaded.predict_action_chunk(step)
    torch.testing.assert_close(restored, original)


def test_processor_then_select_action() -> None:
    torch.manual_seed(9)
    config = tiny_config()
    policy = ImprintPolicy(config)
    pre, post = make_pre_post_processors(config)
    processed = pre({CAM: torch.rand(3, 8, 8), OBS_STATE: torch.randn(4)})
    action = policy.select_action(processed)
    restored = post(action)
    assert tuple(restored.shape) == (1, 4)
    assert restored.device.type == "cpu"
    assert torch.isfinite(restored).all()


def test_pi05_defaults_stay_unchanged() -> None:
    config = PI05Config()
    assert config.use_visual_memory is False
    assert config.chunk_size == 50


def test_episode_progress_uses_frame_and_episode_length() -> None:
    step = ImprintEpisodeProgressStep(episode_indices=[0, 1], episode_lengths=[5, 9])
    output = step.complementary_data(
        {
            "frame_index": torch.tensor([3, 0]),
            "episode_index": torch.tensor([0, 1]),
        }
    )
    torch.testing.assert_close(output["episode_progress"], torch.tensor([0.75, 0.0]))
    already = step.complementary_data(
        {
            "frame_index": torch.tensor([3]),
            "episode_index": torch.tensor([0]),
            "episode_progress": torch.tensor([0.1]),
        }
    )
    torch.testing.assert_close(already["episode_progress"], torch.tensor([0.1]))


class _EpisodeTable:
    def __init__(self, rows: list[tuple[int, int]]) -> None:
        self._rows = [{"episode_index": index, "length": length} for index, length in rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, int]:
        return self._rows[index]


def test_stage1_processor_reads_runtime_dataset_meta() -> None:
    config = tiny_config(training_stage="memory")
    config._runtime_dataset_meta = type("Meta", (), {"episodes": _EpisodeTable([(0, 5), (1, 9)])})()
    preprocessor, _postprocessor = make_pre_post_processors(config)
    batch = preprocessor(
        {
            CAM: torch.rand(2, 3, 8, 8),
            OBS_STATE: torch.randn(2, 4),
            ACTION: torch.randn(2, 4),
            "frame_index": torch.tensor([3, 0]),
            "episode_index": torch.tensor([0, 1]),
        }
    )
    torch.testing.assert_close(batch["episode_progress"], torch.tensor([0.75, 0.0]))
    joint = tiny_config(training_stage="joint")
    joint._runtime_dataset_meta = config._runtime_dataset_meta
    joint_pre, _joint_post = make_pre_post_processors(joint)
    assert not any(isinstance(step, ImprintEpisodeProgressStep) for step in joint_pre.steps)


def test_stage1_memory_transfers_and_inference_resets(tmp_path) -> None:
    torch.manual_seed(10)
    stage1 = ImprintPolicy(tiny_config(training_stage="memory"))
    history = stage1.config.history_length
    loss, _logs = stage1(
        {
            CAM: _visual(2, history),
            OBS_STATE: _state(2, history),
            ACTION: _action(2, history),
            "episode_progress": torch.tensor([0.2, 0.4]),
            "query_steps": torch.tensor([[0, 1], [1, 2]]),
        }
    )
    loss.backward()
    with torch.no_grad():
        stage1.memory.proj_chunk.weight.add_(0.1)
    stage1_dir = tmp_path / "stage1"
    stage1.save_pretrained(stage1_dir)

    joint = ImprintPolicy(tiny_config())
    expert_before = joint.expert.out_proj.weight.detach().clone()
    observation_before = joint.obs_visual.weight.detach().clone()
    report = joint.load_memory_weights(stage1_dir)
    assert "memory.proj_chunk.weight" in report["loaded"]
    assert report["skipped_decoder"]
    assert all(key.startswith("decoder.") for key in report["skipped_decoder"])
    torch.testing.assert_close(joint.memory.proj_chunk.weight, stage1.memory.proj_chunk.weight)
    torch.testing.assert_close(joint.frame_encoder.conv.weight, stage1.frame_encoder.conv.weight)
    torch.testing.assert_close(joint.expert.out_proj.weight, expert_before)
    torch.testing.assert_close(joint.obs_visual.weight, observation_before)

    mismatched = ImprintPolicy(tiny_config(memory_dim=64, delta_head_dim=16))
    try:
        mismatched.load_memory_weights(stage1_dir)
    except ValueError as exc:
        assert "memory geometry" in str(exc)
    else:
        raise AssertionError("a different memory width was accepted")

    joint_dir = tmp_path / "joint"
    joint.save_pretrained(joint_dir)
    loaded = ImprintPolicy.from_pretrained(joint_dir, config=joint.config)
    torch.testing.assert_close(loaded.memory.proj_chunk.weight, stage1.memory.proj_chunk.weight)
    loaded.reset()
    step = {CAM: _visual(1, 1), OBS_STATE: _state(1, 1)}
    assert loaded.finalized_chunk_counts().numel() == 0
    loaded.select_action(step)
    loaded.select_action({CAM: _visual(1, 1), OBS_STATE: _state(1, 1)})
    assert int(loaded.finalized_chunk_counts()[0]) == 1
    chunk = loaded.predict_action_chunk(step)
    assert chunk.shape == (1, loaded.config.chunk_size, 4)
    assert int(loaded.finalized_chunk_counts()[0]) == 1
    loaded.reset()
    assert loaded.finalized_chunk_counts().numel() == 0
