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
from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.inference.observation_history import sample_camera_history
from lerobot.policies.dm05.configuration_dm05 import DM05Config
from lerobot.policies.dm05.core.tokenization import fill_memory_slots
from lerobot.policies.dm05.modeling_dm05 import DM05Policy
from lerobot.policies.dm05.processor_dm05 import MODEL_INPUT_PREFIX, DM05TokenizerProcessorStep
from lerobot.policies.pi05.memory import sample_observation_history
from lerobot.processor import create_transition


def memory_batch(current=None, memory_frames=1):
    tokens = 16 * memory_frames
    inputs = {
        "input_ids": torch.zeros(1, tokens, dtype=torch.long),
        "attention_mask": torch.zeros(1, tokens, dtype=torch.long),
        "token_type_ids": torch.zeros(1, tokens, dtype=torch.long),
        "history_slot_mask": torch.ones(1, tokens, dtype=torch.bool),
        "history_mask": torch.zeros(1, tokens, dtype=torch.bool),
        "history_token_id": torch.tensor(5),
        "history_pad_id": torch.tensor(0),
    }
    if current is not None:
        inputs["history_current_pixel_values"] = torch.full((1, 1, 1, 1), float(current))
    return {f"{MODEL_INPUT_PREFIX}{key}": value for key, value in inputs.items()}


@pytest.fixture
def policy():
    instance = DM05Policy.__new__(DM05Policy)
    torch.nn.Module.__init__(instance)
    instance.config = DM05Config(
        use_memory=True,
        memory_image_key="observation.images.cam_high",
        memory_frames=1,
        memory_stride=25,
        chunk_size=25,
        n_action_steps=25,
        device="cpu",
    )
    instance.reset()
    instance._predict_action_chunk = Mock(return_value=torch.ones(1, 25, 2))
    return instance


@pytest.mark.parametrize("batch", [{}, memory_batch(0)])
def test_direct_chunk_rejects_missing_explicit_history_without_updating_queue(policy, batch):
    with pytest.raises(ValueError, match="explicit history"):
        policy.predict_action_chunk(batch)
    assert policy._memory_steps_seen == 0
    policy._predict_action_chunk.assert_not_called()


def test_select_action_calls_public_entry_once_per_chunk_without_double_enqueue(policy):
    policy.predict_action_chunk = Mock(wraps=policy.predict_action_chunk)
    for frame in range(26):
        policy.select_action(memory_batch(frame))
    assert policy.predict_action_chunk.call_count == 2
    assert policy._memory_steps_seen == 26
    batch = policy.predict_action_chunk.call_args.args[0]
    assert f"{MODEL_INPUT_PREFIX}history_current_pixel_values" not in batch
    assert batch[f"{MODEL_INPUT_PREFIX}history_pixel_values"].flatten().tolist() == [0]
    policy.reset()
    assert not policy._memory_queue and not policy._queues["action"] and policy._memory_steps_seen == 0
    policy.select_action(memory_batch(100))
    assert not policy.predict_action_chunk.call_args.args[0][f"{MODEL_INPUT_PREFIX}history_mask"].any()


@pytest.mark.parametrize("memory_frames", [1, 20])
def test_queued_actions_record_frames_without_assembling_history(policy, memory_frames):
    policy.config.memory_frames = memory_frames
    policy.reset()
    with patch(
        "lerobot.policies.dm05.modeling_dm05.sample_observation_history",
        wraps=sample_observation_history,
    ) as sample:
        for frame in range(25):
            policy.select_action(memory_batch(frame, memory_frames))
        assert sample.call_count == 1
        assert policy._memory_steps_seen == 25
        assert policy._memory_queue[-1].item() == 24
        policy.select_action(memory_batch(25, memory_frames))
        assert sample.call_count == 2
        batch = policy._predict_action_chunk.call_args.args[0]
        assert batch[f"{MODEL_INPUT_PREFIX}history_pixel_values"].item() == 0


@pytest.mark.dataset
@pytest.mark.parametrize("frame", [0, 25, 625])
def test_dataset_online_and_explicit_chunk_history_agree(policy, frame):
    pytest.importorskip("datasets")
    from lerobot.datasets.factory import resolve_delta_timestamps

    camera = policy.config.memory_image_key
    offsets = resolve_delta_timestamps(policy.config, SimpleNamespace(fps=25, features={camera: {}}))[camera]
    assert offsets == [-1.0, 0.0]
    for index in range(frame + 1):
        online = policy._remember_batch(memory_batch(index))
    indices = [max(0, frame + round(offset * 25)) for offset in offsets]
    pixels, padding = sample_camera_history(list(range(frame + 1)), 1, 25)
    assert pixels == (indices[0],) and padding == (frame < 25,)
    inputs = {key.removeprefix(MODEL_INPUT_PREFIX): value for key, value in memory_batch().items()}
    history = None if padding[0] else torch.tensor([[[[float(pixels[0])]]]])
    explicit = fill_memory_slots(inputs, torch.tensor([[not padding[0]]]), history)
    for key in ("input_ids", "attention_mask", "token_type_ids", "history_mask"):
        torch.testing.assert_close(online[f"{MODEL_INPUT_PREFIX}{key}"], explicit[key])
    steps = policy._memory_steps_seen
    policy.predict_action_chunk({f"{MODEL_INPUT_PREFIX}{key}": value for key, value in explicit.items()})
    assert policy._memory_steps_seen == steps


def test_non_memory_direct_prediction_and_select_action_remain_supported(policy):
    policy.config.use_memory = False
    policy.reset()
    policy.predict_action_chunk({})
    policy.select_action({})
    assert policy._predict_action_chunk.call_count == 2


@pytest.mark.parametrize("camera", [None, "observation.images.missing"])
def test_memory_processor_requires_an_active_camera(camera):
    step = DM05TokenizerProcessorStep(
        processor_name_or_path="unused",
        image_keys=["observation.images.cam_high"],
        memory_frames=1,
        memory_image_key=camera,
        add_state=False,
    )
    step._tokenization = SimpleNamespace(
        tokenize_robot_batch=lambda samples: {
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2),
        }
    )
    with pytest.raises(ValueError, match="active camera"):
        step(
            create_transition(
                observation={
                    "observation.state": torch.zeros(1, 2),
                    "observation.images.cam_high": torch.zeros(1, 3, 2, 2),
                }
            )
        )
