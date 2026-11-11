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

import json
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.factory import load_pretrained_policy_processors, make_policy_processor_pipelines
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
    to_relative_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME


@pytest.mark.parametrize("state_index", [0, 1, -1])
@pytest.mark.parametrize("chunked", [False, True])
def test_explicit_state_frame_has_independent_forward_and_inverse_targets(state_index, chunked):
    state = torch.tensor([[[1.0, 2.0, 100.0], [5.0, 6.0, 150.0], [8.0, 9.0, 180.0]]])
    state[:, state_index] = torch.tensor([10.0, 20.0, 200.0])
    absolute = torch.tensor([[13.0, 24.0, 7.0, 99.0]], dtype=torch.float64)
    relative = torch.tensor([[3.0, 4.0, 7.0, 99.0]], dtype=torch.float64)
    if chunked:
        absolute = absolute.unsqueeze(1).repeat(1, 2, 1)
        relative = relative.unsqueeze(1).repeat(1, 2, 1)
    saved_state, saved_actions = state.clone(), absolute.clone()
    mask = [True, True, False]  # Excluded joint and trailing action dimension stay absolute.

    torch.testing.assert_close(to_relative_actions(absolute, state, mask, state_index=state_index), relative)
    # Supply the expected relative target independently: a round trip can hide a wrong anchor.
    torch.testing.assert_close(to_absolute_actions(relative, state, mask, state_index=state_index), absolute)
    torch.testing.assert_close(state, saved_state)
    torch.testing.assert_close(absolute, saved_actions)


def test_legacy_default_keeps_current_first_stacks():
    state = torch.tensor([[[10.0], [100.0]]])
    torch.testing.assert_close(
        to_relative_actions(torch.tensor([[13.0]]), state, [True]), torch.tensor([[3.0]])
    )
    torch.testing.assert_close(
        to_absolute_actions(torch.tensor([[3.0]]), state, [True]), torch.tensor([[13.0]])
    )


@pytest.mark.parametrize("state_index", [-1, 2])
def test_unstacked_state_does_not_use_frame_index(state_index):
    state = torch.tensor([[10.0]])
    torch.testing.assert_close(
        to_relative_actions(torch.tensor([[13.0]]), state, [True], state_index=state_index),
        torch.tensor([[3.0]]),
    )
    torch.testing.assert_close(
        to_absolute_actions(torch.tensor([[3.0]]), state, [True], state_index=state_index),
        torch.tensor([[13.0]]),
    )


def test_selected_frame_is_held_until_the_chunk_drains():
    relative = RelativeActionsProcessorStep(enabled=True, state_index=-1)
    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
    depth = {"value": 0}
    relative.bind_action_queue(lambda: depth["value"])

    def feed(oldest, current):
        relative(create_transition(observation={OBS_STATE: torch.tensor([[[oldest], [current]]])}))
        return absolute(create_transition(action=torch.tensor([[3.0]])))[TransitionKey.ACTION]

    torch.testing.assert_close(feed(1.0, 10.0), torch.tensor([[13.0]]))
    depth["value"] = 1
    torch.testing.assert_close(feed(2.0, 20.0), torch.tensor([[13.0]]))
    depth["value"] = 0
    torch.testing.assert_close(feed(2.0, 20.0), torch.tensor([[23.0]]))
    relative.reset()
    assert relative.get_cached_state() is None


@pytest.mark.parametrize("legacy", [False, True])
def test_saved_pair_preserves_state_selection_and_legacy_default(tmp_path, legacy):
    relative = RelativeActionsProcessorStep(enabled=True, state_index=-1)
    pre, post = make_policy_processor_pipelines(
        [relative], [AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)]
    )
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    if legacy:
        path = tmp_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        config = json.loads(path.read_text())
        del config["steps"][0]["config"]["state_index"]
        path.write_text(json.dumps(config))

    loaded_pre, loaded_post = load_pretrained_policy_processors(tmp_path)
    assert loaded_post.steps[0].relative_step is loaded_pre.steps[0]
    batch = {OBS_STATE: torch.tensor([[[1.0], [10.0]]]), ACTION: torch.tensor([[[13.0]]])}
    expected_relative = 12.0 if legacy else 3.0
    torch.testing.assert_close(loaded_pre(batch)[ACTION], torch.tensor([[[expected_relative]]]))
    torch.testing.assert_close(loaded_post(torch.tensor([[3.0]])), torch.tensor([[4.0 if legacy else 13.0]]))


def test_rtc_reanchors_absolute_prefix_against_selected_frame():
    from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix

    step = RelativeActionsProcessorStep(enabled=True, state_index=-1)
    output = reanchor_relative_rtc_prefix(
        torch.tensor([[13.0], [14.0]]), torch.tensor([[[1.0], [10.0]]]), step, None, "cpu"
    )
    torch.testing.assert_close(output, torch.tensor([[3.0], [4.0]]))


@pytest.mark.parametrize("memory", [False, True])
def test_pi05_factory_anchors_relative_actions_on_delta_zero(monkeypatch, memory):
    pytest.importorskip("transformers")
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.processor import tokenizer_processor

    # Only tokenization is substituted; config, factory, preprocessing and postprocessing are real.
    tokenizer = MagicMock(return_value={"input_ids": torch.zeros(1, 1), "attention_mask": torch.ones(1, 1)})
    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    config = PI05Config(
        device="cpu",
        use_relative_actions=True,
        use_proprioceptive_memory=memory,
        memory_frames=2,
        memory_stride=1,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        action_feature_names=["j0", "j1", "gripper"],
        normalization_mapping={"STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY},
    )
    assert config.state_observation_delta_indices == ([-1, 0] if memory else None)
    pre, post = make_pi05_pre_post_processors(config)
    state = torch.tensor([[[1.0, 2.0, 100.0], [10.0, 20.0, 200.0]]])
    if not memory:
        state = state[:, -1]
    batch = {OBS_STATE: state, ACTION: torch.tensor([[[13.0, 24.0, 7.0]]]), "task": ["pick"]}
    torch.testing.assert_close(pre(batch)[ACTION], torch.tensor([[[3.0, 4.0, 7.0]]]))
    torch.testing.assert_close(post(torch.tensor([[3.0, 4.0, 7.0]])), torch.tensor([[13.0, 24.0, 7.0]]))
