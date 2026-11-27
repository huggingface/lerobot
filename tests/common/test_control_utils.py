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

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.common.control_utils import predict_action, sanity_check_dataset_name
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE


def test_local_repo_id_without_slash_is_valid():
    sanity_check_dataset_name("test_20260918_090318", None)


def test_hub_repo_id_is_valid():
    sanity_check_dataset_name("user/my_dataset", None)


def test_eval_prefix_requires_policy_for_local_and_hub_ids():
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("eval_my_dataset", None)
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("user/eval_my_dataset", None)


def test_policy_requires_eval_prefix_for_local_and_hub_ids():
    policy = SimpleNamespace(type="act")
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("my_dataset", policy)
    with pytest.raises(ValueError, match="eval_"):
        sanity_check_dataset_name("user/my_dataset", policy)
    sanity_check_dataset_name("eval_my_dataset", policy)
    sanity_check_dataset_name("user/eval_my_dataset", policy)


def test_predict_action_keeps_the_relative_anchor_until_the_chunk_drains():
    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=["gripper"], action_names=["arm", "gripper"]
    )
    preprocessor, postprocessor = make_policy_processor_pipelines(
        [relative], [AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)]
    )
    queue = deque()
    deltas = torch.tensor([[1.0, 20.0], [2.0, 50.0], [3.0, 80.0]])

    def select_action(_observation):
        if not queue:
            queue.extend(deltas.unbind(0))
        return queue.popleft().unsqueeze(0)

    policy = SimpleNamespace(select_action=select_action, count_queued_actions=lambda: len(queue))
    states = [10.0, 11.0, 12.0, 30.0, 31.0, 32.0]
    actions = [
        predict_action(
            {OBS_STATE: np.array([state, 5.0], dtype=np.float32)},
            policy,
            torch.device("cpu"),
            preprocessor,
            postprocessor,
            use_amp=False,
        )
        for state in states
    ]
    expected = torch.tensor(
        [[11.0, 20.0], [12.0, 50.0], [13.0, 80.0], [31.0, 20.0], [32.0, 50.0], [33.0, 80.0]]
    )
    torch.testing.assert_close(torch.cat(actions), expected, atol=0, rtol=0)
