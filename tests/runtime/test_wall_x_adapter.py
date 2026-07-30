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

from types import SimpleNamespace

from lerobot.policies.wall_x.inference.wall_x_adapter import WallXPolicyAdapter
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import GenerationConfig
from lerobot.runtime.registry import get_language_adapter_factory


class _Policy:
    def __init__(self):
        self.action_batch = None
        self.text_call = None

    def predict_action_chunk(self, batch):
        self.action_batch = batch
        return "actions"

    def generate_text(self, batch, **kwargs):
        self.text_call = (batch, kwargs)
        return ["reach for the cup"]


def test_wall_x_adapter_routes_subtask_and_text_generation():
    policy = _Policy()
    adapter = WallXPolicyAdapter(policy, GenerationConfig(temperature=0.2, top_p=0.8))
    state = RuntimeState(task="pick up the cup")
    state.language_context["subtask"] = "reach for the cup"
    observation = {"observation.state": SimpleNamespace(shape=(1, 7))}

    assert adapter.select_action(observation, state) == "actions"
    assert policy.action_batch["task"] == "reach for the cup"
    assert adapter.generate_text("subtask", observation, state) == "reach for the cup"
    assert policy.text_call[0]["task"] == "pick up the cup"
    assert policy.text_call[1]["temperature"] == 0.2
    assert policy.text_call[1]["top_p"] == 0.8
    assert not adapter.gen.enable_memory


def test_wall_x_adapter_is_registered_lazily():
    assert get_language_adapter_factory("wall_x") is WallXPolicyAdapter
