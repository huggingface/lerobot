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

import torch

from lerobot.policies.pi0 import PI0Policy


def test_pi0_state_dict_accepts_direct_embed_tokens_key():
    weight = torch.ones(2, 3)
    policy = object.__new__(PI0Policy)

    fixed = PI0Policy._fix_pytorch_state_dict_keys(
        policy,
        {
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": weight,
        },
        model_config=None,
    )

    assert torch.equal(
        fixed["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
        weight,
    )
