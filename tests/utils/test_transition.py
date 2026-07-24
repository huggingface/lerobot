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

from lerobot.utils.constants import ACTION
from lerobot.utils.transition import Transition, move_transition_to_device


def _base_transition() -> Transition:
    return Transition(
        state={"obs": torch.zeros(2)},
        action=torch.ones(2),
        reward=1.0,
        next_state={"obs": torch.ones(2)},
        done=False,
        truncated=False,
    )


def test_complementary_info_not_required():
    # complementary_info is NotRequired — construction without it must type-check
    # and succeed at runtime.
    t = _base_transition()
    assert "complementary_info" not in t
    required = getattr(Transition, "__required_keys__", None)
    if required is not None:
        assert "complementary_info" not in required


def test_move_to_device_without_complementary_info():
    t = _base_transition()
    out = move_transition_to_device(t, "cpu")
    assert out[ACTION].device.type == "cpu"
    assert "complementary_info" not in out or out.get("complementary_info") is None


def test_move_to_device_with_complementary_info():
    t = _base_transition()
    t["complementary_info"] = {"n": torch.tensor(3), "flag": 1}
    out = move_transition_to_device(t, "cpu")
    assert isinstance(out["complementary_info"]["n"], torch.Tensor)
    assert out["complementary_info"]["n"].item() == 3
    assert isinstance(out["complementary_info"]["flag"], torch.Tensor)
