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
"""The export tests rely on `data_dependent_nodes` finding what it promises to find."""

import pytest
import torch

from lerobot.policies.common.input_checks import check_input
from tests.policies.common.fixed_grid_checks import data_dependent_nodes


def _boolean_index(x):
    return x[x > 0]


def _boolean_index_put(x):
    y = x.clone()
    # A tensor value, as the policies wrote embeddings; newer PyTorch traces a scalar one as masked_fill.
    y[x > 0] = x[:1]
    return y


def _checked(x):
    check_input(x > -10, "x is out of range")
    return x * 2


class Program(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


@pytest.mark.parametrize(
    "function, expected",
    [
        (_boolean_index, "with a boolean mask"),
        (_boolean_index_put, "with a boolean mask"),
        (torch.nonzero, "nonzero"),
        (lambda x: x.masked_select(x > 0), "masked_select"),
        (lambda x: x * 2, None),
        (_checked, None),  # A runtime assert reads values only to raise.
    ],
    ids=["boolean-index", "boolean-index-put", "nonzero", "masked-select", "static", "runtime-assert"],
)
def test_data_dependent_nodes_finds_value_dependent_shapes(function, expected):
    program = torch.export.export(Program(function), (torch.randn(6),), strict=False)
    found = data_dependent_nodes(program)
    if expected is None:
        assert found == []
    else:
        assert any(expected in node for node in found), found
