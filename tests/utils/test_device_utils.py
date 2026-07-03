# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from contextlib import nullcontext

import pytest
import torch

from lerobot.utils.device_utils import get_safe_autocast_context


@pytest.mark.parametrize(
    ("device", "enabled", "expect_autocast"),
    [
        ("cpu", True, True),  # AMP-capable device -> real autocast
        (torch.device("cpu"), True, True),  # accepts torch.device
        ("cpu", False, False),  # explicitly disabled -> no-op
        ("mps", True, False),  # AMP unsupported on mps -> no-op
        ("privateuseone", True, False),  # unknown device -> safe no-op
    ],
)
def test_get_safe_autocast_context(device, enabled, expect_autocast):
    ctx = get_safe_autocast_context(device, dtype=torch.bfloat16, enabled=enabled)
    if expect_autocast:
        assert isinstance(ctx, torch.autocast)
        with ctx:
            assert torch.is_autocast_enabled("cpu")
    else:
        assert isinstance(ctx, nullcontext)
