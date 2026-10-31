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

"""Input checks that a policy keeps when `torch.export` or `torch.compile` traces it."""

from collections.abc import Callable

import torch
from torch import Tensor


def check_input(
    condition: Tensor | bool, message: str | Callable[[], str], error_type: type[Exception] = ValueError
) -> None:
    """Raise `error_type` with `message` unless `condition` holds (for every element of a tensor).

    In eager mode this is a plain check. While `torch.export` traces, it becomes a runtime assert in the
    program, which raises a `RuntimeError` without `message` when it fails. Under `torch.compile` the eager
    check runs at a graph break.
    """
    if torch.compiler.is_compiling() or torch.compiler.is_exporting():
        if isinstance(condition, Tensor):
            # Read the failure count, not a bool: PyTorch 2.11 cannot save or decompose an assert on a bool
            # read from a tensor.
            condition = torch.logical_not(condition).sum().item() == 0
        torch._check_with(error_type, condition, message if callable(message) else lambda: message)
    elif not bool(torch.as_tensor(condition).all()):
        raise error_type(message() if callable(message) else message)
