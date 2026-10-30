#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""The readout proof accepts bfloat16 without downloading a checkpoint."""

import torch

from experiments.gar.proof import StopProofError, require_bfloat16


def test_require_bfloat16_accepts_torch_dtype_and_config_string():
    require_bfloat16(torch.bfloat16)
    require_bfloat16("bfloat16")


def test_require_bfloat16_rejects_every_other_dtype():
    for dtype in (torch.float32, torch.float16, "float32", "torch.bfloat16", None):
        try:
            require_bfloat16(dtype)
        except StopProofError as exc:
            assert str(exc) == f"checkpoint dtype is {dtype}, not bfloat16"
        else:
            raise AssertionError(f"{dtype!r} was accepted")
