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

"""Export helpers shared by policies that check how they are being traced."""

import torch


def is_non_strict_exporting() -> bool:
    """True while ``torch.export`` traces in non-strict mode, the only export mode these checks cover.

    Strict export and ``torch.compile`` both run Dynamo, and on PyTorch 2.7 to 2.11 ``torch.compile`` also
    reports ``is_exporting()``, so the two cannot be told apart. Checks guarded by this stay off in both.
    """
    return torch.compiler.is_exporting() and not torch.compiler.is_dynamo_compiling()
