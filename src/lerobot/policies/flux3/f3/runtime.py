# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Small runtime helpers shared by the flux3 model components."""

import torch

MOCK_INIT_SEED = 0


def random_init_(model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16) -> None:
    """Fill a model with small random values and cast to ``dtype`` (weightless wiring/smoke runs).

    Seeded so repeated builds are identical.
    """
    torch.manual_seed(MOCK_INIT_SEED)
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_meta:
                p.normal_(0, 0.02)
    model.to(dtype=dtype)
