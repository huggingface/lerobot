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


def get_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """Resolve a parameter dtype. ``None`` uses PyTorch's default floating dtype."""
    requested_dtype = dtype
    supported = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype is None:
        dtype = torch.get_default_dtype()
    if isinstance(dtype, str):
        dtype = supported.get(dtype)
    if not isinstance(dtype, torch.dtype) or dtype not in supported.values():
        raise ValueError(
            f"Invalid dtype: {requested_dtype!r}. Expected one of {list(supported)} or a matching torch.dtype."
        )
    return dtype
