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

from __future__ import annotations

from typing import Any

from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.language import LANGUAGE_COLUMNS

_PYTHON_LIST_KEYS = {"messages", "message_streams", "target_message_indices", *LANGUAGE_COLUMNS}


def lerobot_collate_fn(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Collate function that preserves Python-list and language fields as lists.

    Drops ``None`` samples (e.g. recipes that yielded no target message), keeps
    rendered-message and language fields as plain Python lists, and delegates
    every other key to PyTorch's ``default_collate``.
    """
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None

    preserved = {
        key: [sample[key] for sample in batch if key in sample]
        for key in _PYTHON_LIST_KEYS
        if any(key in sample for sample in batch)
    }
    tensorizable = [
        {
            key: value
            for key, value in sample.items()
            if key not in _PYTHON_LIST_KEYS and key not in LANGUAGE_COLUMNS
        }
        for sample in batch
    ]
    collated = default_collate(tensorizable)
    collated.update(preserved)
    return collated
