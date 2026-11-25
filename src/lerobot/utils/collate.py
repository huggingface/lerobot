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

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.language import LANGUAGE_COLUMNS
from lerobot.utils.constants import MESSAGES_RENDERED

_PYTHON_LIST_KEYS = {MESSAGES_RENDERED, "message_streams", "target_message_indices", *LANGUAGE_COLUMNS}


def _pad_windows(samples: list[dict[str, Any]]) -> None:
    """Pad every window to the longest of the batch, in place.

    A window (a key with ``{key}_is_pad``) shorter than the others repeats its last row, and the
    added rows are flagged in ``{key}_is_pad``: the same rule as the rows a window reads outside its
    episode. Windows of equal lengths, e.g. every sample of a frame sampler, are left unchanged.
    """
    for pad_key in [key for key in samples[0] if key.endswith("_is_pad")]:
        key = pad_key.removesuffix("_is_pad")
        lengths = [len(sample[pad_key]) for sample in samples]
        longest = max(lengths)
        if min(lengths) == longest:
            continue
        for sample, length in zip(samples, lengths, strict=True):
            if length < longest:
                value = sample[key]
                sample[key] = torch.cat([value, value[-1:].expand(longest - length, *value.shape[1:])])
                sample[pad_key] = torch.cat([sample[pad_key], torch.ones(longest - length, dtype=torch.bool)])


def lerobot_collate_fn(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Collate function for training batches.

    Drops ``None`` samples (e.g. recipes that yielded no target message), pads windows
    of different lengths to the longest of the batch (see :func:`_pad_windows`), keeps
    rendered-message and language fields as plain Python lists, and delegates every
    other key to PyTorch's ``default_collate``.
    """
    samples = [sample for sample in batch if sample is not None]
    if not samples:
        return None
    _pad_windows(samples)

    # All-or-nothing per key: a partial-presence batch (e.g. half the samples
    # carry `messages_rendered` and half don't) is a real bug in the upstream
    # rendering step — silently filtering would hand downstream consumers a
    # preserved list shorter than the tensor batch. Raise instead so the
    # mismatch surfaces at the boundary.
    preserved: dict[str, list[Any]] = {}
    for key in _PYTHON_LIST_KEYS:
        presence = [key in sample for sample in samples]
        if not any(presence):
            continue
        if not all(presence):
            raise ValueError(
                f"Inconsistent batch: {sum(presence)}/{len(samples)} samples carry {key!r}; "
                f"every sample in a batch must agree."
            )
        preserved[key] = [sample[key] for sample in samples]
    tensorizable = [
        {
            key: value
            for key, value in sample.items()
            if key not in _PYTHON_LIST_KEYS and key not in LANGUAGE_COLUMNS
        }
        for sample in samples
    ]
    collated = default_collate(tensorizable)
    collated.update(preserved)
    return collated
