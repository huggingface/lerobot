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

"""Action storage utilities for Real-Time Chunking (RTC)."""

import torch
from torch import Tensor


class ActionStorage:
    """Stores a single action chunk for Real-Time Chunking algorithm.

    This class maintains a single action chunk that can be reused
    in the RTC algorithm for efficient action chunking and temporal processing.

    Args:
        device (str | torch.device): Device to store actions on.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = device
        self._action_chunk: Tensor | None = None

    def reset(self) -> None:
        """Clear the stored action chunk."""
        self._action_chunk = None

    def set_action_chunk(self, action_chunk: Tensor) -> None:
        """Store a new action chunk.

        Args:
            action_chunk (Tensor): Action chunk tensor to store.
        """
        self._action_chunk = action_chunk.to(self.device)

    def get_action_chunk(self) -> Tensor | None:
        """Get the stored action chunk.

        Returns:
            Tensor | None: The stored action chunk, or None if nothing is stored.
        """
        return self._action_chunk

    def has_action_chunk(self) -> bool:
        """Check if an action chunk is stored.

        Returns:
            bool: True if an action chunk is stored, False otherwise.
        """
        return self._action_chunk is not None

    @property
    def is_empty(self) -> bool:
        """Check if the storage is empty."""
        return self._action_chunk is None
