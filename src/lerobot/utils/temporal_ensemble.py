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

"""Temporal ensembling utility for smooth action prediction."""

from collections import deque
from typing import Any


class TemporalEnsembler:
    """
    Temporal ensembling for smooth action prediction.

    This class maintains a buffer of recent actions and computes a weighted average
    to reduce jitter and create smoother robot control.

    Args:
        k: Number of recent actions to keep in the buffer (window size)
        exp: Exponential decay factor for weights:
            - 1.0: uniform weights (simple moving average)
            - <1.0: exponential decay (recent actions weighted more, older actions dampened)
            - >1.0: exponential growth (older actions weighted more, recent actions dampened)
    """

    def __init__(self, k: int = 1, exp: float = 1.0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if exp <= 0:
            raise ValueError(f"exp must be > 0, got {exp}")

        self.k = k
        self.exp = exp
        self.enabled = k > 1
        self.action_buffer: deque = deque(maxlen=k)

        # Precompute weights for efficiency
        if self.enabled:
            self._compute_weights()

    def _compute_weights(self):
        """Compute normalized exponential weights."""
        # Generate weights: [exp^(k-1), exp^(k-2), ..., exp^1, exp^0]
        # Most recent action has weight exp^0 = 1.0
        weights = [self.exp**i for i in range(self.k - 1, -1, -1)]
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def reset(self):
        """Clear the action buffer."""
        self.action_buffer.clear()

    def update(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Update buffer with new action and return smoothed action.

        Args:
            action: Dictionary of action values (e.g., motor positions)

        Returns:
            Smoothed action dictionary
        """
        if not self.enabled:
            return action

        # Add new action to buffer
        self.action_buffer.append(action)

        # If buffer not full yet, return the current action
        if len(self.action_buffer) < self.k:
            return action

        # Compute weighted average
        smoothed_action = {}

        # Get all keys from the most recent action
        for key in action:
            values = []
            weights_to_use = []

            # Collect values from buffer (some old actions might not have all keys)
            for i, buffered_action in enumerate(self.action_buffer):
                if key in buffered_action:
                    values.append(buffered_action[key])
                    weights_to_use.append(self.weights[i])

            if not values:
                # Key not found in any buffered action
                smoothed_action[key] = action[key]
                continue

            # Normalize weights for available values
            weight_sum = sum(weights_to_use)
            normalized_weights = [w / weight_sum for w in weights_to_use]

            # Compute weighted average
            if isinstance(values[0], (int, float)):
                # Scalar value
                smoothed_action[key] = sum(v * w for v, w in zip(values, normalized_weights, strict=True))
            else:
                smoothed_value = sum(v * w for v, w in zip(values, normalized_weights, strict=True))
                smoothed_action[key] = smoothed_value

        return smoothed_action
