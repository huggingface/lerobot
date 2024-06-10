#!/usr/bin/env python

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
"""A protocol that all policies should follow.

This provides a mechanism for type-hinting and isinstance checks without requiring the policies classes
subclass a base class.

The protocol structure, method signatures, and docstrings should be used by developers as a reference for
how to implement new policies.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Policy(Protocol):
    """The required interface for implementing a policy.

    We also expect all policies to subclass torch.nn.Module and PyTorchModelHubMixin.
    """

    name: str

    def __init__(self, cfg, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        """
        Args:
            cfg: Policy configuration class instance or None, in which case the default instantiation of the
                 configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization.
        """

    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """

    def forward(self, batch: dict[str, Tensor]) -> dict:
        """Run the batch through the model and compute the loss for training or validation.

        Returns a dictionary with "loss" and potentially other information. Apart from "loss" which is a Tensor, all
        other items should be logging-friendly, native Python types.
        """

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """


@runtime_checkable
class PolicyWithUpdate(Policy, Protocol):
    def update(self):
        """An update method that is to be called after a training optimization step.

        Implements an additional updates the model parameters may need (for example, doing an EMA step for a
        target model, or incrementing an internal buffer).
        """
