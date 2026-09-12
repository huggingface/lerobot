#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from lerobot.policies.pi0.modeling_pi0 import PI0Policy

from .online import VLAInference


def make_pi0_batch_builder(
    *, task: str, input_features: set[str] | None = None
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the raw PI0 processor input for one HIL environment observation."""

    def build(observation: dict[str, Any]) -> dict[str, Any]:
        batch = {
            key: value
            for key, value in observation.items()
            if input_features is None or key in input_features
        }
        batch["task"] = task
        return batch

    return build


class PI0ContextProvider:
    """Adapt a LeRobot PI0Policy to the VLA context contract used by RLT."""

    def __init__(
        self,
        policy: PI0Policy,
        batch_builder: Callable[[Any], dict[str, torch.Tensor]],
        preprocessor: Callable[[dict[str, Any]], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.policy = policy.eval().requires_grad_(False)
        self.batch_builder = batch_builder
        self.preprocessor = preprocessor

    @torch.no_grad()
    def infer(self, observation: Any) -> VLAInference:
        batch = self.batch_builder(observation)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        _, context = self.policy.predict_action_chunk_with_context(batch)
        return VLAInference(
            final_tokens=context.final_tokens,
            token_mask=context.token_mask,
            reference_actions=context.reference_actions,
            proprio=context.proprio,
        )
