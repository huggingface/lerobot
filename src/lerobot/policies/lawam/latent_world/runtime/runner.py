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

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from lerobot.policies.lawam.latent_world.types import (
    LatentWorldPolicyInferExample,
    LatentWorldPolicyTrainBatch,
)

from .output_mapper import map_policy_infer_output, map_policy_train_output

if TYPE_CHECKING:
    from lerobot.policies.lawam.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
    from lerobot.policies.lawam.vlas.lawam import LatentWorldPolicyBackend


class LatentWorldPolicyRunner:
    def __init__(
        self,
        *,
        policy_backend: LatentWorldPolicyBackend,
        infer_batch_builder: LatentWorldPolicyInferBatchBuilder,
    ) -> None:
        self.policy_backend = policy_backend
        self.infer_batch_builder = infer_batch_builder

    def train_step(self, batch: LatentWorldPolicyTrainBatch) -> dict[str, torch.Tensor]:
        policy_output = self.policy_backend.forward(batch=batch)
        return map_policy_train_output(policy_output)

    @torch.inference_mode()
    def infer_step(
        self,
        examples: Sequence[LatentWorldPolicyInferExample],
        *,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
    ) -> dict[str, Any]:
        if len(examples) == 0:
            raise ValueError("`infer_step` requires at least one example.")
        batch = self.infer_batch_builder.build_infer_batch(examples)
        batch_size = int(batch["action_hz"].shape[0])
        if batch_size != len(examples):
            raise ValueError(
                "Inference batch size mismatch after batch build: "
                f"examples={len(examples)}, batch_size={batch_size}."
            )

        horizon_sec = float(self.policy_backend.flow.config.horizon_sec)
        hz_values = batch["action_hz"].detach().cpu().tolist()
        expected_lens = [int(math.floor(horizon_sec * float(hz))) for hz in hz_values]
        if any(expected_len < 1 for expected_len in expected_lens):
            bad_idx = next(idx for idx, expected_len in enumerate(expected_lens) if expected_len < 1)
            raise ValueError(
                "Invalid effective action length for inference: "
                f"sample={bad_idx}, floor(horizon_sec * action_hz)={expected_lens[bad_idx]}, "
                f"horizon_sec={horizon_sec}, action_hz={float(hz_values[bad_idx])}."
            )
        if len(set(expected_lens)) != 1:
            raise ValueError(
                "Real-time batched inference currently requires all examples to share the same "
                f"effective action length. Got expected_lens={expected_lens} from action_hz={hz_values} "
                f"with horizon_sec={horizon_sec}."
            )
        expected_len = int(expected_lens[0])

        actions = self.policy_backend.predict_action(
            batch=batch,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        actual_len = int(actions.shape[1])
        if actual_len != expected_len:
            raise ValueError(
                "Inference action length mismatch: "
                f"actual_len={actual_len}, expected_len={expected_len}, "
                f"horizon_sec={horizon_sec}, action_hz={hz_values}."
            )
        return map_policy_infer_output(actions)
