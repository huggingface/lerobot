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

"""Stable output mappings between the native backend and LeRobot adapter."""

from __future__ import annotations

import torch


def map_policy_train_output(policy_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map native training losses to the names consumed by the adapter."""
    return {
        "total_loss": policy_output["loss_total"],
        "loss_flow": policy_output["loss_flow"],
        "loss_perceptual": policy_output["loss_perceptual"],
        "loss_distill": policy_output["loss_distill"],
        "loss_vlm": policy_output["loss_vlm"],
    }


def map_policy_infer_output(actions: torch.Tensor) -> dict[str, object]:
    """Convert normalized action tensors to the adapter's NumPy output."""
    return {"normalized_actions": actions.detach().cpu().numpy()}
