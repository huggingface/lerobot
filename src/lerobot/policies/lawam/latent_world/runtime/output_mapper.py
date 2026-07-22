from __future__ import annotations

from typing import Any

import torch


def map_policy_train_output(policy_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "total_loss": policy_output["loss_total"],
        "loss_flow": policy_output["loss_flow"],
        "loss_perceptual": policy_output["loss_perceptual"],
        "loss_distill": policy_output["loss_distill"],
        "loss_vlm": policy_output["loss_vlm"],
    }


def _to_msgpackable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_msgpackable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_msgpackable(item) for item in value)
    return value


def map_policy_infer_output(
    actions: torch.Tensor,
    intermediates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {"normalized_actions": actions.detach().cpu().numpy()}
    if intermediates is not None:
        output["intermediates"] = _to_msgpackable(intermediates)
    return output
