from __future__ import annotations

import torch


def map_policy_train_output(policy_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "total_loss": policy_output["loss_total"],
        "loss_flow": policy_output["loss_flow"],
        "loss_perceptual": policy_output["loss_perceptual"],
        "loss_distill": policy_output["loss_distill"],
        "loss_vlm": policy_output["loss_vlm"],
    }


def map_policy_infer_output(actions: torch.Tensor) -> dict[str, object]:
    return {"normalized_actions": actions.detach().cpu().numpy()}
