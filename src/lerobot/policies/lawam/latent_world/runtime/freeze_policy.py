from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.policies.lawam.vlas.qwen3vl import (
    freeze_qwen3vl,
    keep_first_n_llm_layers,
    unfreeze_last_n_llm_layers,
)


@dataclass(frozen=True)
class LatentWorldPolicyFreezeConfig:
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_embedding: bool = False
    unfreeze_vision_merger: bool = False
    unfreeze_lam_decoder: bool = False
    keep_llm_first_n_layers: int | None = None
    unfreeze_llm_last_n_layers: int | None = None


def parse_policy_freeze_config(freeze_cfg: Any) -> LatentWorldPolicyFreezeConfig:
    if freeze_cfg is None:
        return LatentWorldPolicyFreezeConfig()

    unfreeze_last_n = freeze_cfg.get("unfreeze_llm_last_n_layers", None)
    if unfreeze_last_n is not None:
        unfreeze_last_n = int(unfreeze_last_n)

    keep_first_n = freeze_cfg.get("keep_llm_first_n_layers", None)
    if keep_first_n is not None:
        keep_first_n = int(keep_first_n)
        if keep_first_n <= 0:
            keep_first_n = None

    return LatentWorldPolicyFreezeConfig(
        freeze_vision_backbone=bool(freeze_cfg.get("freeze_vision_backbone", False)),
        freeze_llm_backbone=bool(freeze_cfg.get("freeze_llm_backbone", False)),
        freeze_embedding=bool(freeze_cfg.get("freeze_embedding", False)),
        unfreeze_vision_merger=bool(freeze_cfg.get("unfreeze_vision_merger", False)),
        unfreeze_lam_decoder=bool(freeze_cfg.get("unfreeze_lam_decoder", False)),
        keep_llm_first_n_layers=keep_first_n,
        unfreeze_llm_last_n_layers=unfreeze_last_n,
    )


def apply_policy_freeze(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    freeze_qwen3vl(
        policy_backend.vlm,
        freeze_vision_backbone=freeze_policy.freeze_vision_backbone,
        freeze_llm_backbone=freeze_policy.freeze_llm_backbone,
        freeze_embedding=freeze_policy.freeze_embedding,
        unfreeze_vision_merger=freeze_policy.unfreeze_vision_merger,
    )

    if freeze_policy.keep_llm_first_n_layers is not None:
        keep_first_n_llm_layers(policy_backend.vlm, freeze_policy.keep_llm_first_n_layers)

    if (
        freeze_policy.freeze_llm_backbone
        and freeze_policy.unfreeze_llm_last_n_layers is not None
        and freeze_policy.unfreeze_llm_last_n_layers > 0
    ):
        unfreeze_last_n_llm_layers(
            policy_backend.vlm,
            freeze_policy.unfreeze_llm_last_n_layers,
        )

    for p in policy_backend.lam.parameters():
        p.requires_grad = False
    if freeze_policy.unfreeze_lam_decoder:
        lam_decoder = getattr(policy_backend.lam, "decoder", None)
        if lam_decoder is not None:
            for p in lam_decoder.parameters():
                p.requires_grad = True
