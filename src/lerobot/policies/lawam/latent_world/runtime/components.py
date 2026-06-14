from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.policies.lawam.latent_world.batch_builder import (
    LatentWorldPolicyInferBatchBuilder,
)
from lerobot.policies.lawam.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from lerobot.policies.lawam.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from lerobot.policies.lawam.vlas.lawam import (
    LatentWorldPolicyBackend,
    LatentWorldPolicyConfig,
)

from .contracts import validate_policy_contract
from .runner import LatentWorldPolicyRunner


@dataclass
class LatentWorldPolicyComponents:
    policy_cfg: LatentWorldPolicyConfig
    policy_backend: LatentWorldPolicyBackend
    policy_vlm_adapter: LatentWorldPolicyVLMAdapter
    infer_batch_builder: LatentWorldPolicyInferBatchBuilder
    runner: LatentWorldPolicyRunner


def build_policy_components(config: Any) -> LatentWorldPolicyComponents:
    policy_cfg = LatentWorldPolicyConfigBuilder(config).build()
    validate_policy_contract(config, policy_cfg)

    vlm_model_id = config.framework.qwenvl.base_vlm
    if vlm_model_id is None:
        raise ValueError("Missing `framework.qwenvl.base_vlm` for LaWAM.")

    policy_backend = LatentWorldPolicyBackend.build(policy_cfg, vlm_model_id=str(vlm_model_id))
    policy_vlm_adapter = LatentWorldPolicyVLMAdapter(
        model=policy_backend.vlm,
        processor=policy_backend.processor,
        config=config,
        placeholder_token=policy_cfg.latent_action_placeholder_token,
        act_queries=int(policy_backend.num_action_queries),
        flow_queries=int(policy_backend.flow_action_query.shape[0]),
    )
    infer_batch_builder = LatentWorldPolicyInferBatchBuilder(
        policy_cfg=policy_cfg,
        policy_backend=policy_backend,
        policy_vlm_adapter=policy_vlm_adapter,
        enable_primary_random_resized_crop=bool(config.datasets.vla_data.get("enable_primary_random_resized_crop", False)),
    )
    runner = LatentWorldPolicyRunner(
        policy_backend=policy_backend,
        infer_batch_builder=infer_batch_builder,
    )

    return LatentWorldPolicyComponents(
        policy_cfg=policy_cfg,
        policy_backend=policy_backend,
        policy_vlm_adapter=policy_vlm_adapter,
        infer_batch_builder=infer_batch_builder,
        runner=runner,
    )
