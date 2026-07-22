from .batch_builder import LatentWorldPolicyInferBatchBuilder
from .runtime import LatentWorldPolicyComponents, build_policy_components
from .types import (
    LatentWorldPolicyInferBatch,
    LatentWorldPolicyInferExample,
    LatentWorldPolicyTrainBatch,
    LatentWorldPolicyTrainRawSample,
)
from .vlm_adapter import LatentWorldPolicyVLMAdapter


def __getattr__(name: str):
    if name == "LatentWorldPolicyConfigBuilder":
        from .config_builder import LatentWorldPolicyConfigBuilder

        return LatentWorldPolicyConfigBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatentWorldPolicyInferBatch",
    "LatentWorldPolicyInferBatchBuilder",
    "LatentWorldPolicyInferExample",
    "LatentWorldPolicyComponents",
    "LatentWorldPolicyConfigBuilder",
    "LatentWorldPolicyTrainBatch",
    "LatentWorldPolicyTrainRawSample",
    "LatentWorldPolicyVLMAdapter",
    "build_policy_components",
]
