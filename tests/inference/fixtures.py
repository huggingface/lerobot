"""Canonical schemas for policy conformance tests, independent of deployment examples."""

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.inference import FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.policies import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline


@dataclass(frozen=True)
class RobotSchema:
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec


def preparation_batches(
    policy: PreTrainedPolicy,
    schema: RobotSchema,
    pre: PolicyProcessorPipeline | None = None,
    post: PolicyProcessorPipeline | None = None,
) -> tuple[PolicyRunner, ObservationSnapshot, dict, dict]:
    """Compare transport preparation with the ordinary local observation path."""
    runner = PolicyRunner(
        policy,
        PolicyProcessorPipeline([]) if pre is None else pre,
        PolicyProcessorPipeline([]) if post is None else post,
        action_interval=1 / 30,
        features=tuple(schema.features),
        action_feature=schema.action_feature,
    )
    arrays = {
        feature.name: np.full(feature.shape, 64 + i, dtype=feature.dtype)
        for i, feature in enumerate(schema.features)
    }
    arrays["observation.state"] = np.arange(6, dtype=np.float32)
    source = ObservationSnapshot(arrays, 0.0, "pick up the cube")
    return (
        runner,
        source,
        runner.preprocessor(runner._batch(source)),
        runner.preprocessor(prepare_observation_for_inference(arrays, torch.device("cpu"), source.task)),
    )


def omx_contract(cameras: tuple[tuple[str, str], ...]) -> RobotSchema:
    """Six normalized joints and full-resolution RGB views used by the preparation checks."""
    names = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    semantics = "omx-normalized-joints-gripper-percent-v1"
    return RobotSchema(
        features=(
            FeatureSpec("observation.state", (6,), "float32", names=names, semantics=semantics),
            *(
                FeatureSpec(f"observation.images.{name}", (480, 640, 3), "uint8", kind="rgb", semantics=view)
                for name, view in cameras
            ),
        ),
        action_feature=FeatureSpec("action", (6,), "float32", names=names, semantics=semantics),
    )
