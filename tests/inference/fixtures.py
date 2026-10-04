"""Canonical schemas for policy conformance tests, independent of deployment examples."""

from dataclasses import dataclass

from lerobot.inference import FeatureSpec


@dataclass(frozen=True)
class RobotSchema:
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec


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
