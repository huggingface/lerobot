"""Reject renames that discard data while preserving simultaneous swaps and chains."""

from collections.abc import Callable
from typing import Any

import pytest

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.converters import create_transition
from lerobot.processor.rename_processor import (
    RenameObservationsProcessorStep,
    rename_batch_keys,
    rename_stats,
)


def apply_rename(kind: str, keys: list[str], mapping: dict[str, str]) -> dict[str, Any]:
    """Exercise each public rename entry point using distinguishable source values."""
    values = dict(zip(keys, range(len(keys)), strict=True))
    if kind == "batch":
        return rename_batch_keys(values, mapping)
    if kind == "observation":
        return RenameObservationsProcessorStep(mapping).observation(values)
    if kind == "stats":
        return rename_stats({key: {"mean": value} for key, value in values.items()}, mapping)
    features = {
        PipelineFeatureType.OBSERVATION: {
            key: PolicyFeature(type=FeatureType.STATE, shape=(value + 1,)) for key, value in values.items()
        }
    }
    return RenameObservationsProcessorStep(mapping).transform_features(features)[
        PipelineFeatureType.OBSERVATION
    ]


@pytest.mark.parametrize("kind", ["batch", "observation", "stats", "features"])
@pytest.mark.parametrize("mapping", [{"left": "camera", "right": "camera"}, {"left": "right"}])
@pytest.mark.parametrize("keys", [["left", "right"], ["right", "left"]])
def test_reject_rename_collisions(kind: str, mapping: dict[str, str], keys: list[str]) -> None:
    """Both many-to-one and unchanged-destination collisions must fail in either input order."""
    with pytest.raises(ValueError, match="collision") as error:
        apply_rename(kind, keys, mapping)
    assert "left" in str(error.value) and "right" in str(error.value)


@pytest.mark.parametrize(
    "rename", [rename_batch_keys, RenameObservationsProcessorStep({"old": "new"}).observation]
)
@pytest.mark.parametrize("suffix", ["_is_pad", "_padding_mask"])
def test_reject_implicit_metadata_collision(rename: Callable, suffix: str) -> None:
    """Automatic padding-key renames must not overwrite pre-existing masks."""
    data = {f"old{suffix}": True, f"new{suffix}": False}
    with pytest.raises(ValueError, match="collision"):
        if rename is rename_batch_keys:
            rename(data, {"old": "new"})
        else:
            rename(data)


@pytest.mark.parametrize("kind", ["batch", "observation", "stats", "features"])
@pytest.mark.parametrize("mapping", [{"left": "right", "right": "left"}, {"left": "right", "right": "end"}])
def test_simultaneous_renames_remain_valid(kind: str, mapping: dict[str, str]) -> None:
    """A source name may be reused when its original value moves elsewhere."""
    result = apply_rename(kind, ["left", "right"], mapping)
    assert len(result) == 2
    assert set(result) == set(mapping.values())
    if kind in ("batch", "observation"):
        assert result[mapping["left"]] == 0
        assert result[mapping["right"]] == 1


@pytest.mark.parametrize("kind", ["batch", "observation", "stats", "features"])
def test_absent_source_does_not_create_collision(kind: str) -> None:
    """Multiple aliases are valid when only one source is actually present."""
    assert set(apply_rename(kind, ["left"], {"left": "camera", "right": "camera"})) == {"camera"}


def test_collision_leaves_transition_unchanged() -> None:
    """Fail through the processor API without mutating the caller's observation."""
    observation = {"left": [1], "right": [2]}
    transition = create_transition(observation=observation)
    processor = RenameObservationsProcessorStep({"left": "right"})
    with pytest.raises(ValueError, match="collision"):
        processor(transition)
    assert observation == {"left": [1], "right": [2]}


def test_explicit_metadata_mapping_takes_precedence() -> None:
    """An explicit suffix mapping still overrides the automatic base-key mapping."""
    batch = {"old": 1, "old_is_pad": True, "new_is_pad": False}
    assert rename_batch_keys(batch, {"old": "new", "old_is_pad": "source_mask"}) == {
        "new": 1,
        "source_mask": True,
        "new_is_pad": False,
    }
