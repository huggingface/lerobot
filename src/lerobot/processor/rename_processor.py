#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rename_observations_processor")
class RenameObservationsProcessorStep(ObservationProcessorStep):
    """
    A processor step that renames keys in an observation dictionary.

    This step is useful for creating a standardized data interface by mapping keys
    from an environment's format to the format expected by a LeRobot policy or
    other downstream components.

    Attributes:
        rename_map: A dictionary mapping from old key names to new key names.
                    Keys present in an observation that are not in this map will
                    be kept with their original names.
    """

    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation):
        return rename_keys(observation, self.rename_map)

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transforms:
        - Each key in the observation that appears in `rename_map` is renamed to its value.
        - Keys not in `rename_map` remain unchanged.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = rename_keys(
            features[PipelineFeatureType.OBSERVATION], self.rename_map
        )
        return new_features


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    Renames the top-level keys in a statistics dictionary using a provided mapping.

    This is a helper function typically used to keep normalization statistics
    consistent with renamed observation or action features. It performs a defensive
    deep copy to avoid modifying the original `stats` dictionary.

    Args:
        stats: A nested dictionary of statistics, where top-level keys are
               feature names (e.g., `{"observation.state": {"mean": 0.5}}`).
        rename_map: A dictionary mapping old feature names to new feature names.

    Returns:
        A new statistics dictionary with its top-level keys renamed. Returns an
        empty dictionary if the input `stats` is empty.
    """
    if not stats:
        return {}
    return {
        key: deepcopy(sub_stats) if sub_stats is not None else {}
        for key, sub_stats in rename_keys(stats, rename_map).items()
    }


def _rename_key_with_metadata(key: str, rename_map: dict[str, str]) -> str:
    """Rename a feature key while preserving temporal sampling metadata suffixes."""
    if key in rename_map:
        return rename_map[key]
    for suffix in ("_is_pad", "_padding_mask"):
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in rename_map:
                return f"{rename_map[base]}{suffix}"
    return key


def rename_batch_keys(batch: dict[str, Any], rename_map: dict[str, str] | None) -> dict[str, Any]:
    """Canonicalize raw dataset keys before they are grouped into a transition."""
    if not rename_map:
        return batch
    return rename_keys(batch, rename_map)


def rename_keys[V](mapping: Mapping[str, V], rename_map: dict[str, str] | None) -> dict[str, V]:
    """Rename the keys of `mapping` (and their temporal padding keys) with `rename_map`.

    Raises:
        ValueError: Two keys get the same name, e.g. `{"action.cartesian": "action"}` on data that
            already has an `action`. Rename the existing key as well, e.g. `{"action": "action.joint"}`.
    """
    renamed: dict[str, V] = {}
    sources: dict[str, str] = {}
    for key, value in mapping.items():
        new_key = _rename_key_with_metadata(key, rename_map or {})
        if new_key in sources:
            raise ValueError(
                f"Both {sources[new_key]!r} and {key!r} are named {new_key!r} after rename_map. Map one of "
                f"them to another name as well, e.g. add {{{new_key!r}: {new_key + '.original'!r}}} to rename "
                f"a key onto an existing {new_key!r}."
            )
        renamed[new_key] = value
        sources[new_key] = key
    return renamed
