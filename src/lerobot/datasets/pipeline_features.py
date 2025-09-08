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

from collections.abc import Sequence
from typing import Any

from lerobot.configs.types import PipelineFeatureType
from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import DataProcessorPipeline


def create_initial_features(
    action: dict[str, Any] | None, observation: dict[str, Any] | None
) -> dict[PipelineFeatureType, dict[str, Any]]:
    """
    Creates the initial features dict for the dataset from the action and observation specs.

    - `action`: dict of action feature names to their types/shapes
    - `observation`: dict of observation feature names to their types/shapes
    """
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    if action:
        features[PipelineFeatureType.ACTION] = action
    if observation:
        features[PipelineFeatureType.OBSERVATION] = observation
    return features


def aggregate_pipeline_dataset_features(
    pipeline: DataProcessorPipeline,
    initial_features: dict[PipelineFeatureType, dict[str, Any]],
    *,
    use_videos: bool = True,
    patterns: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Aggregates and filters dataset features based on a data processing pipeline.

    This function determines the final structure of dataset features after applying a series
    of processing steps defined in a pipeline. It starts with an initial set of hardware
    features (e.g., camera image shapes), transforms them using the pipeline, and then
    filters the results.

    Image features are controlled by the `use_videos` flag, while action and state features
    can be selectively included by matching their keys against the provided regex `patterns`.
    The final output is formatted to be compatible with Hugging Face Datasets feature dictionaries.

    Args:
        pipeline (DataProcessorPipeline): The data processing pipeline that defines all
            feature transformations.
        initial_features (dict[str, Any]): A dictionary of initial hardware features, where
            keys are feature names and values are their shapes or types (e.g., camera resolutions).
        use_videos (bool): If `True`, includes image/video features in the output. Defaults to `True`.
        patterns (Sequence[str] | None): An optional sequence of regular expression patterns.
            Only action and state keys that match at least one pattern will be included. If `None`,
            all action and state keys are kept. Defaults to `None`.

    Returns:
        dict[str, dict]: A dictionary representing the final dataset features, structured for
        use with `datasets.Features`.
    """
    import re

    all_features = pipeline.transform_features(initial_features)

    def keep(key: str) -> bool:
        if patterns is None:
            return True
        return any(re.search(pat, key) for pat in patterns)

    hw: dict[str, dict[str, Any]] = {}
    obs_initial = initial_features.get(PipelineFeatureType.OBSERVATION, {})
    if use_videos:
        cams = {
            name: shape for name, shape in obs_initial.items() if isinstance(shape, tuple) and len(shape) == 3
        }
        if cams:
            hw["observation"] = dict(cams)

    # Known prefix tokens to strip if present in keys.
    images_token = OBS_IMAGES.split(".")[-1]
    state_token = OBS_STATE.split(".")[-1]
    action_token = ACTION.split(".")[-1]

    def strip_known_prefix(key: str) -> str:
        # remove any of the known prefixes if present
        for prefix in (
            f"{ACTION}.",
            f"{OBS_STATE}.",
            f"{OBS_IMAGES}.",
            f"{action_token}.",
            f"{state_token}.",
            f"{images_token}.",
        ):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    # all_features is by PipelineFeatureType now; iterate buckets and merge.
    for ptype, feats in all_features.items():
        # feats: dict[str, Any]
        for key, ty in feats.items():
            # Normalize whether the feature key included a prefix or not.
            # For pattern matching, recreate a full-key with the appropriate prefix
            # so existing regexes keep working.
            if ptype == PipelineFeatureType.ACTION:
                # patterns (if any) are applied directly to the feature key
                if not keep(key):
                    continue
                name = strip_known_prefix(key)
                hw.setdefault(ACTION, {})[name] = ty

            elif ptype == PipelineFeatureType.OBSERVATION:
                # Decide whether this observation feature is images vs state.
                is_image = (
                    key.startswith(f"{OBS_IMAGES}.")
                    or key.startswith(f"{images_token}.")
                    or f".{images_token}." in key
                )
                # note: anything not detected as image is treated as state-like

                if is_image:
                    # images obey ONLY the use_videos flag, not patterns
                    if not use_videos:
                        continue
                    name = strip_known_prefix(key)
                    hw.setdefault("observation", {})[name] = ty
                    continue

                # Treat anything not explicitly images as state-like and apply patterns
                if not keep(key):
                    continue
                name = strip_known_prefix(key)
                hw.setdefault("observation", {})[name] = ty

            else:
                # ignore other pipeline feature buckets
                continue

    out: dict[str, dict] = {}
    if ACTION in hw:
        out.update(hw_to_dataset_features(hw[ACTION], ACTION, use_videos))
    if "observation" in hw:
        out.update(hw_to_dataset_features(hw["observation"], "observation", use_videos))

    return out
