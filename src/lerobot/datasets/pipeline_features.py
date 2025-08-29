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

from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor.pipeline import RobotProcessor


def aggregate_pipeline_dataset_features(
    pipeline: RobotProcessor,
    initial_features: dict[str, Any],
    *,
    use_videos: bool = True,
    patterns: Sequence[str] | None = None,
) -> dict[str, dict]:
    """
    Aggregates the pipeline's features and returns a features dict ready for the dataset,
    filtered to only those keys matching any of the given patterns (for action/state only).

    - `initial_features`: raw camera specs, e.g. {"front": (h,w,c), ...}
    - `use_videos`: whether to treat image features as video streams
    - `patterns`: regexes to filter action & state features; images are included
                  whenever use_videos=True, regardless of patterns.
    """
    import re

    # Gather everything the pipeline features specifies, seeded with hardware cams:
    all_features = pipeline.transform_features(initial_features)

    # Helper to decide which action/state keys survive the `patterns` filter:
    def keep(key: str) -> bool:
        if patterns is None:
            return True
        return any(re.search(pat, key) for pat in patterns)

    # Start with hardware dict, injecting initial cameras if videos are ON:
    hw: dict[str, dict[str, Any]] = {}
    if use_videos:
        cams = {
            name: shape
            for name, shape in initial_features.items()
            if isinstance(shape, tuple) and len(shape) == 3
        }
        if cams:
            hw["observation"] = dict(cams)

    # Go over every feature from the pipeline and merge:
    for full_key, ty in all_features.items():
        if full_key.startswith(f"{ACTION}."):
            # action.<feat>
            if not keep(full_key):
                continue
            name = full_key[len(f"{ACTION}.") :]
            hw.setdefault(ACTION, {})[name] = ty

        elif full_key.startswith(f"{OBS_STATE}."):
            # observation.state.<feat>
            if not keep(full_key):
                continue
            name = full_key[len(f"{OBS_STATE}.") :]
            hw.setdefault("observation", {})[name] = ty

        elif full_key.startswith(f"{OBS_IMAGES}."):
            # observation.images.<cam>
            # images obey ONLY the use_videos flag, not patterns
            if not use_videos:
                continue
            name = full_key[len(f"{OBS_IMAGES}.") :]
            hw.setdefault("observation", {})[name] = ty

        else:
            # anything else (e.g. policy-only features) is ignored here
            continue

    out: dict[str, dict] = {}
    if ACTION in hw:
        out.update(hw_to_dataset_features(hw[ACTION], ACTION, use_videos))
    if "observation" in hw:
        out.update(hw_to_dataset_features(hw["observation"], "observation", use_videos))

    return out
