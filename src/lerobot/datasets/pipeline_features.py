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
    filtered to only those keys matching any of the given patterns.

    - `initial_features`: hardware features from the robot.
    - `use_videos`: whether to treat image features as video.
    - `patterns`: list of substrings or regexes to match against full feature names
                  (e.g. 'ee', 'pos', 'observation.state.ee.x'). If None, no filtering.
    """
    import re

    all_features = pipeline.features(initial_features)

    def keep(key: str) -> bool:
        if patterns is None:
            return True
        return any(re.search(pat, key) for pat in patterns)

    hw: dict[str, Any] = {}
    for full_key, ty in all_features.items():
        # full_key is for example: "action.ee.x" or "observation.state.joint.pos"
        if full_key.startswith("action."):
            name = full_key[len("action.") :]
            prefix = "action"
        elif full_key.startswith("observation.state."):
            name = full_key[len("observation.state.") :]
            prefix = "observation"
        elif full_key.startswith("observation.images."):
            name = full_key[len("observation.images.") :]
            prefix = "observation"
        else:
            continue

        if keep(full_key):
            hw.setdefault(prefix, {})[name] = ty

    out: dict[str, dict] = {}
    if "action" in hw:
        out.update(hw_to_dataset_features(hw["action"], "action", use_videos))
    if "observation" in hw:
        out.update(hw_to_dataset_features(hw["observation"], "observation", use_videos))
    return out
