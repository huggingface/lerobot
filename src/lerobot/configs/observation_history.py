# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from typing import Any, cast

from lerobot.utils.constants import OBS_IMAGE, OBS_PREFIX, OBS_STATE


def resolve_observation_delta_indices(config: Any, key: str) -> list[int] | None:
    """Resolve training and inference observation offsets for a canonical policy key."""
    if key.startswith(OBS_IMAGE):
        indices = getattr(config, "image_observation_delta_indices", None)
    elif key == OBS_STATE:
        indices = getattr(config, "state_observation_delta_indices", None)
    elif not key.startswith(OBS_PREFIX):
        return None
    else:
        indices = None
    return cast(
        list[int] | None,
        indices if indices is not None else getattr(config, "observation_delta_indices", None),
    )
