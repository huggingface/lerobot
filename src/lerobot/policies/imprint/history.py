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

PREFIX_LENGTH = "observation.imprint.length"
PREFIX_FRAMES = "observation.imprint.frame_indices"


def prefix_indices(step: int, downsample: int) -> list[int]:
    """Canonical reference framing, including the current unsampled frame."""
    indices = list(range(0, step + 1, downsample))
    if indices[-1] != step:
        indices.append(step)
    return indices
