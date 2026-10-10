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

"""Consumer-facing reward-model capability types.

Reward models expose semantic methods instead of implementing one universal
scalar-reward method. These protocols describe the methods that consumers may
rely on for static type checking. They do not register, construct,
or select reward-model implementations at runtime.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor


@dataclass(frozen=True)
class ProgressPrediction:
    """Per-frame progress values with shape `(batch, time)`."""

    progress: Tensor


class ProgressPredictor(Protocol):
    """A model that predicts frame-aligned progress from encoded inputs."""

    def predict_progress(
        self,
        batch: Mapping[str, Any],
    ) -> ProgressPrediction: ...
