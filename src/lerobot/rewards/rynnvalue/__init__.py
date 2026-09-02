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

"""LeRobot integration for the RynnValue reward model."""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _transformers_available

from .configuration_rynnvalue import RynnValueConfig

if TYPE_CHECKING or _transformers_available:
    from .modeling_rynnvalue import RynnValuePrediction, RynnValueRewardModel
    from .processor_rynnvalue import make_rynnvalue_pre_post_processors

    __all__ = [
        "RynnValueConfig",
        "RynnValuePrediction",
        "RynnValueRewardModel",
        "make_rynnvalue_pre_post_processors",
    ]
else:
    __all__ = ["RynnValueConfig"]
