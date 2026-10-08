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

"""Real-Time Chunking (RTC) utilities for action-chunking policies."""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports

from .configuration_rtc import RTCConfig
from .latency_tracker import LatencyTracker

# These import torch, so each is imported the first time it is used.
if TYPE_CHECKING:
    from .action_interpolator import ActionInterpolator
    from .action_queue import ActionQueue
    from .modeling_rtc import RTCProcessor
    from .relative import reanchor_relative_rtc_prefix
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "ActionInterpolator": ".action_interpolator.ActionInterpolator",
            "ActionQueue": ".action_queue.ActionQueue",
            "RTCProcessor": ".modeling_rtc.RTCProcessor",
            "reanchor_relative_rtc_prefix": ".relative.reanchor_relative_rtc_prefix",
        },
    )

__all__ = [
    "ActionInterpolator",
    "ActionQueue",
    "LatencyTracker",
    "RTCConfig",
    "RTCProcessor",
    "reanchor_relative_rtc_prefix",
]
