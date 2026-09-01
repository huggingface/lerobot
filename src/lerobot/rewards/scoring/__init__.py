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

"""Reusable offline scoring for frame-aligned reward-model signals."""

from .reader import (
    SCORING_FORMAT,
    SCORING_SCHEMA_VERSION,
    get_scoring_provenance,
    get_signal_descriptors,
    read_frame_signals,
)
from .runner import score_dataset, score_dataset_with_reward_model
from .types import FrameSignals, ScoringSummary, SignalDescriptor

__all__ = [
    "SCORING_FORMAT",
    "SCORING_SCHEMA_VERSION",
    "FrameSignals",
    "ScoringSummary",
    "SignalDescriptor",
    "get_scoring_provenance",
    "get_signal_descriptors",
    "read_frame_signals",
    "score_dataset",
    "score_dataset_with_reward_model",
]
