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

"""
Pluggable reward models for LeRobot.

This package provides the abstract interface (RewardModel / RewardModelConfig)
and concrete implementations used by both IL and RL pipelines.

"""

from lerobot.rewards.base import RewardModel, RewardModelConfig

__all__ = ["RewardModel", "RewardModelConfig"]
