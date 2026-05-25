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

"""Strategy factory: config type-name → strategy class dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseStrategy
from .core import RolloutStrategy
from .dagger import DAggerStrategy
from .highlight import HighlightStrategy
from .sentry import SentryStrategy

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig


def create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    """Instantiate the appropriate strategy from a config object.

    Dispatches on ``config.type`` (the name registered via
    ``draccus.ChoiceRegistry``).
    """
    if config.type == "base":
        return BaseStrategy(config)
    if config.type == "sentry":
        return SentryStrategy(config)
    if config.type == "highlight":
        return HighlightStrategy(config)
    if config.type == "dagger":
        return DAggerStrategy(config)
    raise ValueError(f"Unknown strategy type '{config.type}'. Available: base, sentry, highlight, dagger")
