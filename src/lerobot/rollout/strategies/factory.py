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

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from ..configs import RolloutStrategyConfig
from .base import BaseStrategy
from .core import RolloutStrategy
from .dagger import DAggerStrategy
from .episodic import EpisodicStrategy
from .highlight import HighlightStrategy
from .sentry import SentryStrategy


def create_strategy(config: RolloutStrategyConfig) -> RolloutStrategy:
    """Instantiate the appropriate strategy from a config object.

    Dispatches on ``config.type`` (the name registered via ``draccus.ChoiceRegistry``)
    for the built-in strategies, and falls back to resolving the implementation class
    by naming convention for anything else — the same open-registry pattern the robot,
    teleoperator and camera factories use.  A third-party strategy therefore needs no
    edit here: it only has to name its class after its config class without the
    trailing ``Config``, and make it importable from the config's package.

    Raises:
        ValueError: If *config* is not registered on ``RolloutStrategyConfig`` — checked
            before resolution, so forgetting ``@register_subclass`` is reported here even
            when the implementation class would have resolved.
        ImportError: If a registered config's implementation class cannot be found.  Its
            message names every module that was tried; it is deliberately not wrapped,
            and neither is anything raised by the strategy's own ``__init__``.
    """
    # ``config.type`` raises for a subclass that was never registered, which is itself
    # one of the mistakes this factory should report clearly.
    try:
        name: str | None = config.type
    except Exception:
        name = None

    if name == "base":
        return BaseStrategy(config)
    if name == "sentry":
        return SentryStrategy(config)
    if name == "highlight":
        return HighlightStrategy(config)
    if name == "dagger":
        return DAggerStrategy(config)
    if name == "episodic":
        return EpisodicStrategy(config)

    registered = RolloutStrategyConfig.get_known_choices()
    if name is None or name not in registered:
        # Refuse before resolving: a config that resolves but was never registered cannot
        # be selected from the CLI, and letting it through here would trade this message
        # for draccus's "Cannot find choice name" from whatever reads ``.type`` next.
        raise ValueError(
            f"Unknown strategy type '{name or type(config).__name__}'. "
            f"Registered: {', '.join(sorted(registered))}. A third-party strategy must "
            f"register its config with @RolloutStrategyConfig.register_subclass('<name>') "
            f"and name its class after its config class without the 'Config' suffix."
        )

    # Registered, so anything below is not a dispatch problem.  Let the real cause
    # through: ``Could not locate device class ...`` names every module that was tried,
    # and anything else came out of the strategy's own constructor.
    return cast(RolloutStrategy, make_device_from_device_class(config))
