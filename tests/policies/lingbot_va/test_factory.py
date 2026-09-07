#!/usr/bin/env python

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

from __future__ import annotations

import pytest

from lerobot.policies.factory import make_policy_config
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig


def test_make_policy_config_returns_lingbot_va() -> None:
    cfg = make_policy_config("lingbot_va", device="cpu")
    assert isinstance(cfg, LingBotVAConfig)


def test_get_policy_class_resolves_lazily() -> None:
    # Importing the policy class pulls in diffusers (Wan2.2 stack); skip if unavailable.
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    from lerobot.policies.factory import get_policy_class

    cls = get_policy_class("lingbot_va")
    assert cls.name == "lingbot_va"
    assert cls.config_class is LingBotVAConfig
