# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Explicit replacements for pretrained encoders in FLUX3 tests."""

import pytest

from lerobot.policies.flux3 import Flux3Policy
from tests.policies.flux3.helpers import MockTextEncoder


@pytest.fixture
def fake_text_encoder(monkeypatch):
    monkeypatch.setattr(
        Flux3Policy,
        "_build_text_encoder",
        lambda self, config: MockTextEncoder(self.dit_params.context_in_dim),
    )
