#!/usr/bin/env python

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

from types import SimpleNamespace

from lerobot.utils.import_utils import register_third_party_plugins


class _FakeDist:
    def __init__(self, name: str):
        self.metadata = {"Name": name}


def test_register_third_party_plugins_imports_processor_prefix(monkeypatch):
    fake_dists = [
        _FakeDist("lerobot_processor_safety_verifier"),
        _FakeDist("lerobot_policy_turbovla"),
        _FakeDist("some_unrelated_package"),
    ]
    monkeypatch.setattr("importlib.metadata.distributions", lambda: iter(fake_dists))

    imported = []
    monkeypatch.setattr("importlib.import_module", lambda name: imported.append(name) or SimpleNamespace())

    register_third_party_plugins()

    assert "lerobot_processor_safety_verifier" in imported
    assert "lerobot_policy_turbovla" in imported
    assert "some_unrelated_package" not in imported
