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

"""Which installed plugin packages `register_third_party_plugins` imports."""

from types import SimpleNamespace

import pytest

from lerobot.utils import import_utils

INSTALLED = [
    "lerobot_robot_arm",
    "lerobot_camera_eye",
    "lerobot_teleoperator_leader",
    "lerobot_policy_brain",
    "lerobot_env_world",
    "lerobot_strategy_plan",
    "numpy",
    "lerobot_robotics_tools",  # not a robot plugin: no "_" right after "robot"
]


@pytest.fixture
def imported(monkeypatch):
    imported = []
    distributions = [SimpleNamespace(metadata={"Name": name}) for name in INSTALLED]
    monkeypatch.setattr(import_utils.importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(import_utils.importlib, "import_module", imported.append)
    return imported


def test_every_plugin_kind_loads_by_default_and_kinds_narrow_it(imported):
    import_utils.register_third_party_plugins()
    assert imported == INSTALLED[:6]

    imported.clear()
    import_utils.register_third_party_plugins(kinds=import_utils.DEVICE_PLUGIN_KINDS)
    assert imported == INSTALLED[:3]

    imported.clear()
    import_utils.register_third_party_plugins(kinds=import_utils.DEVICE_AND_ENV_PLUGIN_KINDS)
    assert imported == [*INSTALLED[:3], "lerobot_env_world"]


def test_kinds_may_be_an_iterator(imported):
    import_utils.register_third_party_plugins(kinds=iter(["camera"]))
    assert imported == ["lerobot_camera_eye"]


def test_unknown_plugin_kind_raises_before_any_import(imported):
    with pytest.raises(ValueError, match="Unknown plugin kinds"):
        import_utils.register_third_party_plugins(kinds=("robot", "teleop"))
    assert imported == []
