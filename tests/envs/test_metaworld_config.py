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

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "envs" / "metaworld_config.json"


def test_metaworld_task_descriptions_are_complete_and_unique() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    descriptions = config["TASK_DESCRIPTIONS"]
    task_ids = config["TASK_NAME_TO_ID"]

    assert descriptions.keys() == task_ids.keys()
    assert len(descriptions) == 50
    assert set(task_ids.values()) == set(range(50))
    assert len(set(descriptions.values())) == len(descriptions)
    assert task_ids["push-back-v3"] == 37
    assert task_ids["push-v3"] == 38
    assert descriptions["push-back-v3"] != descriptions["push-v3"]
