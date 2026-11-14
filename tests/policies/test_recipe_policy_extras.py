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

"""Policies whose processor renders a recipe install the dataset extra with their own extra."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
POLICIES = ROOT / "src" / "lerobot" / "policies"
# Found from the code, so a new recipe policy is checked without being listed here.
RECIPE_POLICIES = sorted(
    {
        path.relative_to(POLICIES).parts[0]
        for path in POLICIES.glob("*/**/*.py")
        if re.search(r"Render(Runtime|Training)MessagesStep", path.read_text(encoding="utf-8"))
    }
)
EXTRA_NAMES = {"wall_x": "wallx"}


def test_recipe_policy_extras_include_dataset_extra() -> None:
    # No CI job installs a policy extra on its own, and the recipe import checks for the dataset packages.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]
    assert RECIPE_POLICIES
    for policy in RECIPE_POLICIES:
        extra = EXTRA_NAMES.get(policy, policy)
        assert extra in extras, f"add {policy!r} to EXTRA_NAMES with the name of its extra"
        assert "lerobot[dataset]" in extras[extra], f"lerobot[{extra}] misses lerobot[dataset]"
