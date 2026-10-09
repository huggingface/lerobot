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

"""Read native simulator configuration with explicit CLI overrides."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    """Override existing dotted YAML keys; reject typos and malformed paths."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Simulator configuration must be a YAML mapping")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected a dotted key=value override, got {override!r}")
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown simulator override path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown simulator override key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return data
