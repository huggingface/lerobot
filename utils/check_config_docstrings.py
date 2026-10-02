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

"""Check that every registered hardware config documents the fields users have to get right.

Modelled on `transformers/utils/check_config_docstrings.py`, which checks that every model config links a
checkpoint. LeRobot's equivalent question is the one every new user hits: which port is the device on, and
what happens on calibration. A config that leaves those undocumented sends people to the source.

Only fields the config actually declares are required — a config without a `port` is not asked to document
one.

```bash
python utils/check_config_docstrings.py
```
"""

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_docstrings import _re_args, _re_parse_arg, find_indent, iter_objects_to_check  # noqa: E402

# Fields whose semantics are not obvious from the name and that a user must set correctly on first run.
REQUIRED_FIELDS = ["port"]

# A config must say something about calibration if it participates in it at all.
CALIBRATION_PATTERN = re.compile(r"calibrat", re.IGNORECASE)

MODULES_TO_CHECK = ["lerobot.robots"]

# Configs that document their fields with `#` comments above each field, which doc-builder cannot see.
# Each entry is removed as that config's comments are converted to an `Args:` block.
OBJECTS_TO_IGNORE: set[str] = {
    "BiOpenArmFollowerConfig",
    "BiRebotB601FollowerConfig",
    "BiSOFollowerConfig",
    "EarthRoverMiniPlusConfig",
    "HopeJrArmConfig",
    "HopeJrHandConfig",
    "KochFollowerConfig",
    "LeKiwiConfig",
    "OmxFollowerConfig",
    "OpenArmFollowerConfig",
    "Reachy2RobotConfig",
    "RebotB601FollowerRobotConfig",
    "SOFollowerRobotConfig",
}


def documented_args(obj: object) -> set[str]:
    """Return the argument names documented in an object's `Args:` block.

    Args:
        obj (`object`):
            The class to inspect.

    Returns:
        `set[str]`: The documented argument names, empty if there is no `Args:` section.
    """
    doc = getattr(obj, "__doc__", None)
    if not doc:
        return set()

    lines = doc.split("\n")
    idx = 0
    while idx < len(lines) and _re_args.search(lines[idx]) is None:
        idx += 1
    if idx == len(lines):
        return set()

    indent = find_indent(lines[idx])
    names = set()
    idx += 1
    while idx < len(lines) and (len(lines[idx].strip()) == 0 or find_indent(lines[idx]) > indent):
        if find_indent(lines[idx]) == indent + 4:
            match = _re_parse_arg.search(lines[idx])
            if match is not None:
                names.add(match.groups()[1])
        idx += 1
    return names


def check_config_docstrings() -> list[str]:
    """Check every registered config in `MODULES_TO_CHECK`.

    Returns:
        `list[str]`: One message per config that is missing a required field or calibration semantics.
    """
    from lerobot.robots import RobotConfig

    failures = []
    for module_name in MODULES_TO_CHECK:
        for obj in iter_objects_to_check(module_name):
            if not inspect.isclass(obj) or not issubclass(obj, RobotConfig) or obj is RobotConfig:
                continue
            if inspect.isabstract(obj) or obj.__qualname__ in OBJECTS_TO_IGNORE:
                continue

            try:
                fields = set(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                continue

            doc = getattr(obj, "__doc__", "") or ""
            documented = documented_args(obj)
            name = f"{obj.__module__}.{obj.__qualname__}"

            for field in REQUIRED_FIELDS:
                if field in fields and field not in documented:
                    failures.append(f"{name}: does not document `{field}`")

            if "calibration_dir" in fields and CALIBRATION_PATTERN.search(doc) is None:
                failures.append(f"{name}: says nothing about calibration")

    return failures


def main() -> int:
    """Run the check.

    Returns:
        `int`: `0` when every registered config is documented, `1` otherwise.
    """
    failures = check_config_docstrings()
    if failures:
        print(
            "The following robot configs are missing documentation a user needs on first run. See "
            "docs/source/writing_docstrings.mdx:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
