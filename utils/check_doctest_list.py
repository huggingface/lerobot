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

"""Keep the doctest list honest: every path exists, and the file stays sorted.

Adapted from `transformers/utils/check_doctest_list.py`. It is agnostic to whether the list is an allowlist
(what we have now) or a denylist (where transformers ended up), so it survives that inversion unchanged.

Check, as CI does:

```bash
python utils/check_doctest_list.py
```

Sort in place:

```bash
python utils/check_doctest_list.py --fix_and_overwrite
```
"""

import argparse
import sys
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parent.parent
DOCTEST_FILE_PATHS = ["documentation_tests.txt"]


def split_header(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a list file into its leading comment header and its path entries.

    Args:
        lines (`list[str]`):
            The file's lines, without trailing newlines.

    Returns:
        `tuple[list[str], list[str]]`: The leading comment/blank lines, and the remaining lines.
    """
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return lines[:i], lines[i:]
    return lines, []


def clean_doctest_list(doctest_file: Path, overwrite: bool = False) -> None:
    """Check, and optionally fix, one doctest list file.

    Args:
        doctest_file (`Path`):
            The list file to check or clean.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to fix problems in place. When `False`, raises instead.

    Raises:
        ValueError: If the file lists a path that does not exist, or is not alphabetically sorted and
            `overwrite` is `False`.
    """
    lines = doctest_file.read_text(encoding="utf-8").splitlines()
    header, entries = split_header(lines)
    paths = [line.strip().split(" ")[0] for line in entries if line.strip()]

    non_existent = [p for p in paths if not (REPO_PATH / p).exists()]
    if non_existent:
        listed = "\n".join(f"- {p}" for p in non_existent)
        raise ValueError(f"`{doctest_file.name}` contains non-existent paths:\n{listed}")

    if paths != sorted(paths):
        if not overwrite:
            raise ValueError(
                f"Files in `{doctest_file.name}` are not in alphabetical order, run "
                "`make fix-docstrings` to fix this automatically."
            )
        doctest_file.write_text("\n".join(header + sorted(paths)) + "\n", encoding="utf-8")


def main() -> int:
    """Run the check over every doctest list file.

    Returns:
        `int`: A process exit code — `0` when every file is clean, `1` otherwise.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    failed = False
    for name in DOCTEST_FILE_PATHS:
        try:
            clean_doctest_list(REPO_PATH / "utils" / name, args.fix_and_overwrite)
        except ValueError as error:
            print(error, file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
