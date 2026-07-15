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

"""Compatibility entry point for ``lerobot-language-runtime``.

New commands should use ``lerobot-rollout --language`` or ``--sim``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from lerobot.runtime.cli import run

    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
