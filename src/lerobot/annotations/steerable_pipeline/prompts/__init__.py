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
"""Prompt templates loaded as plain text.

One file per use site. Templates use ``str.format(**vars)`` substitution; we
intentionally avoid jinja2 here so the templates remain inspectable in
plain editors and roundtrip cleanly through ``ruff format``.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Read prompt template ``name.txt`` from the ``prompts/`` directory."""
    path = _DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
