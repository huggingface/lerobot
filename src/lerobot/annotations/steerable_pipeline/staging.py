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
"""Per-episode staging.

Each module writes its raw output as a JSONL file under
``<staging_dir>/episode_{ep:06d}/<module>.jsonl``. The writer reads back this
staging tree and partitions rows into the two language columns.

JSONL is preferred over parquet here because the staging artifact is meant to
be human-inspectable, easy to diff between prompt iterations, and trivially
appended to. The final dataset format is parquet; staging is just an
intermediate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ModuleName = str

_MODULES: tuple[ModuleName, ...] = (
    "plan",
    "interjections",
    "vqa",
)


@dataclass
class EpisodeStaging:
    """Filesystem layout for a single episode's staged module outputs."""

    root: Path
    episode_index: int

    @property
    def episode_dir(self) -> Path:
        return self.root / f"episode_{self.episode_index:06d}"

    def path_for(self, module: ModuleName) -> Path:
        if module not in _MODULES:
            raise ValueError(f"Unknown module {module!r}; expected one of {_MODULES}")
        return self.episode_dir / f"{module}.jsonl"

    def write(self, module: ModuleName, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.path_for(module)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: a crash mid-write would otherwise leave a
        # half-written JSONL file that ``read()`` would then fail to
        # parse. Write to a sibling .tmp and rename so the target path
        # only ever points at a complete file.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                f.write("\n")
        tmp_path.replace(path)
        return path

    def read(self, module: ModuleName) -> list[dict[str, Any]]:
        path = self.path_for(module)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def read_all(self) -> dict[ModuleName, list[dict[str, Any]]]:
        return {m: self.read(m) for m in _MODULES}

    def has(self, module: ModuleName) -> bool:
        return self.path_for(module).exists()
