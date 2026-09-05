#!/usr/bin/env python

# Copyright 2026 Gangelia. All rights reserved.
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

"""Structured JSONL logging for fault-injection events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FaultEventLogger:
    """Append-only JSONL writer for meaningful fault events only.

    Callers that want a clean file for a new evaluation run should truncate the
    path once before constructing loggers (see ``eval_main``). Individual
    injectors open the path in append mode so multi-task sequential evals can
    share one ``fault_events.jsonl`` without wiping earlier tasks.
    """

    def __init__(self, path: Path | str | None, *, append: bool = True):
        self.path = Path(path) if path is not None else None
        self._file = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            self._file = self.path.open(mode, encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(event, default=_json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> FaultEventLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")
