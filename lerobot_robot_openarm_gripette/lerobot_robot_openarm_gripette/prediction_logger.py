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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class PredictionLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = log_dir / f"predictions-{timestamp}.jsonl"
        self._file = self.path.open("w", encoding="utf-8")

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, default=self._json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
