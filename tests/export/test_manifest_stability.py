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

from __future__ import annotations

import json
from pathlib import Path

from tests.export.conftest import create_act_policy_and_batch, require_onnx


@require_onnx
def test_manifest_is_stable_across_reexports(tmp_path: Path) -> None:
    from lerobot.export import export_policy

    policy, batch = create_act_policy_and_batch()
    first = export_policy(policy, tmp_path / "first", backend="onnx", example_batch=batch)
    second = export_policy(policy, tmp_path / "second", backend="onnx", example_batch=batch)

    first_bytes = _normalized_manifest_bytes(first / "manifest.json")
    second_bytes = _normalized_manifest_bytes(second / "manifest.json")

    assert first_bytes == second_bytes


def _normalized_manifest_bytes(path: Path) -> bytes:
    manifest = json.loads(path.read_text())
    metadata = manifest.get("metadata")
    if metadata is not None:
        metadata.pop("created_at", None)
    return json.dumps(manifest, indent=2).encode()
