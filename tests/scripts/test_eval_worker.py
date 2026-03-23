#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path

from lerobot.scripts import lerobot_eval_worker


def test_worker_main_writes_results(monkeypatch, tmp_path: Path):
    cfg = lerobot_eval_worker.EvalWorkerConfig(
        env="pusht",  # type: ignore[arg-type]
        instance_id=3,
        output_path=tmp_path / "worker.json",
    )

    monkeypatch.setattr(lerobot_eval_worker, "run_worker", lambda _cfg: {"per_task": []})

    lerobot_eval_worker.worker_main(cfg)

    assert cfg.output_path.exists()
    assert cfg.output_path.read_text().strip() == '{\n  "per_task": []\n}'
