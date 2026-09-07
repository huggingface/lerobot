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
"""The DCP wrappers must hand accelerate exact shard directories.

accelerate 1.14 resolves the load directory with a substring check ("optimizer" /
"pytorch_model_fsdp" in the path -> use as-is) while the save side joins the shard name
unconditionally, so a run path like `--job_name=optimizer_sweep` would save to
`training_state/optimizer_0/` but load from `training_state/` itself. Passing the exact
shard dir makes the containment check deterministically a no-op.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")


def fake_accelerator() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(fsdp_plugin=object()))


# A parent path that trips both of accelerate's substring checks at once.
POISONED_PARENT = Path("/outputs/train/optimizer_sweep_pytorch_model_fsdp_repro/training_state")


def test_load_sharded_optimizer_passes_exact_shard_dir(monkeypatch):
    import accelerate.utils

    from lerobot.distributed.checkpoint import load_sharded_optimizer

    seen = {}
    monkeypatch.setattr(
        accelerate.utils,
        "load_fsdp_optimizer",
        lambda plugin, accelerator, optimizer, model, input_dir: seen.update(path=input_dir),
    )
    load_sharded_optimizer(fake_accelerator(), optimizer=object(), model=object(), input_dir=POISONED_PARENT)
    assert seen["path"] == str(POISONED_PARENT / "optimizer_0")
    assert isinstance(seen["path"], str)  # str, never Path (accelerate does string checks)


def test_load_sharded_model_passes_exact_shard_dir(monkeypatch):
    import accelerate.utils

    from lerobot.distributed.checkpoint import load_sharded_model

    seen = {}
    monkeypatch.setattr(
        accelerate.utils,
        "load_fsdp_model",
        lambda plugin, accelerator, model, input_dir: seen.update(path=input_dir),
    )
    load_sharded_model(fake_accelerator(), model=object(), input_dir=POISONED_PARENT)
    assert seen["path"] == str(POISONED_PARENT / "pytorch_model_fsdp_0")
    assert isinstance(seen["path"], str)
