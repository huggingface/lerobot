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
"""Legacy-checkpoint contracts.

Two contracts are pinned here so they are documented behavior, not accidents:

- **The v0.6.0 hard break.** The v0.6.0 #3810 FSDP checkpoint layout
  (full gathered ``model.safetensors`` + full ``optimizer_state.safetensors``, no DCP dirs,
  no ``checkpoint_format`` in ``train_config.json``) is a hard break with ZERO v0.6.0-aware
  runtime code — not even layout detection. A sharded resume pointed at such a checkpoint
  must fail through the ORDINARY missing-artifact path (torch DCP erroring on the absent
  ``training_state/optimizer_0/``), while the model weights remain loadable forever via
  ``from_pretrained`` and the old ``num_processes`` key keeps feeding the topology reader.
- **Converter equivalence.** ``dcp_to_safetensors`` (real ``merge_fsdp_weights``, no mocks)
  on accelerate's ``save_fsdp_model`` DCP layout reproduces exactly the tensors that the
  direct-gather ``save_pretrained`` artifact contains.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

import torch
import torch.distributed.checkpoint as dist_cp
from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME
from safetensors.torch import load_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.fsdp import FSDPModule

from lerobot.common.train_utils import (
    load_training_dp_world_size,
    resume_after_prepare,
    resume_before_prepare,
)
from lerobot.configs.accelerator import FSDPConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, CheckpointFormat, TrainPipelineConfig
from lerobot.distributed.checkpoint import dcp_to_safetensors, is_sharded_module
from lerobot.optim.optimizers import save_optimizer_state
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import save_rng_state
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointPolicy, make_dummy_policy


@pytest.fixture
def accelerate_state():
    """accelerate's process state, as the trainer's `Accelerator()` would have initialized it.

    `load_fsdp_optimizer` and `merge_fsdp_weights` both consult `PartialState` internals
    (logging and main-process gating). Single-process CPU state; reset on teardown so no
    global accelerate state leaks into other tests.
    """
    from accelerate.state import AcceleratorState, PartialState

    PartialState()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def make_v060_fsdp_checkpoint(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Reproduce the v0.6.0 #3810 FSDP checkpoint layout with real artifacts.

    - ``pretrained_model/``: ``config.json`` + full gathered ``model.safetensors`` (real
      ``save_pretrained`` outputs) and a ``train_config.json`` predating the v0.7 fields
      (``checkpoint_format``/``parallelism``/``accelerator`` stripped from the draccus dump);
    - ``training_state/``: old-style ``training_step.json`` (``{"step", "num_processes"}``,
      no ``dp_world_size``), ``rng_state.safetensors``, and the gathered full optimizer
      channel (``optimizer_state.safetensors`` + ``optimizer_param_groups.json``) — and,
      crucially, NO ``optimizer_0/`` DCP directory.

    Returns the saved model weights for later comparison.
    """
    policy = make_dummy_policy()
    optimizer = torch.optim.Adam(policy.parameters())
    policy.forward({"observation.state": torch.randn(2, 4)})[0].backward()
    optimizer.step()  # real optimizer state, applied before the weights are saved

    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), batch_size=3)
    cfg._save_pretrained(pretrained_dir)
    config_path = pretrained_dir / TRAIN_CONFIG_NAME
    raw = json.loads(config_path.read_text())
    assert "checkpoint_format" in raw  # draccus dumps defaults; a v0.6.0 config predates the key
    for key in ("checkpoint_format", "parallelism", "accelerator"):
        raw.pop(key, None)
    config_path.write_text(json.dumps(raw, indent=4))

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    training_state_dir.mkdir()
    write_json({"step": 5000, "num_processes": 4}, training_state_dir / TRAINING_STEP)
    save_rng_state(training_state_dir)
    save_optimizer_state(optimizer, training_state_dir)
    return {key: tensor.clone() for key, tensor in policy.state_dict().items()}


def as_fsdp2_module(policy: DummyCheckpointPolicy) -> DummyCheckpointPolicy:
    """Give the policy FSDP2's runtime identity via the in-place class swap `fully_shard` performs.

    torch's `fully_shard` swaps ``module.__class__`` to a ``(FSDPModule, type(module))``
    subclass; mirroring that swap is what makes `is_sharded_module` (and thus the sharded
    branch of `resume_after_prepare`) see a sharded model on a CPU-only single process. The
    parameters stay plain tensors — sufficient here, because the resume must fail at the DCP
    read before any sharded state is touched.
    """
    policy.__class__ = type(f"FSDP{type(policy).__name__}", (FSDPModule, type(policy)), {})
    assert is_sharded_module(policy)
    return policy


def sharded_passthrough_accelerator() -> SimpleNamespace:
    """The accelerator surface the sharded resume touches, carrying the trainer's real plugin.

    `FSDPConfig.build_plugin()` is the exact FSDP2 plugin construction `make_accelerator`
    hands to accelerate (state_dict_type stays at the FSDP2 default, SHARDED_STATE_DICT).
    """
    return SimpleNamespace(
        unwrap_model=lambda m: m,
        wait_for_everyone=lambda: None,
        state=SimpleNamespace(fsdp_plugin=FSDPConfig().build_plugin()),
    )


class TestV060HardBreak:
    """Pin the v0.6.0 hard break as a contract.

    Zero v0.6.0-aware code ships — not even layout detection — so every assertion here must
    hold through ORDINARY code paths only: the recorded config parses with plain defaults,
    phase-1 resume and the weights stay loadable, and the sharded phase-2 resume fails with
    torch DCP's own missing-artifact error, never a bespoke v0.6.0 message.
    """

    def test_sharded_resume_fails_with_ordinary_missing_artifact_error(self, tmp_path, accelerate_state):
        make_v060_fsdp_checkpoint(tmp_path)

        # No checkpoint_format recorded -> plain draccus default, no layout detection anywhere.
        cfg = TrainPipelineConfig.from_pretrained(tmp_path / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME)
        assert cfg.checkpoint_format is CheckpointFormat.SAFETENSORS
        cfg.checkpoint_path = tmp_path

        # Phase 1 (RNG + step counter) is format-independent and still succeeds.
        assert resume_before_prepare(cfg) == 5000

        # Phase 2 under sharding: the recorded format skips the DCP model preflight (the
        # weights were already loaded by from_pretrained), then the sharded optimizer load
        # hits the absent optimizer_0/ and fails inside torch DCP — the ordinary error path.
        assert not (tmp_path / TRAINING_STATE_DIR / f"{OPTIMIZER_NAME}_0").exists()
        policy = as_fsdp2_module(make_dummy_policy())
        optimizer = torch.optim.Adam(policy.parameters())
        with pytest.raises(CheckpointException) as excinfo:
            resume_after_prepare(cfg, sharded_passthrough_accelerator(), policy, optimizer, None)
        message = str(excinfo.value)
        assert "lerobot-convert-dcp" not in message  # the converter hint belongs to recorded-format=DCP
        assert "v0.6" not in message  # no bespoke wording: the explanation lives in the migration docs

    def test_weights_remain_loadable_via_from_pretrained(self, tmp_path):
        saved_weights = make_v060_fsdp_checkpoint(tmp_path)
        policy = DummyCheckpointPolicy.from_pretrained(tmp_path / PRETRAINED_MODEL_DIR)
        for key, tensor in policy.state_dict().items():
            assert torch.equal(tensor, saved_weights[key]), key

    def test_topology_reader_falls_back_to_legacy_num_processes(self, tmp_path):
        make_v060_fsdp_checkpoint(tmp_path)
        assert load_training_dp_world_size(tmp_path) == 4


class TestConverterEquivalence:
    def test_dcp_to_safetensors_output_equals_direct_gather(self, tmp_path, accelerate_state):
        """DCP -> safetensors conversion is exactly the direct-gather artifact.

        The DCP checkpoint is written with torch's real `dist_cp.save` (single process, no
        process group), replicating accelerate's `save_fsdp_model` SHARDED_STATE_DICT branch
        byte for byte: the ``{"model": state_dict}`` nesting and the ``pytorch_model_fsdp_0``
        directory name. The conversion runs the real `merge_fsdp_weights` — no mocks.
        """
        policy = make_dummy_policy()
        with torch.no_grad():
            for param in policy.parameters():
                param.add_(torch.randn_like(param))  # make every tensor distinct from init
        reference = {key: tensor.clone() for key, tensor in policy.state_dict().items()}

        # The direct-gather artifact (on a single process the gather is state_dict itself).
        direct_dir = tmp_path / "direct"
        policy.save_pretrained(direct_dir)

        # The DCP artifact, laid out exactly as accelerate's save_fsdp_model writes it.
        pretrained_dir = tmp_path / "checkpoint" / PRETRAINED_MODEL_DIR
        dcp_dir = pretrained_dir / f"{FSDP_MODEL_NAME}_0"
        dcp_dir.mkdir(parents=True)
        dist_cp.save(
            state_dict={"model": policy.state_dict()},
            storage_writer=dist_cp.FileSystemWriter(str(dcp_dir)),
        )

        merged_file = dcp_to_safetensors(dcp_dir, pretrained_dir)
        assert merged_file == pretrained_dir / "model.safetensors"
        merged = load_file(merged_file)
        direct = load_file(direct_dir / "model.safetensors")
        assert set(merged) == set(direct) == set(reference)
        for key, tensor in reference.items():
            assert torch.equal(merged[key], tensor), key
            assert torch.equal(direct[key], tensor), key
            assert merged[key].dtype == tensor.dtype, key
