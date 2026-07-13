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
"""Version canaries for the accelerate/torch seams LeRobot's distributed engine relies on.

LeRobot deliberately builds on a few accelerate internals that are not covered by a public
stability promise. These tests exist to fail LOUDLY on a
dependency upgrade — on a CPU runner, before any distributed job can be corrupted — whenever one
of those seams moves. If a canary fails, re-audit the corresponding integration seam before bumping
the pin; do not simply update the assertion.
"""

import inspect

import pytest


def test_fsdp_checkpoint_name_constants():
    """Checkpoint dir names are imported from accelerate; the on-disk layout depends on them."""
    from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME

    assert FSDP_MODEL_NAME == "pytorch_model_fsdp"
    assert OPTIMIZER_NAME == "optimizer"


def test_parallelism_config_mesh_dim_contract():
    """FSDP2 shards over the flattened dp_shard_cp dim; the dataloader keys on exact root names."""
    from accelerate.parallelism_config import ParallelismConfig

    pc = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, cp_size=2)
    assert pc.fsdp_dim_names == ["dp_replicate", "dp_shard_cp"]
    assert pc.dp_shard_cp_dim_names == ["dp_shard", "cp"]
    assert pc.dp_cp_dim_names == ["dp_replicate", "dp_shard", "cp"]
    # Degenerate FSDP-only case still shards over the flattened name.
    pc_fsdp = ParallelismConfig(dp_replicate_size=1, dp_shard_size=4)
    assert pc_fsdp.fsdp_dim_names == ["dp_shard_cp"]


def test_accelerator_accepts_parallelism_config():
    from accelerate import Accelerator

    params = inspect.signature(Accelerator.__init__).parameters
    assert "parallelism_config" in params
    assert "fsdp_plugin" in params
    assert "gradient_accumulation_plugin" in params


def test_dataloader_is_mesh_aware():
    """prepare_data_loader must accept the device mesh that makes CP peers share batches."""
    from accelerate.data_loader import prepare_data_loader

    assert "torch_device_mesh" in inspect.signature(prepare_data_loader).parameters


def test_cp_mask_stripping_hook_seam():
    """finalize_sharded_policy strips this exact hook.

    If accelerate renames or moves it, the strip becomes a silent no-op and CP training would
    inherit mask-corrupting hooks — hence a canary rather than a runtime hasattr.
    """
    from accelerate.big_modeling import _attach_context_parallel_hooks

    assert callable(_attach_context_parallel_hooks)
    assert _attach_context_parallel_hooks.__module__ == "accelerate.big_modeling"


def test_fsdp_plugin_mirrored_fields_exist():
    """AcceleratorConfig mirrors a plain-typed subset of the plugin; the fields must survive."""
    from accelerate.utils import FullyShardedDataParallelPlugin

    fields = {f.name for f in FullyShardedDataParallelPlugin.__dataclass_fields__.values()}
    assert {
        "fsdp_version",
        "reshard_after_forward",
        "auto_wrap_policy",
        "transformer_cls_names_to_wrap",
        "min_num_params",
        "cpu_offload",
        "ignored_modules",
        "activation_checkpointing",
        "state_dict_type",
    } <= fields


def test_merge_fsdp_weights_signature():
    """The DCP->safetensors converter is a thin wrapper over this accelerate utility."""
    from accelerate.utils import merge_fsdp_weights

    params = inspect.signature(merge_fsdp_weights).parameters
    assert {"checkpoint_dir", "output_path", "safe_serialization"} <= set(params)


def test_fsdp_save_load_helpers_exist():
    from accelerate.utils import (
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    for fn in (save_fsdp_model, load_fsdp_model, save_fsdp_optimizer, load_fsdp_optimizer):
        assert callable(fn)


def test_torch_fsdp2_seams():
    """isinstance(FSDPModule) discrimination + non-forward entry registration + full gather."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,  # noqa: F401
    )
    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method  # noqa: F401

    options = inspect.signature(StateDictOptions).parameters
    assert {"full_state_dict", "cpu_offload"} <= set(options)


def test_accelerate_version_floor():
    import accelerate
    from packaging import version

    if version.parse(accelerate.__version__) < version.parse("1.14.0"):
        pytest.fail(
            f"accelerate {accelerate.__version__} < 1.14.0: the FSDP2 auto-wrap fallback fix "
            "(#3999) and the bf16->fp32 master-weight upcast this design relies on are absent."
        )
