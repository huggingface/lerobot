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

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")


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


def test_fp16_scaler_seams():
    """The fp16 path: a plain GradScaler for FSDP2, plus the skip flag the loop reads."""
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer
    from accelerate.utils import GradScalerKwargs
    from accelerate.utils.fsdp_utils import get_fsdp2_grad_scaler

    # FSDP1's ShardedGradScaler would be wrong for FSDP2 (DTensor gradients unscale natively);
    # accelerate routes around it through this helper, which LeRobot relies on.
    assert callable(get_fsdp2_grad_scaler)
    # The mirrored subset of GradScalerConfig.
    fields = set(GradScalerKwargs.__dataclass_fields__)
    assert {"init_scale", "growth_factor", "backoff_factor", "growth_interval"} <= fields
    # `update_policy` gates the policy's update()/EMA on these.
    assert isinstance(Accelerator.optimizer_step_was_skipped, property)
    assert isinstance(AcceleratedOptimizer.step_was_skipped, property)


def test_dtensor_reduces_the_overflow_flag():
    """What makes an overflow on one rank skip the update on all of them.

    `GradScaler.unscale_` inspects gradients through
    `aten._amp_foreach_non_finite_check_and_unscale_`; DTensor dispatches that op to a handler
    that reduces the found-inf flag across the mesh. Without it, ranks would disagree about
    whether to step and the sharded run would silently diverge.
    """
    import torch
    from torch.distributed.tensor import DTensor

    handlers = DTensor._op_dispatcher._custom_op_handlers
    assert torch.ops.aten._amp_foreach_non_finite_check_and_unscale_.default in handlers


def test_dcp_optimizer_init_still_routes_through_the_scaler():
    """Why `lerobot.distributed.checkpoint` hands DCP the *unwrapped* optimizer.

    torch's optimizer DCP APIs call `_init_optim_state`, which materializes an empty optimizer
    state with one zero-gradient `optimizer.step()`. Through an `AcceleratedOptimizer` carrying
    an fp16 scaler, that dummy step enters `GradScaler.step` — which asserts while the scaler
    is still lazy, and advances the scale once it is not. If this canary stops failing, torch
    or accelerate has fixed the seam and the unwrap can be reconsidered.
    """
    import torch
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer
    from accelerate.state import AcceleratorState
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    Accelerator(cpu=True)  # AcceleratedOptimizer reads the global AcceleratorState
    try:

        def case():
            model = torch.nn.Linear(2, 2)
            scaler = torch.amp.GradScaler("cpu", init_scale=32.0, growth_interval=1)
            return model, AcceleratedOptimizer(torch.optim.AdamW(model.parameters()), scaler=scaler), scaler

        model, wrapped, scaler = case()
        with pytest.raises(AssertionError, match="_scale is None"):
            get_optimizer_state_dict(model, wrapped)

        # Pre-initializing the scaler (what accelerate's own load_state does) only trades the
        # assert for silent corruption: the dummy step is scaler-stepped and scaler-updated.
        model, wrapped, scaler = case()
        scaler.scale(torch.zeros(()))
        before = scaler.state_dict()
        get_optimizer_state_dict(model, wrapped)
        assert scaler.state_dict() != before

        # The unwrapped optimizer leaves the scaler untouched — LeRobot's fix.
        model, wrapped, scaler = case()
        scaler.scale(torch.zeros(()))
        before = scaler.state_dict()
        state = get_optimizer_state_dict(model, wrapped.optimizer)
        assert scaler.state_dict() == before
        assert state["state"]  # the optimizer state really was materialized
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_accelerate_version_floor():
    import accelerate
    from packaging import version

    if version.parse(accelerate.__version__) < version.parse("1.14.0"):
        pytest.fail(
            f"accelerate {accelerate.__version__} < 1.14.0: the FSDP2 auto-wrap fallback fix "
            "(#3999) and the bf16->fp32 master-weight upcast this design relies on are absent."
        )
