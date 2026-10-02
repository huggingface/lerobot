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

import draccus
import pytest

from lerobot.configs.accelerator import (
    AcceleratorConfig,
    ActivationCheckpointingConfig,
    ActivationCheckpointingMode,
    CompileConfig,
    DDPConfig,
    FSDPConfig,
    GradientAccumulationConfig,
    GradScalerConfig,
)
from lerobot.configs.parallelism import ParallelismConfig


class TestFieldValidation:
    def test_wrap_policies_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            FSDPConfig(wrap_modules=["Block"], min_num_params=1000)

    def test_min_num_params_positive(self):
        with pytest.raises(ValueError, match="min_num_params"):
            FSDPConfig(min_num_params=0)

    def test_mixed_precision_choices(self):
        with pytest.raises(ValueError, match="mixed_precision"):
            AcceleratorConfig(mixed_precision="tf32")

    def test_gradient_accumulation_positive(self):
        with pytest.raises(ValueError, match="gradient_accumulation.steps"):
            GradientAccumulationConfig(steps=0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("init_scale", 0.0),
            ("growth_factor", 1.0),  # a scale that can never recover after a backoff
            ("backoff_factor", 1.0),  # an overflow that can never be escaped
            ("backoff_factor", 0.0),
            ("growth_interval", 0),
        ],
    )
    def test_grad_scaler_invariants(self, field, value):
        with pytest.raises(ValueError, match=f"grad_scaler.{field}"):
            GradScalerConfig(**{field: value})


class TestDraccusRoundTrip:
    @pytest.mark.parametrize(
        "cfg",
        [
            AcceleratorConfig(),
            AcceleratorConfig(
                mixed_precision="bf16",
                gradient_accumulation=GradientAccumulationConfig(steps=4),
                fsdp=FSDPConfig(
                    reshard_after_forward=False,
                    wrap_modules=["ACTEncoderLayer", "ACTDecoderLayer"],
                    cpu_offload=True,
                    ignored_modules=r".*pos_embed.*",
                ),
                ddp=DDPConfig(find_unused_parameters=False, static_graph=True),
                grad_scaler=GradScalerConfig(init_scale=1024.0, growth_interval=500),
                compile=CompileConfig(enabled=True, mode="max-autotune", regional=False),
                activation_checkpointing=ActivationCheckpointingConfig(mode=ActivationCheckpointingMode.FULL),
            ),
            AcceleratorConfig(fsdp=FSDPConfig(min_num_params=1_000_000)),
        ],
    )
    def test_encode_json_decode_identity(self, cfg):
        payload = json.loads(json.dumps(draccus.encode(cfg)))
        assert draccus.decode(AcceleratorConfig, payload) == cfg

    def test_pre_existing_config_without_fields_gets_defaults(self):
        assert draccus.decode(AcceleratorConfig, {}) == AcceleratorConfig()


class TestRuntimeBuilders:
    """The mirrors must translate into real accelerate objects (plugins built lazily)."""

    @pytest.fixture(autouse=True)
    def _requires_accelerate(self):
        pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

    def test_fsdp_plugin_translation(self):
        plugin = FSDPConfig(
            reshard_after_forward=False, wrap_modules=["MyBlock"], cpu_offload=True
        ).build_plugin()
        assert plugin.fsdp_version == 2
        assert plugin.reshard_after_forward is False
        assert plugin.transformer_cls_names_to_wrap == ["MyBlock"]
        # bools are normalized into torch offload policies by the plugin itself
        assert type(plugin.cpu_offload).__name__ == "CPUOffloadPolicy"
        # LeRobot never switches state_dict_type: FSDP2's SHARDED default must hold
        assert plugin.state_dict_type.name == "SHARDED_STATE_DICT"
        assert not plugin.activation_checkpointing

    def test_fsdp_plugin_size_based_policy(self):
        plugin = FSDPConfig(min_num_params=1024).build_plugin()
        assert plugin.min_num_params == 1024
        assert plugin.transformer_cls_names_to_wrap is None

    def test_ddp_kwargs_translation(self):
        handler = DDPConfig(find_unused_parameters=False, gradient_as_bucket_view=True).build_kwargs_handler()
        assert handler.find_unused_parameters is False
        assert handler.gradient_as_bucket_view is True

    def test_gradient_accumulation_plugin_translation(self):
        plugin = GradientAccumulationConfig(steps=4).build_plugin()
        assert plugin.num_steps == 4
        assert plugin.sync_with_dataloader is False

    def test_grad_scaler_kwargs_translation(self):
        handler = GradScalerConfig(
            init_scale=1024.0, growth_factor=4.0, backoff_factor=0.25, growth_interval=500
        ).build_kwargs_handler()
        assert handler.init_scale == 1024.0
        assert handler.growth_factor == 4.0
        assert handler.backoff_factor == 0.25
        assert handler.growth_interval == 500
        # `enabled` is not mirrored: fp16 without loss scaling is never a valid configuration.
        assert handler.enabled is True

    @pytest.mark.parametrize(
        ("topology", "world_size", "expects_ddp_handler"),
        [
            ({}, 1, False),  # single process
            ({"dp_replicate": 4}, 4, True),  # DDP
            ({"dp_shard": 4}, 4, False),  # FSDP2
            ({"dp_replicate": 2, "dp_shard": 2}, 4, False),  # HSDP
        ],
    )
    @pytest.mark.parametrize("mixed_precision", ["no", "bf16", "fp16"])
    def test_scaler_handler_is_passed_for_fp16_on_every_topology(
        self, monkeypatch, topology, world_size, expects_ddp_handler, mixed_precision
    ):
        """The scaler handler is orthogonal to the topology matrix.

        accelerate reads it only on the branch that builds the fp16 `GradScaler`, so appending
        it anywhere else would be inert — but it is still left out, to keep a "no"/bf16
        `Accelerator` byte-identical to one built without any scaler support.
        """
        from accelerate.utils import DistributedDataParallelKwargs, GradScalerKwargs

        captured = {}

        class FakeAccelerator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("accelerate.Accelerator", FakeAccelerator)
        parallelism = ParallelismConfig(**topology)
        parallelism.resolve(world_size)
        AcceleratorConfig(mixed_precision=mixed_precision).build(parallelism, cpu=True)

        handlers = captured.get("kwargs_handlers", [])
        scalers = [h for h in handlers if isinstance(h, GradScalerKwargs)]
        ddps = [h for h in handlers if isinstance(h, DistributedDataParallelKwargs)]
        assert len(scalers) == (1 if mixed_precision == "fp16" else 0)
        assert len(ddps) == (1 if expects_ddp_handler else 0)

    def test_gradient_accumulation_never_syncs_with_dataloader(self, monkeypatch):
        """The loop cycles a finite dataloader, so accelerate's default
        sync_with_dataloader=True would force an optimizer step at every dataset epoch
        boundary instead of every num_steps micro-batches."""
        captured = {}

        class FakeAccelerator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("accelerate.Accelerator", FakeAccelerator)
        parallelism = ParallelismConfig()
        parallelism.resolve(1)
        AcceleratorConfig(gradient_accumulation=GradientAccumulationConfig(steps=4)).build(
            parallelism, cpu=True
        )
        ga_plugin = captured["gradient_accumulation_plugin"]
        assert ga_plugin.num_steps == 4
        assert ga_plugin.sync_with_dataloader is False
        assert "gradient_accumulation_steps" not in captured
