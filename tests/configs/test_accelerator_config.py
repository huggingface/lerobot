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
