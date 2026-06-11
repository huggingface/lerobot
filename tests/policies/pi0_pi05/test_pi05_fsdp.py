#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for Pi0.5 FSDP compatibility (JointDecoderLayerPair wrapping)."""

import re
from functools import partial

import pytest
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

pytest.importorskip("transformers")

from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    JointDecoderLayerPair,
    PI05Policy,
    PI05Pytorch,
    _JointLayerView,
)


@pytest.fixture
def model():
    config = PI05Config(dtype="float32")
    return PI05Pytorch(config)


class TestJointLayerStructure:
    def test_joint_layers_created(self, model):
        pwe = model.paligemma_with_expert
        assert hasattr(pwe, "joint_layers")
        assert len(pwe.joint_layers) == 18

    def test_joint_layers_type(self, model):
        pwe = model.paligemma_with_expert
        for pair in pwe.joint_layers:
            assert isinstance(pair, JointDecoderLayerPair)
            assert hasattr(pair, "paligemma_layer")
            assert hasattr(pair, "expert_layer")

    def test_layer_views_proxy_correctly(self, model):
        pwe = model.paligemma_with_expert
        pali_layers = pwe.paligemma.model.language_model.layers
        expert_layers = pwe.gemma_expert.model.layers

        assert isinstance(pali_layers, _JointLayerView)
        assert isinstance(expert_layers, _JointLayerView)
        assert len(pali_layers) == 18
        assert len(expert_layers) == 18

        # Views should return the correct sub-layers
        assert pali_layers[0] is pwe.joint_layers[0].paligemma_layer
        assert expert_layers[0] is pwe.joint_layers[0].expert_layer

    def test_no_duplicate_params(self, model):
        param_ids = set()
        duplicate_count = 0
        for p in model.parameters():
            if id(p) in param_ids:
                duplicate_count += 1
            param_ids.add(id(p))
        assert duplicate_count == 0

    def test_state_dict_keys_under_joint_layers(self, model):
        sd = model.state_dict()
        layer_keys = [k for k in sd if "joint_layers" in k]
        old_keys = [k for k in sd if "language_model.layers." in k or "gemma_expert.model.layers." in k]
        assert len(layer_keys) == 360  # 18 pairs x 20 params each
        assert len(old_keys) == 0


class TestFSDPWrapPolicy:
    def test_auto_wrap_identifies_pairs(self, model):
        # Verify the policy can be constructed (used by accelerate FSDP config)
        _ = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={JointDecoderLayerPair},
        )
        wrapped = [name for name, m in model.named_modules() if isinstance(m, JointDecoderLayerPair)]
        assert len(wrapped) == 18
        assert wrapped[0] == "paligemma_with_expert.joint_layers.0"
        assert wrapped[-1] == "paligemma_with_expert.joint_layers.17"

    def test_params_per_pair(self, model):
        pair = model.paligemma_with_expert.joint_layers[0]
        params = sum(p.numel() for p in pair.parameters())
        # Each pair should have ~134M params (2B/18 + 300M/18)
        assert params > 100_000_000
        assert params < 300_000_000


class TestForwardPass:
    def test_training_forward(self, model):
        model.eval()
        batch_size = 1
        loss = model(
            images=[torch.randn(batch_size, 3, 224, 224)],
            img_masks=[torch.ones(batch_size, dtype=torch.bool)],
            tokens=torch.randint(0, 1000, (batch_size, 10)),
            masks=torch.ones(batch_size, 10, dtype=torch.bool),
            actions=torch.randn(batch_size, 50, 32),
            noise=torch.randn(batch_size, 50, 32),
            time=torch.rand(batch_size),
        )
        assert loss.shape == (1, 50, 32)

    def test_inference_forward(self, model):
        model.eval()
        batch_size = 1
        with torch.no_grad():
            actions = model.sample_actions(
                images=[torch.randn(batch_size, 3, 224, 224)],
                img_masks=[torch.ones(batch_size, dtype=torch.bool)],
                tokens=torch.randint(0, 1000, (batch_size, 10)),
                masks=torch.ones(batch_size, 10, dtype=torch.bool),
                num_steps=2,
            )
        assert actions.shape == (1, 50, 32)


class TestCheckpointCompatibility:
    def test_old_format_keys_remap(self, model):
        """Verify that old-format checkpoint keys (pre-joint_layers) load correctly."""
        config = PI05Config(dtype="float32")
        sd = model.state_dict()

        # Convert to old format
        old_sd = {}
        for key, value in sd.items():
            new_key = key
            m = re.match(r"paligemma_with_expert\.joint_layers\.(\d+)\.paligemma_layer\.(.*)", key)
            if m:
                new_key = (
                    f"paligemma_with_expert.paligemma.model.language_model.layers.{m.group(1)}.{m.group(2)}"
                )
            m = re.match(r"paligemma_with_expert\.joint_layers\.(\d+)\.expert_layer\.(.*)", key)
            if m:
                new_key = f"paligemma_with_expert.gemma_expert.model.layers.{m.group(1)}.{m.group(2)}"
            old_sd[new_key] = value

        # Remap through the policy's key fixing logic
        policy = PI05Policy(config)
        fixed = policy._fix_pytorch_state_dict_keys(old_sd, config)

        # Add model. prefix
        remapped = {}
        for key, value in fixed.items():
            if not key.startswith("model."):
                remapped[f"model.{key}"] = value
            else:
                remapped[key] = value

        missing, unexpected = policy.load_state_dict(remapped, strict=True)
        assert len(missing) == 0, f"Missing keys: {missing[:5]}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected[:5]}"
