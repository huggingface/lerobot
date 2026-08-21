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

from lerobot.policies.smolvla.configuration_smolvla import (
    SUPPORTED_VLM_BACKBONES,
    SmolVLAConfig,
)


def test_default_backbone_is_supported():
    cfg = SmolVLAConfig()
    assert cfg.vlm_model_name in SUPPORTED_VLM_BACKBONES


def test_unsupported_vlm_backbone_warns(caplog):
    with caplog.at_level("WARNING"):
        SmolVLAConfig(vlm_model_name="google/paligemma-3b-pt-224")
    assert "not a validated SmolVLA backbone" in caplog.text


def test_supported_backbone_does_not_warn(caplog):
    with caplog.at_level("WARNING"):
        SmolVLAConfig(vlm_model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    assert "not a validated SmolVLA backbone" not in caplog.text
