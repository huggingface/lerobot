# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

import pytest

pytest.importorskip("transformers")

from lerobot.policies import factory
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

# Override keys of the custom LingBot steps. Written out (not derived) on purpose.
SLOT_MAPPING_STEP = "lingbot_vla_v2_slot_mapping"
INVERSE_SLOT_MAPPING_STEP = "lingbot_vla_v2_inverse_slot_mapping"


def test_saved_checkpoint_forwards_standard_and_config_overrides(monkeypatch):
    """The pipeline is standard now: normalizer overrides pass through, and the
    config-derived custom-step overrides are forwarded so fine-tuning on a new
    embodiment wins over the checkpoint's saved slot mapping."""
    loaded_calls = []

    class DummyPipeline:
        steps = []

    def fake_from_pretrained(**kwargs):
        loaded_calls.append((kwargs["config_filename"], kwargs["overrides"]))
        return DummyPipeline()

    monkeypatch.setattr(factory.PolicyProcessorPipeline, "from_pretrained", fake_from_pretrained)

    preprocessor, postprocessor = factory.make_pre_post_processors(
        LingbotVLAV2Config(),
        pretrained_path="/tmp/saved_lingbot_vla_v2_checkpoint",
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"stats": {}},
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={"unnormalizer_processor": {"stats": {}}},
    )

    assert isinstance(preprocessor, DummyPipeline)
    assert isinstance(postprocessor, DummyPipeline)
    assert [name for name, _ in loaded_calls] == [
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]
    pre_overrides = loaded_calls[0][1]
    post_overrides = loaded_calls[1][1]
    # The standard steps exist in the pipeline now, so their overrides pass through
    # (no more filtering of normalizer / unnormalizer keys).
    assert pre_overrides["normalizer_processor"] == {"stats": {}}
    assert post_overrides["unnormalizer_processor"] == {"stats": {}}
    # Generic overrides pass through untouched.
    assert pre_overrides["device_processor"] == {"device": "cuda"}
    assert pre_overrides["rename_observations_processor"] == {"rename_map": {}}
    # The postprocessor keeps its own device (CPU).
    assert "device_processor" not in post_overrides

    # Only the slot mapping is taken from the active config, on both pipelines.
    cfg = LingbotVLAV2Config()
    assert pre_overrides[SLOT_MAPPING_STEP] == {
        "state_spans": cfg.slot_spans("observation.state"),
        "action_spans": cfg.slot_spans("action"),
    }
    assert post_overrides[INVERSE_SLOT_MAPPING_STEP] == {"action_spans": cfg.slot_spans("action")}
    assert set(pre_overrides) == {
        "device_processor",
        "normalizer_processor",
        "rename_observations_processor",
        SLOT_MAPPING_STEP,
    }
