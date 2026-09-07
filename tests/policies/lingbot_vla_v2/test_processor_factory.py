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

from lerobot.policies import factory
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config

# Override key for the LingBot feature-transform step. Written out (not derived) on
# purpose: a typo here fails with a confusing KeyError, see the assertion below.
FEATURE_TRANSFORM_STEP = "lingbot_vla_v2_feature_transform"


def test_saved_checkpoint_filters_normalizer_overrides(monkeypatch):
    """The LingBot pipeline has no LeRobot normalizer steps; generic normalizer
    overrides from the training script must be dropped before the pipeline loader
    (which rejects override keys matching no step) sees them."""
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
    # Normalizer overrides must be filtered out (the point of this test): the LingBot
    # pipeline has no LeRobot normalizer / unnormalizer steps.
    assert "normalizer_processor" not in pre_overrides
    assert "unnormalizer_processor" not in post_overrides
    # Generic overrides pass through untouched.
    assert pre_overrides["device_processor"] == {"device": "cuda"}
    assert pre_overrides["rename_observations_processor"] == {"rename_map": {}}
    assert post_overrides == {"device_processor": {"device": "cuda"}}
    # The config-derived feature-transform overrides are forwarded so fine-tuning on a
    # new embodiment wins over the checkpoint's saved slot mapping (same rule as
    # ``resolve_robot_config_and_stats``; see
    # ``make_lingbot_vla_v2_pre_post_processors_from_pretrained``).
    assert FEATURE_TRANSFORM_STEP in pre_overrides, (
        f"expected the config-derived step overrides under {FEATURE_TRANSFORM_STEP!r}, "
        f"got {sorted(pre_overrides)}"
    )
    cfg = LingbotVLAV2Config()
    ft_overrides = pre_overrides[FEATURE_TRANSFORM_STEP]
    assert ft_overrides["chunk_size"] == cfg.chunk_size
    assert ft_overrides["cameras"] == cfg.canonical_cameras
    assert ft_overrides["processor_path"] == (cfg.processor_path or cfg.tokenizer_path)
