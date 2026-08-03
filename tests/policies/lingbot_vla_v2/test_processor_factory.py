# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from lerobot.policies import factory
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config


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
    assert loaded_calls == [
        (
            "policy_preprocessor.json",
            {
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        ),
        ("policy_postprocessor.json", {"device_processor": {"device": "cuda"}}),
    ]
