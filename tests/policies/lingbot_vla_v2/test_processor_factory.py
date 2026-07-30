# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from lerobot.policies import factory
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import (
    make_lingbot_vla_v2_pre_post_processors_from_pretrained,
)


def test_raw_upstream_checkpoint_builds_fresh_processors(monkeypatch):
    calls = []

    def fake_make_processors(config, dataset_stats=None):
        calls.append((config, dataset_stats))
        return object(), object()

    monkeypatch.setattr(
        ("lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2.make_lingbot_vla_v2_pre_post_processors"),
        fake_make_processors,
    )
    config = LingbotVLAV2Config(robot_config_path="robot.yaml", norm_stats_path="stats.json")

    preprocessor, postprocessor = make_lingbot_vla_v2_pre_post_processors_from_pretrained(
        config=config,
        pretrained_path="robbyant/lingbot-vla-v2-6b",
        dataset_stats={"observation.state": {}},
    )

    assert preprocessor is not None
    assert postprocessor is not None
    assert calls == [(config, {"observation.state": {}})]


def test_saved_checkpoint_imports_processor_module_before_pipeline_load(monkeypatch):
    imported_modules = []
    loaded_configs = []

    real_import_module = factory.importlib.import_module

    def import_module_spy(module_path):
        imported_modules.append(module_path)
        return real_import_module(module_path)

    class DummyPipeline:
        steps = []

    def fake_from_pretrained(**kwargs):
        loaded_configs.append(kwargs["config_filename"])
        return DummyPipeline()

    monkeypatch.setattr(factory.importlib, "import_module", import_module_spy)
    monkeypatch.setattr(factory.PolicyProcessorPipeline, "from_pretrained", fake_from_pretrained)

    preprocessor, postprocessor = factory.make_pre_post_processors(
        LingbotVLAV2Config(),
        pretrained_path="/tmp/saved_lingbot_vla_v2_checkpoint",
    )

    assert isinstance(preprocessor, DummyPipeline)
    assert isinstance(postprocessor, DummyPipeline)
    assert "lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2" in imported_modules
    assert loaded_configs == ["policy_preprocessor.json", "policy_postprocessor.json"]
