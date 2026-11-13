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

"""Processor step settings must survive `save_pretrained` / `from_pretrained`.

A step whose `get_config()` leaves a field out is rebuilt with the class default on load, so a
checkpoint silently runs with different settings than it was trained with.
"""

from pathlib import Path

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi0_fast.processor_pi0_fast import Pi0FastPrepareStateAndLanguageTokenizerProcessorStep
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import DataProcessorPipeline
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize(
    ("step_cls", "non_defaults"),
    [
        (MapTensorToDeltaActionDictStep, {"use_gripper": False}),
        (MapDeltaActionToRobotActionStep, {"position_scale": 0.5, "noise_threshold": 0.01}),
        (
            Pi05PrepareStateTokenizerProcessorStep,
            {"max_state_dim": 14, "task_key": "instruction", "include_state_in_prompt": False},
        ),
        (
            Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
            {"max_state_dim": 14, "task_key": "instruction"},
        ),
    ],
)
def test_step_settings_survive_save_and_load(tmp_path, step_cls, non_defaults):
    pipeline = DataProcessorPipeline(steps=[step_cls(**non_defaults)], name="pipeline")
    pipeline.save_pretrained(tmp_path)

    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path, config_filename="pipeline.json", local_files_only=True
    )

    step = loaded.steps[0]
    assert {key: getattr(step, key) for key in non_defaults} == non_defaults


@pytest.mark.parametrize(
    "step_cls",
    [
        MapTensorToDeltaActionDictStep,
        MapDeltaActionToRobotActionStep,
        Pi05PrepareStateTokenizerProcessorStep,
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
    ],
)
def test_step_saved_without_config_loads_defaults(step_cls):
    """Pipelines saved before these steps serialized their settings still load, with the defaults."""
    registry_name = step_cls._registry_name
    loaded = DataProcessorPipeline.from_config({"steps": [{"registry_name": registry_name, "config": {}}]})

    assert loaded.steps[0] == step_cls()


def _local_tokenizer(tmp_path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[PAD]": 1, "pick": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        path
    )
    return path


def test_pi05_proprioceptive_memory_checkpoint_keeps_state_out_of_prompt(tmp_path, monkeypatch):
    """A PI0.5 checkpoint trained with `use_proprioceptive_memory=True` reloads without the state in the prompt."""
    pytest.importorskip("transformers")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = PI05Config(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        text_tokenizer_name=str(_local_tokenizer(tmp_path)),
        use_proprioceptive_memory=True,
        device="cpu",
    )
    stats = {
        key: {"mean": torch.zeros(6), "std": torch.ones(6), "q01": -torch.ones(6), "q99": torch.ones(6)}
        for key in (OBS_STATE, ACTION)
    }
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, dataset_stats=stats)
    checkpoint = tmp_path / "checkpoint"
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)

    reloaded, _ = make_pre_post_processors(config, pretrained_path=str(checkpoint))

    step = next(s for s in reloaded.steps if isinstance(s, Pi05PrepareStateTokenizerProcessorStep))
    assert step.include_state_in_prompt is False
