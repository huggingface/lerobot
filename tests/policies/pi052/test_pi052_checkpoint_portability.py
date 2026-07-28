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
import shutil
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.policies import make_pre_post_processors
from lerobot.processor import ActionTokenizerProcessorStep, DataProcessorPipeline, NormalizerProcessorStep
from lerobot.processor.converters import identity_transition
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.utils.constants import ACTION


class _ActionTokenizer:
    def __call__(self, actions):
        return np.asarray(actions).round().astype(np.int64)

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        (path / "processor_config.json").write_text('{"processor_class": "_ActionTokenizer"}\n')


class _PaligemmaTokenizer:
    vocab_size = 4096
    bos_token_id = 2

    def encode(self, text, **kwargs):
        return [10, 11] if text == "Action: " else [12]


def _make_pipeline(action_tokenizer_path):
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    stats = {ACTION: {"min": torch.tensor([-1.0, -2.0]), "max": torch.tensor([1.0, 2.0])}}
    normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats=stats,
    )
    action_tokenizer = ActionTokenizerProcessorStep(
        action_tokenizer_name=str(action_tokenizer_path),
        max_action_tokens=16,
        fast_skip_tokens=128,
    )
    return DataProcessorPipeline(
        [normalizer, RenderMessagesStep(recipe), action_tokenizer],
        name="policy_preprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    )


def test_pi052_pipeline_embeds_and_loads_fitted_action_tokenizer(tmp_path, monkeypatch):
    original_cache = tmp_path / "original_fast_cache"
    original_cache.mkdir()
    tokenizer = _ActionTokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoProcessor.from_pretrained",
        lambda path, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _PaligemmaTokenizer(),
    )
    monkeypatch.setattr(
        "lerobot.policies.pi052.fit_fast_tokenizer.fit_fast_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FAST fitting must not run")),
    )

    pipeline = _make_pipeline(original_cache)
    expected_tokens = pipeline.steps[-1]._tokenize_action(torch.tensor([[[0.2, 0.8]]]))[0]
    expected_recipe = asdict(pipeline.steps[1].recipe)
    expected_state = pipeline.steps[0].state_dict()
    checkpoint = tmp_path / "checkpoint"
    pipeline.save_pretrained(checkpoint)
    DataProcessorPipeline(
        [],
        name="policy_postprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    ).save_pretrained(checkpoint)

    saved_config = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    tokenizer_step = saved_config["steps"][2]
    assert tokenizer_step["config"]["action_tokenizer_name"] == "action_tokenizer"
    assert tokenizer_step["artifacts"] == {"action_tokenizer_name": "action_tokenizer"}
    assert (checkpoint / "action_tokenizer" / "processor_config.json").is_file()

    shutil.rmtree(original_cache)
    loaded, _ = make_pre_post_processors(
        SimpleNamespace(type="pi052", auto_fit_fast_tokenizer=True),
        pretrained_path=str(checkpoint),
        dataset_repo_id="org/dataset-that-must-not-be-read",
    )

    assert asdict(loaded.steps[1].recipe) == expected_recipe
    for key, tensor in expected_state.items():
        torch.testing.assert_close(loaded.steps[0].state_dict()[key], tensor)
    torch.testing.assert_close(
        loaded.steps[-1]._tokenize_action(torch.tensor([[[0.2, 0.8]]]))[0],
        expected_tokens,
    )


def test_pi052_pipeline_rejects_missing_fitted_action_tokenizer(tmp_path, monkeypatch):
    tokenizer = _ActionTokenizer()
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoProcessor.from_pretrained",
        lambda path, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _PaligemmaTokenizer(),
    )

    pipeline = _make_pipeline(tmp_path / "original_fast_cache")
    checkpoint = tmp_path / "checkpoint"
    pipeline.save_pretrained(checkpoint)
    shutil.rmtree(checkpoint / "action_tokenizer")

    with pytest.raises(FileNotFoundError, match="Checkpoint artifacts are incomplete"):
        DataProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_preprocessor.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )
