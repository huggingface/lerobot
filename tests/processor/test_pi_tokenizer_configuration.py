#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.processor import TokenizerProcessorStep  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

TOKENIZER_NAME = "google/paligemma-3b-pt-224"


@pytest.fixture
def local_tokenizer(tmp_path: Path) -> tuple[Path, int]:
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "pick": 2, "the": 3, "cube": 4}
    tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer_path = tmp_path / "tokenizer"
    fast_tokenizer.save_pretrained(tokenizer_path)
    return tokenizer_path, len(vocabulary)


@pytest.mark.parametrize("config_class", [PI0Config, PI05Config])
def test_pi_text_tokenizer_default(config_class):
    assert config_class().text_tokenizer_name == TOKENIZER_NAME


@pytest.mark.parametrize(
    ("config_class", "processor_factory"),
    [
        (PI0Config, make_pi0_pre_post_processors),
        (PI05Config, make_pi05_pre_post_processors),
    ],
)
def test_pi_processor_uses_local_tokenizer_and_config_round_trips(
    config_class,
    processor_factory,
    local_tokenizer,
    tmp_path,
    monkeypatch,
):
    tokenizer_path, vocabulary_size = local_tokenizer
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = config_class(
        text_tokenizer_name=str(tokenizer_path),
        tokenizer_max_length=8,
    )

    preprocessor, _ = processor_factory(config)
    tokenizer_step = next(step for step in preprocessor.steps if isinstance(step, TokenizerProcessorStep))
    transition = tokenizer_step(
        {
            TransitionKey.OBSERVATION: {},
            TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick the cube"]},
        }
    )

    token_ids = transition[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS]
    attention_mask = transition[TransitionKey.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK]
    assert tokenizer_step.tokenizer_name == str(tokenizer_path)
    assert token_ids.shape == attention_mask.shape == (1, config.tokenizer_max_length)
    assert token_ids.max().item() < vocabulary_size

    config_path = tmp_path / "config"
    config.save_pretrained(config_path)
    reloaded_config = PreTrainedConfig.from_pretrained(config_path)
    assert isinstance(reloaded_config, config_class)
    assert reloaded_config.text_tokenizer_name == str(tokenizer_path)

    reloaded_preprocessor, _ = processor_factory(reloaded_config)
    reloaded_tokenizer_step = next(
        step for step in reloaded_preprocessor.steps if isinstance(step, TokenizerProcessorStep)
    )
    assert reloaded_tokenizer_step.tokenizer_name == str(tokenizer_path)
