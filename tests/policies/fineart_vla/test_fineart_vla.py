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

"""FineART-VLA: registration, config/processor round-trips, and forward parity with PI0.5."""

import json
import shutil
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.policies import (  # noqa: E402
    FineARTVLAConfig,
    get_policy_class,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from tests.utils import require_cuda  # noqa: E402


@dataclass
class _CLIConfig:
    policy: PreTrainedConfig


def test_policy_is_registered_under_its_canonical_name():
    config = make_policy_config("fineart_vla", device="cpu", enable_fast_action_loss=False)
    parsed = draccus.parse(_CLIConfig, args=["--policy.type=fineart_vla", "--policy.device=cpu"]).policy
    assert type(config) is type(parsed) is FineARTVLAConfig
    assert config.type == parsed.type == "fineart_vla"

    policy_class = get_policy_class("fineart_vla")
    assert policy_class.__name__ == "FineARTVLAPolicy"
    assert policy_class.config_class is FineARTVLAConfig
    assert "fineart_vla" in PreTrainedConfig.get_known_choices()


def test_config_save_load_roundtrip(tmp_path):
    config = FineARTVLAConfig(device="cpu", flow_num_repeats=3, text_loss_weight=0.5)
    config.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["type"] == "fineart_vla"

    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert type(restored) is FineARTVLAConfig
    assert restored.recipe == config.recipe
    assert restored.flow_num_repeats == 3
    assert restored.text_loss_weight == 0.5


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


def test_processor_save_load_roundtrip_embeds_action_tokenizer(tmp_path, monkeypatch):
    pytest.importorskip("datasets", reason="recipes require lerobot[dataset]")
    pytest.importorskip("av", reason="recipes require lerobot[dataset]")
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe
    from lerobot.processor import ActionTokenizerProcessorStep, DataProcessorPipeline, NormalizerProcessorStep
    from lerobot.processor.converters import identity_transition
    from lerobot.processor.render_messages_processor import RenderMessagesStep

    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoProcessor.from_pretrained",
        lambda path, **kwargs: _ActionTokenizer(),
    )
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _PaligemmaTokenizer(),
    )

    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="high_level", target=True),
        ]
    )
    normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats={ACTION: {"min": torch.tensor([-1.0, -2.0]), "max": torch.tensor([1.0, 2.0])}},
    )
    source_tokenizer = tmp_path / "fast_tokenizer"
    source_tokenizer.mkdir()
    action_tokenizer = ActionTokenizerProcessorStep(
        action_tokenizer_name=str(source_tokenizer), max_action_tokens=16, fast_skip_tokens=128
    )
    pipeline = DataProcessorPipeline(
        [normalizer, RenderMessagesStep(recipe), action_tokenizer],
        name="policy_preprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    action = torch.tensor([[[0.2, 0.8]]])
    expected_tokens = pipeline.steps[-1]._tokenize_action(action)[0]

    checkpoint = tmp_path / "checkpoint"
    pipeline.save_pretrained(checkpoint)
    DataProcessorPipeline(
        [], name="policy_postprocessor", to_transition=identity_transition, to_output=identity_transition
    ).save_pretrained(checkpoint)
    assert (checkpoint / "action_tokenizer" / "processor_config.json").is_file()

    # The checkpoint must be self-contained: the original tokenizer directory is gone.
    shutil.rmtree(source_tokenizer)
    loaded, _ = make_pre_post_processors(
        SimpleNamespace(type="fineart_vla", auto_fit_fast_tokenizer=False), pretrained_path=str(checkpoint)
    )
    assert asdict(loaded.steps[1].recipe) == asdict(recipe)
    torch.testing.assert_close(loaded.steps[0].state_dict()["action.min"], torch.tensor([-1.0, -2.0]))
    torch.testing.assert_close(loaded.steps[-1]._tokenize_action(action)[0], expected_tokens)


def _parity_features():
    return {
        "input_features": {
            "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    }


@require_cuda
def test_forward_matches_pi05_when_language_losses_are_off():
    """With text CE, FAST and knowledge insulation off, FineART-VLA must reduce to PI0.5."""
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    common = {"device": "cuda", "dtype": "float32", "chunk_size": 10, "n_action_steps": 10}
    pi05 = PI05Policy(PI05Config(**common, **_parity_features())).cuda()
    fineart = FineARTVLAPolicy(
        FineARTVLAConfig(
            **common,
            **_parity_features(),
            text_loss_weight=0.0,
            enable_fast_action_loss=False,
            knowledge_insulation=False,
            use_liger_kernels=False,
        )
    ).cuda()
    fineart.load_state_dict(pi05.state_dict(), strict=True)
    pi05.eval()
    fineart.eval()

    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, 16
    batch = {
        "observation.images.base": torch.rand(batch_size, 3, 224, 224, generator=generator),
        "observation.state": torch.randn(batch_size, 8, generator=generator),
        ACTION: torch.randn(batch_size, 10, 8, generator=generator),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (batch_size, seq_len), generator=generator),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, seq_len, dtype=torch.bool),
    }
    batch = {key: value.cuda() for key, value in batch.items()}

    with torch.no_grad():
        torch.manual_seed(1)
        pi05_loss, _ = pi05.forward(batch)
        torch.manual_seed(1)
        fineart_loss, _ = fineart.forward(batch)
        torch.testing.assert_close(fineart_loss, pi05_loss, rtol=1e-4, atol=1e-5)

        torch.manual_seed(2)
        pi05_actions = pi05.predict_action_chunk(batch)
        torch.manual_seed(2)
        fineart_actions = fineart.predict_action_chunk(batch)
        torch.testing.assert_close(fineart_actions, pi05_actions, rtol=1e-4, atol=1e-4)
