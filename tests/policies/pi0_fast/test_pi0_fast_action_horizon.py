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

"""Exercise chunk decoding and execution without downloading policy weights."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from scipy.fft import dct, idct  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

PolicyCase = tuple[PI0FastPolicy, dict[str, torch.Tensor], torch.Tensor, Mock]


@pytest.fixture
def policy_and_actions(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> PolicyCase:
    use_kv_cache, n_action_steps = request.param
    horizon, action_dim = 8, 2
    actions = np.stack(
        [
            np.ones((horizon, action_dim)),
            np.linspace(-1, 1, horizon * action_dim).reshape(horizon, action_dim),
        ]
    )
    coefficients = np.rint(dct(actions, axis=1, norm="ortho") * 10).astype(int)
    min_token = int(coefficients.min())
    encoded = {
        7 + index: "".join(chr(int(value) - min_token) for value in sample.flatten())
        for index, sample in enumerate(coefficients)
    }
    expected = torch.tensor(idct(coefficients / 10, axis=1, norm="ortho"), dtype=torch.float32)

    # Skip weight/tokenizer loading, but exercise the real policy and DCT decoding methods.
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = PI0FastConfig(
        device="cpu",
        chunk_size=horizon,
        n_action_steps=n_action_steps,
        use_kv_cache=use_kv_cache,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )
    vocab_size = 4096
    code_ids = [vocab_size - 1 - policy.config.fast_skip_tokens - code for code in encoded]
    vocabulary = {1: "Action", 2: ":", 3: "|", code_ids[0]: "code0", code_ids[1]: "code1"}
    inverse_vocabulary = {value: key for key, value in vocabulary.items()}
    policy._paligemma_tokenizer = SimpleNamespace(
        vocab_size=vocab_size,
        encode=lambda text, **kwargs: [1, 2],
        convert_ids_to_tokens=lambda ids: [vocabulary[token] for token in ids],
        convert_tokens_to_ids=lambda tokens: [inverse_vocabulary[token] for token in tokens],
    )
    policy.action_tokenizer = SimpleNamespace(
        min_token=min_token,
        scale=10,
        bpe_tokenizer=SimpleNamespace(decode=lambda tokens: encoded[int(tokens[0])]),
    )
    generated = torch.tensor([[1, 2, code_id, 3] for code_id in code_ids])
    sampler = Mock(return_value=generated)
    unused_sampler = Mock(side_effect=AssertionError("Wrong sampling path"))
    policy.model = SimpleNamespace(
        sample_actions_fast_kv_cache=sampler if use_kv_cache else unused_sampler,
        sample_actions_fast=unused_sampler if use_kv_cache else sampler,
    )
    policy._action_queue = deque()
    monkeypatch.setattr(policy, "_preprocess_images", lambda batch: ([], []))
    monkeypatch.setattr(policy, "_rtc_enabled", lambda: False)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    return policy, batch, expected, sampler


@pytest.mark.parametrize(
    "policy_and_actions",
    [(cached, steps) for cached in (False, True) for steps in (1, 3, 8)],
    indirect=True,
)
def test_predict_action_chunk_decodes_full_training_horizon(policy_and_actions: PolicyCase) -> None:
    policy, batch, expected, _ = policy_and_actions

    actual = policy.predict_action_chunk(batch)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "policy_and_actions",
    [(cached, steps) for cached in (False, True) for steps in (1, 3, 8)],
    indirect=True,
)
def test_select_action_executes_only_requested_prefix(policy_and_actions: PolicyCase) -> None:
    policy, batch, expected, sampler = policy_and_actions

    for timestep in range(policy.config.n_action_steps):
        torch.testing.assert_close(policy.select_action(batch), expected[:, timestep])
    assert sampler.call_count == 1
    assert len(policy._action_queue) == 0

    torch.testing.assert_close(policy.select_action(batch), expected[:, 0])
    assert sampler.call_count == 2
