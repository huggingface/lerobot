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

"""Characterize how PI0-FAST turns generated PaliGemma tokens back into FAST actions."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

from lerobot.policies.pi0_fast import modeling_pi0_fast, processor_pi0_fast  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.processor import fast_ids_to_paligemma_ids, paligemma_ids_to_fast_ids  # noqa: E402
from lerobot.processor.tokenizer_processor import ACTION_TOKENIZER_DIR  # noqa: E402

# PaliGemma's vocabulary, including location and segmentation tokens.
PALIGEMMA_VOCAB_SIZE = 257152
# Tokenizer a checkpoint was trained with, fitted on a machine whose path no longer exists.
SAVED_NAME = "/fitted/on/another/machine"


@pytest.mark.parametrize("fast_skip_tokens", [128, 1152], ids=["pi0_fast", "fineart_vla"])
def test_paligemma_mapping_round_trips_every_fast_id(fast_skip_tokens: int) -> None:
    fast_ids = torch.arange(2048)

    model_ids = fast_ids_to_paligemma_ids(fast_ids, PALIGEMMA_VOCAB_SIZE, fast_skip_tokens)

    # FAST IDs fill the vocabulary downward from just below the skipped tail tokens.
    assert model_ids[0] == PALIGEMMA_VOCAB_SIZE - 1 - fast_skip_tokens
    assert model_ids[-1] == PALIGEMMA_VOCAB_SIZE - fast_skip_tokens - 2048
    torch.testing.assert_close(
        paligemma_ids_to_fast_ids(model_ids, PALIGEMMA_VOCAB_SIZE, fast_skip_tokens), fast_ids
    )
    # Plain ints map the same way, as FineART uses them for its text-generation mask bounds.
    assert fast_ids_to_paligemma_ids(2047, PALIGEMMA_VOCAB_SIZE, fast_skip_tokens) == int(model_ids[-1])


def test_detokenize_actions_decodes_full_chunks_in_relaxed_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    horizon, action_dim, fast_id = 10, 3, 7
    decode = Mock(return_value=np.ones((1, horizon, action_dim)))
    monkeypatch.setattr(modeling_pi0_fast, "decode_fast_actions", decode)
    code_id = PALIGEMMA_VOCAB_SIZE - 1 - 128 - fast_id
    vocabulary = {1: "Action", 2: ":", 3: "|", code_id: "code"}
    policy = SimpleNamespace(
        config=SimpleNamespace(validate_action_token_prefix=True, fast_skip_tokens=128),
        action_tokenizer=object(),
        _paligemma_tokenizer=SimpleNamespace(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            encode=lambda text, **kwargs: [1, 2],
            convert_ids_to_tokens=lambda ids: [vocabulary[token] for token in ids],
            convert_tokens_to_ids=lambda tokens: [code_id for _ in tokens],
        ),
    )

    # Wrapper tokens and everything after the "|" end marker are stripped before decoding.
    generated = torch.tensor([[1, 2, code_id, 3, code_id]])
    actions = PI0FastPolicy.detokenize_actions(policy, generated, horizon, action_dim)

    decode.assert_called_once_with(
        policy.action_tokenizer, [[fast_id]], action_dim, time_horizon=horizon, relaxed=True
    )
    torch.testing.assert_close(actions, torch.ones(1, horizon, action_dim))


def _write_checkpoint(root: Path, action_tokenizer_name: str, *, with_tokenizer: bool) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"action_tokenizer_name": action_tokenizer_name}))
    if with_tokenizer:
        (root / ACTION_TOKENIZER_DIR).mkdir()
        (root / ACTION_TOKENIZER_DIR / "processor_config.json").write_text("{}")
    return root


# (checkpoint saves its tokenizer, configured tokenizer, decoding and encoding use the saved copy)
TOKENIZER_CASES = [
    (True, SAVED_NAME, True),
    # Checkpoints saved before the processor stored its tokenizer.
    (False, SAVED_NAME, False),
    # Fine-tuning with another tokenizer keeps the configured one.
    (True, "lerobot/fast-action-tokenizer", False),
]


@pytest.mark.parametrize(("with_tokenizer", "configured_name", "uses_saved_copy"), TOKENIZER_CASES)
def test_saved_tokenizer_is_used_only_when_the_checkpoint_was_trained_with_it(
    tmp_path: Path, with_tokenizer: bool, configured_name: str, uses_saved_copy: bool
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", SAVED_NAME, with_tokenizer=with_tokenizer)

    folder = processor_pi0_fast.find_saved_action_tokenizer(checkpoint, configured_name)

    assert folder == (str(checkpoint / ACTION_TOKENIZER_DIR) if uses_saved_copy else None)


def test_saved_tokenizer_lookup_falls_back_when_the_hub_checkpoint_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot_download(*args, **kwargs):
        raise LocalEntryNotFoundError("offline")

    monkeypatch.setattr(processor_pi0_fast, "snapshot_download", snapshot_download)

    assert processor_pi0_fast.find_saved_action_tokenizer("user/checkpoint", SAVED_NAME) is None


@pytest.mark.parametrize(("with_tokenizer", "configured_name", "uses_saved_copy"), TOKENIZER_CASES)
def test_policy_decodes_with_the_tokenizer_of_its_pretrained_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_tokenizer: bool,
    configured_name: str,
    uses_saved_copy: bool,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", SAVED_NAME, with_tokenizer=with_tokenizer)
    loaded = []

    def from_pretrained(name, **kwargs):
        loaded.append(name)
        raise RuntimeError("stop after choosing the tokenizer")

    monkeypatch.setattr(modeling_pi0_fast, "AutoProcessor", SimpleNamespace(from_pretrained=from_pretrained))
    # Set by every load path: from_pretrained, sharded-checkpoint resume and PEFT adapters.
    config = PI0FastConfig(action_tokenizer_name=configured_name, pretrained_path=checkpoint)

    with pytest.raises(RuntimeError, match="Failed to load required tokenizers"):
        PI0FastPolicy(config)

    assert loaded == [str(checkpoint / ACTION_TOKENIZER_DIR) if uses_saved_copy else configured_name]


@pytest.mark.parametrize(("with_tokenizer", "configured_name", "uses_saved_copy"), TOKENIZER_CASES)
def test_pretrained_processor_encodes_with_the_policy_decoding_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_tokenizer: bool,
    configured_name: str,
    uses_saved_copy: bool,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", SAVED_NAME, with_tokenizer=with_tokenizer)
    load = Mock(return_value=(None, None))
    monkeypatch.setattr(processor_pi0_fast, "load_pretrained_policy_processors", load)

    processor_pi0_fast.make_pi0_fast_pre_post_processors_from_pretrained(
        PI0FastConfig(action_tokenizer_name=configured_name), str(checkpoint)
    )

    # Without an override, the step loads the copy saved in the checkpoint.
    overrides = load.call_args.kwargs["preprocessor_overrides"]
    override = overrides.get("action_tokenizer_processor", {}).get("action_tokenizer_name")
    assert override == (None if uses_saved_copy else configured_name)
