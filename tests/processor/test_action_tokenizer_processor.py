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

"""Shape and formatting contracts for FAST action tokenization, without Hub access."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import ActionTokenizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION_CODE_TOKEN_MASK, ACTION_TOKEN_MASK, ACTION_TOKENS


@pytest.fixture
def processor() -> ActionTokenizerProcessorStep:
    processor = object.__new__(ActionTokenizerProcessorStep)
    processor.max_action_tokens = 12
    processor.fast_skip_tokens = 128
    processor.allow_truncation = False
    processor.action_tokenizer = Mock(side_effect=lambda chunk: [[int(chunk.sum().item()) % 100, 2, 3]])
    processor._paligemma_tokenizer = SimpleNamespace(
        vocab_size=1000,
        bos_token_id=2,
        encode=lambda text, **kwargs: [10, 11] if text == "Action: " else [12, 1],
    )
    return processor


@pytest.mark.parametrize("shape", [(1, 2), (4, 2), (1, 4, 2), (2, 4, 2)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_action_chunks_preserve_horizon_and_formatting(
    processor: ActionTokenizerProcessorStep, shape: tuple[int, ...], device: str
) -> None:
    actions = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    original = actions.clone()

    result = processor(create_transition(action=actions))
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    single_sample = len(shape) == 2
    chunks = actions.unsqueeze(0) if single_sample else actions
    expected_shape = (12,) if single_sample else (shape[0], 12)

    assert result[TransitionKey.ACTION] is actions
    torch.testing.assert_close(actions, original)
    assert processor.action_tokenizer.call_count == len(chunks)
    for call, chunk in zip(processor.action_tokenizer.call_args_list, chunks, strict=True):
        backend_input = call.args[0]
        assert backend_input.shape == (1, *chunk.shape)
        assert backend_input.device.type == "cpu"
        torch.testing.assert_close(backend_input, chunk.unsqueeze(0).cpu())

    raw_codes = [int(chunk.sum().item()) % 100 for chunk in chunks]
    expected_tokens = torch.tensor(
        [[2, 10, 11, 871 - code, 869, 868, 12, 1, 0, 0, 0, 0] for code in raw_codes], device=device
    )
    expected_mask = torch.tensor([True] * 8 + [False] * 4, device=device).expand(len(chunks), -1)
    expected_code_mask = torch.tensor([False] * 3 + [True] * 3 + [False] * 6, device=device).expand(
        len(chunks), -1
    )
    for key, expected in (
        (ACTION_TOKENS, expected_tokens),
        (ACTION_TOKEN_MASK, expected_mask),
        (ACTION_CODE_TOKEN_MASK, expected_code_mask),
    ):
        actual = complementary[key]
        assert actual.shape == expected_shape
        assert actual.device == actions.device
        torch.testing.assert_close(actual, expected.squeeze(0) if single_sample else expected)


@pytest.mark.parametrize("shape", [(), (6,), (1, 2, 3, 4)])
def test_action_tokenizer_rejects_unsupported_ranks(
    processor: ActionTokenizerProcessorStep, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match=r"\(H, D\).*\(B, H, D\)"):
        processor._tokenize_action(torch.zeros(shape))
    processor.action_tokenizer.assert_not_called()
