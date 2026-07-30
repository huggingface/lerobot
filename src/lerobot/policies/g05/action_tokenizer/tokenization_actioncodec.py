# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .modeling_actioncodec import G05ActionCodecModel


def count_valid_sequences(sequence_length: int, min_block_length: int) -> int:
    counts = [0] * (sequence_length + 1)
    for index in range(1, min(min_block_length + 2, sequence_length + 1)):
        counts[index] = 2 * index
    for index in range(min_block_length + 2, sequence_length + 1):
        counts[index] = counts[index - 1] + counts[index - min_block_length - 1]
    return counts[sequence_length]


class ConstrainedSequenceTokenizer:
    """Lossless base-N codec for gripper sequences without one-frame glitches."""

    def __init__(
        self, sequence_length: int, min_block_length: int, vocab_size: int, threshold: float
    ) -> None:
        self.sequence_length = sequence_length
        self.min_block_length = min_block_length
        self.vocab_size = vocab_size
        self.threshold = threshold
        self.num_valid_sequences = count_valid_sequences(sequence_length, min_block_length)
        self.num_tokens = max(1, math.ceil(math.log(self.num_valid_sequences, vocab_size)))
        self.count: Callable = self._make_count_function()

    def _make_count_function(self) -> Callable:
        min_length = self.min_block_length

        @functools.cache
        def count(remaining: int, last_value: int, run_length: int, is_first_run: bool) -> int:
            if remaining == 0:
                return 1
            result = count(remaining - 1, last_value, min(run_length + 1, min_length + 1), is_first_run)
            if is_first_run or run_length > min_length:
                result += count(remaining - 1, 1 - last_value, 1, False)
            return result

        return count

    def _count_if_zero(self, remaining: int, last_value: int, run_length: int, is_first_run: bool) -> int:
        if last_value == -1:
            return self.count(remaining, 0, 1, True)
        if last_value == 0:
            return self.count(
                remaining,
                0,
                min(run_length + 1, self.min_block_length + 1),
                is_first_run,
            )
        if is_first_run or run_length > self.min_block_length:
            return self.count(remaining, 0, 1, False)
        return 0

    def repair(self, sequence: list[int]) -> list[int]:
        sequence = list(sequence)
        while True:
            boundaries = (
                [0] + [i for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1]] + [len(sequence)]
            )
            short_run = next(
                (
                    (boundaries[i], boundaries[i + 1])
                    for i in range(1, len(boundaries) - 2)
                    if boundaries[i + 1] - boundaries[i] <= self.min_block_length
                ),
                None,
            )
            if short_run is None:
                return sequence
            start, end = short_run
            sequence[start:end] = [sequence[start - 1]] * (end - start)

    def encode(self, values: Tensor) -> list[int]:
        sequence = self.repair([int(value >= self.threshold) for value in values.tolist()])
        index = 0
        last_value, run_length, is_first_run = -1, 0, True
        for position, bit in enumerate(sequence):
            remaining = self.sequence_length - position - 1
            if bit:
                index += self._count_if_zero(remaining, last_value, run_length, is_first_run)
            if last_value == -1:
                last_value, run_length = bit, 1
            elif bit == last_value:
                run_length = min(run_length + 1, self.min_block_length + 1)
            else:
                last_value, run_length, is_first_run = bit, 1, False
        digits = [0] * self.num_tokens
        for position in range(self.num_tokens - 1, -1, -1):
            digits[position], index = index % self.vocab_size, index // self.vocab_size
        return digits

    def decode(self, tokens: Sequence[int]) -> Tensor:
        index = 0
        for token in tokens:
            index = index * self.vocab_size + max(0, min(int(token), self.vocab_size - 1))
        index = min(index, self.num_valid_sequences - 1)
        sequence = []
        last_value, run_length, is_first_run = -1, 0, True
        for position in range(self.sequence_length):
            remaining = self.sequence_length - position - 1
            zero_count = self._count_if_zero(remaining, last_value, run_length, is_first_run)
            if index < zero_count:
                bit = 0
            else:
                bit, index = 1, index - zero_count
            sequence.append(1.0 if bit else -1.0)
            if last_value == -1:
                last_value, run_length = bit, 1
            elif bit == last_value:
                run_length = min(run_length + 1, self.min_block_length + 1)
            else:
                last_value, run_length, is_first_run = bit, 1, False
        return torch.tensor(sequence)


class G05ActionTokenizer:
    """G0.5 action tensor ↔ grouped ActionCodec token stream.

    Action indices are local to the codec: codebook entries start at zero and
    group markers follow them. When a text tokenizer is supplied, ``encode``
    and ``decode`` translate this local vocabulary to its exact VLM token ids.
    """

    def __init__(self, model: G05ActionCodecModel, text_tokenizer=None) -> None:
        self.model = model.eval()
        self.config = model.config
        if not self.config.parts_meta:
            raise ValueError("ActionCodec config must include parts_meta")
        self.neural_keys = [key for key in self.config.parts_meta if not self._is_rule_key(key)]
        self.rule_keys = [key for key in self.config.parts_meta if self._is_rule_key(key)]
        self.rule_tokenizer = ConstrainedSequenceTokenizer(
            self.config.horizon,
            self.config.rule_based_min_block_len,
            self.config.codebook_size,
            self.config.rule_based_binarize_threshold,
        )
        self.action_tokens = [f"<action{index:04d}>" for index in range(self.config.codebook_size)]
        self.action_tokens += [
            f"<{key}_{level}>" for level in range(self.config.n_codebooks) for key in self.neural_keys
        ]
        self.action_tokens += [f"<{key}>" for key in self.rule_keys]
        self.marker_indices = {
            token: self.config.codebook_size + index
            for index, token in enumerate(self.action_tokens[self.config.codebook_size :])
        }
        self.text_tokenizer = text_tokenizer
        self._local_to_text: Tensor | None = None
        self._text_to_local: dict[int, int] | None = None
        if text_tokenizer is not None:
            ids = text_tokenizer.convert_tokens_to_ids(self.action_tokens)
            if len(set(ids)) != len(ids) or any(index is None or index < 0 for index in ids):
                raise ValueError("VLM tokenizer does not contain the complete ActionCodec vocabulary")
            self._local_to_text = torch.tensor(ids, dtype=torch.long)
            self._text_to_local = {text_id: local_id for local_id, text_id in enumerate(ids)}

    def _is_rule_key(self, key: str) -> bool:
        return any(pattern in key for pattern in self.config.rule_based_key_patterns)

    def split_action(self, action: Tensor) -> dict[str, Tensor]:
        expected_width = sum(self.config.parts_meta.values())
        if action.ndim != 3 or action.shape[-1] < expected_width:
            raise ValueError(f"action must have shape [B,T,D] with D >= {expected_width}")
        return dict(
            zip(
                self.config.parts_meta,
                action[..., :expected_width].split(list(self.config.parts_meta.values()), -1),
                strict=True,
            )
        )

    @torch.inference_mode()
    def encode_action_indices(self, action: Tensor) -> Tensor:
        parts = self.split_action(action.to(next(self.model.parameters()).device, dtype=torch.float32))
        neural_codes = self.model.encode({key: parts[key] for key in self.neural_keys})
        rows: list[list[int]] = [[] for _ in range(action.shape[0])]
        # G0.5 serializes RVQ codes level-first, with a marker before every
        # component-level segment. This order is part of the trained VLM ABI.
        for level in range(self.config.num_residuals):
            for key in self.neural_keys:
                marker = self.marker_indices[f"<{key}_{level}>"]
                for batch_index, codes in enumerate(neural_codes[key][:, level]):
                    rows[batch_index].extend((marker, *codes.tolist()))
        for key in self.rule_keys:
            marker = self.marker_indices[f"<{key}>"]
            for batch_index, values in enumerate(parts[key][..., 0]):
                rows[batch_index].extend((marker, *self.rule_tokenizer.encode(values)))
        return torch.tensor(rows, dtype=torch.long, device=action.device)

    def encode(self, action: Tensor) -> Tensor:
        local_ids = self.encode_action_indices(action)
        if self._local_to_text is None:
            return local_ids
        return self._local_to_text.to(local_ids.device)[local_ids]

    @torch.inference_mode()
    def decode_action_indices(self, rows: Tensor) -> Tensor:
        if rows.ndim != 2:
            raise ValueError("action token rows must have shape [B,S]")
        batch_size = rows.shape[0]
        parsed_neural = {
            key: torch.zeros(
                batch_size,
                self.config.num_residuals,
                self.config.code_length,
                dtype=torch.long,
                device=next(self.model.parameters()).device,
            )
            for key in self.neural_keys
        }
        parsed_rule = {
            key: torch.zeros(batch_size, self.rule_tokenizer.num_tokens, dtype=torch.long)
            for key in self.rule_keys
        }
        neural_markers = {
            self.marker_indices[f"<{key}_{level}>"]: (key, level)
            for level in range(self.config.num_residuals)
            for key in self.neural_keys
        }
        rule_markers = {self.marker_indices[f"<{key}>"]: key for key in self.rule_keys}
        for batch_index, row in enumerate(rows.tolist()):
            position = 0
            while position < len(row):
                token = row[position]
                if token in neural_markers:
                    key, level = neural_markers[token]
                    values = row[position + 1 : position + 1 + self.config.code_length]
                    parsed_neural[key][batch_index, level, : len(values)] = torch.tensor(
                        values, device=parsed_neural[key].device
                    )
                    position += 1 + self.config.code_length
                elif token in rule_markers:
                    key = rule_markers[token]
                    values = row[position + 1 : position + 1 + self.rule_tokenizer.num_tokens]
                    parsed_rule[key][batch_index, : len(values)] = torch.tensor(
                        values, device=parsed_rule[key].device
                    )
                    position += 1 + self.rule_tokenizer.num_tokens
                else:
                    position += 1
        decoded = self.model.decode(parsed_neural, self.config.parts_meta)
        for key, tokens in parsed_rule.items():
            decoded[key] = torch.stack([self.rule_tokenizer.decode(row) for row in tokens]).unsqueeze(-1)
        return torch.cat([decoded[key].to(rows.device) for key in self.config.parts_meta], dim=-1)

    def decode(self, rows: Tensor) -> Tensor:
        if self._text_to_local is None:
            return self.decode_action_indices(rows)
        try:
            local_rows = [[self._text_to_local[int(token)] for token in row] for row in rows.tolist()]
        except KeyError as error:
            raise ValueError(f"token id {error.args[0]} is outside the ActionCodec vocabulary") from error
        return self.decode_action_indices(torch.tensor(local_rows, dtype=torch.long, device=rows.device))
