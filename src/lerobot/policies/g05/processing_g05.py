#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Native G0.5 prompt serialization and tokenizer registration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

IGNORE_INDEX = -100


class G05TokenType:
    """Token categories stored in G0.5's attention-mask tensor."""

    PADDING = 0
    IMAGE = 1
    PROPRIO = 2
    ACTION = 3
    TEXT = 4
    COT = 5
    PRED_TEXT = 6


@dataclass
class G05SequenceBatch:
    input_ids: Tensor
    labels: Tensor
    token_types: Tensor
    split_index: int | None = None


@dataclass
class _Segment:
    kind: str
    content: str = ""
    sample_key: str = ""
    processor: str = ""
    masked: bool = False
    max_tokens: int | None = None


class G05Tokenizer:
    """Checkpoint-compatible G0.5 tokenizer and template serializer.

    G0.5 extends Qwen3.5's tokenizer with ActionCodec codes, per-group
    residual markers, ``<EOV>``, and ``<state>``. Registration order is model
    state: changing it changes the rows used by the tied language head.
    """

    _PLACEHOLDER = re.compile(r"<([^<>|]+)>")

    def __init__(self, processor_path: str | Path, model_config: dict[str, Any]) -> None:
        self.processor_path = Path(processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        self.model_config = model_config
        at_config = model_config["AT_CONFIG"]
        architecture = at_config["model_arch"]

        self.action_tokens = [f"<action{index:04d}>" for index in range(int(architecture["codebook_size"]))]
        parts = list(at_config["parts_meta"])
        rule_patterns = tuple(at_config.get("rule_based_key_patterns") or ())
        self.rule_parts = [name for name in parts if any(pattern in name for pattern in rule_patterns)]
        self.neural_parts = [name for name in parts if name not in self.rule_parts]
        self.group_tokens = [
            f"<{part}_{residual}>"
            for residual in range(int(architecture["n_codebooks"]))
            for part in self.neural_parts
        ] + [f"<{part}>" for part in self.rule_parts]
        self.tokenizer.add_tokens(self.action_tokens + self.group_tokens + ["<EOV>", "<state>"])

        self.pad_token_id = int(model_config["pad_token_id"])
        self.eos_token_id = int(model_config["eos_token_id"])
        self.image_token_id = int(model_config["image_token_index"])
        self.vision_start_token_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.vision_end_token_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        self.eov_token_id = int(self.tokenizer.convert_tokens_to_ids("<EOV>"))
        self.state_token_id = int(self.tokenizer.convert_tokens_to_ids("<state>"))
        self.action_token_begin = int(self.tokenizer.convert_tokens_to_ids(self.action_tokens[0]))
        self.action_token_end = self.action_token_begin + len(self.action_tokens)
        self.action_token_end_with_markers = self.action_token_end + len(self.group_tokens)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: Tensor | list[int]) -> str:
        if isinstance(ids, Tensor):
            ids = ids.detach().cpu().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @staticmethod
    def _resolve_template(template: str) -> str:
        replacements = {
            "<bos>": "",
            "<eos>": "<|endoftext|>",
            "<chat_user_prefix>": "",
            "<chat_user_suffix>": "",
            "<chat_assistant_prefix>": "",
        }
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        return template

    def _parse(self, template: str) -> list[_Segment]:
        template = self._resolve_template(template)
        segments: list[_Segment] = []
        last = 0
        for match in self._PLACEHOLDER.finditer(template):
            if match.start() > last:
                segments.append(_Segment("static", template[last : match.start()]))
            raw = match.group(1).strip()
            if raw in {"EOC", "EOV"}:
                segments.append(_Segment("control", raw))
                last = match.end()
                continue

            max_tokens = None
            limit = re.match(r"^(.+)_(\d+)$", raw)
            if limit:
                raw, max_tokens = limit.group(1), int(limit.group(2))
            masked = raw.endswith("_!")
            key = raw[:-2] if masked else raw
            if "_" not in key:
                token = f"<{raw}>"
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is None:
                    raise ValueError(f"Unknown G0.5 template token {token!r}.")
                segments.append(_Segment("static", token))
            else:
                sample_key, processor = key.rsplit("_", 1)
                segments.append(
                    _Segment(
                        "dynamic",
                        sample_key=sample_key,
                        processor=processor,
                        masked=masked,
                        max_tokens=max_tokens,
                    )
                )
            last = match.end()
        if last < len(template):
            segments.append(_Segment("static", template[last:]))
        return segments

    @staticmethod
    def _slice_segments(segments: list[_Segment], mode: str | None, *, pred_eov: bool) -> list[_Segment]:
        eoc = next(
            (
                index
                for index, segment in enumerate(segments)
                if segment.kind == "control" and segment.content == "EOC"
            ),
            None,
        )
        eov = next(
            (
                index
                for index, segment in enumerate(segments)
                if segment.kind == "control" and segment.content == "EOV"
            ),
            None,
        )

        def strip(values: list[_Segment], keep_eov: bool) -> list[_Segment]:
            output: list[_Segment] = []
            after_eoc = False
            for segment in values:
                if segment.kind == "control":
                    if segment.content == "EOC":
                        after_eoc = True
                    elif segment.content == "EOV" and keep_eov:
                        output.append(
                            _Segment(
                                "dynamic",
                                content="<EOV>",
                                processor="text",
                                masked=not pred_eov,
                            )
                        )
                    continue
                if after_eoc and segment.kind == "static":
                    output.append(
                        _Segment("dynamic", content=segment.content, processor="text", masked=False)
                    )
                else:
                    output.append(segment)
            return output

        if mode == "context":
            return strip(segments if eoc is None else segments[:eoc], keep_eov=True)
        if mode == "prefix":
            return strip(segments if eov is None else segments[: eov + 1], keep_eov=True)
        if mode == "suffix":
            return [] if eov is None else strip(segments[eov + 1 :], keep_eov=False)
        return strip(segments, keep_eov=True)

    def _serialize_segment(
        self,
        segment: _Segment,
        sample: dict[str, Any],
        *,
        action_codec: Any | None,
    ) -> tuple[list[int], list[int], list[float]]:
        if segment.kind == "static":
            ids = self.encode_text(segment.content)
            return ids, [IGNORE_INDEX] * len(ids), [float(G05TokenType.TEXT)] * len(ids)

        if segment.sample_key:
            if segment.sample_key not in sample:
                raise KeyError(f"G0.5 prompt is missing sample field {segment.sample_key!r}.")
            value = sample[segment.sample_key]
        else:
            value = segment.content

        if segment.processor == "text":
            ids = self.encode_text(value if isinstance(value, str) else str(value))
            token_type = G05TokenType.TEXT if segment.masked else G05TokenType.PRED_TEXT
            labels = [IGNORE_INDEX] * len(ids) if segment.masked else ids.copy()
        elif segment.processor == "image":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("G0.5 image placeholders require an (height, width) pair.")
            height, width = (int(item) for item in value)
            vision = self.model_config["vision"]
            count = (height // int(vision["patch_size"]) // int(vision["spatial_merge_size"])) * (
                width // int(vision["patch_size"]) // int(vision["spatial_merge_size"])
            )
            ids = [self.vision_start_token_id] + [self.image_token_id] * count + [self.vision_end_token_id]
            labels = [IGNORE_INDEX] * len(ids)
            types = (
                [float(G05TokenType.TEXT)] + [float(G05TokenType.IMAGE)] * count + [float(G05TokenType.TEXT)]
            )
            return ids, labels, types
        elif segment.processor == "proprio":
            state = value["value"] if isinstance(value, dict) else value
            count = 1 if torch.as_tensor(state).ndim <= 1 else int(torch.as_tensor(state).shape[0])
            ids = [self.state_token_id] * count
            labels = [IGNORE_INDEX] * count
            token_type = G05TokenType.PROPRIO
        elif segment.processor == "action":
            if action_codec is None:
                raise RuntimeError(
                    "This G0.5 training template includes ActionCodec targets, but the "
                    "checkpoint has no native ActionCodec sidecar loaded."
                )
            ids = action_codec.encode_for_language(value)
            labels = ids.copy()
            token_type = G05TokenType.ACTION
        else:
            ids = self.encode_text(value if isinstance(value, str) else str(value))
            labels = [IGNORE_INDEX] * len(ids) if segment.masked else ids.copy()
            token_type = G05TokenType.COT

        if segment.max_tokens is not None:
            ids = ids[: segment.max_tokens]
            labels = labels[: segment.max_tokens]
        return ids, labels, [float(token_type)] * len(ids)

    def _serialize(
        self,
        sample: dict[str, Any],
        *,
        mode: str | None,
        action_codec: Any | None,
    ) -> tuple[list[int], list[int], list[float]]:
        pred_eov = bool(self.model_config.get("input_preprocessor", {}).get("pred_eov", False))
        segments = self._slice_segments(self._parse(sample["template"]), mode, pred_eov=pred_eov)
        ids: list[int] = []
        labels: list[int] = []
        types: list[float] = []
        for segment in segments:
            segment_ids, segment_labels, segment_types = self._serialize_segment(
                segment, sample, action_codec=action_codec
            )
            ids.extend(segment_ids)
            labels.extend(segment_labels)
            types.extend(segment_types)
        return ids, labels, types

    def _pad(
        self,
        rows: list[tuple[list[int], list[int], list[float]]],
        *,
        right_align: bool,
        device: torch.device,
    ) -> G05SequenceBatch:
        length = max(len(row[0]) for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long, device=device)
        labels = torch.full((len(rows), length), IGNORE_INDEX, dtype=torch.long, device=device)
        token_types = torch.zeros((len(rows), length), dtype=torch.float32, device=device)
        for index, (ids, row_labels, types) in enumerate(rows):
            start = length - len(ids) if right_align else 0
            stop = start + len(ids)
            input_ids[index, start:stop] = torch.tensor(ids, dtype=torch.long, device=device)
            labels[index, start:stop] = torch.tensor(row_labels, dtype=torch.long, device=device)
            token_types[index, start:stop] = torch.tensor(types, dtype=torch.float32, device=device)
        return G05SequenceBatch(input_ids, labels, token_types)

    def encode_inference(
        self,
        samples: list[dict[str, Any]],
        *,
        device: torch.device,
    ) -> G05SequenceBatch:
        rows = [self._serialize(sample, mode="context", action_codec=None) for sample in samples]
        return self._pad(rows, right_align=True, device=device)

    def encode_train(
        self,
        samples: list[dict[str, Any]],
        *,
        device: torch.device,
        action_codec: Any | None,
    ) -> G05SequenceBatch:
        prefix_rows = [
            self._serialize(sample, mode="prefix", action_codec=action_codec) for sample in samples
        ]
        suffix_rows = [
            self._serialize(sample, mode="suffix", action_codec=action_codec) for sample in samples
        ]
        prefix = self._pad(prefix_rows, right_align=True, device=device)
        suffix = self._pad(suffix_rows, right_align=False, device=device)
        return G05SequenceBatch(
            input_ids=torch.cat((prefix.input_ids, suffix.input_ids), dim=1),
            labels=torch.cat((prefix.labels, suffix.labels), dim=1),
            token_types=torch.cat((prefix.token_types, suffix.token_types), dim=1),
            split_index=prefix.input_ids.shape[1],
        )
