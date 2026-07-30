#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_dm05 import resolve_dm05_action_mode
from .normalization_dm05 import DM05Normalizer, build_dm05_action_target_chunk
from .utils import (
    build_meta,
    collate_dm05_instances,
    get_image_keys,
    infer_dm05_non_delta_indices,
    normalize_task_batch,
    pad_action_chunk,
    pad_vector,
    tensor_to_pil,
)


class DM05LerobotBatchConverter:
    def __init__(
        self, config: Any, tokenization_cls: type, processor: Any, normalizer: DM05Normalizer | None = None
    ):
        self.config = config
        self.normalizer = normalizer
        self._tokenizer = tokenization_cls(
            processor=processor,
            n_bins=config.n_bins,
            max_length=config.tokenizer_max_length,
            add_state=False,
        )

    def convert_lerobot_batch(self, batch: dict[str, Any], include_labels: bool) -> dict[str, Any]:
        tokenizer = self._tokenizer.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0)

        if OBS_STATE not in batch:
            raise ValueError(f"DM05 raw LeRobot batch requires `{OBS_STATE}` or pre-tokenized `input_ids`.")
        if include_labels and self.normalizer is None:
            raise ValueError(
                "DM05 training requires DM05 norm stats. Set policy.norm_stats_path, "
                "or compute one with resolve_dm05_normalizer(config, dataset=...) before training."
            )

        state = batch[OBS_STATE]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = int(state.shape[0])
        image_keys = get_image_keys(batch, self.config.image_keys)
        if not image_keys:
            raise ValueError("DM05 raw LeRobot batch requires at least one visual observation key.")

        action_batch = batch.get(ACTION)
        if action_batch is not None:
            if action_batch.dim() == 1:
                action_batch = action_batch.unsqueeze(0).unsqueeze(1)
            elif action_batch.dim() == 2:
                action_batch = (
                    action_batch.unsqueeze(1)
                    if action_batch.shape[0] == batch_size
                    else action_batch.unsqueeze(0)
                )
        tasks = normalize_task_batch(batch.get("task"), batch_size, "Execute the robot action.")
        meta = build_meta(self.config, image_keys)
        action_mode = resolve_dm05_action_mode(self.config) if include_labels else None
        non_delta_indices = infer_dm05_non_delta_indices(self.config, meta) if include_labels else ()

        action_dim, state_dim = int(self.config.max_action_dim), int(self.config.max_state_dim)
        normalizer = self.normalizer
        tokenized_samples: list[dict[str, Tensor]] = []
        for idx in range(batch_size):
            raw_state = state[idx].to(torch.float32)
            norm_state = normalizer.normalize_state(raw_state) if normalizer else raw_state.float()
            state_np = norm_state.detach().cpu().numpy().astype("float32")
            token_kwargs = {
                "prompt": tasks[idx],
                "images": [tensor_to_pil(batch[key][idx]) for key in image_keys],
                "state": state_np,
                "meta_data": meta,
            }
            if include_labels:
                assert normalizer is not None
                assert action_batch is not None
                raw_action = action_batch[idx].to(torch.float32)
                norm_action = normalizer.normalize_action(
                    build_dm05_action_target_chunk(
                        raw_state=raw_state,
                        raw_action=raw_action,
                        chunk_size=self.config.chunk_size,
                        action_mode=action_mode,
                        non_delta_indices=non_delta_indices,
                        action_target_offset=getattr(self.config, "action_target_offset", 0),
                    )
                )
                action_for_text = (
                    norm_action[norm_action.shape[0] // 2].detach().cpu().numpy().astype("float32")
                )
                tokenized = self._tokenizer.tokenize_robot(**token_kwargs, action=action_for_text)
                tokenized["actions"] = (
                    pad_action_chunk(norm_action, self.config.chunk_size, action_dim).detach().cpu()
                )
                action_dim_mask = torch.zeros(action_dim, dtype=torch.bool)
                valid_mask = meta.get("valid_action_dim_mask")
                if valid_mask is None:
                    action_dim_mask[: min(int(norm_action.shape[-1]), action_dim)] = True
                else:
                    valid = torch.as_tensor(valid_mask, dtype=torch.bool)[:action_dim]
                    action_dim_mask[: valid.shape[0]] = valid
                tokenized["action_dim_mask"] = action_dim_mask
                tokenized["has_actions"] = torch.tensor(True, dtype=torch.bool)
            else:
                tokenized = self._tokenizer.tokenize_robot_infer(**token_kwargs)
            tokenized["states"] = pad_vector(norm_state, state_dim).detach().cpu()
            tokenized_samples.append(tokenized)

        return collate_dm05_instances(
            tokenized_samples,
            pad_token_id=int(pad_token_id),
            max_length=self.config.tokenizer_max_length,
        )
