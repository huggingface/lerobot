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

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from tests.policies.pi0_pi05.openpi_pytorch import preprocessing_pytorch as openpi_preprocessing

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TOKENIZER_NAME = "google/paligemma-3b-pt-224"


@dataclass
class OpenPIObservation:
    state: torch.Tensor
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor


@lru_cache(maxsize=1)
def paligemma_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def clone_batch(batch: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value) for key, value in batch.items()
    }


def pad_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad last dimension {tensor.shape[-1]} down to {target_dim}")
    return F.pad(tensor, (0, target_dim - tensor.shape[-1]))


def mean_std_normalize(tensor: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"].to(device=tensor.device, dtype=tensor.dtype)
    std = stats["std"].to(device=tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / (std + 1e-8)


def quantile_normalize(tensor: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    q01 = stats["q01"].to(device=tensor.device, dtype=tensor.dtype)
    q99 = stats["q99"].to(device=tensor.device, dtype=tensor.dtype)
    denom = torch.where(q99 == q01, torch.full_like(q99, 1e-8), q99 - q01)
    return 2.0 * (tensor - q01) / denom - 1.0


def openpi_model_state_from_raw(
    batch: dict[str, torch.Tensor],
    *,
    action_dim: int,
    dataset_stats: dict[str, dict[str, torch.Tensor]],
    pi05: bool,
) -> torch.Tensor:
    state = batch[OBS_STATE].to(dtype=torch.float32)
    if pi05:
        state = quantile_normalize(state, dataset_stats[OBS_STATE])
    else:
        state = mean_std_normalize(state, dataset_stats[OBS_STATE])
    return pad_last_dim(state, action_dim)


def openpi_model_actions_from_raw(
    batch: dict[str, torch.Tensor],
    *,
    action_dim: int,
    dataset_stats: dict[str, dict[str, torch.Tensor]],
    pi05: bool,
) -> torch.Tensor:
    actions = batch[ACTION].to(dtype=torch.float32)
    if pi05:
        actions = quantile_normalize(actions, dataset_stats[ACTION])
    else:
        actions = mean_std_normalize(actions, dataset_stats[ACTION])
    return pad_last_dim(actions, action_dim)


def _tasks_from_raw(batch: dict, batch_size: int) -> list[str]:
    tasks = batch.get("task")
    if tasks is None:
        raise ValueError("The parity batch must include a task prompt.")
    if isinstance(tasks, str):
        return [tasks] * batch_size
    if len(tasks) == 1:
        return [tasks[0]] * batch_size
    if len(tasks) != batch_size:
        raise ValueError(f"Expected {batch_size} task prompts, got {len(tasks)}")
    return list(tasks)


def _format_pi0_prompts(tasks: list[str]) -> list[str]:
    return [f"{task.strip().replace('_', ' ').replace(chr(10), ' ')}\n" for task in tasks]


def _format_pi05_prompts(tasks: list[str], normalized_state: torch.Tensor) -> list[str]:
    state_np = normalized_state.detach().cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    prompts = []
    for task, state in zip(tasks, discretized_states, strict=True):
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, state))
        prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")
    return prompts


def _tokenize_prompts(prompts: list[str], *, max_token_len: int, device: torch.device | str):
    tokenized = paligemma_tokenizer()(
        prompts,
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=max_token_len,
        return_tensors="pt",
    )
    tokens = tokenized["input_ids"].to(device)
    masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
    return tokens, masks


def make_openpi_observation_from_raw(
    batch: dict[str, torch.Tensor],
    *,
    action_dim: int,
    max_token_len: int,
    dataset_stats: dict[str, dict[str, torch.Tensor]],
    pi05: bool,
) -> OpenPIObservation:
    batch_size = batch[OBS_STATE].shape[0]
    device = batch[OBS_STATE].device
    state = openpi_model_state_from_raw(
        batch,
        action_dim=action_dim,
        dataset_stats=dataset_stats,
        pi05=pi05,
    )

    tasks = _tasks_from_raw(batch, batch_size)
    prompts = _format_pi05_prompts(tasks, state) if pi05 else _format_pi0_prompts(tasks)
    tokens, masks = _tokenize_prompts(prompts, max_token_len=max_token_len, device=device)

    images = {
        key: batch[f"observation.images.{key}"].to(device=device, dtype=torch.float32) * 2.0 - 1.0
        for key in IMAGE_KEYS
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool, device=device) for key in IMAGE_KEYS}

    return OpenPIObservation(
        state=state,
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=masks,
        token_ar_mask=torch.zeros_like(tokens, dtype=torch.int32),
        token_loss_mask=torch.ones_like(masks, dtype=torch.bool),
    )


def assert_processor_inputs_match_lerobot(
    lerobot_policy,
    lerobot_batch: dict[str, torch.Tensor],
    openpi_observation: OpenPIObservation,
    *,
    compare_state: bool,
):
    openpi_processed = openpi_preprocessing.preprocess_observation_pytorch(openpi_observation, train=False)
    lerobot_images, lerobot_image_masks = lerobot_policy._preprocess_images(lerobot_batch)

    # Token IDs, token masks, images, image masks, and PI0 state are intentionally built from the same
    # raw batch through independent LeRobot/OpenPI-style processor logic. They must be bitwise equal.
    torch.testing.assert_close(
        openpi_observation.tokenized_prompt, lerobot_batch[OBS_LANGUAGE_TOKENS], rtol=0, atol=0
    )
    torch.testing.assert_close(
        openpi_observation.tokenized_prompt_mask,
        lerobot_batch[OBS_LANGUAGE_ATTENTION_MASK],
        rtol=0,
        atol=0,
    )

    for openpi_image, lerobot_image in zip(openpi_processed.images.values(), lerobot_images, strict=True):
        torch.testing.assert_close(openpi_image, lerobot_image, rtol=0, atol=0)

    for openpi_mask, lerobot_mask in zip(
        openpi_processed.image_masks.values(), lerobot_image_masks, strict=True
    ):
        torch.testing.assert_close(openpi_mask, lerobot_mask, rtol=0, atol=0)

    if compare_state:
        torch.testing.assert_close(
            openpi_processed.state, lerobot_policy.prepare_state(lerobot_batch), rtol=0, atol=0
        )


def load_openpi_reference_state_dict(repo_id: str) -> dict[str, torch.Tensor]:
    cache_dir = Path(snapshot_download(repo_id=repo_id, repo_type="model"))
    return safetensors.torch.load_file(cache_dir / "model.safetensors")


def fix_reference_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fixed_state_dict = dict(state_dict)
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    embed_tokens_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if lm_head_key in fixed_state_dict and embed_tokens_key not in fixed_state_dict:
        fixed_state_dict[embed_tokens_key] = fixed_state_dict[lm_head_key].clone()
    return fixed_state_dict


@contextmanager
def fixed_flow_sampling(model, *, noise: torch.Tensor, time: torch.Tensor) -> Iterator[None]:
    original_sample_noise = model.sample_noise
    original_sample_time = model.sample_time

    def sample_noise(shape, device):
        if tuple(shape) != tuple(noise.shape):
            raise ValueError(f"Expected noise shape {tuple(noise.shape)}, got {tuple(shape)}")
        return noise.to(device=device)

    def sample_time(batch_size, device):
        if batch_size != time.shape[0]:
            raise ValueError(f"Expected time batch size {time.shape[0]}, got {batch_size}")
        return time.to(device=device)

    model.sample_noise = sample_noise
    model.sample_time = sample_time
    try:
        yield
    finally:
        model.sample_noise = original_sample_noise
        model.sample_time = original_sample_time


@contextmanager
def deterministic_openpi_forward_preprocess(openpi_policy) -> Iterator[None]:
    """Disable OpenPI's training-time image augmentation only inside a parity forward block.

    OpenPI's `forward()` calls `_preprocess_observation(..., train=True)`, which can apply stochastic
    image augmentation. LeRobot's policy forward path does not apply that augmentation, so parity would
    otherwise compare two different image tensors rather than two model implementations. The context manager
    keeps the public `openpi_policy.forward(observation, ...)` call while making preprocessing deterministic.

    `yield` marks the body of the caller's `with` block. The `try/finally` restores the original method even
    if the assertion inside the block fails, so the temporary monkeypatch cannot leak into later tests.
    """

    original_preprocess_observation = openpi_policy._preprocess_observation

    def preprocess_observation(observation, *, train=True):
        return original_preprocess_observation(observation, train=False)

    openpi_policy._preprocess_observation = preprocess_observation
    try:
        yield
    finally:
        openpi_policy._preprocess_observation = original_preprocess_observation
