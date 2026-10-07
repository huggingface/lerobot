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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT = (
    "Your task is {instruction}.\n\nFrom main-view observations, infer the robot arm's motion."
)
DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT = (
    "Then use observations and the inferred motion to produce the robot policy actions."
)


def _format_prompt(prompt: str, *, instruction: str) -> str:
    return str(prompt).replace("{instruction}", str(instruction))


def build_prompt_segments(
    *,
    instruction: str,
    placeholder_token: str,
    act_queries: int,
    flow_queries: int,
    cot_prompt_before_wrist: str = DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
    cot_prompt_after_wrist: str = DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
) -> tuple[str, str]:
    act_placeholder_block = " ".join([str(placeholder_token)] * int(act_queries))
    flow_placeholder_block = " ".join([str(placeholder_token)] * int(flow_queries))
    prompt_before_wrist = _format_prompt(cot_prompt_before_wrist, instruction=instruction)
    prompt_after_wrist = _format_prompt(cot_prompt_after_wrist, instruction=instruction)
    text_before_wrist = (
        f"{prompt_before_wrist}\n{act_placeholder_block}"
        if act_placeholder_block
        else str(prompt_before_wrist)
    )
    text_after_wrist = (
        f"{prompt_after_wrist}\n{flow_placeholder_block}"
        if prompt_after_wrist and flow_placeholder_block
        else str(prompt_after_wrist or flow_placeholder_block)
    )
    return text_before_wrist, text_after_wrist


def build_qwenvl_messages(
    *,
    images: Sequence[Sequence[Any]],
    wrist_images: Sequence[Sequence[Any] | None],
    instructions: Sequence[str],
    placeholder_token: str,
    act_queries: int,
    flow_queries: int,
    cot_prompt_before_wrist: str = DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
    cot_prompt_after_wrist: str = DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
) -> list[list[dict[str, Any]]]:
    if len(images) != len(instructions):
        raise ValueError("Images and instructions must have the same length.")
    if len(wrist_images) != len(instructions):
        raise ValueError("Wrist images and instructions must have the same length.")

    messages = []
    for imgs, wrist_imgs, instruction in zip(images, wrist_images, instructions, strict=True):
        content = [{"type": "image", "image": img} for img in imgs]
        text_before_wrist, text_after_wrist = build_prompt_segments(
            instruction=instruction,
            placeholder_token=placeholder_token,
            act_queries=act_queries,
            flow_queries=flow_queries,
            cot_prompt_before_wrist=cot_prompt_before_wrist,
            cot_prompt_after_wrist=cot_prompt_after_wrist,
        )
        if text_before_wrist:
            content.append({"type": "text", "text": text_before_wrist})
        if wrist_imgs:
            content.extend({"type": "image", "image": img} for img in wrist_imgs)
        if text_after_wrist:
            content.append({"type": "text", "text": text_after_wrist})
        messages.append([{"role": "user", "content": content}])
    return messages
