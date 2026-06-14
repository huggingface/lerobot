from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT = (
    "Your task is {instruction}.\n\n"
    "From main-view observations, infer the robot arm's motion."
)
DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT = (
    "Then use observations and the inferred motion to produce the robot policy actions."
)


def _format_prompt(prompt: str, *, instruction: str) -> str:
    return str(prompt).replace("{instruction}", str(instruction))


def resolve_cot_prompt_templates_from_config(config: Any) -> tuple[str, str]:
    dataset_cfg = getattr(getattr(config, "datasets", None), "vla_data", None)
    if dataset_cfg is None:
        return DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT, DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT

    cot_prompt_before_wrist = (
        dataset_cfg.get("CoT_prompt_before_wrist")
        if "CoT_prompt_before_wrist" in dataset_cfg
        else DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT
    )
    cot_prompt_after_wrist = (
        dataset_cfg.get("CoT_prompt_after_wrist")
        if "CoT_prompt_after_wrist" in dataset_cfg
        else DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT
    )
    return str(cot_prompt_before_wrist), str(cot_prompt_after_wrist)


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
        f"{prompt_before_wrist}\n{act_placeholder_block}" if act_placeholder_block else str(prompt_before_wrist)
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
    for imgs, wrist_imgs, instruction in zip(images, wrist_images, instructions):
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


class LatentWorldPolicyVLMAdapter(nn.Module):
    """Policy-side adapter for Qwen-VL prompt/processor encoding."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        config: Any,
        placeholder_token: str,
        act_queries: int,
        flow_queries: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config
        self.placeholder_token = str(placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)

    def build_qwenvl_inputs(
        self,
        *,
        images: Sequence[Sequence[Any]],
        wrist_images: Sequence[Sequence[Any] | None],
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        cot_prompt_before_wrist, cot_prompt_after_wrist = resolve_cot_prompt_templates_from_config(self.config)
        messages = build_qwenvl_messages(
            images=images,
            wrist_images=wrist_images,
            instructions=instructions,
            placeholder_token=self.placeholder_token,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            cot_prompt_before_wrist=cot_prompt_before_wrist,
            cot_prompt_after_wrist=cot_prompt_after_wrist,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
