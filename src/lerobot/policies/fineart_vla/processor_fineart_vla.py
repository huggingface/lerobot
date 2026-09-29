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

"""FineART-VLA processors: recipe rendering, text tokenization, and the pre/post-processor factory.

Without a recipe the factory delegates to the standard PI0.5 pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ActionTokenizerProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

# Import directly to keep optional language dependencies out of ``lerobot.processor``.
from lerobot.processor.render_messages_processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from ..pi05.processor_pi05 import make_pi05_pre_post_processors
from .configuration_fineart_vla import FineARTVLAConfig

logger = logging.getLogger(__name__)


def discretize_state_str(state_row: Any) -> str:
    """Format one normalized state row with PI0.5's 256-bin convention."""
    arr = state_row.detach().cpu().numpy() if hasattr(state_row, "detach") else np.asarray(state_row)
    disc = np.digitize(arr, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    return " ".join(str(int(x)) for x in disc.reshape(-1).tolist())


def _state_row_at(state_all: Any, pos: int) -> Any:
    """Select the per-sample state row from a (possibly batched) state tensor."""
    if state_all is None:
        return None
    if hasattr(state_all, "ndim") and state_all.ndim >= 2:
        return state_all[pos]
    return state_all


def _content_to_text(content: Any) -> str:
    """Collapse a message's ``content`` (string or multimodal blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _flatten_say_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Move ``say`` tool calls into text markers that PaliGemma can learn."""
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return message
    say_texts: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if fn.get("name") != "say":
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                import json  # noqa: PLC0415

                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        text = args.get("text", "") if isinstance(args, dict) else ""
        if text:
            say_texts.append(str(text))
    new = dict(message)
    new.pop("tool_calls", None)
    if not say_texts:
        return new
    base = _content_to_text(new.get("content")).strip()
    marker = "".join(f"<say>{t}</say>" for t in say_texts)
    new["content"] = f"{base}\n{marker}" if base else marker
    return new


def _strip_blocks(message: dict[str, Any]) -> dict[str, Any]:
    """Flatten text blocks and drop image blocks handled by observation inputs."""
    new = dict(message)
    new.pop("stream", None)
    new.pop("target", None)
    content = new.get("content")
    if content is None:
        new["content"] = ""
    elif isinstance(content, str):
        pass
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        new["content"] = "\n".join(parts)
    else:
        new["content"] = str(content)
    return new


def _is_batched_messages(messages: Any) -> bool:
    return isinstance(messages, list) and bool(messages) and isinstance(messages[0], list)


_VQA_COORD_SCALE = 1000.0


def register_paligemma_loc_tokens(tokenizer: Any) -> Any:
    """Register PaliGemma's reserved ``<locDDDD>`` strings as single tokens.

    Without registration, the stock tokenizer splits each location into generic text pieces.
    """
    if "<loc0000>" in getattr(tokenizer, "added_tokens_encoder", {}):
        return tokenizer
    tokenizer.add_tokens([f"<loc{i:04d}>" for i in range(1024)])
    return tokenizer


def _loc_token(coord: float, scale: float = _VQA_COORD_SCALE) -> str:
    """PaliGemma ``<locNNNN>`` for a coord on a ``[0, scale]`` axis."""
    idx = round(float(coord) / scale * 1023) if scale > 0 else 0
    return f"<loc{max(0, min(1023, idx)):04d}>"


def _vqa_answer_to_loc(answer: dict[str, Any]) -> str | None:
    """Convert normalized bbox/keypoint answers to label-first PaliGemma locations.

    Label-first targets prevent location tokens from dominating every assistant turn; non-spatial answers return ``None``.
    """
    point = answer.get("point")
    if isinstance(point, list | tuple) and len(point) == 2 and "point_format" in answer:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        label = str(answer.get("label", "")).strip()
        if not label:
            return None
        return f"{label} {_loc_token(y)}{_loc_token(x)}"

    detections = answer.get("detections")
    if isinstance(detections, list) and detections:
        parts: list[str] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            box = det.get("bbox")
            if not (isinstance(box, list | tuple) and len(box) == 4):
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            label = str(det.get("label", "")).strip()
            if not label:
                continue
            toks = f"{_loc_token(y1)}{_loc_token(x1)}{_loc_token(y2)}{_loc_token(x2)}"
            parts.append(f"{label} {toks}")
        return " ; ".join(parts) if parts else None
    return None


def _messages_vqa_to_loc(
    messages: list[dict[str, Any]],
    target_indices: list[int],
) -> list[dict[str, Any]]:
    """Rewrite spatial VQA target JSON as camera-independent ``<loc>`` text."""
    if not target_indices:
        return messages
    out = list(messages)
    for idx in target_indices:
        if not (0 <= idx < len(out)):
            continue
        content = out[idx].get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            answer = json.loads(content)
        except (ValueError, TypeError):
            continue
        if not isinstance(answer, dict):
            continue
        loc_text = _vqa_answer_to_loc(answer)
        if loc_text is not None:
            out[idx] = {**out[idx], "content": loc_text}
    return out


def _format_messages(
    messages: list[dict[str, Any]],
    target_indices: list[int] | None = None,
    eos_token: str | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Build the flat PI0.5 prompt and each message's payload span.

    Supervised targets include EOS so generation learns when to stop.
    """
    targets = set(target_indices or [])
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        header = f"{role.capitalize()}: "
        body = content + eos_token if (eos_token and i in targets) else content
        full = header + body + "\n"
        start = cursor + len(header)
        end = start + len(body)
        parts.append(full)
        spans.append((start, end))
        cursor += len(full)
    return "".join(parts), spans


@dataclass
@ProcessorStepRegistry.register(name="fineart_vla_text_tokenizer")
class FineARTVLATextTokenizerStep(ProcessorStep):
    """Convert flat role-delimited messages into tokens and supervision masks."""

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_length: int = 200
    padding: str = "max_length"
    padding_side: str = "right"

    def __post_init__(self) -> None:
        self._tokenizer: Any = None

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "padding": self.padding,
            "padding_side": self.padding_side,
        }

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._tokenizer = register_paligemma_loc_tokens(AutoTokenizer.from_pretrained(self.tokenizer_name))
        return self._tokenizer

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        messages = complementary.get("messages_rendered") or complementary.get("messages") or []

        if not messages:
            tasks = complementary.get("task")
            if tasks is None:
                return transition
            tasks = [tasks] if isinstance(tasks, str) else list(tasks)
            messages = [[{"role": "user", "content": task}] for task in tasks]
            complementary = {
                **complementary,
                "message_streams": [["low_level"] for _ in tasks],
                "target_message_indices": [[] for _ in tasks],
            }

        tokenizer = self._ensure_tokenizer()
        state_all = (transition.get(TransitionKey.OBSERVATION) or {}).get(OBS_STATE)
        if _is_batched_messages(messages):
            encoded = [
                self._encode_messages(
                    tokenizer,
                    msg,
                    list(streams),
                    list(tgt_indices),
                    complementary,
                    state_row=_state_row_at(state_all, pos),
                )
                for pos, (msg, streams, tgt_indices) in enumerate(
                    zip(
                        messages,
                        complementary.get("message_streams") or [[] for _ in messages],
                        complementary.get("target_message_indices") or [[] for _ in messages],
                        strict=False,
                    )
                )
            ]
        else:
            encoded = [
                self._encode_messages(
                    tokenizer,
                    messages,
                    list(complementary.get("message_streams") or []),
                    list(complementary.get("target_message_indices") or []),
                    complementary,
                    state_row=_state_row_at(state_all, 0),
                )
            ]

        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        obs[OBS_LANGUAGE_TOKENS] = torch.stack([ids for ids, _, _, _, _ in encoded])
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.stack([attn for _, attn, _, _, _ in encoded])
        transition[TransitionKey.OBSERVATION] = obs

        transition[TransitionKey.COMPLEMENTARY_DATA] = {
            **complementary,
            "text_labels": torch.stack([labels for _, _, labels, _, _ in encoded]),
            "predict_actions": torch.stack([pred for _, _, _, pred, _ in encoded]),
        }
        return transition

    def _encode_messages(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        message_streams: list[str | None],
        target_indices: list[int],
        complementary: dict[str, Any],
        state_row: Any = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
        messages = _messages_vqa_to_loc(messages, target_indices)

        messages = [_strip_blocks(_flatten_say_tool_calls(m)) for m in messages]
        # Only low-level prompts carry PI0.5-style proprioception.
        if state_row is not None and any(s == "low_level" for s in message_streams):
            state_str = discretize_state_str(state_row)
            for m in reversed(messages):
                if m.get("role") == "user":
                    base = _content_to_text(m.get("content", ""))
                    m["content"] = f"{base}, State: {state_str};"
                    break
        prompt, spans = _format_messages(messages, target_indices, getattr(tokenizer, "eos_token", None))
        if "message_streams" not in complementary and not message_streams and not target_indices:
            # Runtime queries contain semantic prompt turns without training labels.
            # Match the prefix immediately preceding an assistant target in training.
            prompt += "Assistant:"

        encoded = tokenizer(
            prompt,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding_side=self.padding_side,
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].bool()
        offsets = encoded["offset_mapping"][0]

        labels = torch.full_like(input_ids, fill_value=-100)
        for idx in target_indices:
            if idx >= len(spans):
                continue
            char_start, char_end = spans[idx]
            for token_pos in range(input_ids.shape[0]):
                if not attention_mask[token_pos]:
                    continue
                tok_start, tok_end = int(offsets[token_pos, 0]), int(offsets[token_pos, 1])
                if tok_end <= char_start or tok_start >= char_end:
                    continue
                labels[token_pos] = input_ids[token_pos]

        predict_actions = torch.tensor(
            bool(any(s == "low_level" for s in message_streams)),
            dtype=torch.bool,
        )
        return input_ids, attention_mask, labels, predict_actions, prompt

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


if TYPE_CHECKING:
    from lerobot.datasets.recipe import TrainingRecipe


def make_fineart_vla_pre_post_processors(
    config: FineARTVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_repo_id: str | None = None,
    dataset_root: str | None = None,
    dataset_revision: str | None = None,
    episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build FineART-VLA's pre/post-processor pipelines.

    Falls through to π0.5's stock pipeline when ``recipe_path`` is unset. ``dataset_meta``
    (forwarded by the policy factory) supplies the dataset source for FAST tokenizer fitting.
    """
    if dataset_meta is not None:
        dataset_repo_id = dataset_repo_id or getattr(dataset_meta, "repo_id", None)
        dataset_root = dataset_root or getattr(dataset_meta, "root", None)
        dataset_revision = dataset_revision or getattr(dataset_meta, "revision", None)
    if config.recipe is None:
        if getattr(config, "enable_fast_action_loss", False):
            raise ValueError("FineART-VLA FAST action loss requires recipe_path to build action supervision.")
        return make_pi05_pre_post_processors(config, dataset_stats=dataset_stats)

    if config.input_features is None or config.output_features is None or config.device is None:
        raise ValueError("FineART-VLA processors require input/output features and a device")

    from lerobot.datasets.recipe import TrainingRecipe  # recipes need the dataset extras

    recipe = TrainingRecipe.from_dict(config.recipe)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        RenderRuntimeMessagesStep(recipe=recipe),
        RenderTrainingMessagesStep(recipe=recipe),
        FineARTVLATextTokenizerStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
        ),
    ]

    # Add FAST action-token supervision only when explicitly enabled.
    if getattr(config, "enable_fast_action_loss", False):
        from .fit_fast_tokenizer import resolve_fast_tokenizer  # noqa: PLC0415

        input_steps.append(
            ActionTokenizerProcessorStep(
                action_tokenizer_name=resolve_fast_tokenizer(
                    config,
                    dataset_repo_id,
                    dataset_root,
                    dataset_stats,
                    dataset_revision,
                    episodes,
                    exclude_episodes,
                ),
                max_action_tokens=config.max_action_tokens,
                fast_skip_tokens=config.fast_skip_tokens,
                paligemma_tokenizer_name="google/paligemma-3b-pt-224",
                allow_truncation=True,
            )
        )

    input_steps.append(DeviceProcessorStep(device=config.device))

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions,
            relative_step=relative_step,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _load_recipe(path_str: str) -> TrainingRecipe:
    """Resolve ``path_str`` to a ``TrainingRecipe``.

    Accepts an absolute path or a path relative to
    ``src/lerobot/configs/``.
    """
    from lerobot.datasets.recipe import TrainingRecipe  # recipes need the dataset extras

    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        configs_dir = Path(__file__).parents[2] / "configs"
        candidate = configs_dir / path_str
        if candidate.exists():
            p = candidate
    return TrainingRecipe.from_yaml(p)
