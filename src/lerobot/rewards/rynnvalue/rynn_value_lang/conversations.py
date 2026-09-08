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

"""Checkpoint-compatible conversation builders for RynnValue."""

from abc import ABC, abstractmethod


def _meta_block(robot_description: str | None, camera_description: str | None) -> list[dict]:
    sentences = []
    if robot_description is not None:
        sentences.append(f"The agent is {robot_description}.")
    if camera_description is not None:
        sentences.append(f"The observation is captured from {camera_description}.")
    return [{"type": "text", "text": " ".join(sentences)}]


def _analysis_block(
    description: str | None = None,
    match_answer: str | None = None,
    success_answer: str | None = None,
) -> list[dict]:
    lines = []
    if description:
        lines.append(f"- Video Description: {description}")
    if match_answer is not None:
        lines.append(f"- Match: {match_answer}")
    if success_answer is not None:
        lines.append(f"- Success: {success_answer}")
    if not lines:
        return []
    return [
        {"type": "text", "text": "Analysis: \n"},
        {"type": "text", "text": "\n".join(lines)},
    ]


class ConversationBuilder(ABC):
    name = "base"
    PROGRESS_QUESTION = "Estimate the minimum remaining time in seconds until the agent completes the task."

    def __init__(
        self,
        value_token: str,
        relative_value_token: str | None,
        use_meta: bool = False,
        value_token_repeat: int = 1,
        relative_value_token_repeat: int = 1,
    ):
        self.value_token = value_token
        self.relative_value_token = relative_value_token
        self.use_meta = use_meta
        if value_token_repeat < 1:
            raise ValueError(f"value_token_repeat must be >= 1, got {value_token_repeat}")
        if relative_value_token_repeat < 1:
            raise ValueError(f"relative_value_token_repeat must be >= 1, got {relative_value_token_repeat}")
        self.value_token_repeat = int(value_token_repeat)
        self.relative_value_token_repeat = int(relative_value_token_repeat)

    def _maybe_meta(self, robot_description: str | None, camera_description: str | None) -> list[dict]:
        if not self.use_meta:
            return []
        if robot_description is None and camera_description is None:
            raise ValueError(
                "use_meta=True requires at least one of `robot_description` or "
                "`camera_description` to be provided."
            )
        return _meta_block(robot_description, camera_description)

    @staticmethod
    def _instruction_block(instruction: str) -> list[dict]:
        return [{"type": "text", "text": f"The agent is performing the following task: {instruction}."}]

    def _append_value_tokens(self, content: list[dict]) -> None:
        for _ in range(self.value_token_repeat):
            content.append({"type": "text", "text": self.value_token})

    def _append_relative_value_tokens(self, content: list[dict]) -> None:
        for _ in range(self.relative_value_token_repeat):
            content.append({"type": "text", "text": self.relative_value_token})

    @abstractmethod
    def progress_value_count(self, num_frames: int) -> int: ...

    @abstractmethod
    def build_progress(self, **kwargs) -> list[dict]: ...


class InterleavedHistoryConversationBuilder(ConversationBuilder):
    name = "interleaved_history"
    RELATIVE_DISTANCE_QUESTION = (
        "For each frame after the first, what is the time delta from the previous frame?"
    )

    @staticmethod
    def _build_analysis_prompt(has_description: bool) -> str:
        lines = ["Analyze this trajectory. Provide:"]
        if has_description:
            lines.append("- Video Description: a brief description of what the agent is doing in the video.")
        lines.extend(
            [
                "- Match: whether the video matches the stated task (Yes/No).",
                "- Success: whether the agent has completed the task (Yes/No).",
            ]
        )
        return "\n".join(lines)

    def progress_value_count(self, num_frames: int) -> int:
        return num_frames

    def build_progress(
        self,
        instruction: str,
        description: str | None,
        num_frames: int,
        robot_description: str | None,
        camera_description: str | None,
        match_answer: str | None = None,
        success_answer: str | None = None,
        is_inference: bool = False,
    ) -> list[dict]:
        if self.relative_value_token is None:
            raise ValueError(
                "interleaved_history conversation requires a registered "
                "<relative_value> token (enable relative_value_head_config)."
            )
        content = self._maybe_meta(robot_description, camera_description)
        content.extend(self._instruction_block(instruction))
        content.append({"type": "text", "text": f"Question: {self.RELATIVE_DISTANCE_QUESTION}"})
        content.append({"type": "text", "text": f"Question: {self.PROGRESS_QUESTION}"})
        for i in range(num_frames):
            content.append({"type": "image"})
            if i > 0:
                self._append_relative_value_tokens(content)
            self._append_value_tokens(content)
        analysis_prompt = self._build_analysis_prompt(
            has_description=True if is_inference else bool(description)
        )
        analysis = _analysis_block(description, match_answer, success_answer)
        content.append({"type": "text", "text": analysis_prompt})
        if not analysis and is_inference:
            analysis = [{"type": "text", "text": "Analysis: \n"}]
        content.extend(analysis)
        return content


_REGISTRY = {InterleavedHistoryConversationBuilder.name: InterleavedHistoryConversationBuilder}


def build_conversation_builder(
    style: str,
    *,
    value_token: str,
    relative_value_token: str | None,
    use_meta: bool = False,
    value_token_repeat: int = 1,
    relative_value_token_repeat: int = 1,
) -> ConversationBuilder:
    if style not in _REGISTRY:
        raise ValueError(f"Unknown conversation_style: {style!r}. Expected one of {sorted(_REGISTRY)}.")
    return _REGISTRY[style](
        value_token=value_token,
        relative_value_token=relative_value_token,
        use_meta=use_meta,
        value_token_repeat=value_token_repeat,
        relative_value_token_repeat=relative_value_token_repeat,
    )


__all__ = [
    "ConversationBuilder",
    "InterleavedHistoryConversationBuilder",
    "build_conversation_builder",
]
