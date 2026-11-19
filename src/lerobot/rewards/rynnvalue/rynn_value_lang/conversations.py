# Copyright 2026 Alibaba DAMO Academy and The HuggingFace Inc. team. All rights reserved.
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

# conversations.py
"""Conversation builders for RynnValueLangProcessor.

Each builder turns ``(instruction, num_frames, ...)`` into the chat-template
content list and tells the processor how many ``<value>`` / ``<relative_value>``
tokens to expect via ``progress_value_count``. The interleaved-history layout
predicts the remaining time to task completion with per-frame ``<value>``
tokens, the frame-to-frame time delta with ``<relative_value>``, and appends a
trailing analysis block.

The processor stays agnostic to the prompt layout — it only reads
``progress_value_count`` to align target tensors.
"""

from abc import ABC, abstractmethod


def _meta_block(
    robot_description: str | None,
    camera_description: str | None,
) -> list[dict]:
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
    """Strategy object that owns the prompt layout for one processor.

    Subclasses implement ``build_progress`` and expose
    ``progress_value_count(num_frames)`` so the processor knows how many
    ``<value>`` tokens each progress sample contains.
    """

    name: str = "base"

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
        self.value_token_repeat = int(value_token_repeat)
        if relative_value_token_repeat < 1:
            raise ValueError(f"relative_value_token_repeat must be >= 1, got {relative_value_token_repeat}")
        self.relative_value_token_repeat = int(relative_value_token_repeat)

    def _maybe_meta(
        self,
        robot_description: str | None,
        camera_description: str | None,
    ) -> list[dict]:
        if not self.use_meta:
            return []
        if robot_description is None and camera_description is None:
            raise ValueError(
                "use_meta=True requires at least one of `robot_description` "
                "or `camera_description` to be provided."
            )
        return _meta_block(robot_description, camera_description)

    def _instruction_block(self, instruction: str) -> list[dict]:
        return [
            {"type": "text", "text": f"The agent is performing the following task: {instruction}."},
        ]

    def _append_value_tokens(self, content: list[dict]) -> None:
        """Append ``value_token_repeat`` copies of ``<value>``.

        Repeated tokens supply extra positions per prediction slot; the model
        mean-pools their hidden states before feeding the value head.
        """
        for _ in range(self.value_token_repeat):
            content.append({"type": "text", "text": f"{self.value_token}"})

    def _append_relative_value_tokens(self, content: list[dict]) -> None:
        """Append ``relative_value_token_repeat`` copies of ``<relative_value>``.

        Mirrors :meth:`_append_value_tokens` for the relative head; the model
        mean-pools the R repeated hidden states into one prediction per slot.
        """
        for _ in range(self.relative_value_token_repeat):
            content.append({"type": "text", "text": f"{self.relative_value_token}"})

    @abstractmethod
    def progress_value_count(self, num_frames: int) -> int:
        """Number of <value> prediction slots (targets) emitted by the progress prompt.

        Each slot may be represented by ``value_token_repeat`` consecutive
        ``<value>`` tokens in the prompt; the model pools them back down to
        one prediction per slot, so this returns the slot/target count, not
        the raw token count.
        """

    @abstractmethod
    def build_progress(self, **kwargs) -> list[dict]:
        """Return the chat-template content list for a progress sample."""


class InterleavedHistoryConversationBuilder(ConversationBuilder):
    """History layout with per-frame value + relative-value tokens + trailing analysis.

    Layout::

        [meta]
        The agent is performing the following task: {instruction}.
        Question: <RELATIVE_DISTANCE_QUESTION>
        Question: <PROGRESS_QUESTION>
        # per-frame, i = 0..N-1:
        [img]
        (if i > 0) <relative_value>×R
        <value>×R
        Analyze this trajectory in terms of video description, match, and success.
        Analysis:
        Video Description: {description}
        Match: Yes/No
        Success: Yes/No
    """

    name = "interleaved_history"

    RELATIVE_DISTANCE_QUESTION = (
        "For each frame after the first, what is the time delta from the previous frame?"
    )

    @staticmethod
    def _build_analysis_prompt(has_description: bool) -> str:
        lines = ["Analyze this trajectory. Provide:"]
        if has_description:
            lines.append("- Video Description: a brief description of what the agent is doing in the video.")
        lines.append("- Match: whether the video matches the stated task (Yes/No).")
        lines.append("- Success: whether the agent has completed the task (Yes/No).")
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
        content: list[dict] = []
        content.extend(self._maybe_meta(robot_description, camera_description))
        content.extend(self._instruction_block(instruction))
        content.append({"type": "text", "text": f"Question: {self.RELATIVE_DISTANCE_QUESTION}"})
        content.append({"type": "text", "text": f"Question: {self.PROGRESS_QUESTION}"})
        for i in range(num_frames):
            content.append({"type": "image"})
            if i > 0:
                self._append_relative_value_tokens(content)
            self._append_value_tokens(content)
        # Inference: always include all questions so the model sees full context.
        # Training: only include description question when description data exists.
        prompt_has_description = True if is_inference else bool(description)
        analysis_prompt = self._build_analysis_prompt(has_description=prompt_has_description)
        analysis = _analysis_block(
            description=description,
            match_answer=match_answer,
            success_answer=success_answer,
        )
        if not analysis and is_inference:
            # Inference: emit the analysis prompt questions together with
            # the "Analysis: \n" marker so generation has full context.
            content.append({"type": "text", "text": analysis_prompt})
            analysis = [{"type": "text", "text": "Analysis: \n"}]
        else:
            content.append({"type": "text", "text": analysis_prompt})
        content.extend(analysis)
        return content


_REGISTRY = {
    InterleavedHistoryConversationBuilder.name: InterleavedHistoryConversationBuilder,
}


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
