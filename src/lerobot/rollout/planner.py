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

"""External text backend: an OpenAI-compatible VLM answers the engine's text queries.

The prompt follows the in-context planner of *Steerable Vision-Language-Action Policies*
(arXiv 2602.13193, Fig. 22): the request, then past (images, applied command) pairs in
order, then the current images. The VLM replies with one JSON field.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from lerobot.annotations.steerable_pipeline.config import VlmConfig
from lerobot.annotations.steerable_pipeline.vlm_client import VlmClient, make_vlm_client
from lerobot.processor import RenderRuntimeMessagesStep
from lerobot.utils.constants import MESSAGES_RENDERED, QUERY_KIND, QUERY_TEXT

from .inference import EXTERNAL_HISTORY_DEFAULT, PolicyQuery, QueryKind

logger = logging.getLogger(__name__)

REPLY_FIELD = {QueryKind.VQA: "answer", QueryKind.NEXT_SUBTASK: "instruction"}

# scene and previous_command only make the model reason before it answers, parse_reply drops them
REPLY_FORMAT = {
    QueryKind.VQA: 'Reply as JSON: {"answer": "<your answer>"}',
    QueryKind.NEXT_SUBTASK: (
        "Reply as one JSON object with exactly these three keys, in this order, all filled in:\n"
        '"scene": one short sentence on the scene now and what changed since the previous observation;\n'
        '"previous_command": one of "completed", "in progress", "failed", "none";\n'
        '"instruction": the next instruction, or "done" when every object named in the request is already in '
        "its goal state. Never reply with the instruction alone."
    ),
}

SUBTASK_RULES = (
    "Each earlier observation is followed by the command the robot was given right after it, and the next "
    "observation shows what the robot did with that command. A command in the history was only sent to the "
    "robot; judge whether it was completed from the images alone.\n"
    "Think in this order: describe the current scene and what changed since the previous observation; decide "
    "whether the previous command is completed, still in progress, or failed; then choose the next instruction. "
    "Repeat the previous instruction word for word while the robot is still working on it. Change it only when "
    "its object is visibly in the goal state, or when the same instruction produced no visible change over two "
    "or more consecutive observations. A change means picking an instruction whose object is not yet in its "
    "goal state — re-issuing the stalled instruction counts. Never pick an instruction whose object is already "
    'in its goal state; when every object named in the request is done, answer "done".\n'
    "Identify objects by their shape and printed label in the images; the command text does not tell you which "
    "object moved."
)


@dataclass
class PlannerConfig(VlmConfig):
    """VLM client settings plus the planner's own knobs.

    ``instructions`` is an optional closed vocabulary: when set, a next-subtask reply must be
    one of these strings verbatim. ``history`` is the number of past (observation, applied
    command) pairs shown to the VLM; zero disables it.
    """

    auto_serve: bool = False
    system_prompt: str = "Help operate the robot using its current camera images."
    instructions: list[str] = field(default_factory=list)
    history: int = EXTERNAL_HISTORY_DEFAULT

    def __post_init__(self) -> None:
        if self.history < 0:
            raise ValueError("planner.history must be non-negative")


class VlmPlanner:
    """Asks a vision-language model what the robot should do, from its camera images.

    The engine calls an instance like a function. Each call sends one fresh chat request and
    returns the model's answer as text. Nothing is kept between calls.

    Two kinds of question are handled. A visual question returns the answer. A next-subtask
    question returns the instruction the robot should follow next.
    """

    def __init__(
        self,
        config: PlannerConfig,
        robot_type: str,
        runtime_messages: RenderRuntimeMessagesStep | None = None,
        client: VlmClient | None = None,
    ) -> None:
        self.config = config
        self.robot_type = robot_type
        self.runtime_messages = runtime_messages
        self.client = client or make_vlm_client(config)

    def __call__(self, obs_processed: dict, query: PolicyQuery, task: str) -> str:
        messages = self.build_messages(obs_processed, query, task)
        reply = self.client.generate_json([messages])[0]
        return self.parse_reply(reply, query, task)

    def build_messages(self, obs_processed: dict, query: PolicyQuery, task: str) -> list[dict]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if query.kind is QueryKind.NEXT_SUBTASK:
            messages += self.recipe_turns(query)
        messages.append({"role": "user", "content": self.build_user_content(obs_processed, query, task)})
        return messages

    def recipe_turns(self, query: PolicyQuery) -> list[dict]:
        """The turns the checkpoint was trained with for this request, rendered from its recipe."""
        if self.runtime_messages is None:
            return []
        rendered = self.runtime_messages.complementary_data(
            {QUERY_KIND: query.kind.value, QUERY_TEXT: query.text}
        )
        return rendered[MESSAGES_RENDERED]

    def build_user_content(self, obs_processed: dict, query: PolicyQuery, task: str) -> list[dict]:
        content = [text_block(self.request_text(query, task))]
        for index, (observation, instruction) in enumerate(query.history, start=1):
            content += self.observation_blocks(f"Observation {index}:", observation)
            content.append(text_block(f"Command given after observation {index}: {instruction}"))
        content += self.observation_blocks("Current observation:", obs_processed)
        return content

    def request_text(self, query: PolicyQuery, task: str) -> str:
        lines = [f"Robot: {self.robot_type}", f"Current instruction: {task}", f"Request: {query.text}"]
        if query.kind is QueryKind.NEXT_SUBTASK:
            lines.append(SUBTASK_RULES)
            if self.config.instructions:
                lines.append(
                    f"Choose exactly one of these allowed instructions: {self.config.instructions!r}"
                )
        lines.append(REPLY_FORMAT[query.kind])
        return "\n".join(lines)

    def observation_blocks(self, label: str, obs_processed: dict) -> list[dict]:
        """One label, then a caption and an image per selected camera."""
        blocks = [text_block(label)]
        for key, value in obs_processed.items():
            if self.config.camera_key is not None and key != self.config.camera_key:
                continue
            if isinstance(value, np.ndarray) and value.ndim == 3:
                blocks.append(text_block(f"Camera: {key}"))
                blocks.append({"type": "image", "image": Image.fromarray(value)})
        return blocks

    def parse_reply(self, reply: object, query: PolicyQuery, task: str) -> str:
        """The reply field as text; a next-subtask reply of ``done`` holds the current instruction."""
        reply_field = REPLY_FIELD[query.kind]
        logger.info("Planner reply (%s): %r", query.kind.value, reply)
        text = reply.get(reply_field) if isinstance(reply, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Planner returned no non-empty {reply_field}: {reply!r}")
        text = text.strip()
        if query.kind is QueryKind.NEXT_SUBTASK and text == "done":
            return task
        if (
            query.kind is QueryKind.NEXT_SUBTASK
            and self.config.instructions
            and text not in self.config.instructions
        ):
            raise ValueError(f"Planner instruction is outside the allowed list: {text!r}")
        return text


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}
