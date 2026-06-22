#!/usr/bin/env python

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

"""Adapter step that flattens rendered chat messages back into a task string.

Bridges RenderMessagesStep (which outputs structured messages) to policies
that expect a plain task string in complementary_data["task"] (e.g. PI05).
"""

from __future__ import annotations

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register(name="rendered_messages_to_task")
class RenderedMessagesToTaskStep(ComplementaryDataProcessorStep):
    """Extract user-role message content from rendered messages into the task string.

    After RenderMessagesStep renders a recipe into structured messages, this
    step extracts content from all user-role messages, joins them, and writes
    the result to complementary_data["task"]. This allows downstream steps
    (like Pi05PrepareStateTokenizerProcessorStep) to consume the
    advantage-conditioned prompt without modification.

    No-ops when the "messages" key is absent (backward compatible with
    pipelines that don't use language annotations).
    """

    def complementary_data(self, complementary_data: dict) -> dict:
        messages = complementary_data.get("messages")
        if messages is None:
            return complementary_data

        user_parts = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    user_parts.append(content)
                elif isinstance(content, list):
                    # HF multimodal blocks: extract text blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                user_parts.append(text)

        new_complementary_data = dict(complementary_data)

        if user_parts:
            task = complementary_data.get("task")
            # Wrap in list if the original task was a list (batched)
            joined = "\n".join(user_parts)
            if isinstance(task, list):
                new_complementary_data["task"] = [joined] * len(task)
            else:
                new_complementary_data["task"] = joined

        # Remove consumed rendering outputs
        new_complementary_data.pop("messages", None)
        new_complementary_data.pop("message_streams", None)
        new_complementary_data.pop("target_message_indices", None)

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
