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

"""Checkpoint-compatible native RynnValueLang Transformers implementation."""

from transformers import AutoConfig, AutoModel, AutoProcessor

from . import attention_impl as attention_impl
from .configuration_rynn_value_lang import (
    RynnValueLangConfig,
    ValueHeadConfig,
    ValueTokenizerConfig,
)
from .conversations import (
    ConversationBuilder,
    InterleavedHistoryConversationBuilder,
    build_conversation_builder,
)
from .modeling_rynn_value_lang import RynnValueLangModel, RynnValueLangOutputWithPast
from .processing_rynn_value_lang import RynnValueLangProcessor

AutoConfig.register("rynn_value_lang", RynnValueLangConfig, exist_ok=True)
AutoModel.register(RynnValueLangConfig, RynnValueLangModel, exist_ok=True)
AutoProcessor.register(RynnValueLangConfig, RynnValueLangProcessor, exist_ok=True)

__all__ = [
    "ConversationBuilder",
    "InterleavedHistoryConversationBuilder",
    "RynnValueLangConfig",
    "RynnValueLangModel",
    "RynnValueLangOutputWithPast",
    "RynnValueLangProcessor",
    "ValueHeadConfig",
    "ValueTokenizerConfig",
    "attention_impl",
    "build_conversation_builder",
]
