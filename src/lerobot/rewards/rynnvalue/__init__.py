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

"""Native, checkpoint-compatible RynnValue core."""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _transformers_available

from .configuration_rynnvalue import RynnValueConfig
from .conversations import (
    ConversationBuilder,
    InterleavedHistoryConversationBuilder,
    build_conversation_builder,
)
from .value_heads import BroValueHead, LinearValueHead, build_value_head
from .value_tokenizer import ValueTokenizer, to_symexp, to_symlog

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModel, AutoProcessor

    from .configuration_rynn_value_lang import (
        RynnValueLangConfig,
        ValueHeadConfig,
        ValueTokenizerConfig,
    )
    from .modeling_rynn_value_lang import RynnValueLangModel, RynnValueLangOutputWithPast
    from .modeling_rynnvalue import RynnValueRewardModel
    from .processing_rynn_value_lang import RynnValueLangProcessor
    from .processor_rynnvalue import make_rynnvalue_pre_post_processors

    AutoConfig.register("rynn_value_lang", RynnValueLangConfig, exist_ok=True)
    AutoModel.register(RynnValueLangConfig, RynnValueLangModel, exist_ok=True)
    AutoProcessor.register(RynnValueLangConfig, RynnValueLangProcessor, exist_ok=True)

    __all__ = [
        "BroValueHead",
        "ConversationBuilder",
        "InterleavedHistoryConversationBuilder",
        "LinearValueHead",
        "RynnValueLangConfig",
        "RynnValueLangModel",
        "RynnValueLangOutputWithPast",
        "RynnValueLangProcessor",
        "RynnValueConfig",
        "RynnValueRewardModel",
        "ValueHeadConfig",
        "ValueTokenizer",
        "ValueTokenizerConfig",
        "build_conversation_builder",
        "build_value_head",
        "make_rynnvalue_pre_post_processors",
        "to_symexp",
        "to_symlog",
    ]
else:
    __all__ = [
        "BroValueHead",
        "ConversationBuilder",
        "InterleavedHistoryConversationBuilder",
        "LinearValueHead",
        "RynnValueConfig",
        "ValueTokenizer",
        "build_conversation_builder",
        "build_value_head",
        "to_symexp",
        "to_symlog",
    ]
