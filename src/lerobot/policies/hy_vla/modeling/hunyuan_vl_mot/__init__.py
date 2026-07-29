# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""HunYuanVL-MoT model classes.

Registers the ``hunyuan_vl_mot`` model family into the HuggingFace
transformers ``Auto*`` registries at import time so that standard
``AutoConfig`` and model entrypoints work with ``tencent/HY-Embodied-*``
checkpoints. LeRobot supplies the policy preprocessing pipeline, so the
upstream processor is intentionally not bundled.

Implementation note: the original upstream ``__init__.py`` used
``transformers.utils._LazyModule`` which replaces ``sys.modules[__name__]``
and therefore prevents any code appearing after the replacement from
running on the actual module object the user receives. We instead use
eager imports plus an idempotent ``Auto*.register(...)`` call -- the
same pattern as the upstream parent package ``hunyuan_vla/__init__.py``.
"""

from contextlib import suppress

from .configuration_hunyuan_vl_mot import (
    HunYuanVLMoTConfig,
    HunYuanVLMoTTextConfig,
    HunYuanVLMoTVisionConfig,
)
from .modeling_hunyuan_vl_mot import (
    HunYuanVLMoTForConditionalGeneration,
    HunYuanVLMoTModel,
    HunYuanVLMoTPreTrainedModel,
)


def _register_hunyuan_vl_mot() -> None:
    """Register HunYuanVL-MoT into the transformers Auto* registries.

    Idempotent: safe to call multiple times. Duplicate-registration errors
    raised by ``Auto*.register`` are swallowed via ``contextlib.suppress``,
    which is the documented way to make these helpers re-entrant under
    Jupyter autoreload, DDP fork-children, and other re-import scenarios.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForImageTextToText,
    )

    # AutoConfig is keyed by the ``model_type`` string in ``config.json``.
    # HunYuanVLMoTConfig.model_type == "hunyuan_vl_mot".
    with suppress(ValueError):
        AutoConfig.register("hunyuan_vl_mot", HunYuanVLMoTConfig)

    # Base-class auto model (no LM head). Used when the checkpoint's
    # ``architectures`` does not point at the conditional-generation head.
    with suppress(ValueError):
        AutoModel.register(HunYuanVLMoTConfig, HunYuanVLMoTModel)

    # Image-text-to-text auto model: matches HY-Embodied-0.5's
    # ``architectures: ["HunYuanVLMoTForConditionalGeneration"]`` and is
    # what the LeRobot Hy-VLA model uses to instantiate the VLM half of the
    # dual tower.
    with suppress(ValueError):
        AutoModelForImageTextToText.register(HunYuanVLMoTConfig, HunYuanVLMoTForConditionalGeneration)


# Register at import time so importing this LeRobot module unlocks the
# standard Transformers AutoModel entrypoints for HY-Embodied checkpoints.
_register_hunyuan_vl_mot()


__all__ = [
    "HunYuanVLMoTConfig",
    "HunYuanVLMoTTextConfig",
    "HunYuanVLMoTVisionConfig",
    "HunYuanVLMoTModel",
    "HunYuanVLMoTForConditionalGeneration",
    "HunYuanVLMoTPreTrainedModel",
    "_register_hunyuan_vl_mot",
]
