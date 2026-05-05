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

from __future__ import annotations

from typing import Any

from ..manifest import ProcessorSpec

PI05_TOKENIZER_NAME = "google/paligemma-3b-pt-224"


def build_pi05_processor_specs(
    config: Any,
    *,
    tokenizer_artifact: str | None = None,
) -> tuple[list[ProcessorSpec], list[ProcessorSpec]]:
    """Build the ordered preprocessor and postprocessor spec lists for PI05.

    Constructs the full pipeline of PI05-specific transforms:

    - **Preprocessors** (in order): relative-actions conversion,
      state padding, tokenisation.
    - **Postprocessors**: absolute-actions conversion.

    Args:
        config: PI05 policy config object with attributes
            ``use_relative_actions``, ``relative_exclude_joints``,
            ``action_feature_names``, ``max_state_dim``,
            ``tokenizer_max_length``.
        tokenizer_artifact: Relative path to the bundled tokenizer directory,
            or ``None`` if not bundled.

    Returns:
        A 2-tuple ``(preprocessors, postprocessors)`` of processor specs.
    """
    preprocessors = [
        ProcessorSpec(
            type="relative_actions",
            extra_params={
                "enabled": getattr(config, "use_relative_actions", False),
                "exclude_joints": list(getattr(config, "relative_exclude_joints", [])),
                "action_names": getattr(config, "action_feature_names", None),
            },
        ),
        ProcessorSpec(type="pi05_prepare_state", extra_params={"max_state_dim": config.max_state_dim}),
        ProcessorSpec(
            type="tokenize",
            artifact=tokenizer_artifact,
            extra_params={
                "tokenizer_name": PI05_TOKENIZER_NAME,
                "max_length": config.tokenizer_max_length,
                "padding_side": "right",
                "padding": "max_length",
                "truncation": True,
            },
        ),
    ]
    postprocessors = [
        ProcessorSpec(
            type="absolute_actions",
            extra_params={"enabled": getattr(config, "use_relative_actions", False)},
        )
    ]
    return preprocessors, postprocessors


__all__ = [
    "PI05_TOKENIZER_NAME",
    "build_pi05_processor_specs",
]
