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

from ..manifest import ProcessorSpec


def build_normalization_processor_specs(
    input_groups: list[tuple[str, list[str]]],
    output_groups: list[tuple[str, list[str]]],
    *,
    artifact: str | None,
) -> tuple[list[ProcessorSpec], list[ProcessorSpec]]:
    """Build executable normalize/denormalize specs from mode groups.

    Args:
        input_groups: List of ``(mode, features)`` tuples where ``mode`` is the
            normalisation mode and ``features`` is the list of input feature keys.
        output_groups: List of ``(mode, features)`` tuples where ``mode`` is the
            normalisation mode and ``features`` is the list of output feature keys.
        artifact: Relative path to the stats file shared by all specs.

    Returns:
        Ordered ``(preprocessors, postprocessors)`` specs ready for the manifest.
    """
    preprocessors = [
        ProcessorSpec(type="normalize", mode=mode, artifact=artifact, features=features)
        for mode, features in input_groups
    ]
    postprocessors = [
        ProcessorSpec(type="denormalize", mode=mode, artifact=artifact, features=features)
        for mode, features in output_groups
    ]
    return preprocessors, postprocessors


__all__ = [
    "build_normalization_processor_specs",
]
