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
"""Steerable annotation pipeline producing ``language_persistent`` and
``language_events`` columns for LeRobot datasets.

The pipeline is decomposed into three independently runnable modules whose
outputs are staged per-episode before a final parquet rewrite:

- :mod:`.modules.plan_subtasks_memory` (the ``plan`` module) — persistent styles
- :mod:`.modules.interjections_and_speech` (the ``interjections`` module) — event styles + speech
- :mod:`.modules.general_vqa` (the ``vqa`` module) — event-style VQA pairs
"""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports

from .config import AnnotationPipelineConfig

# These import lerobot.datasets, so each is imported the first time it is used.
if TYPE_CHECKING:
    from .validator import StagingValidator, ValidationReport
    from .writer import LanguageColumnsWriter
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "StagingValidator": ".validator.StagingValidator",
            "ValidationReport": ".validator.ValidationReport",
            "LanguageColumnsWriter": ".writer.LanguageColumnsWriter",
        },
    )

__all__ = [
    "AnnotationPipelineConfig",
    "LanguageColumnsWriter",
    "StagingValidator",
    "ValidationReport",
]
