#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

# Register DM05 processor steps for PolicyProcessorPipeline.from_pretrained.
from . import processor_dm05 as processor_dm05  # noqa: F401
from .configuration_dm05 import DM05Config


def __getattr__(name: str):
    if name == "DM05Policy":
        from .modeling_dm05 import DM05Policy

        return DM05Policy
    if name in {
        "compute_dm05_norm_stats_from_lerobot_dataset",
        "prepare_dm05_norm_stats_from_dataset",
        "resolve_dm05_normalizer",
    }:
        from . import normalization_dm05

        return getattr(normalization_dm05, name)
    if name == "make_dm05_pre_post_processors":
        from .processor_dm05 import make_dm05_pre_post_processors

        return make_dm05_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DM05Config",
    "DM05Policy",
    "compute_dm05_norm_stats_from_lerobot_dataset",
    "prepare_dm05_norm_stats_from_dataset",
    "resolve_dm05_normalizer",
    "make_dm05_pre_post_processors",
]
