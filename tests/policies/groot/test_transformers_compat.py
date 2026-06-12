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

"""Regression tests for #3765/#3775: importing the GR00T module with an incompatible
`transformers` version must raise an actionable error instead of a cryptic
`StrictDataclassDefinitionError`."""

from dataclasses import dataclass

import pytest

from lerobot.policies.groot.groot_n1 import _ensure_strict_compatible_config_base


def test_object_base_passes():
    # `PretrainedConfig` falls back to `object` when transformers is not installed.
    _ensure_strict_compatible_config_base(object)


def test_dataclass_base_passes():
    # transformers>=5.4.0 makes `PretrainedConfig` a dataclass.
    @dataclass
    class DataclassConfigBase:
        pass

    _ensure_strict_compatible_config_base(DataclassConfigBase)


def test_non_dataclass_base_raises_actionable_error():
    # transformers<5.4.0 ships a non-dataclass `PretrainedConfig`.
    class LegacyConfigBase:
        pass

    with pytest.raises(ImportError, match=r"transformers>=5\.4\.0"):
        _ensure_strict_compatible_config_base(LegacyConfigBase)
