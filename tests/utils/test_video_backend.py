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

import importlib.machinery
import sys
import types

import pytest

from lerobot.utils.import_utils import get_safe_default_video_backend


@pytest.mark.parametrize(
    "core_attrs,expected",
    [
        ({"ffmpeg_major_version": 9}, "torchcodec"),
        ({"ffmpeg_major_version": None}, "pyav"),
        ({}, "torchcodec"),
    ],
    ids=["ffmpeg-loaded", "ffmpeg-missing", "older-torchcodec"],
)
def test_default_backend_checks_ffmpeg(monkeypatch, core_attrs, expected):
    for name, attrs in (("torchcodec", {}), ("torchcodec._core", core_attrs)):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    assert get_safe_default_video_backend() == expected
