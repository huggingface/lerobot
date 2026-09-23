#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import patch

import pytest

import lerobot
from lerobot.utils.import_utils import _require_package_cache, require_package


def test_version():
    """Verify the package exposes a version string."""
    assert isinstance(lerobot.__version__, str)
    assert len(lerobot.__version__) > 0


def test_require_package_raises_when_missing():
    """require_package raises ImportError with install instructions when a package is missing."""
    with patch("lerobot.utils.import_utils.is_package_available", return_value=False):
        # Clear the cache so the mock takes effect
        _require_package_cache.clear()
        try:
            with pytest.raises(ImportError, match=r"pip install 'lerobot\[dataset\]'"):
                require_package("datasets", extra="dataset")
        finally:
            _require_package_cache.clear()


def test_require_package_passes_when_available():
    """require_package does not raise when the package is installed."""
    with patch("lerobot.utils.import_utils.is_package_available", return_value=True):
        _require_package_cache.clear()
        try:
            # Should not raise
            require_package("datasets", extra="dataset")
        finally:
            _require_package_cache.clear()


def test_require_standalone_package_does_not_invent_a_lerobot_extra():
    with patch("lerobot.utils.import_utils.is_package_available", return_value=False) as available:
        _require_package_cache.clear()
        try:
            with pytest.raises(ImportError, match="uv pip install 'SAM-2'"):
                require_package("SAM-2", extra=None, import_name="sam2")
            available.assert_called_once_with("SAM-2", "sam2")
        finally:
            _require_package_cache.clear()


def test_require_package_error_message_includes_uv():
    """Error message includes both pip and uv install commands."""
    with patch("lerobot.utils.import_utils.is_package_available", return_value=False):
        _require_package_cache.clear()
        try:
            with pytest.raises(ImportError, match=r"uv pip install"):
                require_package("grpcio", extra="async", import_name="grpc")
        finally:
            _require_package_cache.clear()
