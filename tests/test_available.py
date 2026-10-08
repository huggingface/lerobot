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


def test_require_package_error_message_includes_uv():
    """Error message includes both pip and uv install commands."""
    with patch("lerobot.utils.import_utils.is_package_available", return_value=False):
        _require_package_cache.clear()
        try:
            with pytest.raises(ImportError, match=r"uv pip install"):
                require_package("grpcio", extra="async", import_name="grpc")
        finally:
            _require_package_cache.clear()


def _write_broken_package(tmp_path, name="broken_dep_for_test"):
    """Create a fake package that is present per find_spec but raises on import."""
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise ImportError('simulated broken install')\n")


def _write_broken_transformers_with_metadata(tmp_path):
    """Fake 'transformers' with dist metadata: present AND versioned, but unimportable."""
    _write_broken_package(tmp_path, name="transformers")
    dist_info = tmp_path / "transformers-4.57.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: transformers\nVersion: 4.57.1\n")


def test_is_importable_broken_package(tmp_path, monkeypatch, caplog):
    """A package that exists per find_spec but raises on import is not importable."""
    import importlib.util

    from lerobot.utils.import_utils import _is_importable

    _write_broken_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert importlib.util.find_spec("broken_dep_for_test") is not None

    with caplog.at_level("WARNING", logger="lerobot.utils.import_utils"):
        assert _is_importable("broken_dep_for_test") is False
    assert "treating it as unavailable" in caplog.text


def test_is_importable_healthy_module():
    from lerobot.utils.import_utils import _is_importable

    assert _is_importable("json") is True
    assert _is_importable("definitely_not_a_real_module_xyz") is False


def test_transformers_flag_false_for_broken_install(tmp_path, monkeypatch, caplog):
    """_transformers_available is False when transformers is present but unimportable (#4332)."""
    import importlib
    import importlib.util

    import lerobot.utils.import_utils as import_utils

    _write_broken_transformers_with_metadata(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert importlib.util.find_spec("transformers") is not None
    assert import_utils.is_package_available("transformers") is True

    with caplog.at_level("WARNING", logger="lerobot.utils.import_utils"):
        reloaded = importlib.reload(import_utils)
    try:
        assert reloaded._transformers_available is False
    finally:
        importlib.reload(import_utils)
