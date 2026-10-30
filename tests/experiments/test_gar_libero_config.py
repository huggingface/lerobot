#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""The LIBERO config writer does not import LIBERO or download a checkpoint."""

import json
import sys
from pathlib import Path

import pytest

from experiments.gar.proof import StopProofError, find_libero_benchmark_root, write_libero_config


def _fake_package(root: Path) -> Path:
    package = root / "site" / "libero" / "libero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package marker\n", encoding="utf-8")
    for name in ("bddl_files", "init_files"):
        (package / name).mkdir()
    return root / "site"


def _values(config_path: Path) -> dict[str, str]:
    values = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(": ", 1)
        values[key] = json.loads(raw)
    return values


def test_write_libero_config_uses_assets_outside_the_package(tmp_path, monkeypatch):
    before = {name for name in sys.modules if name == "libero" or name.startswith("libero.")}
    site = _fake_package(tmp_path)
    assets = tmp_path / "outside-assets"
    assets.mkdir()
    (assets / "marker.txt").write_text("present\n", encoding="utf-8")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "config-home"))

    def fail_download(destination: Path) -> None:
        raise AssertionError(f"downloaded into {destination}")

    monkeypatch.setattr("experiments.gar.proof._download_libero_assets", fail_download)
    config_path = write_libero_config(search_path=[str(site)], assets_dir=assets)

    after = {name for name in sys.modules if name == "libero" or name.startswith("libero.")}
    assert after == before
    assert config_path == tmp_path / "config-home" / "config.yaml"
    values = _values(config_path)
    package = site / "libero" / "libero"
    assert values["benchmark_root"] == str(package.resolve())
    assert values["bddl_files"] == str((package / "bddl_files").resolve())
    assert values["init_states"] == str((package / "init_files").resolve())
    assert values["assets"] == str(assets.resolve())
    assert not (package / "assets").exists()
    datasets = Path(values["datasets"])
    assert datasets.is_dir()
    assert set(values) == {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
    for path in values.values():
        assert Path(path).is_dir()


def test_find_libero_benchmark_root_rejects_a_path_without_the_package(tmp_path):
    with pytest.raises(StopProofError):
        find_libero_benchmark_root([str(tmp_path)])
