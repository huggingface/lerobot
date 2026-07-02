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

import sys

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_edit_dataset as edit_dataset_module
from lerobot.scripts.lerobot_edit_dataset import get_output_path


def test_get_output_path_non_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(edit_dataset_module, "HF_LEROBOT_HOME", tmp_path)
    src = tmp_path / "user/dataset"
    src.mkdir(parents=True)

    _, output_path, backup_path = get_output_path("user/dataset", "user/dataset-edited", None, None)

    assert output_path == (tmp_path / "user/dataset-edited").resolve()
    assert backup_path is None
    assert src.exists()


def test_get_output_path_in_place_backs_up_original(tmp_path, monkeypatch):
    monkeypatch.setattr(edit_dataset_module, "HF_LEROBOT_HOME", tmp_path)
    src = tmp_path / "user/dataset"
    src.mkdir(parents=True)
    (src / "marker").write_text("x")

    _, _, backup_path = get_output_path("user/dataset", None, None, None)

    assert backup_path == tmp_path / "user/dataset_old"
    assert (backup_path / "marker").exists()
    assert not src.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_get_output_path_in_place_with_symlinked_cache(tmp_path, monkeypatch):
    # Regression: with a symlinked HF_LEROBOT_HOME, the previous caller-side
    # path-equality check missed the in-place case and the backup was skipped.
    real_home = tmp_path / "real"
    real_home.mkdir()
    link_home = tmp_path / "linked"
    link_home.symlink_to(real_home)
    monkeypatch.setattr(edit_dataset_module, "HF_LEROBOT_HOME", link_home)

    src = link_home / "user/dataset"
    src.mkdir(parents=True)
    (src / "marker").write_text("x")

    _, _, backup_path = get_output_path("user/dataset", None, None, None)

    assert backup_path is not None
    assert (backup_path / "marker").exists()
