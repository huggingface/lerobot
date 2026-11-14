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

import errno
import json
import os
import stat
from pathlib import Path

import pytest

from lerobot.utils.io_utils import load_json, write_json


@pytest.mark.parametrize("existing", [False, True])
def test_write_json_serialization_failure_keeps_previous_file(tmp_path, existing):
    path = tmp_path / "state.json"
    original = b'{"step": 42}\n'
    if existing:
        path.write_bytes(original)

    with pytest.raises(TypeError):
        write_json({"step": 43, "invalid": object()}, path)

    assert set(tmp_path.iterdir()) == ({path} if existing else set())
    if existing:
        assert path.read_bytes() == original


def test_write_json_partial_write_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    original = b'{"step": 42}\n'
    path.write_bytes(original)

    def fail_after_write(data, stream, **kwargs):
        stream.write('{"step":')
        raise OSError("disk write failed")

    monkeypatch.setattr(json, "dump", fail_after_write)
    with pytest.raises(OSError, match="disk write failed"):
        write_json({"step": 43}, path)

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("existing", [False, True])
def test_write_json_replace_failure_keeps_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    path = tmp_path / "state.json"
    original = b'{"step": 42}\n'
    if existing:
        path.write_bytes(original)
    replacements = []

    def fail_replace(source: Path, target: Path) -> None:
        replacements.append((source, target))
        assert source.read_text(encoding="utf-8") == '{\n    "step": 43\n}'
        raise PermissionError("replacement denied")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="replacement denied"):
        write_json({"step": 43}, path)

    assert len(replacements) == 1
    assert replacements[0][1] == path
    assert set(tmp_path.iterdir()) == ({path} if existing else set())
    if existing:
        assert path.read_bytes() == original


def test_write_json_creates_parents_and_preserves_format(tmp_path):
    path = tmp_path / "nested" / "state.json"
    data = {"label": "机器人", "step": 43}

    write_json(data, path)

    assert load_json(path) == data
    assert path.read_text(encoding="utf-8") == json.dumps(data, indent=4, ensure_ascii=False)
    assert set(path.parent.iterdir()) == {path}


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX permission bits")
def test_write_json_preserves_existing_permissions(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o640)

    write_json({"step": 43}, path)

    assert load_json(path) == {"step": 43}
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_write_json_follows_existing_symlink(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    link = tmp_path / "latest.json"
    try:
        link.symlink_to(path.name)
    except NotImplementedError:
        pytest.skip("Symbolic links are not supported")
    except OSError as exc:
        if (
            exc.errno in (errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP)
            or getattr(exc, "winerror", None) == 1314
        ):
            pytest.skip(f"Cannot create symbolic links: {exc}")
        raise

    write_json({"step": 43}, link)

    assert link.is_symlink()
    assert load_json(path) == {"step": 43}
