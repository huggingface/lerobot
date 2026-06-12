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

import json
from types import SimpleNamespace

import pytest

# ``lerobot.scripts.lerobot_annotate`` (and the ``_push_to_hub`` path it
# exercises) imports ``lerobot.datasets``, which only ships under the
# ``dataset`` extra. Skip in tiers without it instead of erroring.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")


def test_push_to_hub_tags_uploaded_dataset_revision(tmp_path, monkeypatch):
    from lerobot.scripts.lerobot_annotate import _push_to_hub

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))

    calls = {}

    class FakeHfApi:
        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs
            return SimpleNamespace(oid="abc123")

        def delete_tag(self, repo_id, **kwargs):
            import requests
            from huggingface_hub.errors import RevisionNotFoundError

            calls["delete_tag"] = {"repo_id": repo_id, **kwargs}
            # Simulate the common case: no stale tag to delete.
            raise RevisionNotFoundError("no such tag", response=requests.Response())

        def create_tag(self, **kwargs):
            calls["create_tag"] = kwargs

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    cfg = SimpleNamespace(
        repo_id="source/dataset",
        new_repo_id="annotated/dataset",
        push_private=True,
        push_commit_message=None,
    )

    _push_to_hub(root, cfg)

    assert calls["create_repo"] == {
        "repo_id": "annotated/dataset",
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    assert calls["upload_folder"]["repo_id"] == "annotated/dataset"
    # A stale tag (e.g. from a previous annotation run) is deleted first so
    # the new tag always points at the upload we just made.
    assert calls["delete_tag"] == {
        "repo_id": "annotated/dataset",
        "tag": "v3.0",
        "repo_type": "dataset",
    }
    assert calls["create_tag"] == {
        "repo_id": "annotated/dataset",
        "tag": "v3.0",
        "repo_type": "dataset",
        "revision": "abc123",
    }
