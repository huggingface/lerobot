#!/usr/bin/env python

import json
from types import SimpleNamespace


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

        def create_tag(self, **kwargs):
            calls["create_tag"] = kwargs

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    cfg = SimpleNamespace(
        repo_id="source/dataset",
        dest_repo_id="annotated/dataset",
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
    assert calls["create_tag"] == {
        "repo_id": "annotated/dataset",
        "tag": "v3.0",
        "repo_type": "dataset",
        "exist_ok": True,
        "revision": "abc123",
    }
