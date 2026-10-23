# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Public model assembly preserves weights and rejects changed or escaped assets."""

import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "docker/annotation_hands/build_model_assets.py"
SPEC = importlib.util.spec_from_file_location("hand_model_build", SCRIPT)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_public_assembly_omits_training_state_without_changing_tensors(tmp_path, monkeypatch):
    state = {"weight": torch.arange(12).reshape(3, 4), "bias": torch.tensor([0.25, -1.0])}
    original = tmp_path / "full.ckpt"
    torch.save({"state_dict": state, "optimizer_states": [{"step": 5000}]}, original)
    expected = tmp_path / "expected" / "inference-only.pt"
    expected.parent.mkdir()
    torch.save({"state_dict": state}, expected)
    root = tmp_path / "assets"
    root.mkdir()
    manifest = root / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "inference-only.pt",
                        "size": expected.stat().st_size,
                        "sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )

    template = tmp_path / "template.zip"
    build.create_template(expected, template)

    def download(source, target):
        assert source["repo"] == "public/model" and source["revision"] == "a" * 40
        shutil.copyfile(original, target)

    monkeypatch.setattr(build, "download_public", download)
    sources = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "downloads": [
            {
                "repo": "public/model",
                "file": "full.ckpt",
                "revision": "a" * 40,
                "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "output": "inference-only.pt",
                "template": template.name,
                "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            }
        ],
    }
    build.assemble(root, sources, tmp_path)
    payload = torch.load(root / "inference-only.pt", weights_only=True)
    assert set(payload) == {"state_dict"}
    for name, tensor in state.items():
        assert torch.equal(payload["state_dict"][name], tensor)
    (root / "inference-only.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        build.verify(root, sources)


def test_rejects_unverified_upstream_before_extracting_tensors(tmp_path, monkeypatch):
    manifest = tmp_path / "assets.json"
    manifest.write_text('{"files": []}')
    original = tmp_path / "bad.ckpt"
    original.write_bytes(b"unverified")
    monkeypatch.setattr(build, "download_public", lambda source, target: shutil.copyfile(original, target))
    rebuild = Mock(side_effect=AssertionError("Never extract an unverified checkpoint"))
    monkeypatch.setattr(build, "rebuild_checkpoint", rebuild)
    sources = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "downloads": [
            {
                "repo": "public/model",
                "file": "bad.ckpt",
                "revision": "a" * 40,
                "sha256": "0" * 64,
                "output": "inference-only.pt",
                "template": "template.zip",
            }
        ],
    }
    with pytest.raises(ValueError, match="Upstream model changed"):
        build.assemble(tmp_path, sources, tmp_path)
    rebuild.assert_not_called()


def test_public_http_download_never_uses_job_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private-job-token")
    request = Mock(return_value=io.BytesIO(b"public model"))
    monkeypatch.setattr(build.urllib.request, "urlopen", request)
    build.download_public(
        {"repo": "public/model", "revision": "a" * 40, "file": "model.pt"}, tmp_path / "model.pt"
    )
    assert request.call_args.args == (
        "https://huggingface.co/public/model/resolve/" + "a" * 40 + "/model.pt",
    )
    assert (tmp_path / "model.pt").read_bytes() == b"public model"


def test_tensor_records_must_match_the_accepted_checkpoint(tmp_path):
    accepted, changed = tmp_path / "accepted.pt", tmp_path / "changed.pt"
    torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, accepted)
    torch.save({"state_dict": {"weight": torch.tensor([2.0])}}, changed)
    template = tmp_path / "template.zip"
    build.create_template(accepted, template)
    with pytest.raises(ValueError, match="Upstream tensor differs"):
        build.rebuild_checkpoint(changed, tmp_path / "output.pt", template)


@pytest.mark.parametrize("name", ["../outside", "/outside", "MANO_RIGHT.pkl"])
def test_model_package_paths_cannot_escape_or_include_mano(tmp_path, name):
    with pytest.raises(ValueError, match="Unexpected path"):
        build.asset_path(tmp_path, name)
