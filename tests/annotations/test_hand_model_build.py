# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Public model assembly preserves weights and rejects changed or escaped assets."""

import hashlib
import importlib.util
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

    def download(repo, filename, **kwargs):
        assert repo == "public/model" and kwargs["token"] is False and kwargs["revision"] == "a" * 40
        target = Path(kwargs["local_dir"]) / filename
        shutil.copyfile(original, target)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    sources = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "downloads": [
            {
                "repo": "public/model",
                "file": "full.ckpt",
                "revision": "a" * 40,
                "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "output": "inference-only.pt",
                "inference_tensors": 2,
            }
        ],
    }
    build.assemble(root, sources)
    payload = torch.load(root / "inference-only.pt", weights_only=True)
    assert set(payload) == {"state_dict"}
    for name, tensor in state.items():
        assert torch.equal(payload["state_dict"][name], tensor)
    (root / "inference-only.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        build.verify(root, sources)


def test_rejects_unverified_upstream_before_unpickling(tmp_path, monkeypatch):
    manifest = tmp_path / "assets.json"
    manifest.write_text('{"files": []}')
    original = tmp_path / "bad.ckpt"
    original.write_bytes(b"unverified")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(original))
    load = Mock(side_effect=AssertionError("Never unpickle an unverified checkpoint"))
    monkeypatch.setattr(torch, "load", load)
    sources = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "downloads": [
            {
                "repo": "public/model",
                "file": "bad.ckpt",
                "revision": "a" * 40,
                "sha256": "0" * 64,
                "output": "inference-only.pt",
                "inference_tensors": 2,
            }
        ],
    }
    with pytest.raises(ValueError, match="Upstream model changed"):
        build.assemble(tmp_path, sources)
    load.assert_not_called()


@pytest.mark.parametrize("name", ["../outside", "/outside", "MANO_RIGHT.pkl"])
def test_model_package_paths_cannot_escape_or_include_mano(tmp_path, name):
    with pytest.raises(ValueError, match="Unexpected path"):
        build.asset_path(tmp_path, name)
