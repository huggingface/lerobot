# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from lerobot.policies.zero_wam.prompt_cache import (
    atomic_torch_save,
    load_prompt_text,
    resolve_prompt_path,
    save_shared_prompt_text,
)


def test_shared_caption_preserves_padding_and_owns_just_its_storage(tmp_path):
    # Full-size blank-caption layout: keep its nonzero token and all512 padded positions.
    backing = torch.ones(1024, 4096, dtype=torch.bfloat16)
    embedding = backing[:512]
    embedding[1:] = 0
    first = save_shared_prompt_text(embedding, tmp_path)
    assert first == save_shared_prompt_text(embedding.clone(), tmp_path)
    stored = torch.load(tmp_path / first, weights_only=True)
    torch.testing.assert_close(stored, embedding, rtol=0, atol=0)
    assert stored.untyped_storage().nbytes() == 512 * 4096 * 2
    assert (tmp_path / first).stat().st_size < 512 * 4096 * 2 + 4096
    changed = embedding.clone()
    changed[0, 0] = 2
    assert save_shared_prompt_text(changed, tmp_path) != first
    assert len(list((tmp_path / "text_embeddings").iterdir())) == 2


def test_references_follow_snapshot_directories_without_resolving_blob_symlinks(tmp_path, monkeypatch):
    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshot"
    blobs.mkdir()
    snapshot.mkdir()
    (blobs / "manifest-blob").touch()
    (snapshot / "pools.jsonl").symlink_to(blobs / "manifest-blob")
    torch.save(torch.arange(8).reshape(2, 4), blobs / "text-blob")
    (snapshot / "text.pth").symlink_to(blobs / "text-blob")
    (blobs / "latent-blob").touch()
    (snapshot / "demo.pth").symlink_to(blobs / "latent-blob")
    monkeypatch.chdir(tmp_path)
    assert resolve_prompt_path("demo.pth", "snapshot/pools.jsonl") == str(snapshot / "demo.pth")
    assert resolve_prompt_path(str(blobs / "latent-blob"), snapshot / "pools.jsonl") == str(
        blobs / "latent-blob"
    )
    text = load_prompt_text({"text_emb_path": "text.pth"}, snapshot / "demo.pth")
    torch.testing.assert_close(text, torch.arange(8).reshape(2, 4))


def test_missing_shared_text_cannot_silently_use_the_target_caption(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompt_text({"text_emb_path": "missing.pth"}, tmp_path / "demo.pth")
    with pytest.raises(ValueError, match="only one"):
        load_prompt_text(
            {"text_emb_path": "missing.pth", "text_emb": torch.ones(2, 4)}, tmp_path / "demo.pth"
        )
    assert load_prompt_text({}, tmp_path / "legacy-video-only.pth") is None
    invalid = tmp_path / "invalid.pth"
    torch.save({"wrong": "format"}, invalid)
    with pytest.raises(ValueError, match=r"\[tokens, hidden\]"):
        load_prompt_text({"text_emb_path": invalid.name}, tmp_path / "demo.pth")


def test_interrupted_shared_write_does_not_publish_a_partial_entry(tmp_path, monkeypatch):
    target = tmp_path / "caption.pth"

    def interrupted(value, path):
        path.write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_torch_save(torch.ones(2, 4), target)
    assert not target.exists() and not list(tmp_path.iterdir())
