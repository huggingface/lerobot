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
"""Unit tests for the fine-tuned-checkpoint existence probe used by
`GrootPolicy.from_pretrained` (#4711).

The probe must treat "definitively absent" (missing file/repo, not a repo id)
differently from "could not tell" (transient hub failures): only the former may
fall back to the base-GR00T path silently. Online, a transient failure must
propagate — silently rerouting a user's fine-tuned checkpoint to a different
model is the failure class of #4577.
"""

import logging

import pytest

pytest.importorskip("transformers", reason="groot requires the `groot` extra (transformers)")

from huggingface_hub.errors import HFValidationError  # noqa: E402

from lerobot.policies.groot import modeling_groot as mod  # noqa: E402


class _FakeApi:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.calls: list[dict] = []

    def file_exists(self, repo_id, filename, *, revision=None, token=None, **kwargs):
        self.calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        if self._exc is not None:
            raise self._exc
        return self._result


def _probe(**kwargs):
    return mod._probe_finetuned_checkpoint(
        "user/my_finetuned_checkpoint",
        revision=None,
        cache_dir=None,
        proxies=None,
        token=None,
        local_files_only=kwargs.pop("local_files_only", False),
        **kwargs,
    )


def test_present_file_is_finetuned(monkeypatch):
    api = _FakeApi(result=True)
    monkeypatch.setattr(mod, "HfApi", lambda: api)
    assert _probe() is True
    assert api.calls == [
        {"repo_id": "user/my_finetuned_checkpoint", "filename": "model.safetensors", "revision": None}
    ]


def test_absent_file_is_not_finetuned(monkeypatch):
    # Online, the HEAD probe reports absence as a plain False.
    monkeypatch.setattr(mod, "HfApi", lambda: _FakeApi(result=False))
    assert _probe() is False


def test_not_a_repo_id_is_not_finetuned(monkeypatch):
    monkeypatch.setattr(mod, "HfApi", lambda: _FakeApi(exc=HFValidationError("not a valid repo id")))
    assert _probe() is False


def test_online_transient_error_propagates(monkeypatch):
    # Online, "can't tell" must reach the caller instead of silently rerouting
    # to the base-GR00T path.
    def download(**kwargs):
        raise AssertionError("online probe must not download, only HEAD-check")

    monkeypatch.setattr(mod, "hf_hub_download", download)
    monkeypatch.setattr(mod, "HfApi", lambda: _FakeApi(exc=ConnectionError("hub hiccup")))
    with pytest.raises(ConnectionError, match="hub hiccup"):
        _probe()


def test_offline_transient_error_degrades_to_base_path(monkeypatch, caplog):
    # Offline, cache-based base-model loading is a supported flow: degrade to
    # "no" with a warning instead of hard-failing the probe.
    def offline_error(**kwargs):
        assert kwargs.get("local_files_only") is True
        raise OSError("offline mode is enabled")

    monkeypatch.setattr(mod, "hf_hub_download", offline_error)
    monkeypatch.setattr(mod, "HF_HUB_OFFLINE", True)
    with caplog.at_level(logging.WARNING):
        assert _probe() is False
    assert "assuming a base GR00T model" in caplog.text


def test_offline_cache_hit_is_finetuned(monkeypatch):
    # Offline with the checkpoint cached: the cache-only hf_hub_download path
    # must resolve it (a HEAD probe cannot see the local cache).
    def cached(**kwargs):
        assert kwargs.get("local_files_only") is True
        return "/cache/model.safetensors"

    def no_head(self):
        raise AssertionError("offline probe must not hit the hub")

    monkeypatch.setattr(mod, "hf_hub_download", cached)
    monkeypatch.setattr(mod, "HfApi", no_head)
    assert _probe(local_files_only=True) is True


def test_local_dir_probe(tmp_path):
    # Local directory: presence of model.safetensors decides, no hub call.
    monkey_free_dir = tmp_path / "finetuned"
    monkey_free_dir.mkdir()
    assert mod._probe_finetuned_checkpoint(
        str(monkey_free_dir),
        revision=None,
        cache_dir=None,
        proxies=None,
        token=None,
        local_files_only=False,
    ) is False
    (monkey_free_dir / "model.safetensors").write_bytes(b"weights")
    assert mod._probe_finetuned_checkpoint(
        str(monkey_free_dir),
        revision=None,
        cache_dir=None,
        proxies=None,
        token=None,
        local_files_only=False,
    ) is True
