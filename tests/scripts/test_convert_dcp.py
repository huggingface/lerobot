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
"""lerobot-convert-dcp: locating, converting, and graceful-degradation publishing."""

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from lerobot.scripts.lerobot_convert_dcp import (
    ConvertDcpConfig,
    _locate_pretrained_dir,
    _publish_converted,
    convert_checkpoint,
)
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig


@pytest.fixture
def fake_merge(monkeypatch):
    """Stand in for accelerate.utils.merge_fsdp_weights: writes a marker safetensors file."""
    import accelerate.utils

    def merge(checkpoint_dir, output_path, safe_serialization=True, remove_checkpoint_dir=False):
        assert isinstance(checkpoint_dir, str) and isinstance(output_path, str)  # str, not Path
        (Path(output_path) / "model.safetensors").write_bytes(b"merged")
        # Mirror accelerate: the shard directory is removed by the merge itself, when asked.
        if remove_checkpoint_dir:
            shutil.rmtree(checkpoint_dir)

    monkeypatch.setattr(accelerate.utils, "merge_fsdp_weights", merge)


def make_dcp_checkpoint(tmp_path: Path, config: DummyCheckpointConfig | None = None) -> Path:
    pretrained = tmp_path / PRETRAINED_MODEL_DIR
    dcp_dir = pretrained / "pytorch_model_fsdp_0"
    dcp_dir.mkdir(parents=True)
    (dcp_dir / "__0_0.distcp").write_bytes(b"shard")
    (dcp_dir / ".metadata").write_bytes(b"metadata")
    if config is None:
        (pretrained / "config.json").write_text("{}")
    else:
        config.save_pretrained(pretrained)
    return tmp_path


class TestConvert:
    def test_locate_accepts_step_dir_or_pretrained_dir(self, tmp_path):
        step_dir = make_dcp_checkpoint(tmp_path)
        pretrained = step_dir / PRETRAINED_MODEL_DIR
        assert _locate_pretrained_dir(step_dir) == pretrained
        assert _locate_pretrained_dir(pretrained) == pretrained

    def test_convert_keeps_dcp_by_default(self, tmp_path, fake_merge):
        step_dir = make_dcp_checkpoint(tmp_path, DummyCheckpointConfig(device="cpu"))
        out = convert_checkpoint(ConvertDcpConfig(checkpoint_dir=step_dir))
        assert out.read_bytes() == b"merged"
        assert (step_dir / PRETRAINED_MODEL_DIR / "pytorch_model_fsdp_0").is_dir()

    def test_convert_delete_dcp(self, tmp_path, fake_merge):
        step_dir = make_dcp_checkpoint(tmp_path, DummyCheckpointConfig(device="cpu"))
        convert_checkpoint(ConvertDcpConfig(checkpoint_dir=step_dir, delete_dcp=True))
        assert not (step_dir / PRETRAINED_MODEL_DIR / "pytorch_model_fsdp_0").exists()

    def test_convert_names_a_peft_adapter(self, tmp_path, fake_merge):
        pytest.importorskip("peft")
        step_dir = make_dcp_checkpoint(tmp_path, DummyCheckpointConfig(device="cpu", use_peft=True))
        pretrained = step_dir / PRETRAINED_MODEL_DIR
        out = convert_checkpoint(ConvertDcpConfig(checkpoint_dir=step_dir))
        assert out == pretrained / "adapter_model.safetensors"
        assert out.read_bytes() == b"merged"
        assert not (pretrained / "model.safetensors").exists()

    def test_missing_shards_error_names_the_format(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="checkpoint_format=dcp"):
            convert_checkpoint(ConvertDcpConfig(checkpoint_dir=tmp_path))


class TestPublishGracefulDegradation:
    def _mock_api(self, monkeypatch):
        calls = {}

        class FakeApi:
            def create_repo(self, repo_id, private=None, exist_ok=False):
                return SimpleNamespace(repo_id=repo_id)

            def upload_folder(
                self, *, repo_id, folder_path, allow_patterns=None, ignore_patterns=None, **kwargs
            ):
                calls["repo_id"] = repo_id
                root = Path(folder_path)
                paths = (p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
                calls["files"] = set(
                    filter_repo_objects(paths, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
                )
                return SimpleNamespace(repo_url=SimpleNamespace(url=f"https://huggingface.co/{repo_id}"))

        import lerobot.scripts.lerobot_convert_dcp as mod

        monkeypatch.setattr(mod, "HfApi", FakeApi)
        return calls

    def test_missing_train_config_warns_and_uploads_core(self, tmp_path, monkeypatch, caplog):
        calls = self._mock_api(monkeypatch)
        pretrained = make_dcp_checkpoint(tmp_path) / PRETRAINED_MODEL_DIR
        (pretrained / "model.safetensors").write_bytes(b"w")
        with caplog.at_level(logging.WARNING):
            _publish_converted(pretrained, "user/converted", private=None)
        assert any("train_config.json missing" in m for m in caplog.messages)
        assert "model.safetensors" in calls["files"]
        # The DCP shard directory is still on disk (--delete_dcp defaults to False) but neither
        # its `.distcp` shards nor their `.metadata` sidecar are uploaded.
        assert not any(name.startswith("pytorch_model_fsdp_0/") for name in calls["files"])
        # config.json is not parseable as a policy config here -> card skipped with a warning
        assert any("model card" in m for m in caplog.messages)

    def test_upload_keeps_every_saved_artifact_but_dcp_shards(self, tmp_path, monkeypatch):
        calls = self._mock_api(monkeypatch)
        pretrained = make_dcp_checkpoint(tmp_path) / PRETRAINED_MODEL_DIR
        # Files a tokenizer and a remote-code action tokenizer save next to the model.
        saved = {
            "tokenizer/tokenizer.json": "{}",
            "tokenizer/chat_template.jinja": "default",
            "tokenizer/additional_chat_templates/tools.jinja": "tools",
            "action_tokenizer/preprocessor_config.json": "{}",
            "action_tokenizer/processing_action_tokenizer.py": "class FAST: ...",
        }
        for name, text in saved.items():
            (pretrained / name).parent.mkdir(parents=True, exist_ok=True)
            (pretrained / name).write_text(text)

        _publish_converted(pretrained, "user/converted", private=None)

        assert calls["files"] == {"config.json", *saved}
