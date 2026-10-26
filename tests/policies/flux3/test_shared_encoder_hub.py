"""Shared encoders resolve a pinned base-repository location without local rewriting."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.flux3 import utils
from lerobot.policies.flux3.f3 import text_encoder


@pytest.mark.parametrize("local", [False, True])
def test_text_model_and_processor_share_location(monkeypatch, tmp_path, local):
    pytest.importorskip("transformers")
    calls = []

    def model(spec, **kwargs):
        calls.append(("model", spec, kwargs))
        return torch.nn.Identity()

    def processor(spec, **kwargs):
        calls.append(("processor", spec, kwargs))
        return SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    monkeypatch.setattr(text_encoder.Qwen3VLForConditionalGeneration, "from_pretrained", model)
    monkeypatch.setattr(text_encoder.AutoProcessor, "from_pretrained", processor)
    spec = str(tmp_path) if local else "org/base:text_encoder@immutable"
    text_encoder.Qwen3VLEmbedder(spec)
    expected = {} if local else {"revision": "immutable", "subfolder": "text_encoder"}
    repo = spec if local else "org/base"
    assert calls == [
        ("model", repo, {"torch_dtype": torch.bfloat16, **expected}),
        ("processor", repo, expected),
    ]


def test_vae_uses_pinned_revision_and_keeps_local_paths(monkeypatch, tmp_path):
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return "cached/vae.safetensors"

    monkeypatch.setattr(utils, "hf_hub_download", download)
    assert (
        utils.resolve_weights("org/base:video_vae.safetensors@immutable", "unused")
        == "cached/vae.safetensors"
    )
    assert calls == [(("org/base", "video_vae.safetensors"), {"revision": "immutable"})]
    assert utils.resolve_weights(str(tmp_path), "unused") == str(tmp_path)
    assert len(calls) == 1
