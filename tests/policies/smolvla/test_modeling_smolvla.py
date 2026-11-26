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
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tests.utils import skip_if_package_missing


def _build_tiny_vlm_dirs(tmp_path: Path, *, tie_word_embeddings: bool = False) -> tuple[Path, Path]:
    """Create matching tiny VLM directories: one complete and one with metadata only."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        SmolVLMConfig,
        SmolVLMForConditionalGeneration,
        SmolVLMImageProcessor,
        SmolVLMProcessor,
        SmolVLMVideoProcessor,
    )

    full_dir = tmp_path / "smolvlm_full"
    meta_dir = tmp_path / "smolvlm_metadata"
    full_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    vlm_cfg = SmolVLMConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "use_cache": False,
            "torch_dtype": "bfloat16",
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 32,
            "patch_size": 16,
            "num_channels": 3,
        },
        scale_factor=2,
        image_token_id=10,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype="float32",
    )
    vlm_cfg.text_config.tie_word_embeddings = tie_word_embeddings
    torch.manual_seed(0)
    vlm = SmolVLMForConditionalGeneration(vlm_cfg)
    vlm.save_pretrained(full_dir)

    vocab = {f"t{i}": i for i in range(64)}
    vocab.update(
        {
            "<unk>": 0,
            "<image>": 10,
            "<fake_token_around_image>": 11,
            "<global-img>": 12,
        }
    )
    tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="<unk>",
        extra_special_tokens={
            "fake_image_token": "<fake_token_around_image>",
            "global_image_token": "<global-img>",
            "image_token": "<image>",
        },
    )
    img_proc = SmolVLMImageProcessor(
        size={"longest_edge": 32},
        max_image_size={"longest_edge": 32},
        do_resize=True,
        do_image_splitting=False,
    )
    vid_proc = SmolVLMVideoProcessor(
        size={"longest_edge": 32},
        max_image_size={"longest_edge": 32},
        do_resize=True,
        do_image_splitting=False,
    )
    processor = SmolVLMProcessor(image_processor=img_proc, tokenizer=tok, video_processor=vid_proc)
    processor.save_pretrained(full_dir)

    # Copy only config, tokenizer, and processor files into meta_dir (no weight files).
    for item in full_dir.iterdir():
        if item.suffix in {".safetensors", ".bin"} or item.name.endswith(".index.json"):
            continue
        (meta_dir / item.name).write_bytes(item.read_bytes())

    return full_dir, meta_dir


def _make_tiny_smolvla_config(
    vlm_path: Path,
    *,
    load_vlm_weights: bool = True,
    train_expert_only: bool = True,
    freeze_vision_encoder: bool = False,
) -> SmolVLAConfig:
    return SmolVLAConfig(
        vlm_model_name=str(vlm_path),
        load_vlm_weights=load_vlm_weights,
        train_expert_only=train_expert_only,
        freeze_vision_encoder=freeze_vision_encoder,
        num_vlm_layers=2,
        num_expert_layers=-1,
        expert_width_multiplier=0.5,
        attention_mode="cross_attn",
        self_attn_every_n_layers=2,
        resize_imgs_with_padding=(32, 32),
        chunk_size=4,
        n_action_steps=4,
        num_steps=2,
        max_state_dim=8,
        max_action_dim=8,
        device="cpu",
        push_to_hub=False,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )


def _make_dummy_batch(chunk_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]], dtype=torch.float32),
        "observation.images.camera1": torch.full((1, 3, 32, 32), 0.25, dtype=torch.float32),
        "observation.language.tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "observation.language.attention_mask": torch.tensor([[True, True, True, True]], dtype=torch.bool),
        "action": torch.full((1, chunk_size, 6), 0.05, dtype=torch.float32),
    }


def _assert_policies_match_exactly(actual: SmolVLAPolicy, reference: SmolVLAPolicy) -> None:
    assert actual.config.load_vlm_weights == reference.config.load_vlm_weights

    actual_params = dict(actual.named_parameters(remove_duplicate=False))
    ref_params = dict(reference.named_parameters(remove_duplicate=False))
    assert actual_params.keys() == ref_params.keys()
    for name, ref_param in ref_params.items():
        actual_param = actual_params[name]
        assert actual_param.device.type != "meta", f"{name} remained on meta"
        assert actual_param.dtype == ref_param.dtype, f"{name} dtype mismatch"
        assert actual_param.requires_grad == ref_param.requires_grad, f"{name} requires_grad mismatch"
        torch.testing.assert_close(actual_param, ref_param, atol=0.0, rtol=0.0)

    actual_buffers = dict(actual.named_buffers(remove_duplicate=False))
    ref_buffers = dict(reference.named_buffers(remove_duplicate=False))
    assert actual_buffers.keys() == ref_buffers.keys()
    for name, ref_buf in ref_buffers.items():
        actual_buf = actual_buffers[name]
        assert actual_buf.device.type != "meta", f"buffer {name} remained on meta"
        assert actual_buf.dtype == ref_buf.dtype, f"buffer {name} dtype mismatch"
        torch.testing.assert_close(actual_buf, ref_buf, atol=0.0, rtol=0.0)

    torch.testing.assert_close(
        actual.model.global_image_start_token, reference.model.global_image_start_token, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        actual.model.image_end_token, reference.model.image_end_token, atol=0.0, rtol=0.0
    )

    noise = torch.ones((1, actual.config.chunk_size, actual.config.max_action_dim), dtype=torch.float32)
    actual_actions = actual.predict_action_chunk(
        _make_dummy_batch(actual.config.chunk_size), noise=noise.clone()
    )
    ref_actions = reference.predict_action_chunk(
        _make_dummy_batch(reference.config.chunk_size), noise=noise.clone()
    )
    torch.testing.assert_close(actual_actions, ref_actions, atol=0.0, rtol=0.0)

    time = torch.tensor([0.5], dtype=torch.float32)
    actual_loss, _ = actual.forward(
        _make_dummy_batch(actual.config.chunk_size), noise=noise.clone(), time=time.clone()
    )
    ref_loss, _ = reference.forward(
        _make_dummy_batch(reference.config.chunk_size), noise=noise.clone(), time=time.clone()
    )
    torch.testing.assert_close(actual_loss, ref_loss, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("resize", [(4, 4), None])
def test_prepare_images_normalizes_uint8_inputs_before_resizing(resize):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.camera": object()},
            resize_imgs_with_padding=resize,
            empty_cameras=0,
        )
    )
    image = torch.tensor([[[[0, 255], [128, 64]]]], dtype=torch.uint8).repeat(1, 3, 1, 1)

    images, image_masks = SmolVLAPolicy.prepare_images(policy, {"observation.images.camera": image})

    assert images[0].dtype == torch.float32
    assert images[0].min() >= -1.0
    assert images[0].max() <= 1.0
    assert image_masks[0].tolist() == [True]
    if resize is None:
        torch.testing.assert_close(images[0], image.float() / 255.0 * 2.0 - 1.0)


@skip_if_package_missing("transformers")
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_from_pretrained_complete_checkpoint_skips_backbone_load(
    tmp_path, monkeypatch, strict, tie_word_embeddings
):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path, tie_word_embeddings=tie_word_embeddings)
    torch.manual_seed(7)
    train_cfg = _make_tiny_smolvla_config(full_dir, load_vlm_weights=True, train_expert_only=True)
    source_policy = SmolVLAPolicy(train_cfg)
    ckpt_dir = tmp_path / f"ckpt_{tie_word_embeddings}_{strict}"
    source_policy.save_pretrained(ckpt_dir)

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(cfg, SmolVLAConfig)
    cfg.vlm_model_name = str(meta_dir)

    def _fail_backbone_load(*args, **kwargs):
        raise AssertionError("Backbone weights must not be loaded when checkpoint is complete")

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText, "from_pretrained", _fail_backbone_load
    )
    restored = SmolVLAPolicy.from_pretrained(ckpt_dir, config=cfg, strict=strict, local_files_only=True)

    monkeypatch.undo()
    ref_cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(ref_cfg, SmolVLAConfig)
    ref_cfg.vlm_model_name = str(full_dir)
    reference = SmolVLAPolicy(ref_cfg)
    PreTrainedPolicy._load_weight_files(reference, [ckpt_dir / "model.safetensors"], "cpu", strict)
    reference.eval()

    _assert_policies_match_exactly(restored, reference)
    if tie_word_embeddings:
        text_embed = restored.model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight
        lm_head = restored.model.vlm_with_expert.vlm.lm_head.weight
        assert text_embed is lm_head


@skip_if_package_missing("transformers")
def test_from_pretrained_sharded_complete_checkpoint_skips_backbone_load(tmp_path, monkeypatch):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path, tie_word_embeddings=False)
    torch.manual_seed(11)
    train_cfg = _make_tiny_smolvla_config(full_dir, load_vlm_weights=True, train_expert_only=False)
    source_policy = SmolVLAPolicy(train_cfg)

    single_dir = tmp_path / "single"
    source_policy.save_pretrained(single_dir)
    full_tensors = load_file(str(single_dir / "model.safetensors"))

    sharded_dir = tmp_path / "sharded"
    sharded_dir.mkdir()
    sharded_dir.joinpath("config.json").write_text((single_dir / "config.json").read_text())
    keys = sorted(full_tensors)
    mid = len(keys) // 2
    shard0 = {k: full_tensors[k] for k in keys[:mid]}
    shard1 = {k: full_tensors[k] for k in keys[mid:]}
    save_file(shard0, str(sharded_dir / "model-00001-of-00002.safetensors"))
    save_file(shard1, str(sharded_dir / "model-00002-of-00002.safetensors"))
    weight_map = dict.fromkeys(shard0, "model-00001-of-00002.safetensors") | dict.fromkeys(
        shard1, "model-00002-of-00002.safetensors"
    )
    (sharded_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    cfg = PreTrainedConfig.from_pretrained(sharded_dir, local_files_only=True)
    assert isinstance(cfg, SmolVLAConfig)
    cfg.vlm_model_name = str(meta_dir)

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected backbone load")),
    )
    restored = SmolVLAPolicy.from_pretrained(sharded_dir, config=cfg, strict=True, local_files_only=True)
    assert cfg.load_vlm_weights is True

    source_policy.eval()
    _assert_policies_match_exactly(restored, source_policy)


@skip_if_package_missing("transformers")
def test_from_pretrained_forwards_local_files_only_to_backbone_metadata(tmp_path, monkeypatch):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path)
    source_policy = SmolVLAPolicy(_make_tiny_smolvla_config(full_dir, load_vlm_weights=True))
    ckpt_dir = tmp_path / "offline_ckpt"
    source_policy.save_pretrained(ckpt_dir)

    seen_calls: list[tuple[str, bool | None]] = []
    orig_auto_config = smolvlm_with_expert.AutoConfig.from_pretrained
    orig_auto_processor = smolvlm_with_expert.AutoProcessor.from_pretrained

    def _spy_auto_config(*args, **kwargs):
        seen_calls.append(("AutoConfig", kwargs.get("local_files_only")))
        return orig_auto_config(*args, **kwargs)

    def _spy_auto_processor(*args, **kwargs):
        seen_calls.append(("AutoProcessor", kwargs.get("local_files_only")))
        return orig_auto_processor(*args, **kwargs)

    monkeypatch.setattr(smolvlm_with_expert.AutoConfig, "from_pretrained", _spy_auto_config)
    monkeypatch.setattr(smolvlm_with_expert.AutoProcessor, "from_pretrained", _spy_auto_processor)
    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected backbone load")),
    )

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(cfg, SmolVLAConfig)
    cfg.vlm_model_name = str(meta_dir)
    restored = SmolVLAPolicy.from_pretrained(ckpt_dir, config=cfg, strict=True, local_files_only=True)

    assert ("AutoConfig", True) in seen_calls
    assert ("AutoProcessor", True) in seen_calls
    source_policy.eval()
    _assert_policies_match_exactly(restored, source_policy)


@skip_if_package_missing("transformers")
def test_from_pretrained_legacy_vision_keys_complete_skips_backbone_load(tmp_path, monkeypatch):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path)
    torch.manual_seed(13)
    source_policy = SmolVLAPolicy(_make_tiny_smolvla_config(full_dir, load_vlm_weights=True))
    ckpt_dir = tmp_path / "legacy_ckpt"
    source_policy.save_pretrained(ckpt_dir)

    state = load_file(str(ckpt_dir / "model.safetensors"))
    # Simulate a pre-flatten SigLIP checkpoint key `vision_tower.vision_model.*` by renaming one
    # submodule key while asserting `rename_legacy_vision_model_keys` normalization lets completeness
    # succeed without loading backbone weights.
    target_key = "model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight"
    renamed_target_key = "model.vlm_with_expert.vlm.model.vision_tower.weight"
    legacy_file_key = "model.vlm_with_expert.vlm.model.vision_tower.vision_model.weight"
    state[legacy_file_key] = state[target_key].clone()
    save_file(state, str(ckpt_dir / "model.safetensors"))

    orig_build_skeleton = SmolVLAPolicy._build_meta_skeleton

    def _skeleton_with_vision_tower(cls_arg, *args, **kwargs):
        model = orig_build_skeleton(*args, **kwargs)
        vision_model = model.model.vlm_with_expert.vlm.model.vision_model
        vision_model.register_parameter(
            "vision_tower", torch.nn.Parameter(torch.empty_like(vision_model.post_layernorm.weight))
        )
        # Attach at `.vlm.model.vision_tower.weight` so legacy key normalizes directly onto it.
        tower_mod = torch.nn.Module()
        tower_mod.register_parameter(
            "weight", torch.nn.Parameter(torch.empty_like(vision_model.post_layernorm.weight))
        )
        del vision_model.vision_tower
        model.model.vlm_with_expert.vlm.model.add_module("vision_tower", tower_mod)
        return model

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected backbone load")),
    )
    monkeypatch.setattr(SmolVLAPolicy, "_build_meta_skeleton", classmethod(_skeleton_with_vision_tower))

    restored = SmolVLAPolicy.from_pretrained(
        ckpt_dir,
        config=_make_tiny_smolvla_config(meta_dir, load_vlm_weights=True),
        strict=True,
        local_files_only=True,
    )
    torch.testing.assert_close(
        restored.state_dict()[renamed_target_key],
        state[target_key].to(dtype=restored.state_dict()[renamed_target_key].dtype),
        atol=0.0,
        rtol=0.0,
    )


@skip_if_package_missing("transformers")
def test_from_pretrained_partial_checkpoint_falls_back_to_backbone_when_non_strict(tmp_path):
    full_dir, _meta_dir = _build_tiny_vlm_dirs(tmp_path)
    torch.manual_seed(17)
    source_policy = SmolVLAPolicy(_make_tiny_smolvla_config(full_dir, load_vlm_weights=True))
    ckpt_dir = tmp_path / "partial_ckpt"
    source_policy.save_pretrained(ckpt_dir)

    state = load_file(str(ckpt_dir / "model.safetensors"))
    partial_state = {k: v for k, v in state.items() if ".vlm_with_expert.vlm." not in k}
    save_file(partial_state, str(ckpt_dir / "model.safetensors"))

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(cfg, SmolVLAConfig)
    cfg.vlm_model_name = str(full_dir)

    restored = SmolVLAPolicy.from_pretrained(ckpt_dir, config=cfg, strict=False, local_files_only=True)
    for k, tensor in restored.model.vlm_with_expert.vlm.state_dict().items():
        assert tensor.device.type != "meta", f"VLM tensor {k} stayed on meta"
    noise = torch.ones((1, restored.config.chunk_size, restored.config.max_action_dim), dtype=torch.float32)
    actions = restored.predict_action_chunk(_make_dummy_batch(restored.config.chunk_size), noise=noise)
    assert actions.shape == (1, restored.config.chunk_size, 6)
    assert torch.isfinite(actions).all()


@skip_if_package_missing("transformers")
def test_from_pretrained_partial_checkpoint_raises_when_strict(tmp_path, monkeypatch):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path)
    source_policy = SmolVLAPolicy(_make_tiny_smolvla_config(full_dir, load_vlm_weights=True))
    ckpt_dir = tmp_path / "partial_strict_ckpt"
    source_policy.save_pretrained(ckpt_dir)

    state = load_file(str(ckpt_dir / "model.safetensors"))
    partial_state = {k: v for k, v in state.items() if k != "model.state_proj.weight"}
    save_file(partial_state, str(ckpt_dir / "model.safetensors"))

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(cfg, SmolVLAConfig)
    cfg.vlm_model_name = str(meta_dir)

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected backbone load")),
    )
    with pytest.raises(RuntimeError, match="state_proj.weight"):
        SmolVLAPolicy.from_pretrained(ckpt_dir, config=cfg, strict=True, local_files_only=True)


@skip_if_package_missing("transformers")
@pytest.mark.parametrize("strict", [True, False])
def test_from_pretrained_unexpected_keys_behavior(tmp_path, monkeypatch, caplog, strict):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path)
    source_policy = SmolVLAPolicy(_make_tiny_smolvla_config(full_dir, load_vlm_weights=True))
    ckpt_dir = tmp_path / f"unexpected_{strict}"
    source_policy.save_pretrained(ckpt_dir)

    state = load_file(str(ckpt_dir / "model.safetensors"))
    state["model.unexpected_extra_tensor"] = torch.ones(2, 2)
    save_file(state, str(ckpt_dir / "model.safetensors"))

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(cfg, SmolVLAConfig)
    cfg.vlm_model_name = str(meta_dir)

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected backbone load")),
    )
    if strict:
        with pytest.raises(RuntimeError, match="unexpected_extra_tensor"):
            SmolVLAPolicy.from_pretrained(ckpt_dir, config=cfg, strict=True, local_files_only=True)
    else:
        with caplog.at_level(logging.WARNING):
            restored = SmolVLAPolicy.from_pretrained(
                ckpt_dir, config=cfg, strict=False, local_files_only=True
            )
        assert "unexpected_extra_tensor" in caplog.text
        source_policy.eval()
        _assert_policies_match_exactly(restored, source_policy)


@skip_if_package_missing("transformers")
@pytest.mark.parametrize(
    ("load_vlm_weights", "train_expert_only", "freeze_vision_encoder"),
    [
        (False, True, False),
        (True, False, True),
    ],
)
def test_from_pretrained_preserves_requires_grad_and_dtype_modes(
    tmp_path, monkeypatch, load_vlm_weights, train_expert_only, freeze_vision_encoder
):
    from lerobot.policies.smolvla import smolvlm_with_expert

    full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path)
    torch.manual_seed(23)
    cfg = _make_tiny_smolvla_config(
        full_dir,
        load_vlm_weights=load_vlm_weights,
        train_expert_only=train_expert_only,
        freeze_vision_encoder=freeze_vision_encoder,
    )
    source_policy = SmolVLAPolicy(cfg)
    ckpt_dir = tmp_path / f"modes_{load_vlm_weights}_{train_expert_only}_{freeze_vision_encoder}"
    source_policy.save_pretrained(ckpt_dir)

    load_cfg = PreTrainedConfig.from_pretrained(ckpt_dir, local_files_only=True)
    assert isinstance(load_cfg, SmolVLAConfig)
    load_cfg.vlm_model_name = str(meta_dir)

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected backbone load")),
    )
    restored = SmolVLAPolicy.from_pretrained(ckpt_dir, config=load_cfg, strict=True, local_files_only=True)
    source_policy.eval()
    _assert_policies_match_exactly(restored, source_policy)


@skip_if_package_missing("transformers")
def test_direct_init_with_load_vlm_weights_still_loads_backbone(tmp_path):
    _full_dir, meta_dir = _build_tiny_vlm_dirs(tmp_path)
    cfg = _make_tiny_smolvla_config(meta_dir, load_vlm_weights=True)
    with pytest.raises(OSError):
        SmolVLAPolicy(cfg)
