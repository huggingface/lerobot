# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""LingBot-VLA 2.0 tests on a tiny randomly initialized model, fully offline.

The Qwen3-VL config is patched to a 2-layer model and the tokenizer / image processor are
built locally, so every test runs the real processors and policy without the Hub.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config, SlotMapping
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.utils import skip_if_package_missing

CAMERA = "observation.images.camera_top"
CHUNK = 4


def _slots(key: str, arm: list, effector: list) -> dict[str, SlotMapping]:
    def mapping(spans):
        return SlotMapping(origin_keys=[{key: {"start": s, "end": e}} for s, e in spans])

    return {f"{key}.arm.position": mapping(arm), f"{key}.effector.position": mapping(effector)}


def _slot_kwargs(arm: list, effector: list) -> dict:
    return {"state_slots": _slots(OBS_STATE, arm, effector), "action_slots": _slots(ACTION, arm, effector)}


def _ds_meta(dim: int) -> SimpleNamespace:
    """The dataset metadata make_policy reads: features and stats of a ``dim``-D robot."""
    names = [f"joint_{i}" for i in range(dim - 1)] + ["gripper"]
    vector = {"dtype": "float32", "shape": (dim,), "names": names}
    features = {
        OBS_STATE: vector,
        ACTION: vector,
        CAMERA: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
    }
    g = torch.Generator().manual_seed(dim)
    stats = {}
    for key in (OBS_STATE, ACTION):
        lo, hi = -1 - torch.rand(dim, generator=g), 1 + torch.rand(dim, generator=g)
        stats[key] = {"min": lo, "max": hi, "q01": lo, "q99": hi, "mean": lo + hi, "std": hi}
    return SimpleNamespace(features=features, stats=stats)


def _raw_batch(dim: int, device: str = "cpu", batch_size: int = 2) -> dict:
    g = torch.Generator().manual_seed(0)
    batch = {
        OBS_STATE: torch.randn(batch_size, dim, generator=g),
        CAMERA: torch.rand(batch_size, 3, 32, 32, generator=g),
        ACTION: torch.randn(batch_size, CHUNK, dim, generator=g),
    }
    return {**{k: v.to(device) for k, v in batch.items()}, "task": ["pick up the cube"] * batch_size}


@pytest.fixture
@skip_if_package_missing("transformers")
def tiny_backbone(tmp_path, monkeypatch):
    """Patch the Qwen3-VL config to a tiny model and save an offline tokenizer + image processor."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor
    from transformers.models.qwen3_vl import Qwen3VLConfig

    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    vlm = Qwen3VLConfig()
    t, v = vlm.text_config, vlm.vision_config
    t.num_hidden_layers, t.hidden_size, t.intermediate_size = 2, 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_parameters = {**t.rope_parameters, "mrope_section": [2, 3, 3]}
    t.eos_token_id = 3
    v.hidden_size, v.intermediate_size, v.num_heads, v.depth, v.out_hidden_size = 64, 128, 2, 2, 64
    v.deepstack_visual_indexes = [0, 1]
    monkeypatch.setattr(modeling.AutoConfig, "from_pretrained", lambda *args, **kwargs: vlm)

    words = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "user", "pick", "up", "the", "cube"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|>{% endfor %}"
    )
    assets = tmp_path / "qwen_assets"
    tokenizer.save_pretrained(assets)
    Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2).save_pretrained(assets)
    return assets


def _tiny_config(assets, device: str = "cpu", **kwargs) -> LingbotVLAV2Config:
    return LingbotVLAV2Config(
        tokenizer_path=str(assets),
        device=device,
        dtype=torch.float32,
        expert_hidden_size=64,
        expert_intermediate_size=128,
        action_num_attention_heads=4,
        action_num_key_value_heads=2,
        action_head_dim=16,
        token_moe_layers=[0, 1],
        token_num_experts=4,
        token_top_k=2,
        token_moe_intermediate_size=32,
        token_shared_intermediate_size=32,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        num_steps=2,
        tokenizer_max_length=16,
        resize_imgs_with_padding=(64, 64),
        image_max_pixels=64 * 64,
        image_min_pixels=64 * 64,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
        **kwargs,
    )


def _select_action(policy, preprocessor, postprocessor, dim: int, device: str) -> torch.Tensor:
    batch = _raw_batch(dim, device)
    batch.pop(ACTION)
    policy.reset()
    return postprocessor(policy.select_action(preprocessor(batch)))


def test_train_and_select_action(tiny_backbone):
    """7-D robot (arm 0-6, gripper 6-7, canonical ``end.position`` unmapped): train step + inference."""
    cfg = _tiny_config(tiny_backbone, **_slot_kwargs(arm=[[0, 6]], effector=[[6, 7]]))
    meta = _ds_meta(7)
    policy = make_policy(cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    loss, _ = policy.forward(preprocessor(_raw_batch(7)))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())

    action = _select_action(policy, preprocessor, postprocessor, 7, "cpu")
    assert action.shape == (2, 7)
    assert torch.isfinite(action).all()


def test_align_query_tokens(tiny_backbone):
    """A tiny RoboTwin-style align recipe appends 2 x num_task_tokens query tokens to the prefix."""
    head = {"num_layers": 1, "num_heads": 2, "dim_head": 8, "ff_mult": 1, "num_backbone_tokens": 8}
    shared = {"share_future_depth_query": True, "use_shared_future_task_proj": True}
    align = {"mode": "query", "num_task_tokens": 4, "use_future_video": True, "llm": {"dim_out": 64}}
    align["depth"] = {**head, "dim_out": 16, "use_future_depth": True}
    align["video"] = {**head, **shared, "dim_out": 16, "use_current_patch_loss": True}
    slots, meta = _slot_kwargs(arm=[[0, 6]], effector=[[6, 7]]), _ds_meta(7)
    cfg = _tiny_config(tiny_backbone, align_params=align, **slots)
    policy = make_policy(cfg, ds_meta=meta)
    plain = make_policy(_tiny_config(tiny_backbone, **slots), ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    images, img_masks, tokens, masks, _, grid = policy._extract_model_inputs(preprocessor(_raw_batch(7)))
    prefix_len = policy.model.embed_prefix(images, img_masks, tokens, masks, grid)[0].shape[1]
    assert prefix_len == plain.model.embed_prefix(images, img_masks, tokens, masks, grid)[0].shape[1] + 8
    assert _select_action(policy, preprocessor, postprocessor, 7, "cpu").shape == (2, 7)


@pytest.mark.parametrize(
    ("arm", "effector"),
    [([[0, 6]], [[6, 7]]), ([[0, 6], [7, 13]], [[6, 7], [13, 14]])],
    ids=["single_arm", "robotwin"],
)
def test_slot_mapping_round_trip(tiny_backbone, arm, effector):
    """Preprocessor (normalize + slot map) then postprocessor (inverse map + unnormalize) is the identity."""
    dim = max(end for _, end in arm + effector)
    cfg = _tiny_config(tiny_backbone, **_slot_kwargs(arm, effector))
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(dim,)),
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(dim,))}
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_ds_meta(dim).stats)

    raw = _raw_batch(dim)
    canonical = preprocessor(raw)[ACTION]
    assert canonical.shape == (2, CHUNK, cfg.max_action_dim)
    # The gripper lands in the canonical effector slot (after 14 arm + 14 end-effector dims).
    effector_start = cfg.canonical_joints["arm.position"] + cfg.canonical_joints["end.position"]
    assert canonical[..., effector_start : effector_start + len(effector)].abs().sum() > 0
    torch.testing.assert_close(postprocessor(canonical), raw[ACTION], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_finetune_save_reload_select_action(tiny_backbone, tmp_path, device):
    """Fine-tune a 7-D checkpoint on a 6-D robot, save, then reload it the way eval/rollout do."""
    base_dir, finetuned_dir = tmp_path / "base", tmp_path / "finetuned"
    base_cfg = _tiny_config(tiny_backbone, device=device, **_slot_kwargs(arm=[[0, 6]], effector=[[6, 7]]))
    base_meta = _ds_meta(7)
    base = make_policy(base_cfg, ds_meta=base_meta)
    for obj in (base, *make_pre_post_processors(base_cfg, dataset_stats=base_meta.stats)):
        obj.save_pretrained(base_dir)

    # lerobot-train --policy.path=<base> --policy.{state,action}_slots=<6-D robot>
    cfg = PreTrainedConfig.from_pretrained(base_dir)
    cfg.pretrained_path = base_dir
    cfg.device = device
    cfg.state_slots, cfg.action_slots = _slot_kwargs(arm=[[0, 5]], effector=[[5, 6]]).values()
    meta = _ds_meta(6)
    policy = make_policy(cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=base_dir,
        dataset_stats=meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": meta.stats,
                "features": cfg.output_features,
                "norm_map": cfg.normalization_mapping,
            },
        },
    )
    # The accelerate dataloader hands the preprocessor tensors already on the training device.
    loss, _ = policy.forward(preprocessor(_raw_batch(6, device)))
    loss.backward()
    assert torch.isfinite(loss)
    for obj in (policy, preprocessor, postprocessor):
        obj.save_pretrained(finetuned_dir)
    saved = json.loads((finetuned_dir / "config.json").read_text())
    assert saved["input_features"][OBS_STATE]["shape"] == [6]
    assert saved["output_features"][ACTION]["shape"] == [6]

    # lerobot-eval / lerobot-rollout: everything comes from the fine-tuned checkpoint.
    cfg = PreTrainedConfig.from_pretrained(finetuned_dir)
    cfg.pretrained_path = finetuned_dir
    policy = make_policy(cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=finetuned_dir)
    action = _select_action(policy, preprocessor, postprocessor, 6, device)
    assert action.shape == (2, 6)
    assert torch.isfinite(action).all()


@skip_if_package_missing("transformers")
def test_sparse_moe_matches_per_token_reference():
    """The padded sparse expert path equals a naive per-token top-k mixture."""
    import torch.nn.functional as F  # noqa: N812

    from lerobot.policies.lingbot_vla_v2.model_core.qwen2_action_expert import Qwen2FusedExperts

    torch.manual_seed(0)
    experts = Qwen2FusedExperts(num_experts=6, hidden_size=16, intermediate_size=8)
    hidden = torch.randn(40, 16)
    scores = torch.randn(40, 6).sigmoid()
    scores[:, 4:] = 0  # experts 4 and 5 receive no tokens
    weights, selected = torch.topk(scores, k=2, dim=-1)

    expected = torch.zeros_like(hidden)
    for k in range(selected.shape[-1]):
        e = selected[:, k]
        gate = torch.einsum("td,tjd->tj", hidden, experts.gate_proj[e])
        up = torch.einsum("td,tjd->tj", hidden, experts.up_proj[e])
        expected += weights[:, k, None] * torch.einsum("tj,tdj->td", F.silu(gate) * up, experts.down_proj[e])

    actual = experts._sparse_forward(weights, selected, hidden)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
