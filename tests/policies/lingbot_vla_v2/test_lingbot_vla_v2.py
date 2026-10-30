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
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.utils import skip_if_package_missing

CAMERA = "observation.images.camera_top"
WRIST = "observation.images.camera_wrist"
CHUNK = 4


def _slots(key: str, arm: list, effector: list) -> dict[str, SlotMapping]:
    def mapping(spans):
        return SlotMapping(origin_keys=[{key: {"start": s, "end": e}} for s, e in spans])

    return {f"{key}.arm.position": mapping(arm), f"{key}.effector.position": mapping(effector)}


def _slot_kwargs(arm: list, effector: list) -> dict:
    return {"state_slots": _slots(OBS_STATE, arm, effector), "action_slots": _slots(ACTION, arm, effector)}


def _ds_meta(dim: int, cameras: tuple[str, ...] = (CAMERA,)) -> SimpleNamespace:
    """The dataset metadata make_policy reads: features and stats of a ``dim``-D robot."""
    names = [f"joint_{i}" for i in range(dim - 1)] + ["gripper"]
    vector = {"dtype": "float32", "shape": (dim,), "names": names}
    image = {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
    features = {OBS_STATE: vector, ACTION: vector, **dict.fromkeys(cameras, image)}
    g = torch.Generator().manual_seed(dim)
    stats = {}
    for key in (OBS_STATE, ACTION):
        lo, hi = -1 - torch.rand(dim, generator=g), 1 + torch.rand(dim, generator=g)
        stats[key] = {"min": lo, "max": hi, "q01": lo, "q99": hi, "mean": lo + hi, "std": hi}
    return SimpleNamespace(features=features, stats=stats)


def _raw_batch(
    dim: int, device: str = "cpu", batch_size: int = 2, cameras: tuple[str, ...] = (CAMERA,)
) -> dict:
    g = torch.Generator().manual_seed(0)
    batch = {
        OBS_STATE: torch.randn(batch_size, dim, generator=g),
        **{camera: torch.rand(batch_size, 3, 32, 32, generator=g) for camera in cameras},
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


def _tiny_config(
    assets, device: str = "cpu", dtype: torch.dtype = torch.float32, **kwargs
) -> LingbotVLAV2Config:
    return LingbotVLAV2Config(
        tokenizer_path=str(assets),
        device=device,
        dtype=dtype,
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


def _policy_and_batch(tiny_backbone, cameras, vision_boundaries=True, **config):
    """A policy, a preprocessed batch without actions, and noise."""
    slots = _slot_kwargs(arm=[[0, 6]], effector=[[6, 7]])
    meta = _ds_meta(7, cameras)
    cfg = _tiny_config(tiny_backbone, qwen3vl_use_vision_boundaries=vision_boundaries, **slots, **config)
    policy = make_policy(cfg, ds_meta=meta).eval()
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    raw = _raw_batch(7, cameras=cameras)
    raw.pop(ACTION)
    return policy, preprocessor(raw), torch.randn(2, CHUNK, cfg.max_action_dim, dtype=cfg.dtype)


class _PredictActionChunk(torch.nn.Module):
    """`predict_action_chunk` with the batch tensors as positional inputs, so torch.export can trace it."""

    def __init__(self, policy, batch, grid=None):
        super().__init__()
        self.policy, self.batch, self.grid = policy, batch, grid
        self.keys = sorted(key for key, value in batch.items() if isinstance(value, torch.Tensor))

    def forward(self, noise, *tensors):
        return self.policy.predict_action_chunk(
            {**self.batch, **dict(zip(self.keys, tensors, strict=True))}, noise=noise, vision_grid=self.grid
        )

    def inputs(self, noise, batch):
        # The sampler updates its noise in place, so every call gets its own copy.
        return (noise.clone(), *(batch[key] for key in self.keys))


def _grid(policy, batch):
    from lerobot.policies.common.qwen3_vision import Qwen3VisionGrid

    visual = policy.model.qwenvl_with_expert.qwenvl.model.visual
    return Qwen3VisionGrid.build(visual, batch["image_grid_thw"].flatten(0, -2))


def test_prediction_on_a_vision_grid_exports_and_matches_eager(tiny_backbone, monkeypatch):
    """Eager prediction keeps transformers' prefix positions; on a vision grid, prediction exports. Two
    cameras are visible, with vision boundaries, in bfloat16: the default layout and dtype."""
    from tests.policies.common.fixed_grid_checks import data_dependent_nodes

    # The runtime asserts must not depend on transformers' own checks, which this switch turns off.
    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA, WRIST), dtype=torch.bfloat16)
    core = policy.model.qwenvl_with_expert
    build_prefix_position_ids, prefixes = core.build_prefix_position_ids, []

    def recorded(input_ids, attention_mask, **kwargs):
        positions = build_prefix_position_ids(input_ids, attention_mask, **kwargs)
        prefixes.append((input_ids, attention_mask, positions))
        return positions

    monkeypatch.setattr(core, "build_prefix_position_ids", recorded)
    stock = _PredictActionChunk(policy, batch)
    module = _PredictActionChunk(policy, batch, _grid(policy, batch))
    with torch.no_grad():
        expected = stock(*stock.inputs(noise, batch)).clone()
        # On the grid, eager prediction runs the export path and gives the same actions.
        assert torch.equal(module(*module.inputs(noise, batch)), expected)
        # As loading weights leaves it, the experts' inference cache is empty, and tracing must not fill it.
        for experts in (m for m in policy.modules() if hasattr(m, "clear_inference_cache")):
            experts.clear_inference_cache()
        program = torch.export.export(module, module.inputs(noise, batch), strict=False)
    assert data_dependent_nodes(program) == []
    assert not [name for name, _ in policy.named_buffers() if "cache" in name]
    # A grid other than the one the program holds, and a sample with no visible prefix token, on which eager rope
    # positions fail, are refused.
    changed = {**batch, "image_grid_thw": batch["image_grid_thw"].clone()}
    changed["image_grid_thw"][..., 1:] = torch.tensor([2, 8])
    empty = {key: batch[key].clone() for key in ("img_masks", OBS_LANGUAGE_ATTENTION_MASK)}
    for mask in empty.values():
        mask[1] = False
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*module.inputs(noise, batch)), expected, rtol=0, atol=0)
        for wrong in (changed, {**batch, **empty}):
            with pytest.raises(RuntimeError, match="Runtime assertion failed"):
                program.module()(*module.inputs(noise, wrong))

    # The positions the grid gives, computed from the grid ints of every camera, match transformers'.
    input_ids, attention_mask, positions = prefixes[0]
    every_grid = batch["image_grid_thw"].reshape(-1, 3).tolist()
    assert torch.equal(core.build_fixed_grid_position_ids(input_ids, attention_mask, every_grid), positions)


def test_fixed_grid_positions_match_get_rope_index_with_several_cameras(tiny_backbone):
    """Camera grids with one token count but different shapes, and another camera hidden in each sample."""
    policy, *_ = _policy_and_batch(tiny_backbone, (CAMERA,))
    core = policy.model.qwenvl_with_expert
    cfg = core.qwenvl.config
    shapes = [[1, 4, 4], [1, 2, 8], [1, 8, 2]]
    shown = {1: [[1], [1]], 2: [[0, 1], [1, 0]], 3: [[0, 1, 1], [1, 0, 1]]}
    for boundaries in (True, False):
        image = [cfg.image_token_id] * 4
        if boundaries:
            image = [cfg.vision_start_token_id, *image, cfg.vision_end_token_id]
        for cameras, rows in shown.items():
            visible = torch.tensor(rows, dtype=torch.bool)
            if not boundaries:  # eager allows one visible camera per sample without them
                visible &= visible.cumsum(dim=1) == 1
            grids = [shapes[(sample + camera) % 3] for sample in range(2) for camera in range(cameras)]
            input_ids = torch.tensor([image * cameras + [5, 6, 7]] * 2)
            text = torch.ones(2, 3, dtype=torch.bool)
            mask = torch.cat([visible.repeat_interleave(len(image), dim=1), text], dim=1).long()
            # Eager prediction gives `get_rope_index` the grids of the visible cameras only.
            visible_grids = torch.tensor(grids)[visible.flatten()]
            eager = core.build_prefix_position_ids(input_ids, mask, image_grid_thw=visible_grids)
            assert torch.equal(core.build_fixed_grid_position_ids(input_ids, mask, grids), eager)


def test_export_follows_the_vision_grid_not_the_precompute_cache(tiny_backbone):
    """`precompute_grid_thw` caches the grid of its first call; export uses `vision_grid`."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA,))
    precompute, *_ = _policy_and_batch(tiny_backbone, (CAMERA,), precompute_grid_thw=True)
    precompute.load_state_dict(policy.state_dict())
    module = _PredictActionChunk(precompute, batch, _grid(precompute, batch))
    # Another grid with the same token count, so the option keeps the cache it builds here.
    other = {**batch, "image_grid_thw": batch["image_grid_thw"].clone()}
    other["image_grid_thw"][..., 1:] = torch.tensor([2, 8])
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch, noise=noise.clone())
        precompute.predict_action_chunk(other, noise=noise.clone())
        stock = _PredictActionChunk(precompute, batch)
        with pytest.raises(ValueError, match="needs a vision_grid"):
            torch.export.export(stock, stock.inputs(noise, batch), strict=False)
        program = torch.export.export(module, module.inputs(noise, batch), strict=False).module()
        torch.testing.assert_close(program(*module.inputs(noise, batch)), expected)


def test_adjacent_cameras_without_vision_boundaries_are_refused(tiny_backbone):
    """Without boundary tokens two visible cameras form one image block, which has no rope positions."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA, WRIST), vision_boundaries=False)
    with torch.no_grad(), pytest.raises(ValueError, match="at most one visible camera per sample"):
        policy.predict_action_chunk(batch, noise=noise)


def test_vision_grid_without_vision_boundaries_allows_one_visible_camera(tiny_backbone):
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA, WRIST), vision_boundaries=False)
    wrist_hidden = {**batch, "img_masks": batch["img_masks"].clone()}
    wrist_hidden["img_masks"][:, 1] = False
    module = _PredictActionChunk(policy, wrist_hidden, _grid(policy, wrist_hidden))
    with torch.no_grad():
        expected = policy.predict_action_chunk(wrist_hidden, noise=noise.clone())
        with pytest.raises(ValueError, match="at most one visible camera per sample"):
            module(*module.inputs(noise, batch))
        program = torch.export.export(module, module.inputs(noise, wrist_hidden), strict=False).module()
        torch.testing.assert_close(program(*module.inputs(noise, wrist_hidden)), expected)
        with pytest.raises(RuntimeError, match="Runtime assertion failed"):
            program(*module.inputs(noise, batch))


def test_vision_grid_refuses_cameras_with_different_token_counts(tiny_backbone):
    """The grid path splits the tower output evenly per camera, as eager stacks it: one token count only."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA, WRIST))
    # Each sample keeps its 32 patches, split 8 and 24: the tower runs, an even split would mix the cameras.
    uneven = {**batch, "image_grid_thw": batch["image_grid_thw"].clone()}
    uneven["image_grid_thw"][:, 0, 1:] = torch.tensor([2, 4])
    uneven["image_grid_thw"][:, 1, 1:] = torch.tensor([4, 6])
    with torch.no_grad(), pytest.raises(ValueError, match="same number of image tokens"):
        policy.predict_action_chunk(uneven, noise=noise, vision_grid=_grid(policy, uneven))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_full_cuda_graph_prefix_takes_a_vision_grid(tiny_backbone):
    """With a grid, `use_cudagraph_prefix_full` leaves the stock vision tower out of its graphs and captures
    only the KV fill, as `use_cudagraph_prefix` does; the first call captures it, the second replays it."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA,))
    policy = policy.to("cuda")
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    noise, grid = noise.cuda(), _grid(policy, batch)
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch, noise=noise.clone())
        policy.config.use_cudagraph_prefix_full = True
        for _ in range(2):
            actions = policy.predict_action_chunk(batch, noise=noise.clone(), vision_grid=grid)
            torch.testing.assert_close(actions, expected, rtol=0, atol=0)
    assert getattr(policy.model, "_prefix_graph_state", None) is not None
    assert getattr(policy.model, "_vision_graph_state", None) is None


def test_export_leaves_compile_predict_velocity_to_eager_calls(tiny_backbone, monkeypatch):
    """torch.compile under torch.export gives back the plain method, which would then be cached for every
    later eager call."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA,))
    compiled = []

    def compile_with_mode(self, fn):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(type(policy.model), "_compile_with_mode", compile_with_mode)
    module = _PredictActionChunk(policy, batch, _grid(policy, batch))
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch, noise=noise.clone())
        policy.config.compile_predict_velocity = True
        program = torch.export.export(module, module.inputs(noise, batch), strict=False).module()
        assert not compiled and getattr(policy.model, "_compiled_predict_velocity", None) is None
        torch.testing.assert_close(program(*module.inputs(noise, batch)), expected, rtol=0, atol=0)
        # The next eager call compiles, as the option asks.
        torch.testing.assert_close(policy.predict_action_chunk(batch, noise=noise.clone()), expected)
    assert len(compiled) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("graph", ["use_cudagraph_prefix", "use_cudagraph_denoise"])
def test_export_with_cuda_graphs_leaves_the_policy_intact(tiny_backbone, graph):
    """Export skips the CUDA graphs: captured while torch.export traces, they would hold traced tensors."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA,))
    policy = policy.to("cuda")
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    noise = noise.cuda()
    module = _PredictActionChunk(policy, batch, _grid(policy, batch))
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch, noise=noise.clone())
        setattr(policy.config, graph, True)
        program = torch.export.export(module, module.inputs(noise, batch), strict=False)
        torch.testing.assert_close(program.module()(*module.inputs(noise, batch)), expected, rtol=0, atol=0)
        # The first eager call after the export captures the graph, which then matches plain prediction.
        for _ in range(2):
            torch.testing.assert_close(
                policy.predict_action_chunk(batch, noise=noise.clone()), expected, rtol=0, atol=0
            )


def test_export_refuses_float32_flash_attention_in_the_vision_tower(tiny_backbone):
    """Eager runs float32 flash attention in bfloat16, which the exported tower does not reproduce."""
    policy, batch, noise = _policy_and_batch(tiny_backbone, (CAMERA,))
    module = _PredictActionChunk(policy, batch, _grid(policy, batch))
    with torch.no_grad():
        # Set after building, which would need flash_attn; export refuses before attention runs.
        policy.model.qwenvl_with_expert.qwenvl.model.visual.config._attn_implementation = "flash_attention_2"
        with pytest.raises(ValueError, match="flash_attention_2 needs bfloat16 weights"):
            torch.export.export(module, module.inputs(noise, batch), strict=False)


def test_align_query_tokens(tiny_backbone, monkeypatch):
    """A tiny RoboTwin-style align recipe appends 2 x num_task_tokens query tokens to the prefix, and a
    training step on [current, future] frames adds the distillation losses (teachers stubbed: no download)."""
    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    head = {"num_layers": 1, "num_heads": 2, "dim_head": 8, "ff_mult": 1, "num_backbone_tokens": 8}
    shared = {"share_future_depth_query": True, "use_shared_future_task_proj": True}
    align = {"mode": "query", "num_task_tokens": 4, "use_future_video": True, "depth_loss_weight": 0.004}
    align["llm"] = {"dim_out": 64, "image_token_size": 2}
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

    class Teachers:  # the frozen teachers' target shapes: [B, num_backbone_tokens, dim_out]
        def __init__(self, params, device):
            pass

        def targets(self, current, future, fps):
            assert current.shape == future.shape == (2, 3, 64, 64)
            names = ("depth", "future_depth", "future_video", "current_video")
            return {name: torch.randn(2, 8, 16) for name in names}

    monkeypatch.setattr(modeling, "DistillationTeachers", Teachers)
    assert cfg.image_observation_delta_indices == [0, CHUNK - 1]
    batch = _raw_batch(7)
    batch[CAMERA] = torch.stack([batch[CAMERA], batch[CAMERA].flip(-1)], dim=1)  # current, future
    policy.train()
    loss, logs = policy.forward(preprocessor(batch))
    assert {"depth_loss", "future_depth_loss", "future_video_loss", "current_video_loss"} <= logs.keys()
    loss.backward()
    assert policy.model.future_video_align_head.projector.proj_out.weight.grad.abs().sum() > 0


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

    # An action dim the slots leave out can't be produced by the inverse map, so it is rejected up front.
    cfg.action_slots = _slots(ACTION, [[0, 5], *arm[1:]], effector)
    with pytest.raises(ValueError, match=r"unmapped indices: \[5\]"):
        cfg.validate_features()


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
    cfg.tokenizer_max_length = 24  # --policy.tokenizer_max_length, applied over the saved processor's 16
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
    batch = preprocessor(_raw_batch(6, device))
    assert batch[OBS_LANGUAGE_TOKENS].shape[-1] == 24
    loss, _ = policy.forward(batch)
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


def test_default_image_settings_match_upstream(tiny_backbone):
    """Upstream feeds 64 tokens per camera (256x256 frames at 65536 px), which the distillation heads assume."""
    from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import LingbotVLAV2ImageProcessorStep
    from lerobot.processor import TransitionKey

    defaults = LingbotVLAV2Config()
    step = LingbotVLAV2ImageProcessorStep(
        tokenizer_path=str(tiny_backbone),
        cameras=["camera_top"],
        resize_imgs_with_padding=defaults.resize_imgs_with_padding,
        image_max_pixels=defaults.image_max_pixels,
        image_min_pixels=defaults.image_min_pixels,
    )
    out = step({TransitionKey.OBSERVATION: {CAMERA: torch.rand(1, 3, 480, 640)}})
    grid = out[TransitionKey.OBSERVATION]["image_grid_thw"][0, 0]
    assert int(grid.prod()) // 4 == 64 == 8 * 8  # 2x2 merge; upstream `llm.image_token_size: 8`


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


def test_a_given_noise_runs_in_the_model_dtype(tiny_backbone):
    """A float32 starting noise, as callers draw it, runs in the dtype of a bfloat16 policy."""
    policy, batch, _ = _policy_and_batch(tiny_backbone, (CAMERA,), dtype=torch.bfloat16)
    noise = torch.randn(2, CHUNK, policy.config.max_action_dim)

    with torch.no_grad():
        given = policy.predict_action_chunk(batch, noise=noise)
        cast = policy.predict_action_chunk(batch, noise=noise.to(torch.bfloat16))

    torch.testing.assert_close(given, cast, rtol=0, atol=0)
