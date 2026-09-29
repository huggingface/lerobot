#!/usr/bin/env python

# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""CPU tests of the flux3 policy: registration, config, processors, the packing contract, a tiny end-to-end
model with a fake video VAE and the mock text encoder, checkpoint round trips and the key remap."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.schedulers import FrozenWarmupConstantSchedulerConfig
from lerobot.policies import factory
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors
from lerobot.policies.flux3.f3 import (
    JointSingleSeq,
    JointSingleSeqParams,
    action_dit_params,
    fresh_head_state_dict,
    load_text_encoder,
    packing,
    sampling,
    stream_of_key,
)
from lerobot.policies.flux3.f3.wiring import (
    head_parameter_names,
    load_action_checkpoint,
    remap_shared_action_keys,
)
from lerobot.policies.flux3.utils import grid_shape
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _datasets_available, _diffusers_available, _peft_available
from tests.policies.flux3.helpers import (
    DROID_CAMS,
    FRAME_REFERENCE_SETTINGS,
    TINY_DIT,
    FakeVideoVAE,
    droid_config,
    droid_features,
    single_config,
    task_config,
)

if TYPE_CHECKING or _datasets_available:
    from lerobot.scripts.lerobot_train import _ema_parameters, _ema_weights
if TYPE_CHECKING or _diffusers_available:
    from diffusers.training_utils import EMAModel
if TYPE_CHECKING or _peft_available:
    from peft import PeftModel


@pytest.fixture
def fake_vae(monkeypatch, fake_text_encoder):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())


def droid_batch(b: int = 2, frames: int | None = None, pad_second: bool = False) -> dict:
    t = frames or 1
    gen = torch.Generator().manual_seed(1)
    batch = {}
    for k in DROID_CAMS:
        img = torch.rand(b, t, 3, 360, 640, generator=gen)
        batch[k] = img if frames else img[:, 0]
    batch[OBS_STATE] = torch.rand(b, 1, 8, generator=gen) if frames else torch.rand(b, 8, generator=gen)
    batch[ACTION] = torch.rand(b, 32, 8, generator=gen)
    batch["task"] = ["pick the pen from the purple mat", "put it on the tray"][:b]
    if frames:
        pad = torch.zeros(b, 32, dtype=torch.bool)
        if pad_second and b > 1:
            pad[1, -1] = True
        batch["action_is_pad"] = pad
    return batch


# ---------------------------------------------------------------- registration / config
@pytest.mark.parametrize("spec", [None, "", "   "])
def test_missing_text_encoder_is_rejected(spec):
    with pytest.raises(ValueError, match="text_encoder_id must be a non-empty Hub ID or local path"):
        Flux3Config(text_encoder_id=spec)
    with pytest.raises(ValueError, match="text_encoder_id must be a non-empty Hub ID or local path"):
        load_text_encoder(spec)


def test_flux3_is_registered_and_publicly_exported():
    inputs, outputs = droid_features()
    cfg = make_policy_config(
        "flux3",
        **FRAME_REFERENCE_SETTINGS,
        input_features=inputs,
        output_features=outputs,
        camera_keys=DROID_CAMS,
        camera_layout="droid",
    )
    assert isinstance(cfg, Flux3Config) and cfg.type == "flux3"
    assert get_policy_class("flux3") is Flux3Policy
    assert "flux3" in PreTrainedConfig.get_known_choices()


@pytest.mark.parametrize("large_budget", [False, True])
@pytest.mark.parametrize("robot", ["so101", "so101_overrides", "droid"])
def test_shared_peft_recipe_preserves_checkpoint_contract(tmp_path, monkeypatch, large_budget, robot):
    # Use saved full-finetuning settings that differ from the PEFT recipe, including
    # tuple-valued Adam betas, to catch silently dropped config overrides.
    checkpoint = tmp_path / robot
    original = (
        droid_config(action_modality="action_prediction_droid")
        if robot == "droid"
        else task_config(n_action_steps=30, guidance_scale=3.0, guidance_scale_action=None)
    )
    original.save_pretrained(checkpoint)
    saved = (checkpoint / "config.json").read_bytes()
    recipe = Path(__file__).resolve().parents[3] / "examples/flux3/lora.json"
    overrides = json.loads(recipe.read_text())["policy"]
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    args = [
        "lerobot-train",
        f"--config_path={recipe}",
        f"--policy.path={checkpoint}",
        "--policy.device=cpu",
        f"--dataset.repo_id=test/{robot}-task",
        f"--output_dir={tmp_path / 'train'}",
    ]
    runtime_overrides = {}
    if robot == "so101_overrides":
        runtime_overrides = {
            "n_action_steps": 32,
            "guidance_scale": 4.0,
            "guidance_scale_action": 1.0,
            "compile_model": True,
        }
        args.extend(f"--policy.{key}={json.dumps(value)}" for key, value in runtime_overrides.items())
    if large_budget:
        args.extend(["--batch_size=8", "--steps=14000"])
    monkeypatch.setattr(sys, "argv", args)

    @parser.wrap()
    def parse_recipe(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    train = parse_recipe()
    assert str(train.policy.pretrained_path) == str(checkpoint)
    for key in vars(original):
        if key not in overrides and key not in runtime_overrides and key != "pretrained_path":
            assert getattr(train.policy, key) == getattr(original, key), key
    for key, value in runtime_overrides.items():
        assert getattr(train.policy, key) == value
    assert train.dataset.repo_id == f"test/{robot}-task" and train.dataset.eval_split == 0.2
    assert train.optimizer.lr == 1e-4
    assert train.policy.optimizer_lr * train.policy.optimizer_lr_heads_multiplier == 5e-4
    assert train.optimizer.betas == (0.9, 0.999)
    assert train.optimizer.weight_decay == 0 and train.optimizer.grad_clip_norm == 1
    assert all(
        value == 0
        for value in (
            train.scheduler.freeze_steps,
            train.scheduler.num_warmup_steps,
            train.scheduler.warmup_steps_heads,
            train.scheduler.decay_steps,
            train.scheduler.cooldown_steps,
        )
    )
    assert train.peft.r == train.peft.lora_alpha == 32
    assert train.ema.enable and train.ema.decay == 0.999
    assert train.policy.caption_dropout == 0.1 and train.policy.gradient_checkpointing
    assert not train.policy.push_to_hub and not train.policy.augment
    accumulation = train.accelerator.gradient_accumulation.steps
    assert accumulation == 4
    assert train.batch_size * accumulation * (4 if large_budget else 1) == (128 if large_budget else 8)
    assert train.steps // accumulation == (3500 if large_budget else 2500)
    assert train.save_freq == train.eval_steps == 500
    assert (checkpoint / "config.json").read_bytes() == saved


def test_config_droid_reference_settings():
    cfg = droid_config()
    assert cfg.chunk_size == 32 and cfg.fps == 15.0 and cfg.canvas_hw == (544, 736)
    assert cfg.latent_hw == (17, 20) and cfg.action_dim == 8 and cfg.window_frames == 33
    assert math.isclose(cfg.action_loss_weight, 50.0) and cfg.action_scale == 2.0
    assert (cfg.num_inference_steps, cfg.guidance_scale, cfg.sampler_shift) == (4, 4.0, 5.0)
    assert cfg.guidance_scale_action == 1.0
    assert cfg.observation_delta_indices == list(range(33))
    assert cfg.image_observation_delta_indices == list(range(33))
    assert cfg.state_observation_delta_indices == [0]
    assert cfg.action_delta_indices == list(range(32))
    assert cfg.reward_delta_indices is None
    opt = cfg.get_optimizer_preset()
    assert math.isclose(opt.lr, 1.92e-4) and opt.betas == (0.9, 0.99) and opt.weight_decay == 0.05
    sched = cfg.get_scheduler_preset()
    assert (sched.freeze_steps, sched.num_warmup_steps, sched.warmup_steps_heads) == (1000, 2000, 1000)


def test_config_camera_order_and_validation():
    inputs, outputs = droid_features()
    cfg = Flux3Config(input_features=inputs, output_features=outputs, camera_layout="droid")
    assert cfg.camera_order[0] == "observation.images.wrist_image_left"
    with pytest.raises(ValueError, match="needs 3 camera"):
        two = {k: v for k, v in inputs.items() if "exterior_image_2" not in k}
        Flux3Config(input_features=two, output_features=outputs, camera_layout="droid")
    with pytest.raises(ValueError, match="state dim"):
        bad = dict(inputs)
        bad[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
        Flux3Config(input_features=bad, output_features=outputs, camera_layout="droid")
    with pytest.raises(ValueError, match="camera_layout"):
        Flux3Config(camera_layout="mosaic")
    with pytest.raises(ValueError, match="n_action_steps"):
        Flux3Config(n_action_steps=40)
    single = single_config()
    assert single.drop_n_first_frames == 0 and single.drop_n_last_frames == single.chunk_size
    assert single.latent_hw == (2, 3) and single.camera_order == ["observation.images.top"]
    grid = single_config(camera_layout="grid")
    assert grid.latent_hw == (2, 3) and grid.camera_order == ["observation.images.top"]
    five = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48 + 16 * i, 64))
        for i in range(5)
    }
    many = Flux3Config(
        input_features={**five, OBS_STATE: inputs[OBS_STATE]}, output_features=outputs, camera_layout="grid"
    )
    assert len(many.camera_order) == 5 and many.latent_hw == (8, 16)  # any count, any resolution


def test_config_round_trips_through_save_and_load(tmp_path):
    cfg = droid_config()
    cfg._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, Flux3Config) and loaded.type == "flux3"
    assert loaded.camera_keys == DROID_CAMS and loaded.dit_config == TINY_DIT
    assert loaded.input_features[OBS_STATE] == PolicyFeature(type=FeatureType.STATE, shape=(8,))
    for key, value in FRAME_REFERENCE_SETTINGS.items():
        assert getattr(loaded, key) == value


@pytest.mark.parametrize("features", [{}, {"input_features": None, "output_features": None}])
def test_config_allows_features_to_be_inferred_later(features):
    cfg = Flux3Config(camera_layout="single", **features)
    assert not cfg.input_features and not cfg.output_features
    source = task_config()
    cfg.input_features = source.input_features
    cfg.output_features = source.output_features
    cfg.validate_features()
    assert cfg.action_dim == 6 and cfg.camera_order == ["observation.images.top"]


# ---------------------------------------------------------------- processors
@pytest.mark.parametrize("trunk_weights", [None, "missing-trunk.safetensors"])
def test_history_requires_statistics_before_building_models(monkeypatch, trunk_weights):
    # Parsing/export setup is allowed before statistics are available.
    cfg = task_config(normalization_stats=None, trunk_weights=trunk_weights)

    def unexpected_model_build(*args, **kwargs):
        pytest.fail("No model should be allocated or downloaded before validating history statistics")

    for hook in ("_build_dit", "_build_video_vae", "_build_text_encoder"):
        monkeypatch.setattr(Flux3Policy, hook, unexpected_model_build)
    raw_stats = {ACTION: {"mean": torch.zeros(6), "std": torch.ones(6)}}
    with pytest.raises(ValueError, match=r"saved base \(--policy.path\).*normalization_stats"):
        Flux3Policy(cfg, dataset_stats=raw_stats)


def test_processors_identity_for_absolute_actions_and_relative_round_trip():
    cfg = droid_config()
    pre, post = make_flux3_pre_post_processors(cfg, dataset_stats=None)
    assert pre.name == "policy_preprocessor" and post.name == "policy_postprocessor"
    obs = {k: torch.rand(3, 360, 640) for k in DROID_CAMS}
    obs[OBS_STATE] = torch.arange(8.0)
    obs["task"] = "pick"
    out = pre(obs)
    assert torch.equal(out[OBS_STATE], torch.arange(8.0)[None])  # IDENTITY, batch dim added
    assert out[DROID_CAMS[0]].shape == (1, 3, 360, 640)
    act = post(torch.ones(1, 32, 8) * 0.5)
    assert torch.allclose(act, torch.full((1, 32, 8), 0.5))

    rel = single_config(
        use_relative_actions=True,
        action_feature_names=["j0", "j1", "j2", "j3", "j4", "gripper"],
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    stats = {ACTION: {"mean": torch.zeros(6), "std": torch.ones(6)}}
    pre, post = make_flux3_pre_post_processors(rel, dataset_stats=stats)
    state = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.2])
    action = torch.tensor([[1.5, 2.5, 3.5, 4.5, 5.5, 0.9]] * 32)
    out = pre({"observation.images.top": torch.rand(3, 64, 96), OBS_STATE: state, ACTION: action})
    rel_actions = out[ACTION].reshape(-1, 6)
    assert torch.allclose(rel_actions[0, :5], torch.full((5,), 0.5))  # deltas to the state
    assert abs(float(rel_actions[0, 5]) - 0.9) < 1e-5  # gripper stays absolute
    back = post(out[ACTION])
    assert torch.allclose(back.reshape(-1, 6), action)


@pytest.mark.parametrize("config_factory", [droid_config, task_config], ids=["frame", "history"])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"preprocessor_overrides": None, "postprocessor_overrides": None}],
    ids=["omitted", "none"],
)
def test_saved_processors_reload_from_disk(tmp_path, config_factory, overrides):
    cfg = config_factory()
    pre, post = make_flux3_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre2, post2 = make_pre_post_processors(cfg, pretrained_path=tmp_path, **overrides)
    assert len(pre2.steps) == len(pre.steps) and len(post2.steps) == len(post.steps)


# ---------------------------------------------------------------- packing contract
def test_token_budget_and_ids():
    cfg = droid_config()
    b = packing.token_budget(cfg.chunk_size, cfg.latent_hw, text_tokens=80)
    assert (b["x_video_cond"], b["x_video"], b["x_action_cond"], b["x_action"]) == (340, 2720, 1, 32)
    assert packing.latent_frames(33) == 9
    v = packing.pack_video(torch.zeros(1, 96, 9, 17, 20), fps=cfg.fps)
    assert v["x_video"].shape == (1, 2720, 96) and v["x_video_cond"].shape == (1, 340, 96)
    assert sorted(set(v["x_video_ids"][0, :, 0].tolist())) == [26, 53, 80, 106, 133, 160, 186, 213]
    assert v["x_video_ids"][0, :, 3].unique().tolist() == [0]
    a = packing.pack_actions(
        torch.ones(1, 1, 8),
        torch.ones(1, 32, 8),
        packing.default_action_times(1, cfg.chunk_size, cfg.fps),
        "a",
        scale=cfg.action_scale,
    )
    assert a["x_a_ids"][0, :4, 0].tolist() == [6, 13, 20, 26] and a["x_a_cond_ids"][0, 0].tolist() == [
        0,
        0,
        0,
        0,
    ]
    assert torch.allclose(a["x_a_cond"], torch.full((1, 1, 8), 2.0))  # x2 on the state token
    assert a["x_a"].dtype == torch.bfloat16 and float(a["x_a"][0, 0, 0]) == 2.0
    text = packing.pack_text(torch.zeros(1, 160, 32))
    assert text["ctx_ids"][0, :, 3].tolist() == list(range(160)) and text["vector"].shape == (1, 768)
    assert text["ctx_ids"][0, :, :3].abs().max() == 0


def test_noising_and_joint_token_mean_loss():
    x0 = torch.randn(2, 5, 3)
    t = torch.tensor([0.25, 0.9])
    x_t, target = packing.add_noise(x0, t, torch.Generator().manual_seed(0))
    eps = target + x0
    assert torch.allclose(x_t, t[:, None, None] * eps + (1 - t[:, None, None]) * x0, atol=1e-6)
    pred = {"x_video": torch.zeros(1, 4, 2), "x_a": torch.zeros(1, 2, 2)}
    tgt = {"x_video": torch.ones(1, 4, 2), "x_a": torch.ones(1, 2, 2)}
    out = packing.flow_loss(pred, tgt, "a", action_weight=50.0, video_weight=1.0, reduction="joint_tokens")
    assert math.isclose(float(out["loss"]), (4 + 2 * 50.0) / 6, rel_tol=1e-6)  # ONE mean over all tokens
    unweighted = packing.flow_loss(
        pred, tgt, "a", action_weight=1.0, video_weight=1.0, reduction="joint_tokens"
    )
    assert math.isclose(float(unweighted["loss"]), 1.0)
    # DROID token counts: weight 50 under the joint mean is an action:video weight ratio of 0.588
    n_v, n_a = 2720, 32
    assert math.isclose(50.0 * n_a / n_v, 0.588, rel_tol=1e-3)


def test_timestep_sampler_and_augmentation():
    cfg = droid_config()
    t = packing.sample_timesteps(
        4000, torch.Generator().manual_seed(0), width=cfg.train_timestep_width, shift=cfg.train_timestep_shift
    )
    assert t.shape == (4000,) and t.min() > 0 and t.max() < 1 and t.median() > 0.95
    flat = packing.sample_timesteps(
        1000, torch.Generator().manual_seed(1), width=cfg.train_timestep_width, shift=1.0
    )
    assert abs(float(flat.median()) - 0.5) < 0.05
    camera_hw = cfg.image_features[cfg.camera_order[0]].shape[-2:]
    a = packing.sample_augmentation(torch.Generator().manual_seed(3), camera_hw=camera_hw)
    assert (a["crop_height"], a["crop_width"]) == (342, 608) and sorted(a["color_fn_order"]) == [0, 1, 2, 3]
    x = torch.rand(3, 33, 3, 360, 640)
    canvas = packing.materialize_video(x, a, "cpu", layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw)
    assert canvas.shape == (3, 33, 544, 736) and canvas.min() >= -1 and canvas.max() <= 1
    u8 = (x * 255).to(torch.uint8)
    assert torch.allclose(
        packing.materialize_video(u8, None, "cpu", layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw),
        packing.compose_canvas(u8.float() / 255, cfg.camera_layout, cfg.canvas_hw),
    )
    with pytest.raises(ValueError, match="3 cameras"):
        packing.compose_canvas(torch.rand(1, 2, 3, 360, 640), cfg.camera_layout, cfg.canvas_hw)
    single = packing.compose_canvas(torch.rand(1, 2, 3, 64, 96), "single", (128, 192))
    assert single.shape == (3, 2, 128, 192)
    assert grid_shape(1) == (1, 1) and grid_shape(3) == (2, 2) and grid_shape(5) == (2, 3)
    grid = packing.compose_canvas(torch.rand(3, 2, 3, 64, 96), "grid", (128, 192))
    assert grid.shape == (3, 2, 128, 192)
    assert torch.all(grid[:, :, 64:, 96:] == -1)  # the fourth cell of the 2x2 grid stays black
    assert grid[:, :, :64, :96].min() > -1 and grid[:, :, 64:, :96].min() > -1  # cells 1 and 3 carry content


def test_cosmos_unipc_schedule_and_solver():
    sigmas, ticks = sampling.cosmos_unipc_schedule(4, 5.0)
    assert ticks.tolist() == [999, 937, 833, 624]
    assert sigmas.shape == (5,) and sigmas[-1] == 0 and sigmas[0] > 0.999
    x0 = {"x_video": torch.randn(1, 6, 2), "x_a": torch.randn(1, 3, 2)}
    torch.manual_seed(0)
    samples = {k: torch.randn_like(v) for k, v in x0.items()}
    tick_list = ticks.tolist()

    def predict(s, tick):  # a perfect velocity model of one sample
        sigma = sigmas[tick_list.index(int(tick))]
        return {k: (s[k] - x0[k]) / sigma for k in s}

    out = sampling.cosmos_unipc_order2(samples, predict, n_steps=4, shift=5.0)
    for k in x0:
        assert torch.allclose(out[k], x0[k], atol=1e-3), k


def test_remap_shared_action_keys_and_fresh_heads():
    sd = {
        "content_mode_blocks.action_prediction.0.q_proj.weight": torch.zeros(1),
        "early_stream_modulations.action_prediction_cond.lin.weight": torch.zeros(1),
        "emb_in.action_prediction.weight": torch.zeros(1),  # generic head
        "emb_in.action_prediction_other.weight": torch.zeros(1),  # another embodiment
        "final_layer.action_prediction_other_cond.linear.weight": torch.zeros(1),
        "emb_in.action_prediction_mine.weight": torch.zeros(1),  # ours (a finetuned checkpoint)
        "emb_in.video.weight": torch.zeros(1),
    }
    out = remap_shared_action_keys(sd, "action_prediction_mine")
    assert set(out) == {
        "content_mode_blocks.action_prediction_mine.0.q_proj.weight",
        "early_stream_modulations.action_prediction_mine_cond.lin.weight",
        "emb_in.action_prediction_mine.weight",
        "emb_in.video.weight",
    }
    heads = fresh_head_state_dict(64, "myrobot", 16)
    assert heads["emb_in.myrobot.weight"].shape == (64, 16)
    assert heads["final_layer.myrobot_cond.linear.weight"].shape == (16, 64)
    # reference init: xavier input projections drawn from the seed, zeroed final layers
    assert not any(v.any() for k, v in heads.items() if k.startswith("final_layer."))
    assert torch.equal(
        heads["emb_in.myrobot.weight"], fresh_head_state_dict(64, "myrobot", 16)["emb_in.myrobot.weight"]
    )
    assert not torch.equal(
        heads["emb_in.myrobot.weight"],
        fresh_head_state_dict(64, "myrobot", 16, seed=1)["emb_in.myrobot.weight"],
    )
    assert fresh_head_state_dict(64, "myrobot", 6, cond_channels=12)["emb_in.myrobot_cond.weight"].shape == (
        64,
        12,
    )
    params = action_dit_params(JointSingleSeqParams(**TINY_DIT), "myrobot", 16)
    assert params.in_channels["myrobot"] == 16 and params.sequence["x_myrobot_cond"] == "myrobot_cond"


def test_frozen_warmup_constant_scheduler():
    p1, p2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p1], "lr": 1.0}, {"params": [p2], "lr": 5.0}])
    sched = FrozenWarmupConstantSchedulerConfig(
        freeze_steps=10, num_warmup_steps=20, warmup_steps_heads=10
    ).build(opt, num_training_steps=100)
    lrs = []
    for _ in range(45):
        lrs.append(tuple(sched.get_last_lr()))
        opt.step()
        sched.step()
    # heads: (g + 1) / (warmup + 1), never zero, one after the warmup; trunk: zero through microstep 10
    # (the boundary itself), first non-zero factor 1 / 20 at microstep 11, one at microstep 30 (reference classes)
    assert lrs[0][0] == 0.0 and math.isclose(lrs[0][1], 5 / 11) and math.isclose(lrs[9][1], 50 / 11)
    assert lrs[10] == (0.0, 5.0) and math.isclose(lrs[11][0], 1 / 20) and lrs[29][0] < 1.0
    assert math.isclose(lrs[30][0], 1.0) and lrs[44] == (1.0, 5.0)
    cool = FrozenWarmupConstantSchedulerConfig(
        freeze_steps=0, num_warmup_steps=0, warmup_steps_heads=0, cooldown_steps=10
    ).lr_lambda(0, 0, cooldown=10, total_steps=100)
    assert cool(89) == 1.0 and math.isclose(cool(95), 0.5)


# ---------------------------------------------------------------- tiny end-to-end model
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
        ),
    ],
)
def test_valid_windows_stay_on_batch_device(device):
    device = torch.device(device)
    batch = {
        "action_is_pad": torch.tensor([[False, False], [False, True], [False, False]], device=device),
        "observation.images.top_is_pad": torch.tensor([False, False, True], device=device),
        "observation.state_is_pad": torch.zeros(3, 1, dtype=torch.bool),
        "unrelated": torch.ones(3, dtype=torch.bool, device=device),
        "wrong_dtype_is_pad": torch.ones(3, device=device),
        "wrong_batch_is_pad": torch.ones(2, dtype=torch.bool, device=device),
    }
    keep = Flux3Policy._valid_windows(batch, 3, device)
    assert keep.device == batch["action_is_pad"].device and keep.tolist() == [True, False, False]
    keep = Flux3Policy._valid_windows({}, 3, device)
    assert keep.device == batch["action_is_pad"].device and keep.tolist() == [True, True, True]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_grouped_training_keeps_gradients_and_aggregates_metrics(fake_vae, monkeypatch, device):
    cfg = single_config(augment=False, caption_dropout=0)
    policy = Flux3Policy(cfg).to(device).train()
    policy.frozen.video_vae.proj = policy.frozen.video_vae.proj.to(device)
    context = policy._context

    def variable_context(caption, device):
        ctx, ids = context(caption, device)
        length = 16 if caption == "short" else 32
        return ctx[:, :length], ids[:, :length]

    monkeypatch.setattr(policy, "_context", variable_context)
    flow_loss = packing.flow_loss
    group_metrics = []

    def record_loss(pred, targets, *args, **kwargs):
        losses = flow_loss(pred, targets, *args, **kwargs)
        assert all(t.device.type == device for t in (*pred.values(), *targets.values(), *losses.values()))
        group_metrics.append((pred["x_video"].shape[0], losses))
        return losses

    monkeypatch.setattr(packing, "flow_loss", record_loss)
    batch = {
        "observation.images.top": torch.rand(4, cfg.window_frames, 3, 64, 96, device=device),
        OBS_STATE: torch.rand(4, 1, 6, device=device),
        ACTION: torch.rand(4, cfg.chunk_size, 6, device=device),
        "action_is_pad": torch.tensor([False, False, False, True], device=device),
        "task": ["short", "long", "short", "padded"],
    }
    loss, metrics = policy(batch)
    assert loss.device.type == device and loss.requires_grad and torch.isfinite(loss)
    assert metrics["n_valid_windows"] == 3 and [n for n, _ in group_metrics] == [2, 1]
    for key in ("video_mse", "action_mse"):
        expected = sum(n * float(values[key]) for n, values in group_metrics) / 3
        assert isinstance(metrics[key], float) and metrics[key] == pytest.approx(expected)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(g.device.type == device and torch.isfinite(g).all() for g in grads)
    assert any(g.count_nonzero() for g in grads)
    batch["action_is_pad"].fill_(True)
    zero, metrics = policy(batch)
    assert zero.device.type == device and zero.requires_grad and zero.item() == 0
    assert metrics == {"video_mse": 0.0, "action_mse": 0.0, "n_valid_windows": 0}
    zero.backward()
    assert len(group_metrics) == 2


def test_policy_forward_predict_and_select_on_droid_layout(fake_vae):
    cfg = droid_config(n_action_steps=4)
    policy = Flux3Policy(cfg)
    assert set(policy.state_dict()) == {f"dit.{k}" for k in policy.dit.state_dict()}
    assert not any("text_encoder" in k or "video_vae" in k for k in policy.state_dict())
    # training window: 33 frames per camera, the second window is padded -> excluded from the loss
    policy.train()
    batch = droid_batch(b=2, frames=33, pad_second=True)
    loss, info = policy.forward(batch)
    assert torch.isfinite(loss) and info["n_valid_windows"] == 1
    loss.backward()
    head_names = {f"dit.{n}" for n in head_parameter_names(policy.dit, cfg.action_modality)}
    # the conditioning stream's output head gets no loss (its prediction is discarded), the rest must
    trained = {n for n in head_names if "final_layer.action_cond" not in n}
    grads = {n: p.grad for n, p in policy.named_parameters() if n in trained}
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads.values())
    with pytest.raises(ValueError, match="frames per camera"):
        policy.forward(droid_batch(b=1, frames=9))
    # inference on single frames
    policy.eval()
    obs = droid_batch(b=2)
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (2, 32, 8) and chunk.dtype == torch.float32 and torch.isfinite(chunk).all()
    again = policy.predict_action_chunk(obs)
    assert torch.allclose(chunk, again)  # seeded noise: deterministic
    first = policy.select_action(obs)
    assert first.shape == (2, 8) and torch.allclose(first, chunk[:, 0])
    assert len(policy._action_queue) == 3
    policy.reset()
    assert len(policy._action_queue) == 0


def test_grid_layout_runs_any_camera_count_and_mixed_resolutions(fake_vae):
    cams = {
        "observation.images.a": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        "observation.images.b": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
        "observation.images.c": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }
    cfg = single_config(
        n_action_steps=4,
        input_features={**cams, OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        camera_layout="grid",
        canvas_hw=(64, 96),
        dit_config=TINY_DIT,
        dtype="float32",
        video_vae_id=None,
        device="cpu",
    )
    policy = Flux3Policy(cfg)
    pre, _ = make_flux3_pre_post_processors(cfg)
    gen = torch.Generator().manual_seed(4)

    def frames(t):
        return {k: torch.rand(1, t, *f.shape, generator=gen) for k, f in cams.items()}

    policy.train()
    batch = {
        **frames(33),
        OBS_STATE: torch.rand(1, 1, 6, generator=gen),
        ACTION: torch.rand(1, 32, 6, generator=gen),
    }
    loss, info = policy.forward(pre({**batch, "task": ["sort"]}))
    assert torch.isfinite(loss) and info["n_valid_windows"] == 1
    policy.eval()
    obs = {**frames(1), OBS_STATE: torch.rand(1, 6, generator=gen), "task": ["sort"]}
    chunk = policy.predict_action_chunk(pre(obs))
    assert chunk.shape == (1, 32, 6) and torch.isfinite(chunk).all()
    u8 = {k: (v * 255).to(torch.uint8) for k, v in frames(1).items()}
    assert policy._cameras(pre(u8)).dtype == torch.uint8  # mixed resolutions resized, uint8 stays uint8
    with pytest.raises(ValueError, match="share"):
        policy._cameras({**frames(1), "observation.images.a": torch.rand(1, 2, 3, 48, 64)})  # T differs


def test_policy_single_layout_relative_actions_and_cfg_one(fake_vae):
    cfg = single_config(use_relative_actions=True, guidance_scale=1.0, n_action_steps=8)
    policy = Flux3Policy(cfg).eval()
    gen = torch.Generator().manual_seed(2)
    obs = {
        "observation.images.top": torch.rand(1, 3, 64, 96, generator=gen),
        OBS_STATE: torch.rand(1, 6, generator=gen),
        "task": ["stack the cubes"],
    }
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (1, 32, 6) and torch.isfinite(chunk).all()
    policy.train()
    batch = {
        "observation.images.top": torch.rand(2, 33, 3, 64, 96, generator=gen),
        OBS_STATE: torch.rand(2, 1, 6, generator=gen),
        ACTION: torch.rand(2, 32, 6, generator=gen),
        "task": ["a", "b"],
    }
    loss, info = policy.forward(batch)
    assert torch.isfinite(loss) and info["n_valid_windows"] == 2


def test_gripper_flip_is_applied_symmetrically(fake_vae):
    cfg = droid_config()
    policy = Flux3Policy(cfg)
    x = torch.rand(2, 32, 8)
    flipped = policy._flip(x)
    assert torch.allclose(flipped[..., :7], x[..., :7]) and torch.allclose(flipped[..., 7], 1 - x[..., 7])
    assert torch.allclose(policy._flip(flipped), x)
    assert torch.equal(Flux3Policy(single_config())._flip(x[..., :6]), x[..., :6])


def test_optim_param_groups_and_peft_targets(fake_vae):
    cfg = droid_config()
    policy = Flux3Policy(cfg)
    groups = policy.get_optim_params()
    assert len(groups) == 2 and math.isclose(groups[1]["lr"], 1.92e-4 * 5)
    n_heads = sum(p.numel() for p in groups[1]["params"])
    expected = sum(
        p.numel() for n, p in policy.dit.named_parameters() if n in head_parameter_names(policy.dit, "action")
    )
    assert n_heads == expected > 0
    n_all = sum(p.numel() for p in policy.parameters())
    assert n_all == n_heads + sum(p.numel() for p in groups[0]["params"])
    targets = policy._get_default_peft_targets()
    assert "q_proj" in targets["target_modules"] and "dit.emb_in.action" in targets["modules_to_save"]
    with pytest.raises(ValueError, match="saved pretrained flux3 policy"):
        policy._validate_peft_config(None)


def test_frozen_components_stay_unregistered_eval_and_follow_device_moves(monkeypatch, fake_text_encoder):
    class ParamVideoVAE(FakeVideoVAE):
        def __init__(self):
            super().__init__()
            self.module = nn.Linear(
                1, 1
            ).train()  # the loader hands over a module: the container must freeze it

    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: ParamVideoVAE())
    policy = Flux3Policy(droid_config())
    frozen = policy.frozen
    assert "frozen" not in policy._modules and "frozen" not in dict(policy.named_modules())
    assert not any("video_vae" in k or "text_encoder" in k for k in policy.state_dict())
    vae_module, text_encoder = frozen.video_vae.module, frozen.text_encoder
    assert not vae_module.training and not text_encoder.training
    assert not vae_module.weight.requires_grad
    assert all(not p.requires_grad for m in frozen.modules() for p in m.parameters())
    policy.train()
    assert policy.training and policy.dit.training
    assert not vae_module.training and not text_encoder.training  # train() cannot reach them
    policy.to(torch.float64)
    assert vae_module.weight.dtype == torch.float64  # _apply still moves them along
    assert policy.frozen is frozen


def test_compile_model_wraps_the_inference_dit_forward_only(fake_vae, monkeypatch):
    compiled_fns = []

    def fake_compile(fn, *args, **kwargs):
        compiled_fns.append(fn)

        def wrapped(*a, **k):
            wrapped.calls += 1
            return fn(*a, **k)

        wrapped.calls = 0
        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    policy = Flux3Policy(droid_config(compile_model=True, n_action_steps=4))
    assert compiled_fns == []  # lazy: nothing is compiled at construction
    policy.train()
    loss, _ = policy.forward(droid_batch(b=1, frames=33))
    assert torch.isfinite(loss) and compiled_fns == []  # training keeps the eager DiT forward
    policy.eval()
    obs = droid_batch(b=1)
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (1, 32, 8) and torch.isfinite(chunk).all()
    assert len(compiled_fns) == 1 and compiled_fns[0].__self__ is policy.dit
    compiled = policy._compiled_dit
    per_chunk = compiled.calls
    assert per_chunk > 0
    policy.predict_action_chunk(obs)
    assert policy._compiled_dit is compiled and compiled.calls == 2 * per_chunk  # compiled once, reused
    policy.to(torch.float32)
    assert policy._compiled_dit is None  # a device / dtype move compiles afresh on the next prediction
    eager = Flux3Policy(droid_config(n_action_steps=4)).eval()
    eager.predict_action_chunk(obs)
    assert eager._compiled_dit is None and len(compiled_fns) == 1  # flag off: torch.compile never called


def test_save_load_round_trip_excludes_frozen_components(fake_vae, tmp_path):
    cfg = droid_config()
    policy = Flux3Policy(cfg)
    with torch.no_grad():
        policy.dit.emb_in["action"].weight.fill_(1.25)
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    with safe_open(tmp_path / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
    assert keys and all(k.startswith("dit.") for k in keys)
    assert not any("video_vae" in k or "text_encoder" in k for k in keys)
    loaded = Flux3Policy.from_pretrained(tmp_path)
    assert torch.equal(loaded.dit.emb_in["action"].weight, torch.full((64, 8), 1.25))
    # a checkpoint of another embodiment: trunk loads, mismatched heads are skipped (fresh), no error
    other = Flux3Policy.from_pretrained(tmp_path, config=single_config(), strict=False)
    assert other.dit.emb_in["action"].weight.shape == (64, 6)
    assert torch.equal(other.dit.single_blocks[0].q_proj.weight, policy.dit.single_blocks[0].q_proj.weight)


def test_policy_strict_loading_rejects_missing_head_weights(fake_vae, tmp_path):
    policy = Flux3Policy(single_config())
    policy.save_pretrained(tmp_path)
    head_key = "dit.final_layer.action.linear.weight"
    loaded = Flux3Policy.from_pretrained(tmp_path, strict=True)
    assert torch.equal(loaded.state_dict()[head_key], policy.state_dict()[head_key])

    state = {k: v.contiguous() for k, v in policy.state_dict().items() if k != head_key}
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match=f"checkpoint missing 1 keys.*{head_key}"):
        Flux3Policy.from_pretrained(tmp_path, strict=True)

    with pytest.raises(RuntimeError, match="checkpoint missing 1 keys"):
        Flux3Policy.from_pretrained(tmp_path)
    loaded = Flux3Policy.from_pretrained(tmp_path, strict=False)
    assert all(torch.equal(loaded.state_dict()[k], v) for k, v in state.items())


def test_trunk_weights_are_remapped_onto_the_action_modality(fake_vae, tmp_path):
    """An action-pretrained trunk carries `action_prediction*` streams; they seed our modality's blocks."""
    trunk_params = action_dit_params(JointSingleSeqParams(**TINY_DIT), "action_prediction", 32)
    torch.manual_seed(7)
    trunk = JointSingleSeq(trunk_params)
    sd = {k: v.contiguous() for k, v in trunk.state_dict().items()}
    path = tmp_path / "dit.safetensors"
    save_file(sd, str(path))
    cfg = droid_config(trunk_weights=str(path), action_modality="myrobot")
    policy = Flux3Policy(cfg)
    assert torch.equal(
        policy.dit.content_mode_blocks["myrobot"][0].q_proj.weight,
        trunk.content_mode_blocks["action_prediction"][0].q_proj.weight,
    )
    assert torch.equal(
        policy.dit.early_stream_modulations["myrobot_cond"].lin.weight,
        trunk.early_stream_modulations["action_prediction_cond"].lin.weight,
    )
    assert torch.equal(policy.dit.single_blocks[0].mlp_in.weight, trunk.single_blocks[0].mlp_in.weight)
    assert policy.dit.emb_in["myrobot"].weight.shape == (64, 8)  # fresh, sized to the dataset (not 32)
    assert "emb_in.action_prediction" not in {k.rsplit(".", 1)[0] for k in policy.dit.state_dict()}


def test_strict_head_loading_rejects_partial_heads(fake_vae, tmp_path):
    """A checkpoint with some but not all embodiment heads is not a finetune; strict loading must say so."""
    dit = Flux3Policy(droid_config(action_modality="myrobot")).dit
    state = {k: v.contiguous() for k, v in dit.state_dict().items()}
    del state["emb_in.myrobot_cond.weight"]
    path = tmp_path / "partial.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError, match="missing required embodiment heads"):
        load_action_checkpoint(dit, str(path), "myrobot", strict_heads=True)
    state["emb_in.myrobot_cond.weight"] = dit.state_dict()["emb_in.myrobot_cond.weight"].contiguous()
    save_file(state, str(path))
    load_action_checkpoint(dit, str(path), "myrobot", strict_heads=True)


def test_full_checkpoint_reload_does_not_reopen_trunk(fake_vae, tmp_path, monkeypatch):
    policy = Flux3Policy(single_config())
    trunk = tmp_path / "trunk.safetensors"
    save_file({k: v.contiguous() for k, v in policy.dit.state_dict().items()}, str(trunk))
    cfg = single_config(trunk_weights=str(trunk))
    policy = Flux3Policy(cfg)
    saved = tmp_path / "policy"
    policy.save_pretrained(saved)
    trunk.rename(tmp_path / "moved.safetensors")

    def forbidden(*args, **kwargs):
        pytest.fail("complete policy reload attempted to resolve initialization weights")

    monkeypatch.setattr("lerobot.policies.flux3.modeling_flux3.resolve_weights", forbidden)
    loaded = Flux3Policy.from_pretrained(saved, local_files_only=True)
    overridden = Flux3Policy.from_pretrained(saved, config=single_config(trunk_weights=str(trunk)))
    # Deferred checkpoint construction also skips initialization artifacts.
    Flux3Policy(single_config(trunk_weights=str(trunk), pretrained_path=str(saved)))
    assert loaded.config.pretrained_path == str(saved)
    # A second-generation finetune must become the adapter base, not its ancestor.
    second = tmp_path / "second"
    loaded.save_pretrained(second)
    assert Flux3Policy.from_pretrained(second).config.pretrained_path == str(second)
    for name, tensor in policy.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
        assert torch.equal(overridden.state_dict()[name], tensor)


def test_peft_requires_saved_base_and_round_trips(fake_vae, tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    policy = Flux3Policy(single_config(normalization_mapping=normalization_mapping))
    policy.config.trunk_weights = "initialization-only.safetensors"
    with pytest.raises(ValueError, match="saved pretrained flux3 policy"):
        policy.wrap_with_peft(peft_cli_overrides={"method_type": "lora"})
    base = tmp_path / "base"
    policy.save_pretrained(base)
    policy = Flux3Policy.from_pretrained(base)
    wrapped = policy.wrap_with_peft(peft_cli_overrides={"method_type": "lora"})
    with torch.no_grad():
        for parameter in wrapped.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.0125)
    adapter = tmp_path / "adapter"
    wrapped.save_pretrained(adapter)
    adapter_config = peft.PeftConfig.from_pretrained(adapter)
    assert adapter_config.base_model_name_or_path == str(base)
    for folder, mean, std in ((base, 0.0, 1.0), (adapter, 2.0, 2.0)):
        stats = {
            key: {"mean": torch.full((6,), mean), "std": torch.full((6,), std)} for key in (OBS_STATE, ACTION)
        }
        pre, post = make_flux3_pre_post_processors(policy.config, dataset_stats=stats)
        pre.save_pretrained(folder)
        post.save_pretrained(folder)
    cfg = single_config(
        pretrained_path=str(adapter), use_peft=True, normalization_mapping=normalization_mapping
    )
    features = {**cfg.input_features, **cfg.output_features}
    monkeypatch.setattr(factory, "dataset_to_policy_features", lambda _: features)
    restored = factory.make_policy(cfg, ds_meta=SimpleNamespace(features={}, stats={}))
    # The trainer chooses processor files from the caller's config after loading the model.
    pre, post = make_pre_post_processors(cfg, pretrained_path=cfg.pretrained_path)
    torch.testing.assert_close(pre({OBS_STATE: torch.full((6,), 4.0)})[OBS_STATE], torch.ones(1, 6))
    torch.testing.assert_close(post(torch.zeros(1, 32, 6)), torch.full((1, 32, 6), 2.0))
    assert cfg.pretrained_path == str(adapter)
    assert restored.config.pretrained_path == str(base)
    assert restored.config is not cfg
    resaved = tmp_path / "resaved_adapter"
    restored.save_pretrained(resaved)
    assert peft.PeftConfig.from_pretrained(resaved).base_model_name_or_path == str(base)
    for name, parameter in wrapped.named_parameters():
        assert torch.equal(dict(restored.named_parameters())[name], parameter), name
    obs = {"observation.images.top": torch.rand(1, 3, 64, 96), OBS_STATE: torch.rand(1, 6)}
    assert torch.equal(wrapped.predict_action_chunk(obs), restored.predict_action_chunk(obs))


def test_unsupported_relative_execution_and_rtc_arguments_fail_early(fake_vae):
    policy = Flux3Policy(single_config(use_relative_actions=True))
    assert not policy.supports_rtc()
    with pytest.raises(NotImplementedError, match="relative-action execution requires RTC"):
        policy.select_action({})
    absolute = Flux3Policy(single_config())
    for method in (absolute.select_action, absolute.predict_action_chunk):
        with pytest.raises(NotImplementedError, match="RTC inference arguments"):
            method({}, inference_delay=1, prev_chunk_left_over=None)


def test_lora_update_ema_resume_and_reload(tmp_path, fake_vae):
    for package in ("peft", "diffusers", "datasets"):
        pytest.importorskip(package)
    cfg = single_config(
        gradient_checkpointing=True,
        optimizer_lr=1e-4,
        optimizer_lr_heads_multiplier=5,
        augment=False,
        caption_dropout=0,
    )
    base = tmp_path / "base"
    Flux3Policy(cfg).save_pretrained(base)
    policy = Flux3Policy.from_pretrained(base).wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": 32, "lora_alpha": 32}
    )
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or "modules_to_save" in n for n in trainable)
    groups = policy.get_optim_params()
    assert groups[1]["lr"] == 5e-4
    ema = EMAModel(_ema_parameters(policy), decay=0.999, min_decay=0.999)
    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0)
    batch = {
        "observation.images.top": torch.rand(1, cfg.window_frames, 3, 64, 96),
        OBS_STATE: torch.rand(1, 6),
        ACTION: torch.rand(1, cfg.chunk_size, 6),
        "task": ["pick up box"],
    }
    frozen = {n: p.clone() for n, p in policy.named_parameters() if not p.requires_grad}
    policy.train()
    loss, metrics = policy(batch)
    assert torch.isfinite(loss) and metrics["n_valid_windows"] == 1
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
    optimizer.step()
    optimizer.zero_grad()
    ema.step(_ema_parameters(policy))
    assert all(torch.equal(p, dict(policy.named_parameters())[n]) for n, p in frozen.items())
    live = [p.clone() for p in _ema_parameters(policy)]
    adapter = tmp_path / "ema_adapter"
    with _ema_weights(ema, policy):
        policy.save_pretrained(adapter)
        policy.config.save_pretrained(adapter)
        expected = {n: p.clone() for n, p in policy.named_parameters()}
    assert all(torch.equal(a, b) for a, b in zip(live, _ema_parameters(policy), strict=True))
    restored = PeftModel.from_pretrained(Flux3Policy.from_pretrained(base), adapter, is_trainable=True)
    assert all(torch.equal(p, expected[n]) for n, p in restored.named_parameters())
    restored_ema = EMAModel(_ema_parameters(restored), decay=0.999, min_decay=0.999)
    restored_ema.load_state_dict(ema.state_dict())
    assert restored_ema.optimization_step == ema.optimization_step
    # Resume one identical EMA update from the same shadow, not a raw-weight reseed.
    ema.step(_ema_parameters(policy))
    for dst, src in zip(_ema_parameters(restored), _ema_parameters(policy), strict=True):
        dst.data.copy_(src)
    restored_ema.step(_ema_parameters(restored))
    assert all(torch.equal(a, b) for a, b in zip(ema.shadow_params, restored_ema.shadow_params, strict=True))


def test_captions_use_one_paraphrase_per_window():
    batch = {"task": ["pick it up | grab it | take it", "single", None]}
    assert Flux3Policy._captions(batch, 3) == ["pick it up", "single", ""]
    torch.manual_seed(0)
    drawn = {Flux3Policy._captions(batch, 3, training=True)[0] for _ in range(40)}
    assert drawn == {"pick it up", "grab it", "take it"}
    assert Flux3Policy._captions({"task": "a | b"}, 2) == ["a", "a"]


DROPPED_STREAMS = {"image", "image_cond", "audio", "audio_cond"}


def _tiny_trunks(tmp_path):
    """A full action-pretrained tiny trunk and its copy without the image / audio streams."""
    trunk_params = action_dit_params(JointSingleSeqParams(**TINY_DIT), "action_prediction", 32)
    torch.manual_seed(11)
    full_sd = {k: v.contiguous() for k, v in JointSingleSeq(trunk_params).state_dict().items()}
    lean_sd = {k: v for k, v in full_sd.items() if stream_of_key(k) not in DROPPED_STREAMS}
    assert len(lean_sd) < len(full_sd)
    full, lean = tmp_path / "full.safetensors", tmp_path / "lean.safetensors"
    save_file(full_sd, str(full))
    save_file(lean_sd, str(lean))
    return full, lean


def _lean_config(**overrides) -> Flux3Config:
    return droid_config(augment=False, caption_dropout=0.0, **overrides)


def _with_full_streams(policy: Flux3Policy) -> Flux3Policy:
    """Reconstruct an old full policy fixture without enabling that layout in production code."""
    config = policy.config
    policy.dit_params = action_dit_params(
        JointSingleSeqParams(**config.dit_config),
        config.action_modality,
        config.action_dim,
        attn_mode=config.attn_mode,
        conditioning_channels=policy.packer.conditioning_channels(config),
    )
    full = JointSingleSeq(policy.dit_params).to(dtype=policy.dtype_)
    full.load_state_dict({**full.state_dict(), **policy.dit.state_dict()})
    policy.dit = full
    return policy


def _equivalent(p_a: Flux3Policy, p_b: Flux3Policy) -> None:
    obs = droid_batch(b=2)
    p_a.eval()
    p_b.eval()
    assert torch.equal(p_a.predict_action_chunk(obs), p_b.predict_action_chunk(obs))
    batch = droid_batch(b=2, frames=33)
    p_a.train()
    p_b.train()
    torch.manual_seed(3)
    loss_a, _ = p_a.forward(batch)
    torch.manual_seed(3)
    loss_b, _ = p_b.forward(batch)
    assert torch.equal(loss_a, loss_b)


def test_lean_trunk_without_image_audio_streams_predicts_identically(fake_vae, tmp_path):
    """The policy never feeds image / audio tokens, so a trunk without those streams must load and match."""
    full, lean = _tiny_trunks(tmp_path)
    p_full = _with_full_streams(Flux3Policy(_lean_config(trunk_weights=str(full))))
    p_lean = Flux3Policy(_lean_config(trunk_weights=str(lean)))
    assert set(p_full.dit.in_channels) >= DROPPED_STREAMS
    assert set(p_lean.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
    assert not any(stream_of_key(k) in DROPPED_STREAMS for k in p_lean.state_dict())
    n_full = sum(p.numel() for p in p_full.dit.parameters())
    n_lean = sum(p.numel() for p in p_lean.dit.parameters())
    assert n_lean < n_full
    full_sd = p_full.state_dict()
    trunk_keys = [
        k for k in p_lean.state_dict() if not k.startswith(("dit.emb_in.action", "dit.final_layer.action"))
    ]
    for k in trunk_keys:  # the remapped trunk is identical; only the fresh heads carry each model's own init
        assert torch.equal(p_lean.state_dict()[k], full_sd[k]), k
    p_lean.load_state_dict({k: full_sd[k] for k in p_lean.state_dict()})
    _equivalent(p_full, p_lean)


def test_lean_policy_round_trips_and_full_files_load_into_lean_policies(fake_vae, tmp_path):
    full, lean = _tiny_trunks(tmp_path)
    p_lean = Flux3Policy(_lean_config(trunk_weights=str(lean)))
    saved = tmp_path / "lean_policy"
    p_lean.save_pretrained(saved)
    reloaded = Flux3Policy.from_pretrained(saved)
    assert set(reloaded.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
    _equivalent(p_lean, reloaded)
    # an explicit lean build initialized from the FULL trunk: the extra stream weights are ignored
    p_explicit = Flux3Policy(_lean_config(trunk_weights=str(full)))
    p_explicit.load_state_dict(p_lean.state_dict())  # fresh heads carry each build's own init
    _equivalent(p_lean, p_explicit)
    # an older FULL policy file opened as a lean policy (strict load) drops the foreign streams
    p_full = _with_full_streams(Flux3Policy(_lean_config(trunk_weights=str(full))))
    p_full.load_state_dict({**p_full.state_dict(), **p_lean.state_dict()})  # same heads as the lean policy
    saved_full = tmp_path / "full_policy"
    p_full.save_pretrained(saved_full)
    as_lean = Flux3Policy.from_pretrained(saved_full, config=_lean_config())
    assert set(as_lean.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
    _equivalent(p_lean, as_lean)


def test_new_policies_omit_unused_streams_even_from_full_trunks(fake_vae, tmp_path):
    full, _ = _tiny_trunks(tmp_path)
    for cfg in (
        _lean_config(),
        _lean_config(trunk_weights=str(full)),
        _lean_config(pretrained_path="deferred-policy"),
    ):
        policy = Flux3Policy(cfg)
        assert set(policy.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
        assert not any(stream_of_key(k) in DROPPED_STREAMS for k in policy.state_dict())


@pytest.mark.parametrize("override", [False, True])
def test_full_policy_reloads_without_unused_streams(fake_vae, tmp_path, override):
    policy = _with_full_streams(Flux3Policy(_lean_config()))
    policy.save_pretrained(tmp_path)
    kwargs = {"config": _lean_config()} if override else {}
    restored = Flux3Policy.from_pretrained(tmp_path, **kwargs)
    assert set(restored.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
    expected = {k: v for k, v in policy.state_dict().items() if stream_of_key(k) not in DROPPED_STREAMS}
    assert restored.state_dict().keys() == expected.keys()
    for key, value in restored.state_dict().items():
        assert torch.equal(value, expected[key]), key
    _equivalent(policy, restored)
    restored.save_pretrained(tmp_path / "resaved")
    assert "content_streams" not in json.loads((tmp_path / "resaved/config.json").read_text())


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_policy_rejects_corrupt_trunk_even_when_fresh_heads_allowed(fake_vae, tmp_path, strict, corruption):
    policy = Flux3Policy(single_config())
    policy.save_pretrained(tmp_path)
    state = dict(policy.state_dict())
    key = "dit.single_blocks.0.q_proj.weight"
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        # The old substring exception must not swallow an unrelated audio_cond-looking key.
        state["dit.audio_cond_typo.weight"] = torch.zeros(1)
    else:
        state[key] = state[key][:1].contiguous()
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match=corruption):
        Flux3Policy.from_pretrained(tmp_path, strict=strict)


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_trunk_initialization_rejects_corrupt_checkpoint(fake_vae, tmp_path, corruption):
    full, _ = _tiny_trunks(tmp_path)
    state = load_file(full)
    key = "single_blocks.0.q_proj.weight"
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["audio_cond_typo.weight"] = torch.zeros(1)
    else:
        state[key] = state[key][:1].contiguous()
    save_file(state, str(full))
    with pytest.raises(
        (ValueError, RuntimeError), match="size mismatch" if corruption == "shape" else corruption
    ):
        Flux3Policy(_lean_config(trunk_weights=str(full)))


def test_strict_policy_loading_rejects_missing_video_conditioning(fake_vae, tmp_path):
    policy = Flux3Policy(_lean_config())
    policy.save_pretrained(tmp_path)
    state = dict(policy.state_dict())
    del state["dit.emb_in.video_cond.weight"]
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="checkpoint missing 1 keys"):
        Flux3Policy.from_pretrained(tmp_path)


def test_full_trunk_without_optional_audio_conditioning_still_loads(fake_vae, tmp_path):
    full, _ = _tiny_trunks(tmp_path)
    state = {k: v for k, v in load_file(full).items() if stream_of_key(k) != "audio_cond"}
    save_file(state, str(full))
    policy = Flux3Policy(_lean_config(trunk_weights=str(full)))
    assert set(policy.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
    assert torch.equal(policy.dit.single_blocks[0].q_proj.weight, state["single_blocks.0.q_proj.weight"])
