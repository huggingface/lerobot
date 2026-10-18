# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU interface tests, deliberately distinct from full-checkpoint agreement."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies import LongWAMConfig, get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy, _OrderedWorker


def config(**kwargs):
    return LongWAMConfig(
        device="cpu",
        dtype=torch.float32,
        camera_size=(16, 16),
        action_dim=3,
        proprio_dim=2,
        action_horizon=8,
        n_action_steps=6,
        **kwargs,
    )


class ScriptedCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 3)
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.samples, self.calls = [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.delay_anchor = None

    def training_loss(self, sample):
        self.samples.append(sample)
        # Only the current state conditions the action, never supervised futures.
        prediction = self.projection(sample["proprio"][:, 48]).unsqueeze(1)
        loss = (prediction - sample["action"]).square().mean()
        return loss, {"loss_action": loss.detach()}

    def encode_prompt(self, prompts):
        return torch.zeros(len(prompts), 4, 6), torch.ones(len(prompts), 4, dtype=torch.bool)

    def infer_joint_ar(self, **kwargs):
        anchor = round(float(kwargs["proprio"][0, 0]))
        episode = round(float(kwargs["proprio"][0, 1]))
        window = kwargs["obs_window"]
        frames = None if window is None else ((window[0, 0, :, 0, 0] + 1) * 50).round().int().tolist()
        self.calls.append((episode, anchor, frames, kwargs["prompt"]))
        if anchor == self.delay_anchor:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("Scripted test did not release inference")
        offset = torch.arange(kwargs["action_horizon"])
        # Distinct channels: equal sums can no longer conceal wrong chunks.
        actions = torch.stack((torch.full_like(offset, episode), torch.full_like(offset, anchor), offset), -1)
        return {"action": actions.float()}


@pytest.fixture
def build(monkeypatch):
    built = []

    def factory(cfg, *, load_initial_weights, **kwargs):
        model = ScriptedCore()
        model.initialized = load_initial_weights
        model.tokenizer_assets = kwargs.get("tokenizer_assets")
        built.append(model)
        return model

    monkeypatch.setattr(LongWAMPolicy, "_build_core_model", staticmethod(factory))
    yield built


def observation(cfg, step, episode=1):
    batch = {key: torch.full((1, 3, 16, 16), step / 100) for key in cfg.camera_keys}
    batch.update(
        {"observation.state": torch.tensor([[step, episode]], dtype=torch.float32), "task": ["pick up"]}
    )
    return batch


def training_batch(cfg):
    t = len(cfg.observation_delta_indices)
    batch = {
        key: torch.linspace(0, 1, t)[None, :, None, None, None].expand(2, t, 3, 16, 16)
        for key in cfg.camera_keys
    }
    batch["observation.state"] = torch.arange(t).float()[None, :, None].expand(2, t, 2)
    batch["action"] = torch.ones(2, cfg.action_horizon, cfg.action_dim)
    batch["action_is_pad"] = torch.zeros(2, cfg.action_horizon, dtype=torch.bool)
    batch["task"] = ["pick up", "put down"]
    for key in cfg.camera_keys:
        batch[key + "_is_pad"] = torch.zeros(2, t, dtype=torch.bool)
        batch[key + "_is_pad"][:, : cfg.past_obs_steps] = True
    return batch


@pytest.mark.parametrize("past", [0, 48, 96, 192, 384])
def test_config_registration_temporal_offsets_and_roundtrip(tmp_path, past):
    cfg = make_policy_config("longwam", past_obs_steps=past, device="cpu", streaming_vae=True)
    assert isinstance(cfg, LongWAMConfig)
    assert get_policy_class("longwam") is LongWAMPolicy
    assert cfg.observation_delta_indices == list(range(-past, 33))
    assert cfg.observation_delta_indices[past] == 0
    assert cfg.action_delta_indices == list(range(32))
    assert cfg.video_init_repo.endswith("LongLive2.0-Robot-S")
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.past_obs_steps == past
    assert restored.camera_keys == cfg.camera_keys
    assert restored.streaming_vae is True
    assert restored.get_optimizer_preset().betas == (0.9, 0.95)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"past_obs_steps": -1},
        {"past_obs_steps": 49},
        {"video_stride": 2},
        {"future_video_steps": 15},
        {"execution": "unknown"},
        {"trigger_stride": 1},
        {"execution": "async", "trigger_stride": 1},
        {"denoising": "cod"},
        {"denoising": "other"},
        {"inference_action_horizon": 0},
        {"camera_keys": ()},
    ],
)
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_cod_is_an_explicit_separate_configuration():
    assert config(denoising="cod", imagine_sigma=0).denoising == "cod"


@pytest.mark.parametrize("features", [None, {}])
def test_missing_features_keep_inherited_config_defaults(features):
    cfg = config(input_features=features, output_features=features)
    assert set(cfg.image_features) == set(cfg.camera_keys)
    assert cfg.robot_state_feature.shape == (2,)
    assert cfg.action_feature.shape == (3,)


def test_worker_rejects_missing_payload_and_inference_before_observation(build):
    worker = _OrderedWorker(LongWAMPolicy(config()))
    try:
        with pytest.raises(ValueError, match="payload"):
            worker.call("reset")
        worker.call("reset", {"episode": 1, "instruction": "pick up"})
        with pytest.raises(ValueError, match="Observe before"):
            worker.call("infer", {"episode": 1, "anchor": -1})
    finally:
        worker.close()


def test_native_forward_preserves_temporal_alignment_and_trains(build):
    cfg = config()
    policy = LongWAMPolicy(cfg)
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    original = policy.model.projection.weight.detach().clone()
    batch = training_batch(cfg)
    for _ in range(3):
        optimizer.zero_grad()
        loss, metrics = policy(batch)
        assert torch.isfinite(loss) and isinstance(metrics["loss_action"], float)
        loss.backward()
        optimizer.step()
    assert not torch.equal(original, policy.model.projection.weight)
    sample = policy.model.samples[-1]
    assert sample["video"].shape == (2, 3, 21, 16, 32)
    assert sample["proprio"][0, 48, 0] == 48
    assert sample["image_is_pad"][0].tolist() == [True] * 12 + [False] * 9
    assert torch.equal(sample["action_is_pad"], batch["action_is_pad"])
    assert not any(p.requires_grad for p in policy.model.vae.parameters())
    policy.train()
    assert not policy.model.vae.training and not policy.model.text_encoder.training


def test_forward_rejects_a_single_frame_instead_of_fabricating_video(build):
    policy = LongWAMPolicy(config())
    with pytest.raises(ValueError, match="shape"):
        policy(observation(policy.config, 0))


def test_inference_pixels_match_reference_pil_and_bfloat16_order(build):
    cfg = config()
    cfg.dtype = torch.bfloat16
    policy = LongWAMPolicy(cfg)
    pixels = np.random.default_rng(1).integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    image = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255).unsqueeze(0)
    batch = dict.fromkeys(cfg.camera_keys, image)
    expected = np.array(Image.fromarray(pixels).resize((24, 16), Image.BILINEAR).crop((4, 0, 20, 16)))
    expected = torch.from_numpy(expected).permute(2, 0, 1).unsqueeze(0).bfloat16() * (2 / 255) - 1
    torch.testing.assert_close(
        policy._images(batch, temporal=False), torch.cat([expected, expected], -1), atol=0, rtol=0
    )


def test_prompt_encoding_is_cached_within_an_episode(build, monkeypatch):
    policy = LongWAMPolicy(config()).eval()
    calls = []
    original = policy.model.encode_prompt

    def encode(prompts):
        calls.append(prompts)
        return original(prompts)

    monkeypatch.setattr(policy.model, "encode_prompt", encode)
    try:
        for step in range(13):
            policy.select_action(observation(policy.config, step))
        assert len(calls) == 1
        policy.reset()
        policy.select_action(observation(policy.config, 0))
        assert len(calls) == 2
    finally:
        policy.close()


def test_preprocessing_constant_channels_clipping_and_reload(tmp_path):
    cfg = config(toggle_action_dimensions=[-1])
    stats = {
        "observation.state": {"min": torch.tensor([2.0, 7.0]), "max": torch.tensor([4.0, 7.0])},
        "action": {"min": torch.zeros(3), "max": torch.ones(3)},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    data = observation(cfg, 0)
    data["observation.state"] = torch.tensor([[100.0, 7.0]])
    processed = pre(data)
    torch.testing.assert_close(processed["observation.state"], torch.tensor([[5.0, 0.0]]))
    torch.testing.assert_close(processed[cfg.camera_keys[0]], data[cfg.camera_keys[0]])
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    loaded_pre, loaded_post = make_pre_post_processors(cfg, pretrained_path=tmp_path)
    torch.testing.assert_close(loaded_pre(data)["observation.state"], processed["observation.state"])
    action = torch.tensor([[0.0, 1.0, 1.0]])
    torch.testing.assert_close(loaded_post(action), torch.tensor([[0.5, 1.0, -1.0]]))
    torch.testing.assert_close(post(action), loaded_post(action))
    assert stats["observation.state"]["max"][1] == 7


def test_missing_stats_fail_instead_of_silently_skipping_normalization():
    pre, _ = make_pre_post_processors(config())
    with pytest.raises(ValueError, match="statistics"):
        pre(observation(config(), 0))


@pytest.mark.parametrize("execution", ["sync", "async"])
def test_actions_identify_episode_anchor_offset_and_consumed_frames(build, execution):
    cfg = config(execution=execution, trigger_stride=4 if execution == "async" else None)
    policy = LongWAMPolicy(cfg).eval()
    try:
        for step in range(18):
            result = policy.select_action(observation(cfg, step))[0]
            anchor = (step // 6) * 6 if execution == "sync" else (0 if step < 6 else ((step - 2) // 4) * 4)
            torch.testing.assert_close(result, torch.tensor([1.0, float(anchor), float(step - anchor)]))
        for episode, anchor, frames, _ in policy.model.calls:
            assert episode == 1
            assert frames == [max(0, anchor + delta) for delta in range(-48, 1, 4)]
        policy.reset()
        torch.testing.assert_close(
            policy.select_action(observation(cfg, 0, 2)), torch.tensor([[2.0, 0.0, 0.0]])
        )
    finally:
        policy.close()


def test_late_handoff_blocks_then_selects_elapsed_offset(build):
    cfg = config(execution="async", trigger_stride=4)
    policy = LongWAMPolicy(cfg).eval()
    policy.model.delay_anchor = 4
    try:
        for step in range(6):
            policy.select_action(observation(cfg, step))
        assert policy.model.entered.wait(2)
        with ThreadPoolExecutor(max_workers=1) as caller:
            result = caller.submit(policy.select_action, observation(cfg, 6))
            try:
                with pytest.raises(TimeoutError):
                    result.result(timeout=0.05)
                assert policy._runtime.step == 6
            finally:
                policy.model.release.set()
            torch.testing.assert_close(result.result(3), torch.tensor([[1.0, 4.0, 2.0]]))
        assert policy._runtime.metrics["late_handoffs"] == 1
    finally:
        policy.model.release.set()
        policy.close()


def test_reset_drains_pending_request_before_new_episode_action(build):
    cfg = config(execution="async", trigger_stride=4)
    policy = LongWAMPolicy(cfg).eval()
    policy.model.delay_anchor = 4
    try:
        for step in range(5):
            policy.select_action(observation(cfg, step))
        assert policy.model.entered.wait(2)
        policy.reset()
        with ThreadPoolExecutor(max_workers=1) as caller:
            first_action = caller.submit(policy.select_action, observation(cfg, 0, 2))
            try:
                with pytest.raises(TimeoutError):
                    first_action.result(timeout=0.05)
            finally:
                policy.model.release.set()
            torch.testing.assert_close(first_action.result(3), torch.tensor([[2.0, 0.0, 0.0]]))
        assert policy._runtime.episode == 2
        assert policy.model.calls[-1][:3] == (2, 0, [0] * 13)
    finally:
        policy.model.release.set()
        policy.close()


def test_task_changes_and_generic_async_are_rejected_without_advancing_history(build):
    cfg = config()
    policy = LongWAMPolicy(cfg).eval()
    try:
        batch = observation(cfg, 0)
        policy.select_action(batch)
        with pytest.raises(ValueError, match="reset"):
            policy.select_action({**batch, "task": ["new task"]})
        with pytest.raises(ValueError, match="fixed instructions"):
            policy.drop_queued_actions()
        with pytest.raises(NotImplementedError, match="inline"):
            policy.predict_action_chunk(batch)
        assert policy._runtime.step == 1
        assert not policy.supports_rtc()
    finally:
        policy.close()


def test_checkpoint_roundtrip_uses_native_format_and_skips_initialization(build, tmp_path):
    policy = LongWAMPolicy(config())
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert json.loads((tmp_path / "config.json").read_text())["type"] == "longwam"
    loaded = LongWAMPolicy.from_pretrained(tmp_path)
    assert not loaded.model.initialized
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)
    try:
        torch.testing.assert_close(
            loaded.select_action(observation(loaded.config, 0)), torch.tensor([[1.0, 0.0, 0.0]])
        )
    finally:
        loaded.close()


def test_native_config_and_core_contract_for_fresh_training(tmp_path):
    from lerobot.policies.longwam.weights_longwam import build_core_model

    # Reject a missing mapped initializer BEFORE allocating a multi-billion-parameter model.
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="action_init_path"):
        build_core_model(config(), load_initial_weights=True)
    with pytest.raises(FileNotFoundError, match="action_init_path"):
        build_core_model(config(action_init_path=str(tmp_path / "missing.pt")), load_initial_weights=True)


@pytest.mark.parametrize("resume", [False, True])
def test_native_training_processor_overrides_and_resume_stats(tmp_path, resume):
    cfg = config()
    old_stats = {
        "observation.state": {"min": torch.zeros(2), "max": torch.ones(2)},
        "action": {"min": torch.zeros(3), "max": torch.ones(3)},
    }
    new_stats = {
        "observation.state": {"min": torch.zeros(2), "max": torch.full((2,), 2.0)},
        "action": {"min": torch.zeros(3), "max": torch.full((3,), 2.0)},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=old_stats)
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    overrides = {
        "normalizer_processor": {
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
        },
        "device_processor": {"device": "cpu"},
        "rename_observations_processor": {"rename_map": {}},
    }
    loaded, output = make_pre_post_processors(
        cfg,
        pretrained_path=tmp_path,
        dataset_stats=None if resume else new_stats,
        preprocessor_overrides=overrides,
        postprocessor_overrides={
            "unnormalizer_processor": {"features": cfg.output_features, "norm_map": cfg.normalization_mapping}
        },
    )
    data = observation(cfg, 0)
    data["observation.state"] = torch.ones(1, 2)
    torch.testing.assert_close(loaded(data)["observation.state"], torch.full((1, 2), 1.0 if resume else 0.0))
    torch.testing.assert_close(output(torch.zeros(1, 3)), torch.full((1, 3), 0.5 if resume else 1.0))
    assert "stats" not in overrides["normalizer_processor"]


def test_native_make_policy_factory_and_reload(build, tmp_path):
    from types import SimpleNamespace

    from lerobot.policies import make_policy

    cfg = config()
    features = {
        key: {"dtype": "image", "shape": (16, 16, 3), "names": ["height", "width", "channel"]}
        for key in cfg.camera_keys
    }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            "action": {"dtype": "float32", "shape": (3,), "names": None},
        }
    )
    metadata = SimpleNamespace(features=features, stats={})
    policy = make_policy(cfg, ds_meta=metadata)
    assert isinstance(policy, LongWAMPolicy)
    policy.save_pretrained(tmp_path)
    cfg.pretrained_path = tmp_path
    loaded = make_policy(cfg, ds_meta=metadata)
    assert not loaded.model.initialized
    assert not cfg.push_to_hub
    deferred = make_policy(cfg, ds_meta=metadata, defer_weight_load=True)
    assert not deferred.model.initialized
    assert deferred.model.tokenizer_assets == tmp_path / "tokenizer"


@pytest.mark.parametrize("key", ["context", "context_mask"])
def test_partial_cached_text_fails_before_advancing_runtime(build, key):
    cfg = config()
    policy = LongWAMPolicy(cfg)
    data = observation(cfg, 0)
    data[key] = torch.ones(1, 4)
    with pytest.raises(ValueError, match="together"):
        policy.select_action(data)
    assert policy._runtime is None
