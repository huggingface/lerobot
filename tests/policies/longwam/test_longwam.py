# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline end-to-end checks with real, reduced Wan/VAE/UMT5 backbones."""

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from lerobot.policies import make_pre_post_processors
from lerobot.policies.longwam.configuration_longwam import LongWAMConfig
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy
from lerobot.processor import NormalizerProcessorStep, TokenizerProcessorStep
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "place": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(
        tmp_path / "tokenizer"
    )
    return LongWAMConfig(
        device="cpu",
        dtype=torch.float32,
        push_to_hub=False,
        camera_keys=("observation.images.front", "observation.images.wrist"),
        camera_size=(16, 16),
        past_obs_steps=16,
        future_video_steps=16,
        action_horizon=4,
        inference_action_horizon=6,
        n_action_steps=2,
        action_dim=3,
        proprio_dim=2,
        num_inference_steps=2,
        imagine_steps=1,
        action_hidden_dim=8,
        action_ffn_dim=32,
        tokenizer_name=str(tmp_path / "tokenizer"),
        tokenizer_max_length=8,
        prompt_template="{task}",
        video_config={
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
            "patch_size": [1, 2, 2],
            "rope_max_seq_len": 32,
        },
        vae_config={
            "base_dim": 4,
            "z_dim": 4,
            "dim_mult": [1, 2, 2],
            "num_res_blocks": 1,
            "temperal_downsample": [True, True],
            "scale_factor_temporal": 4,
            "scale_factor_spatial": 4,
            "latents_mean": [0.0] * 4,
            "latents_std": [1.0] * 4,
        },
        text_encoder_config={
            "vocab_size": 4,
            "d_model": 16,
            "d_kv": 8,
            "d_ff": 32,
            "num_layers": 1,
            "num_heads": 2,
            "dropout_rate": 0,
            "feed_forward_proj": "gated-gelu",
        },
    )


def stats(cfg):
    return {
        "observation.state": {
            "min": torch.full((cfg.proprio_dim,), -2.0),
            "max": torch.full((cfg.proprio_dim,), 2.0),
        },
        "action": {"min": torch.full((cfg.action_dim,), -2.0), "max": torch.full((cfg.action_dim,), 2.0)},
    }


def batch(cfg, temporal=False, value=0.25, task="pick"):
    shape = (1, len(cfg.observation_delta_indices)) if temporal else (1,)
    result = {
        **{key: torch.full((*shape, 3, 16, 16), value) for key in cfg.camera_keys},
        "observation.state": torch.full((*shape, cfg.proprio_dim), value),
        "task": [task],
    }
    if temporal:
        result["action"] = torch.randn(1, cfg.action_horizon, cfg.action_dim)
        result["action_is_pad"] = torch.tensor([[False, False, True, True]])
        for key in cfg.camera_keys:
            result[f"{key}_is_pad"] = torch.zeros(shape, dtype=torch.bool)
            result[f"{key}_is_pad"][:, -2:] = True
    return result


@pytest.mark.parametrize(
    "input_size,camera_size",
    [
        ((16, 32), (16, 16)),
        ((32, 16), (16, 16)),
        ((16, 16), (16, 16)),
        ((32, 32), (16, 16)),
        ((23, 41), (16, 16)),
        ((41, 23), (16, 16)),
        ((41, 23), (16, 32)),
        ((23, 41), (32, 16)),
    ],
)
def test_training_and_inference_camera_geometry(config, input_size, camera_size):
    config.camera_size = camera_size
    pre, _ = make_pre_post_processors(config, dataset_stats=stats(config))
    height, width = input_size
    images = torch.ones(2, 3, height, width)
    if width >= height:
        images[..., width // 4 : 3 * width // 4] = 0
    else:
        images[..., height // 4 : 3 * height // 4, :] = 0
    images[1] = 1 - images[0]
    video = torch.stack((images, 1 - images), dim=1)

    def inputs(frames):
        return {
            **dict.fromkeys(config.camera_keys, frames),
            "observation.state": torch.zeros(*frames.shape[:-3], config.proprio_dim),
            "task": ["pick"] * frames.shape[0],
        }

    training = pre(inputs(video))
    inference = pre(inputs(video.flatten(0, 1)))
    for key in config.camera_keys:
        expected = inference[key].reshape(2, 2, 3, *camera_size)
        # Tensor training interpolation need not reproduce PIL's uint8 rounding.
        torch.testing.assert_close(training[key], expected, atol=2 / 255, rtol=0)
        if input_size in ((16, 32), (32, 16)):
            torch.testing.assert_close(training[key][0, 0], -torch.ones(3, *camera_size))


@pytest.mark.parametrize("mode", ["idm", "cod"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_train_save_load_and_sync_inference(config, tmp_path, mode, checkpointing):
    config.denoising, config.imagine_sigma = mode, 0 if mode == "cod" else 0.9
    config.use_gradient_checkpointing = checkpointing
    torch.manual_seed(7)
    policy = LongWAMPolicy(config)
    action_rope = policy.dit.action.rope_cos.clone()
    video_rope = policy.dit.video.rope.freqs_cos.clone()
    policy.bfloat16()
    torch.testing.assert_close(policy.dit.action.rope_cos, action_rope, atol=0, rtol=0)
    torch.testing.assert_close(policy.dit.video.rope.freqs_cos, video_rope, atol=0, rtol=0)
    policy.float()
    pre, post = make_pre_post_processors(config, dataset_stats=stats(config))
    assert any(type(step) is NormalizerProcessorStep for step in pre.steps)
    assert any(type(step) is TokenizerProcessorStep for step in pre.steps)
    assert config.observation_delta_indices == list(range(-16, 17, 4))
    frozen = {n: p.clone() for n, p in policy.named_parameters() if not p.requires_grad}
    trainable = {n: p.clone() for n, p in policy.named_parameters() if p.requires_grad}
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    training_batch = batch(config, temporal=True)
    for key, size in zip(config.camera_keys, ((16, 32), (32, 16)), strict=True):
        training_batch[key] = torch.rand(*training_batch[key].shape[:-2], *size)
    loss, metrics = policy(pre(training_batch))
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss_video", "loss_action"}
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    optimizer.step()
    for prefix in ("dit.video.", "dit.action.", "proprio_encoder."):
        assert any(
            not torch.equal(trainable[n], p) for n, p in policy.named_parameters() if n.startswith(prefix)
        )
    assert all(torch.equal(frozen[n], p) for n, p in policy.named_parameters() if n in frozen)
    save = tmp_path / "native"
    policy.save_pretrained(save)
    pre.save_pretrained(save, config_filename="policy_preprocessor.json")
    post.save_pretrained(save, config_filename="policy_postprocessor.json")
    restored = LongWAMPolicy.from_pretrained(save, strict=True)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=save)
    observation = batch(config)
    policy.eval()
    expected = post(policy.select_action(pre(observation)))
    actual = loaded_post(restored.select_action(loaded_pre(observation)))
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert actual.shape == (1, config.action_dim)
    assert len(restored._frames) == 1
    restored.select_action(loaded_pre(batch(config, value=0.5)))
    assert len(restored._frames) == 2
    restored.reset()
    chunk = restored.predict_action_chunk(loaded_pre(observation))
    assert chunk.shape == (1, config.prediction_steps, config.action_dim)
    assert torch.isfinite(chunk).all()
    restored.reset()
    torch.testing.assert_close(restored.select_action(loaded_pre(observation)), chunk[:, 0])
    # Interactive instruction switch must discard old actions and old history.
    changed = loaded_pre(batch(config, value=0.75, task="place"))
    restored.select_action(changed)
    assert len(restored._frames) == 1
    torch.testing.assert_close(restored._language_tokens, changed[OBS_LANGUAGE_TOKENS])
    restored.drop_queued_actions()
    assert not restored._frames and not restored._action_queue and restored._text_context is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["idm", "cod"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_training_parameter_and_optimizer_precision(config, tmp_path, device, mode, dtype, autocast):
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip("requires CUDA with BF16 support")
    torch.manual_seed(7)
    config.denoising, config.imagine_sigma = mode, 0 if mode == "cod" else 0.9
    config.use_gradient_checkpointing = True
    # Save FP32 values first so loading into BF16 cannot silently pass an FP32 test.
    initial = LongWAMPolicy(config)
    checkpoint = tmp_path / "precision"
    initial.save_pretrained(checkpoint)
    config.dtype, config.device = dtype, device
    policy = LongWAMPolicy.from_pretrained(checkpoint, config=config, strict=True).train()
    for name, parameter in policy.named_parameters():
        assert parameter.dtype == dtype
        torch.testing.assert_close(
            parameter, initial.get_parameter(name).to(device=device, dtype=dtype), atol=0, rtol=0
        )
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    pre, _ = make_pre_post_processors(config, dataset_stats=stats(config))
    training_batch = pre(batch(config, temporal=True))
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
        loss, _ = policy(training_batch)
    assert torch.isfinite(loss)
    loss.backward()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    assert trainable
    for parameter in trainable:
        assert parameter.grad is not None
        assert parameter.grad.dtype == dtype
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    for parameter in trainable:
        for moment in ("exp_avg", "exp_avg_sq"):
            assert optimizer.state[parameter][moment].dtype == dtype
            assert torch.isfinite(optimizer.state[parameter][moment]).all()
    for prefix in ("dit.video.", "dit.action.", "proprio_encoder."):
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in policy.named_parameters()
            if name.startswith(prefix)
        )
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["idm", "cod"])
def test_history_eviction_and_padding(config, mode):
    config.denoising, config.imagine_sigma = mode, 0 if mode == "cod" else 0.9
    policy = LongWAMPolicy(config)
    pre, _ = make_pre_post_processors(config, dataset_stats=stats(config))
    for step in range(25):
        policy._observe(pre(batch(config, value=step / 25)))
    assert len(policy._frames) == 17
    values = policy._history()[0, 0, :, 0, 0]
    expected_pixels = (torch.tensor([8, 12, 16, 20, 24]) / 25 * 255).round()
    torch.testing.assert_close(values, expected_pixels * (2 / 255) - 1)
    data = pre(batch(config, temporal=True))
    data["action_is_pad"].fill_(True)
    for key in config.camera_keys:
        data[f"{key}_is_pad"].fill_(True)
    loss, _ = policy(data)
    assert loss.item() == 0


@pytest.mark.parametrize("boundary", ["meta", "assigned_state"])
def test_restore_shared_text_weights_after_loading_boundary(config, boundary):
    policy = LongWAMPolicy(config)
    assert policy.text_encoder.shared.weight is policy.text_encoder.encoder.embed_tokens.weight
    if boundary == "meta":
        policy.to("meta")
    else:
        state = {name: value.clone() for name, value in policy.state_dict().items()}
        policy.load_state_dict(state, assign=True)
    assert policy.text_encoder.shared.weight is not policy.text_encoder.encoder.embed_tokens.weight
    policy.tie_weights()
    assert policy.text_encoder.shared.weight is policy.text_encoder.encoder.embed_tokens.weight
    assert not policy.text_encoder.shared.weight.requires_grad
