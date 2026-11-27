#!/usr/bin/env python

# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""CPU tests of CAE-FastWAM's action VAE training (`lerobot-train-tokenizer --tokenizer_type=cae_action_vae`)
and of `--policy.action_vae_path`."""

import json
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="cae_fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="cae_fastwam requires the `fastwam` extra (diffusers)")

import lerobot.datasets
from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.cae_fastwam import action_vae
from lerobot.policies.cae_fastwam.action_vae import (
    CAEActionVAE,
    CAEActionVAETrainingConfig,
    build_action_chunks,
    load_cae_action_vae_config,
    normalize_action_chunks,
    resolve_schedule,
    save_cae_action_vae,
    train_cae_action_vae,
)
from lerobot.policies.cae_fastwam.configuration_cae_fastwam import (
    CAE_FASTWAM_BASE_MODEL_ID,
    CAEFastWAMConfig,
)
from lerobot.policies.cae_fastwam.modeling_cae_fastwam import CAEFastWAMPolicy
from lerobot.policies.cae_fastwam.processor_cae_fastwam import make_cae_fastwam_pre_post_processors
from lerobot.scripts.lerobot_train_tokenizer import TokenizerTrainingConfig, train_tokenizer
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.cae_fastwam.test_cae_fastwam_policy import (  # noqa: F401
    inference_batch,
    offline_components,
    tiny_config,
)


def fake_dataset(episode_lengths, action_dim=3, seed=0):
    """An object with the `hf_dataset` columns the chunk builder reads, rows shuffled."""
    rng = np.random.default_rng(seed)
    episode_index = np.concatenate([np.full(n, e) for e, n in enumerate(episode_lengths)])
    frame_index = np.concatenate([np.arange(n) for n in episode_lengths])
    actions = rng.normal(size=(len(episode_index), action_dim)).astype(np.float32)
    order = rng.permutation(len(episode_index))
    table = datasets.Dataset.from_dict(
        {
            ACTION: actions[order].tolist(),
            "episode_index": episode_index[order].tolist(),
            "frame_index": frame_index[order].tolist(),
        }
    )
    return SimpleNamespace(hf_dataset=table), actions, episode_lengths


def test_action_chunks_start_at_every_frame_and_pad_like_lerobot_train():
    dataset, actions, lengths = fake_dataset([5, 3])
    chunks, is_pad = build_action_chunks(dataset, action_horizon=4)
    assert chunks.shape == (8, 4, 3) and is_pad.shape == (8, 4) and chunks.dtype == np.float32
    # Episode 1 starts at row 5; its chunk at frame 1 is frames 1, 2 then the last frame repeated (padding).
    np.testing.assert_array_equal(chunks[6], actions[[6, 7, 7, 7]])
    np.testing.assert_array_equal(is_pad[6], [False, False, True, True])
    np.testing.assert_array_equal(chunks[0], actions[0:4])
    assert not is_pad[0].any() and not is_pad[1].any() and is_pad[2, 3]
    first_only, _ = build_action_chunks(dataset, action_horizon=4, max_episodes=1)
    np.testing.assert_array_equal(first_only, chunks[:5])
    sampled, sampled_pad = build_action_chunks(dataset, action_horizon=4, sample_fraction=0.5, seed=3)
    assert len(sampled) == 2 + 1 and sampled_pad.shape == (3, 4)


def test_actions_are_normalized_like_the_policy_preprocessor():
    stats = {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 0.0], "q01": [0.0, 0.0, 0.0], "q99": [0.5, 1.0, 1.0]}
    chunks = np.array([[[0.5, 1.5, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        normalize_action_chunks(chunks, stats, NormalizationMode.MIN_MAX), [[[0.5, 0.5, -1.0]]], atol=1e-6
    )
    np.testing.assert_allclose(
        normalize_action_chunks(chunks, stats, NormalizationMode.QUANTILES), [[[1.0, 2.0, -1.0]]], atol=1e-6
    )
    assert normalize_action_chunks(chunks, stats, NormalizationMode.IDENTITY) is chunks


def tiny_vae(num_suites=1, variational=True, latent_dim=5):
    torch.manual_seed(0)
    return CAEActionVAE(
        action_dim=3,
        action_horizon=8,
        num_tokens=2,
        latent_dim=latent_dim,
        encoder_hidden_dim=8,
        decoder_hidden_dim=8,
        num_suites=num_suites,
        variational=variational,
    )


def test_loss_is_the_original_stage_1_objective():
    vae = tiny_vae()
    actions = torch.randn(4, 8, 3)
    is_pad = torch.zeros(4, 8, dtype=torch.bool)
    is_pad[0, 6:] = True
    ids = torch.zeros(4, dtype=torch.long)
    torch.manual_seed(7)
    losses = vae.compute_loss(actions, is_pad, ids, beta_kl=0.5, free_bits=0.1, velocity_weight=0.1)
    torch.manual_seed(7)
    out = vae(actions, ids)
    error = (out["recon"] - actions).pow(2).mean(-1)
    valid = (~is_pad).float()
    recon = ((error * valid).sum(1) / valid.sum(1)).mean()
    vel = (out["recon"].diff(dim=1) - actions.diff(dim=1)).pow(2).mean()
    kl = (-0.5 * (1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp())).clamp(min=0.1).mean()
    torch.testing.assert_close(losses["loss"], recon + 0.1 * vel + 0.5 * kl)
    torch.testing.assert_close(losses["loss_kl"], kl)


def test_training_fits_chunks_deterministically():
    t = torch.linspace(0, 1, 8)
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 6, size=(512, 1, 3))
    chunks = np.sin(t.numpy()[None, :, None] * 3 + phases).astype(np.float32)
    is_pad = np.zeros(chunks.shape[:2], dtype=bool)
    cfg = CAEActionVAETrainingConfig(
        num_tokens=2, latent_dim=5, encoder_hidden_dim=16, decoder_hidden_dim=16, epochs=6, batch_size=64,
        lr=3e-3, ema_decay=0.9, kl_warmup_epochs=2, device="cpu",
    )  # fmt: skip
    vae, metrics = train_cae_action_vae(chunks, is_pad, cfg)
    history = metrics["history"]
    assert len(history) == 6 and metrics["steps"] == 6 * 8 and metrics["kl_warmup_epochs"] == 2
    assert history[-1]["loss_recon"] < history[0]["loss_recon"]
    assert [h["beta_kl"] for h in history[:3]] == [0.5e-4, 1e-4, 1e-4]
    assert metrics["final"]["mse"] < float(np.mean(chunks**2))
    assert not any(p.requires_grad for p in vae.parameters())
    vae2, metrics2 = train_cae_action_vae(chunks, is_pad, cfg)
    assert metrics2["final"] == metrics["final"]
    for a, b in zip(vae.state_dict().values(), vae2.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_default_schedule_is_the_released_step_budget():
    """About 54k optimizer steps whatever the dataset size, with the released KL schedule (warmup = 2 x epochs). The
    four LIBERO suites (53229 / 67309 / 52895 / 104280 chunks) get the epochs the step-matched validation used."""
    cfg = CAEActionVAETrainingConfig()
    assert [resolve_schedule(n, cfg) for n in (53229, 67309, 52895, 104280)] == [
        (261, 522),
        (206, 412),
        (262, 524),
        (133, 266),
    ]
    assert resolve_schedule(277713, cfg) == (50, 100)  # the four suites together: the released 50 epochs
    assert resolve_schedule(53229, CAEActionVAETrainingConfig(epochs=50)) == (50, 100)
    with pytest.raises(ValueError, match="batch size"):
        resolve_schedule(100, cfg)


def vae_stats():
    return {
        "my_robot": {
            OBS_STATE: {"min": [0.0, -1.0], "max": [2.0, 1.0]},
            ACTION: {"min": [-1.0, -1.0, 0.0], "max": [1.0, 3.0, 1.0]},
        }
    }


def saved_vae(path: Path, latent_dim: int = 5, mode: str = "MIN_MAX") -> tuple[Path, CAEActionVAE]:
    vae = tiny_vae(latent_dim=latent_dim)
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.3)
    config = {
        "action_dim": 3,
        "action_horizon": 8,
        "num_tokens": 2,
        "latent_dim": latent_dim,
        "encoder_hidden_dim": 8,
        "decoder_hidden_dim": 8,
        "variational": True,
        "task_suites": ["my_robot"],
        "normalization_mode": mode,
    }
    return save_cae_action_vae(path, vae, config, vae_stats(), {"note": "test"}), vae


def test_saved_vae_round_trips(tmp_path):
    path, vae = saved_vae(tmp_path / "vae")
    assert {p.name for p in path.iterdir()} == {
        "config.json",
        "model.safetensors",
        "stats.json",
        "metadata.json",
    }
    config, stats = load_cae_action_vae_config(path)
    assert config["type"] == "cae_action_vae" and stats == vae_stats()
    loaded = CAEActionVAE.from_config(config)
    loaded.load_state_dict(action_vae.load_cae_action_vae_state_dict(path))
    for key, value in vae.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    (tmp_path / "not_vae").mkdir()
    (tmp_path / "not_vae" / "config.json").write_text(json.dumps({"type": "cae_fastwam"}))
    with pytest.raises(ValueError, match="not a CAE-FastWAM action VAE"):
        load_cae_action_vae_config(tmp_path / "not_vae")


def test_policy_config_adopts_the_trained_vae(tmp_path):
    path, _ = saved_vae(tmp_path / "vae")
    cfg = tiny_config(action_vae_path=str(path), task_suite=None, task_suite_stats=None)
    assert cfg.task_suites == ["my_robot"] and cfg.task_suite_stats == vae_stats()
    assert cfg.resolve_task_suite() == "my_robot" and cfg.suite_id() == 0
    quantiles, _ = saved_vae(tmp_path / "vae_q", mode="QUANTILES")
    with pytest.raises(ValueError, match="--normalization_mode=MIN_MAX"):
        tiny_config(action_vae_path=str(quantiles), task_suite=None, task_suite_stats=None)
    # The released checkpoint still initializes everything but the VAE (whose fields may differ) ...
    d16, _ = saved_vae(tmp_path / "vae_d16", latent_dim=16)
    default = CAEFastWAMConfig(action_vae_path=str(d16), n_action_steps=4)
    assert default.task_suites == ["my_robot"] and (default.action_horizon, default.action_dim) == (8, 3)
    assert default.pretrained_path == Path(CAE_FASTWAM_BASE_MODEL_ID)
    # ... but not when a weight of the action expert would not fit.
    assert CAEFastWAMConfig(action_vae_path=str(path), n_action_steps=4).pretrained_path is None
    # The processors normalize with the VAE's statistics.
    pre, _ = make_cae_fastwam_pre_post_processors(cfg)
    obs = {
        "observation.images.image": torch.rand(3, 16, 16),
        "observation.images.image2": torch.rand(3, 16, 16),
        OBS_STATE: torch.tensor([1.0, 0.5]),
    }
    torch.testing.assert_close(pre(obs)[OBS_STATE], torch.tensor([[0.0, 0.5]]))


def test_policy_loads_the_vae_weights_over_the_checkpoint_and_saves_them(offline_components, tmp_path):  # noqa: F811
    path, vae = saved_vae(tmp_path / "vae")
    base = CAEFastWAMPolicy(tiny_config(dtype=torch.float32))  # a checkpoint with a 2-suite VAE
    base.save_pretrained(tmp_path / "base")
    cfg = tiny_config(action_vae_path=str(path), task_suite=None, task_suite_stats=None, dtype=torch.float32)
    policy = CAEFastWAMPolicy.from_pretrained(tmp_path / "base", config=cfg)
    for name, module in (("encoder", policy.model.action_encoder), ("decoder", policy.model.action_decoder)):
        for key, value in getattr(vae, name).state_dict().items():
            torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)
        assert not any(p.requires_grad for p in module.parameters())
    # Everything else comes from the checkpoint.
    base_mot = {k: v for k, v in base.state_dict().items() if ".mot." in k}
    for key, value in base_mot.items():
        torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)
    assert policy.predict_action_chunk(inference_batch()).shape == (1, 8, 3)
    # The saved policy is self-contained: VAE weights and statistics, no `action_vae_path`.
    policy.save_pretrained(tmp_path / "trained")
    assert policy.config.action_vae_path == str(path)
    saved = PreTrainedConfig.from_pretrained(tmp_path / "trained")
    assert saved.action_vae_path is None and saved.task_suites == ["my_robot"]
    assert saved.task_suite_stats == vae_stats()
    reloaded = CAEFastWAMPolicy.from_pretrained(tmp_path / "trained")
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)


def test_train_tokenizer_trains_the_cae_action_vae(tmp_path, lerobot_dataset_factory, monkeypatch):
    dataset = lerobot_dataset_factory(
        root=tmp_path / "data", total_episodes=4, total_frames=200, use_videos=False
    )
    monkeypatch.setattr(lerobot.datasets, "LeRobotDataset", lambda **kwargs: dataset)
    action_dim = dataset.meta.features[ACTION]["shape"][0]
    out = tmp_path / "vae"
    cfg = TokenizerTrainingConfig(
        repo_id="user/robot",
        tokenizer_type="cae_action_vae",
        action_horizon=8,
        sample_fraction=1.0,
        normalization_mode="MIN_MAX",
        output_dir=str(out),
        cae=CAEActionVAETrainingConfig(
            num_tokens=2, latent_dim=4, encoder_hidden_dim=8, decoder_hidden_dim=8, steps=12, batch_size=32,
            device="cpu",
        ),
    )  # fmt: skip
    train_tokenizer(cfg)
    config, stats = load_cae_action_vae_config(out)
    assert config["action_dim"] == action_dim and config["task_suites"] == ["user/robot"]
    assert config["normalization_mode"] == "MIN_MAX" and set(stats["user/robot"]) >= {ACTION}
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["num_training_chunks"] == 200 and len(metadata["metrics"]["history"]) == 2
    with pytest.raises(ValueError, match="multiple of"):
        train_tokenizer(TokenizerTrainingConfig(repo_id="user/robot", tokenizer_type="cae_action_vae"))
    with pytest.raises(ValueError, match="Unknown tokenizer_type"):
        train_tokenizer(TokenizerTrainingConfig(repo_id="user/robot", tokenizer_type="vq"))
