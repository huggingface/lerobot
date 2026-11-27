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

"""The base checkpoint: conversion, head initialization of a rig, normalization statistics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.praxis.modeling_praxis import PraxisPolicy
from lerobot.policies.praxis.praxis_core import rig
from lerobot.policies.praxis.praxis_core.checkpoint import (
    BASE_HEAD_TENSORS,
    save_checkpoint,
    validate_base_config,
    verify_checksums,
)
from lerobot.policies.praxis.praxis_core.train.norm_stats import norm_stats

from .test_praxis_training import training_batch
from .tiny_model import (
    FakeAutoencoder,
    FakeTextEncoder,
    make_tiny_policy,
    tiny_bimanual_lerobot_config,
    tiny_praxis_config,
)

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "praxis"
RIG_KEYS = ("action_space", "normalization", "pose_frame")


def _example(name: str):  # noqa: ANN202
    sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def tiny_base_config() -> dict:
    """The tiny single-arm config as a base config: no rig fields, no patch dims, a `heads` note."""
    config = tiny_praxis_config()
    config["embodiment"] = "base"
    config["chunking"] = {**config["chunking"], "action_frames_per_token": 4, "proprio_frames": 1}
    for key in RIG_KEYS:
        config.pop(key, None)
    for key in ("action_patch_dim", "proprio_patch_dim"):
        config["action_expert"].pop(key)
    for key in (
        "height",
        "width",
        "views",
        "tile_bounds_rows_cols",
        "tile_height",
        "tile_width",
        "antialias",
    ):
        config["video"].pop(key, None)
    config["heads"] = {"status": "absent", "tensors": list(BASE_HEAD_TENSORS)}
    return config


def save_tiny_base(path: Path, seed: int = 0) -> dict:
    """Write a tiny base checkpoint: the tiny policy's weights without the six head tensors."""
    policy = make_tiny_policy(seed=seed)
    weights = {k: v for k, v in policy.model.state_dict().items() if k not in BASE_HEAD_TENSORS}
    assert len(weights) == len(policy.model.state_dict()) - len(BASE_HEAD_TENSORS)
    config = tiny_base_config()
    validate_base_config(config)
    save_checkpoint(path, weights, config)
    return config


def test_base_config_fixture_and_rig_config() -> None:
    base = json.loads((Path(__file__).parent / "checkpoint_configs" / "praxis-1-base.json").read_text())
    validate_base_config(base)
    droid = json.loads((Path(__file__).parent / "checkpoint_configs" / "praxis-1-droid-m2.json").read_text())
    rebuilt = rig.single_arm_config(
        base, tile_height=224, tile_width=416, antialias=True, normalization=droid["normalization"]
    )
    # the DROID config rebuilds from the base plus the rig fields, apart from the provenance block
    for key in set(droid) | set(rebuilt):
        if key in ("source",):
            continue
        assert rebuilt.get(key) == droid.get(key), key


def test_init_heads_makes_a_loadable_policy(tmp_path: Path, monkeypatch) -> None:
    """The LeRobot init builds a policy of the rig from a base: the base tensors, six new heads, the processors."""
    init_heads = _example("init_heads")
    base_dir = tmp_path / "base"
    save_tiny_base(base_dir, seed=3)
    stats = {
        "action_q01": [-0.15, -0.2, -0.2, 0.68, -0.58, -0.39, -0.57, 0.69, -0.4, 0.0],
        "action_q99": [0.19, 0.2, 0.2, 1.0, 0.58, 0.45, 0.59, 1.0, 0.4, 1.0],
        "proprio_mean": [0.52, 0.0, 0.32, 0.74, -0.05, 0.09, -0.06, -0.76, 0.01, 0.39],
        "proprio_std": [0.12, 0.18, 0.17, 0.33, 0.48, 0.32, 0.49, 0.34, 0.25, 0.41],
    }
    out = init_heads.init_heads(
        base_dir,
        tmp_path / "rig",
        tile_height=32,
        tile_width=64,
        antialias=False,
        normalization=rig.normalization_from_stats(stats, domain="tiny"),
        seed=7,
        link=True,
        device="cpu",
        attention_backend="eager",
        image_shape=(3, 36, 64),
    )
    config = PreTrainedConfig.from_pretrained(out)
    assert config.embodiment == "single_arm" and config.model_config["normalization"]["domain"] == "tiny"
    verify_checksums(out)  # the base shards, the heads shard, the index and the config are all listed
    monkeypatch.setattr(PraxisPolicy, "_load_frozen_parts", lambda self: None)
    policy = PraxisPolicy.from_pretrained(out, config=config, strict=True)  # every tensor present, strict
    policy.vae, policy.text_encoder = FakeAutoencoder(), FakeTextEncoder()
    heads = rig.init_head_tensors(policy.config.model_config, seed=7)
    state = policy.model.state_dict()
    for name, tensor in heads.items():
        assert torch.equal(state[name].cpu(), tensor), name
    # the base tensors are the base's
    base_policy = make_tiny_policy(seed=3)
    for name, tensor in base_policy.model.state_dict().items():
        if name not in BASE_HEAD_TENSORS:
            assert torch.equal(state[name].cpu(), tensor), name
    # the processors are there and a training step runs
    assert (out / "policy_preprocessor.json").is_file() and (out / "policy_postprocessor.json").is_file()
    policy.train()
    loss, _ = policy.forward(training_batch(policy, seed=1))
    assert torch.isfinite(loss)
    # a base directory does not load as a policy
    with pytest.raises(ValueError, match="init_heads"):
        PraxisPolicy(PreTrainedConfig.from_pretrained(out).__class__(model_config=tiny_base_config()))


def test_init_heads_refuses_a_rig_checkpoint_and_wrong_heads(tmp_path: Path) -> None:
    init_heads = _example("init_heads")
    base_dir = tmp_path / "base"
    save_tiny_base(base_dir)
    norm = rig.normalization_from_stats(
        {
            k: [0.1] * 10 if k != "proprio_std" else [0.5] * 10
            for k in ("action_q01", "action_q99", "proprio_mean", "proprio_std")
        },
        domain="x",
    )
    norm["action_q99"] = [0.2] * 10
    bad_heads = {name: torch.zeros(1) for name in BASE_HEAD_TENSORS}
    with pytest.raises(Exception, match="shape"):
        init_heads.init_heads(
            base_dir,
            tmp_path / "a",
            tile_height=32,
            tile_width=64,
            antialias=False,
            normalization=norm,
            heads=bad_heads,
            device="cpu",
            attention_backend="eager",
        )
    rig_dir = tmp_path / "rig_ckpt"
    policy = make_tiny_policy(seed=1)
    save_checkpoint(rig_dir, policy.model.state_dict(), policy.config.model_config)
    with pytest.raises(Exception, match="base"):
        init_heads.init_heads(
            rig_dir,
            tmp_path / "b",
            tile_height=32,
            tile_width=64,
            antialias=False,
            normalization=norm,
            device="cpu",
            attention_backend="eager",
        )


def test_norm_stats_pin_the_gripper_and_guard_constant_dims() -> None:
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(1000, 10)).astype(np.float64)
    actions[:, 2] = 0.3  # a constant xyz dim
    proprio = rng.normal(size=(100, 10))
    proprio[:, 5] = 1.0  # a constant proprio dim
    stats = norm_stats(actions, proprio)
    assert (stats["action_q01"][9], stats["action_q99"][9]) == (0.0, 1.0)
    assert stats["action_q01"][2] == pytest.approx(-0.2) and stats["action_q99"][2] == pytest.approx(0.8)
    assert stats["proprio_std"][5] == pytest.approx(1e-3)


def test_compute_norm_stats_recovers_the_data_recipe_stats() -> None:
    """Windows drawn as the fine-tune draws them; the quantiles of a known motion come back."""
    stats_module = _example("compute_norm_stats")
    rng = np.random.default_rng(0)
    episodes = []
    for n in (150, 200, 260):
        t = np.arange(n, dtype=np.float32)[:, None] / 15.0
        obs = np.zeros((n, 7), np.float32)
        obs[:, :3] = np.array([0.5, 0.0, 0.4], np.float32) + 0.02 * np.sin(t)  # 2 cm oscillation
        obs[:, 3:6] = np.array([np.pi - 0.1, 0.05, 0.2], np.float32)
        obs[:, 6] = (np.sin(2 * t[:, 0]) > 0).astype(np.float32)
        cmd = obs.copy()
        cmd[:, :3] += 0.005 * rng.standard_normal((n, 3)).astype(np.float32)
        episodes.append((obs, cmd))
    stats = stats_module.sample_window_stats(episodes, n_windows=200, seed=0)
    assert (stats["action_q01"][9], stats["action_q99"][9]) == (0.0, 1.0)
    # relative xyz motion over a 24-frame chunk of a 2 cm oscillation stays within a few cm
    assert all(-0.06 < q < 0.0 for q in stats["action_q01"][:3]) and all(
        0.0 < q < 0.06 for q in stats["action_q99"][:3]
    )
    assert abs(stats["proprio_mean"][0] - 0.5) < 0.01 and stats["proprio_std"][0] < 0.03


def test_compute_norm_stats_slices_episodes_by_their_frame_ranges() -> None:
    stats_module = _example("compute_norm_stats")
    # an episode-filtered table: episodes with the frames 10-14 and 30-32, in order
    frame_index = np.array([10, 11, 12, 13, 14, 30, 31, 32])
    slices = stats_module.episode_slices(frame_index, np.array([10, 30]), np.array([15, 33]))
    assert slices == [slice(0, 5), slice(5, 8)]
    with pytest.raises(ValueError, match="every frame"):
        stats_module.episode_slices(frame_index, np.array([10]), np.array([16]))
    with pytest.raises(ValueError, match="increase"):
        stats_module.episode_slices(frame_index[::-1], np.array([10]), np.array([15]))


def test_compute_norm_stats_loads_the_episodes_of_a_dataset(tmp_path: Path) -> None:
    """One read of the columns, sliced per episode: the rows of each selected episode, in episode order."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    stats_module = _example("compute_norm_stats")
    features = {
        "observation.state.cartesian": {"dtype": "float32", "shape": (7,), "names": None},
        "action.cartesian": {"dtype": "float32", "shape": (7,), "names": None},
    }
    dataset = LeRobotDataset.create(
        "local/praxis_norm_stats", fps=15, features=features, root=tmp_path / "ds", use_videos=False
    )
    rng = np.random.default_rng(0)
    recorded = []
    for n in (5, 3, 4):
        obs = rng.standard_normal((n, 7)).astype(np.float32)
        cmd = rng.standard_normal((n, 7)).astype(np.float32)
        for t in range(n):
            dataset.add_frame(
                {"observation.state.cartesian": obs[t], "action.cartesian": cmd[t], "task": "pick"}
            )
        dataset.save_episode()
        recorded.append((obs, cmd))
    dataset.finalize()
    for selected, expected in ((None, recorded), ([2, 0], [recorded[0], recorded[2]])):
        episodes = stats_module.load_episodes(
            "local/praxis_norm_stats",
            root=str(tmp_path / "ds"),
            episodes=selected,
            state_key="observation.state.cartesian",
            action_key="action.cartesian",
        )
        assert len(episodes) == len(expected)
        for (obs, cmd), (want_obs, want_cmd) in zip(episodes, expected, strict=True):
            np.testing.assert_array_equal(obs, want_obs)
            np.testing.assert_array_equal(cmd, want_cmd)


def test_base_config_is_refused_with_the_init_pointer() -> None:
    base = json.loads(((Path(__file__).parent / "checkpoint_configs") / "praxis-1-base.json").read_text())
    config = tiny_bimanual_lerobot_config(model_config=base)
    with pytest.raises(ValueError, match="init_heads"):
        PraxisPolicy(config)
