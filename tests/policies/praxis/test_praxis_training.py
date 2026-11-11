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

"""Training of the praxis policy on a tiny random model: the window recipe, the objective, the hooks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from lerobot.optim.optimizers import PraxisAdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.praxis.praxis_core.layers import MoEFeedForward
from lerobot.policies.praxis.praxis_core.robot_io import contiguous_chunk_geometry, pose8_to_10d
from lerobot.policies.praxis.praxis_core.single_arm import tile_clip
from lerobot.policies.praxis.praxis_core.train.data import (
    TrainingWindow,
    draw_anchor,
    droid_pose8,
    window_fits,
)
from lerobot.policies.praxis.praxis_core.train.objective import encode_window
from lerobot.policies.praxis.praxis_core.train.optim import AdamW, derive_seed
from lerobot.utils.constants import ACTION, OBS_STATE

from .tiny_model import TILE, FakeAutoencoder, FakeTextEncoder, make_tiny_policy, tiny_praxis_config

pytest.importorskip("transformers", reason="praxis requires the praxis extra (transformers)")

IMAGE = (3, 36, 64)
EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "praxis"


def load_example(name: str) -> ModuleType:
    """A script of ``examples/praxis``, imported by path."""
    spec = importlib.util.spec_from_file_location(f"praxis_example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_batch(policy, seed: int = 0, batch: int = 1, image: tuple[int, int, int] = IMAGE) -> dict:
    """A LeRobot training batch of `batch` windows: random cameras, a smooth pose, a commanded pose."""
    frames = policy.config.window_frames
    rng = np.random.default_rng(seed)
    exterior = (
        torch.from_numpy(rng.integers(0, 256, size=(batch, frames, *image), dtype=np.uint8)).float() / 255
    )
    wrist = torch.from_numpy(rng.integers(0, 256, size=(batch, frames, *image), dtype=np.uint8)).float() / 255
    t = np.arange(frames, dtype=np.float32)[None, :, None] / frames
    base = np.array([0.5, 0.0, 0.4, np.pi - 0.1, 0.05, 0.2, 0.0], dtype=np.float32)
    drift = np.array([0.1, -0.05, 0.08, 0.2, 0.1, -0.3, 1.0], dtype=np.float32)
    state = base + drift * t + rng.normal(0, 1e-3, size=(batch, frames, 7)).astype(np.float32)
    action = state + np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05], dtype=np.float32)
    state[..., 6] = np.clip(state[..., 6], 0, 1)
    action[..., 6] = np.clip(action[..., 6], 0, 1)
    return {
        policy.config.exterior_image_key: exterior,
        policy.config.wrist_image_key: wrist,
        OBS_STATE: torch.from_numpy(state),
        ACTION: torch.from_numpy(action),
        "task": [f"Pick up the red cube {b}." for b in range(batch)],
    }


def reference_window(policy, batch: dict, index: int) -> TrainingWindow:
    """The window as the praxis data module builds it (`DroidWindowDataset.load_window`)."""
    cfg = policy.config.model_config
    chunking = cfg["chunking"]
    geometry = contiguous_chunk_geometry(int(chunking["action_chunk"]), int(chunking["n_chunks"]))
    to_u8 = lambda x: (x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()  # noqa: E731
    ext = to_u8(batch[policy.config.exterior_image_key][index])
    wrist = to_u8(batch[policy.config.wrist_image_key][index])
    clip = tile_clip(
        ext,
        wrist,
        TILE[0],
        TILE[1],
        float(cfg["video"]["aspect_tol"]),
        antialias=bool(cfg["video"].get("antialias", False)),
    )
    obs7 = batch[OBS_STATE][index].numpy()
    cmd7 = batch[ACTION][index].numpy()
    observed = droid_pose8(obs7[:, :6], obs7[:, 6])
    commanded = droid_pose8(cmd7[:, :6], cmd7[:, 6])
    starts = [off for off, _ in geometry]
    actions = [
        pose8_to_10d(commanded[sk : sk + ln], anchor8=observed[sk])
        for sk, (_, ln) in zip(starts, geometry, strict=True)
    ]
    return TrainingWindow(
        frames=clip[0].permute(1, 0, 2, 3).to(torch.uint8).contiguous(),
        prompt=batch["task"][index],
        actions_rel=np.concatenate(actions, axis=0),
        proprio_abs=pose8_to_10d(observed[starts]),
        meta={},
    )


def test_window_recipe_constants() -> None:
    policy = make_tiny_policy(seed=0)
    cfg = policy.config
    assert cfg.window_frames == 93 and cfg.drop_n_last_frames == 93
    assert cfg.observation_delta_indices == list(range(93)) and cfg.action_delta_indices == list(range(93))
    assert window_fits(102, 93, 15.0) and not window_fits(93, 93, 15.0)
    # The recipe's last window of a 200-frame episode starts at frame 106 (ends one frame before the
    # last); the episode-aware sampler draws the same starts with `drop_n_last_frames` dropped.
    anchors = {draw_anchor(200, 93, 15.0, np.random.default_rng(i)) for i in range(3000)}
    assert (min(anchors), max(anchors)) == (0, 200 - 94)
    from lerobot.datasets.sampler import EpisodeAwareSampler

    sampler = EpisodeAwareSampler(
        np.array([0]),
        np.array([200]),
        drop_n_first_frames=0,
        drop_n_last_frames=cfg.drop_n_last_frames,
        shuffle=False,
    )
    starts = list(iter(sampler))
    assert (min(starts), max(starts)) == (min(anchors), max(anchors))


def test_training_windows_follow_the_praxis_data_recipe() -> None:
    policy = make_tiny_policy(seed=1)
    batch = training_batch(policy, seed=3, batch=2)
    windows = policy._training_windows(batch)
    assert len(windows) == 2
    for index, window in enumerate(windows):
        ref = reference_window(policy, batch, index)
        assert window.frames.shape == (93, 3, TILE[0], 2 * TILE[1]) and window.frames.dtype == torch.uint8
        assert torch.equal(window.frames, ref.frames)
        assert np.array_equal(window.actions_rel, ref.actions_rel) and window.actions_rel.shape == (93, 10)
        assert np.array_equal(window.proprio_abs, ref.proprio_abs) and window.proprio_abs.shape == (4, 10)
        assert window.prompt == batch["task"][index]


def test_window_poses_match_the_upstream_praxis_package() -> None:
    """The vendored pose and canvas functions give the bits of the praxis repository's."""
    praxis_data = pytest.importorskip("praxis.train.data")
    praxis_io = pytest.importorskip("praxis.robot_io")
    rng = np.random.default_rng(0)
    cart = rng.uniform(-1, 1, size=(30, 6)).astype(np.float32)
    grip = rng.uniform(-0.2, 1.2, size=30).astype(np.float32)
    ours, theirs = droid_pose8(cart, grip), praxis_data.droid_pose8(cart, grip)
    assert np.array_equal(ours, theirs)
    rel_ours = pose8_to_10d(ours[1:10], anchor8=ours[0])
    rel_theirs = praxis_io.pose8_to_10d(theirs[1:10], anchor8=theirs[0])
    assert np.array_equal(rel_ours, rel_theirs)


def test_forward_matches_the_praxis_objective() -> None:
    """`forward` equals encode_window + FinetuneModel.loss with the same draws (the praxis trainer's path)."""
    policy = make_tiny_policy(seed=2, text_dropout=0.0, proprio_dropout=0.0, activation_checkpointing=False)
    policy.train()
    batch = training_batch(policy, seed=5)
    loss, info = policy.forward(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert set(info) >= {"loss_video", "loss_action", "action_dropped"} and all(
        isinstance(v, float) for v in info.values()
    )
    finetune = policy._finetune
    window = reference_window(policy, batch, 0)
    encoded = encode_window(window, FakeAutoencoder(), FakeTextEncoder(), policy.config.model_config, "cpu")
    seed = derive_seed(policy.config.seed, 0, 0, 0)
    ref_loss, ref_metrics = finetune.loss(
        encoded, finetune.draw(encoded, torch.Generator().manual_seed(seed))
    )
    assert torch.equal(loss.detach(), ref_loss.detach())
    assert info["loss_video"] == float(ref_metrics["loss_video"]) and info["loss_action"] == float(
        ref_metrics["loss_action"]
    )


def test_forward_is_deterministic_per_call_and_changes_between_calls() -> None:
    policy = make_tiny_policy(seed=4, activation_checkpointing=False)
    policy.train()
    batch = training_batch(policy, seed=6)
    loss_a, _ = policy.forward(batch)
    policy.train_counter.calls.zero_()
    loss_b, _ = policy.forward(batch)
    loss_c, _ = policy.forward(batch)  # second call: other draws
    assert torch.equal(loss_a.detach(), loss_b.detach()) and not torch.equal(loss_b.detach(), loss_c.detach())


def test_backward_trains_the_weights_and_update_moves_the_router_biases() -> None:
    policy = make_tiny_policy(seed=7, moe_balance_speed=0.01, activation_checkpointing=False)
    params = policy.get_optim_params()
    assert len(params) == len(list(policy.model.parameters())) and all(p.requires_grad for p in params)
    # fetching the optimizer parameters does not change the model: frozen weights stay frozen
    policy.model.action_expert.requires_grad_(False)
    assert len(policy.get_optim_params()) == len(params) - len(list(policy.model.action_expert.parameters()))
    assert not any(p.requires_grad for p in policy.model.action_expert.parameters())
    policy.model.action_expert.requires_grad_(True)
    policy.train()
    loss, _ = policy.forward(training_batch(policy, seed=8))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.model.action_expert.parameters())
    layers = [m for m in policy.model.modules() if isinstance(m, MoEFeedForward)]
    assert layers
    before = [layer.expert_bias.clone() for layer in layers]
    policy.update()
    moved = [(layer.expert_bias - b) for layer, b in zip(layers, before, strict=True)]
    assert any(bool((m != 0).any()) for m in moved)
    for m in moved:
        assert torch.all((m == 0) | ((m.abs() - 0.01).abs() < 1e-6))
    policy.update()  # no step in between: counts are empty, the biases stay
    assert all(
        torch.equal(layer.expert_bias, b + m) for layer, b, m in zip(layers, before, moved, strict=True)
    )


def test_activation_checkpointing_gives_the_same_loss_and_gradients() -> None:
    batch = None
    results = []
    for checkpointing in (False, True):
        policy = make_tiny_policy(
            seed=9, activation_checkpointing=checkpointing, text_dropout=0.0, proprio_dropout=0.0
        )
        policy.train()
        batch = batch or training_batch(policy, seed=10)
        loss, _ = policy.forward(batch)
        loss.backward()
        grads = [p.grad.clone() for p in policy.model.action_expert.parameters() if p.grad is not None]
        results.append((loss.detach(), grads))
    (loss_plain, grads_plain), (loss_ckpt, grads_ckpt) = results
    assert torch.equal(loss_plain, loss_ckpt)
    assert len(grads_plain) == len(grads_ckpt) > 0
    for a, b in zip(grads_plain, grads_ckpt, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_training_batch_validation() -> None:
    policy = make_tiny_policy(seed=11)
    batch = training_batch(policy, seed=12)
    short = {k: (v[:, :-1] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with pytest.raises(ValueError, match="training window"):
        policy._training_windows(short)
    padded = dict(batch)
    padded[f"{ACTION}_is_pad"] = torch.zeros(1, 93, dtype=torch.bool)
    padded[f"{ACTION}_is_pad"][0, -1] = True
    with pytest.raises(ValueError, match="drop_n_last_frames"):
        policy._training_windows(padded)


def test_optimizer_and_scheduler_presets() -> None:
    policy = make_tiny_policy(seed=13)
    opt_cfg = policy.config.get_optimizer_preset()
    assert isinstance(opt_cfg, PraxisAdamWConfig) and opt_cfg.lr == 3e-6 and opt_cfg.betas == (0.9, 0.95)
    sched = policy.config.get_scheduler_preset()
    assert isinstance(sched, CosineDecayWithWarmupSchedulerConfig) and sched.decay_lr == 3e-7
    weight = torch.nn.Parameter(torch.ones(200_000, dtype=torch.bfloat16))
    weight.grad = torch.full_like(weight, 1e-3)
    optimizer = PraxisAdamWConfig(lr=2e-3, weight_decay=0.0).build([weight])
    assert isinstance(optimizer, AdamW)
    optimizer.step()
    # bfloat16 spacing at 1.0 is 2^-7; a 2e-3 step rounds to nothing without stochastic rounding
    mean_change = (weight.detach().float() - 1.0).mean().item()
    assert -2e-3 * 1.3 < mean_change < -2e-3 * 0.7, mean_change


def test_select_droid_episodes() -> None:
    import pandas as pd

    select_episodes = load_example("select_droid_episodes").select_episodes

    episodes = pd.DataFrame(
        {
            "episode_index": [0, 1, 2, 3, 4],
            "tasks": [[], ["Pick up the cup"], ["Open the drawer"], [" "], ["Push the box"]],
            "length": [300, 266, 101, 400, 102],
        }
    )
    assert select_episodes(episodes) == [1, 4]  # no instruction (0, 3) and too short (2) are out
    assert select_episodes(episodes, max_episodes=1) == [1]


def _antialias_model_config() -> dict:
    model_config = tiny_praxis_config()
    model_config["video"]["antialias"] = True
    return model_config


def test_training_windows_follow_the_checkpoint_antialias() -> None:
    """`video.antialias` sets the tile resize of the training windows like inference (the fit rule of `praxis-1-droid`)."""
    policy = make_tiny_policy(seed=16, model_config=_antialias_model_config())
    batch = training_batch(policy, seed=17, image=(3, 72, 128))
    window = policy._training_windows(batch)[0]
    assert torch.equal(window.frames, reference_window(policy, batch, 0).frames)
    plain = make_tiny_policy(seed=16)._training_windows(batch)[0]
    assert window.frames.shape == plain.frames.shape and not torch.equal(window.frames, plain.frames)


@pytest.mark.parametrize("antialias", [False, True])
def test_training_windows_are_the_same_through_the_processor_pipeline(antialias: bool) -> None:
    """A batch that went through the preprocessor (tiles, pose8) gives the same windows as the raw batch."""
    from lerobot.policies.factory import make_pre_post_processors

    policy = make_tiny_policy(seed=14, **({"model_config": _antialias_model_config()} if antialias else {}))
    batch = training_batch(policy, seed=15, image=(3, 72, 128))
    pre, _ = make_pre_post_processors(policy.config, None)
    processed = pre(dict(batch))
    assert processed[OBS_STATE].shape == (1, 93, 8) and processed[ACTION].shape == (1, 93, 8)
    assert processed[policy.config.exterior_image_key].shape == (1, 93, 3, TILE[0], TILE[1])
    raw, cooked = policy._training_windows(batch)[0], policy._training_windows(processed)[0]
    assert torch.equal(raw.frames, cooked.frames)
    assert np.array_equal(raw.actions_rel, cooked.actions_rel) and np.array_equal(
        raw.proprio_abs, cooked.proprio_abs
    )


def test_train_counter_travels_with_the_checkpoint(tmp_path, monkeypatch) -> None:
    """The number of training windows seen is a buffer: it seeds the draws and comes back after a save."""
    policy = make_tiny_policy(seed=20)
    batch = training_batch(policy, seed=21)
    assert int(policy.train_counter.calls) == 0
    policy.train()
    policy(batch)
    policy(batch)
    assert int(policy.train_counter.calls) == 2
    policy.save_pretrained(tmp_path)
    monkeypatch.setattr(type(policy), "_load_frozen_parts", lambda self: None)  # the fakes stand in
    loaded = type(policy).from_pretrained(tmp_path, config=policy.config)
    assert int(loaded.train_counter.calls) == 2
    # a checkpoint without the counter (converted from praxis) starts at zero
    state = dict(policy.model.state_dict())
    from safetensors.torch import save_file

    (tmp_path / "bare").mkdir()
    policy.config._save_pretrained(tmp_path / "bare")
    save_file(state, str(tmp_path / "bare" / "model.safetensors"))
    bare = type(policy).from_pretrained(tmp_path / "bare", config=policy.config)
    assert int(bare.train_counter.calls) == 0


def test_eval_forward_draws_without_advancing_the_counter() -> None:
    """A held-out evaluation (`eval()`) draws from the current count and leaves it where it was."""
    policy = make_tiny_policy(seed=22, activation_checkpointing=False)
    batch = training_batch(policy, seed=23)
    policy.train()
    policy(batch)
    assert int(policy.train_counter.calls) == 1
    policy.eval()
    with torch.no_grad():
        loss_a, _ = policy(batch)
        loss_b, _ = policy(batch)
    assert int(policy.train_counter.calls) == 1 and torch.equal(loss_a, loss_b)
    policy.train()
    policy(batch)
    assert int(policy.train_counter.calls) == 2


def test_forward_runs_with_the_praxis_numerics_whatever_the_host_sets() -> None:
    """The objective runs without the host's autocast (the router scores stay float32) and with
    the praxis matmul precision; the loss is the same as on PyTorch defaults."""
    policy = make_tiny_policy(seed=24, activation_checkpointing=False)
    policy.train()
    batch = training_batch(policy, seed=25)
    loss_plain, _ = policy(batch)
    policy.train_counter.calls.zero_()
    saved = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss_host, _ = policy(batch)
        assert torch.get_float32_matmul_precision() == "medium"  # the host setting comes back
    finally:
        torch.set_float32_matmul_precision(saved)
    assert torch.equal(loss_plain.detach(), loss_host.detach())


def test_local_flat_follows_the_fsdp_and_hsdp_placements() -> None:
    """`_local_flat` slices a DTensor sharded on dim 0 (FSDP) or sharded on one mesh dim and
    replicated on the other (HSDP), and rejects other placements."""
    import socket

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    from lerobot.policies.praxis.praxis_core.train.optim import _local_flat

    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        full = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        flat, start = _local_flat(full)
        assert start == 0 and torch.equal(flat, full.view(-1))
        mesh_1d = init_device_mesh("cpu", (1,))
        sharded = distribute_tensor(full, mesh_1d, [Shard(0)])
        flat, start = _local_flat(sharded)
        assert start == 0 and torch.equal(flat, full.view(-1))
        mesh_2d = init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))
        hsdp = distribute_tensor(full, mesh_2d, [Replicate(), Shard(0)])
        flat, start = _local_flat(hsdp)
        assert start == 0 and torch.equal(flat, full.view(-1))
        with pytest.raises(ValueError, match="dim-0 shard"):
            _local_flat(distribute_tensor(full, mesh_1d, [Replicate()]))
        with pytest.raises(ValueError, match="dim-0 shard"):
            _local_flat(distribute_tensor(full, mesh_1d, [Shard(1)]))
    finally:
        dist.destroy_process_group()
