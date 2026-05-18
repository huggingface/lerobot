#!/usr/bin/env python

from __future__ import annotations

import os
from copy import deepcopy

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.filterwarnings(
    "ignore:In CPU autocast, but the target dtype is not supported:UserWarning"
)

from conftest import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    BATCH_SIZE,
    EXPECTED_ACTION_CHUNK_SHAPE,
    EXPECTED_SELECT_ACTION_SHAPE,
    N_ACTION_STEPS,
    STATE_DIM,
    make_config,
    make_inference_batch,
    make_train_batch,
    set_seed_all,
)

from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

PRETRAINED_REPO_ID = "ginwind/VLA-JEPA"
PRETRAINED_SUBFOLDER = "LIBERO"

# extended hub tests load the full converted safetensors checkpoints (~5 GB) and are
# skipped by default.  Set VLA_JEPA_EXTENDED=1 to opt in.
_VLA_JEPA_EXTENDED = os.environ.get("VLA_JEPA_EXTENDED", "0") != "0"
extended_test = pytest.mark.skipif(not _VLA_JEPA_EXTENDED, reason="Set VLA_JEPA_EXTENDED=1 to run hub tests")


# ---------------------------------------------------------------------------
# Core training / inference tests
# ---------------------------------------------------------------------------


def test_training_forward_pass(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.train()

    batch = make_train_batch()
    batch_before = deepcopy(batch)

    loss, logs = policy.forward(batch)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert set(logs) == {"action_loss", "wm_loss", "loss"}
    assert logs["action_loss"] > 0
    assert logs["wm_loss"] >= 0

    loss.backward()
    assert any(p.grad is not None for p in policy.model.action_model.parameters() if p.requires_grad)
    # Batch must not be mutated.
    assert set(batch) == set(batch_before)
    for key, value in batch.items():
        if isinstance(value, Tensor):
            assert torch.equal(value, batch_before[key])
        else:
            assert value == batch_before[key]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_training_forward_various_batch_sizes(patch_vla_jepa_external_models: None, batch_size: int) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.train()
    loss, logs = policy.forward(make_train_batch(batch_size=batch_size))
    assert torch.isfinite(loss) and loss > 0
    assert set(logs) == {"action_loss", "wm_loss", "loss"}


@pytest.mark.parametrize(
    "action_dim,state_dim,action_horizon",
    [
        (3, 4, 4),
        (7, 0, 16),
        (6, 8, 8),
    ],
)
def test_training_forward_various_dims(
    patch_vla_jepa_external_models: None,
    action_dim: int,
    state_dim: int,
    action_horizon: int,
) -> None:
    set_seed_all(42)
    config = make_config(action_dim=action_dim, state_dim=state_dim, action_horizon=action_horizon)
    policy = VLAJEPAPolicy(config)
    policy.train()
    batch = make_train_batch(action_dim=action_dim, state_dim=state_dim, action_horizon=action_horizon)
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss) and loss > 0


@torch.no_grad()
def test_action_generation_shape(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    batch = make_inference_batch()

    chunk = policy.predict_action_chunk(batch)
    assert tuple(chunk.shape) == EXPECTED_ACTION_CHUNK_SHAPE
    assert chunk.device.type == "cpu"
    assert torch.isfinite(chunk).all()

    a1 = policy.select_action(batch)
    a2 = policy.select_action(batch)
    assert tuple(a1.shape) == EXPECTED_SELECT_ACTION_SHAPE
    assert tuple(a2.shape) == EXPECTED_SELECT_ACTION_SHAPE
    assert torch.isfinite(a1).all() and torch.isfinite(a2).all()


@torch.no_grad()
@pytest.mark.parametrize("action_dim,state_dim", [(3, 4), (7, 0), (6, 8)])
def test_action_generation_various_dims(
    patch_vla_jepa_external_models: None, action_dim: int, state_dim: int
) -> None:
    set_seed_all(42)
    config = make_config(action_dim=action_dim, state_dim=state_dim)
    policy = VLAJEPAPolicy(config)
    policy.eval()
    batch = make_inference_batch(state_dim=state_dim)
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape[-1] == action_dim
    assert torch.isfinite(chunk).all()


@torch.no_grad()
def test_inference_reproducibility(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    batch = make_inference_batch()

    set_seed_all(123)
    actions_1 = policy.predict_action_chunk(batch)
    set_seed_all(123)
    actions_2 = policy.predict_action_chunk(batch)

    assert tuple(actions_1.shape) == EXPECTED_ACTION_CHUNK_SHAPE
    assert torch.allclose(actions_1, actions_2, atol=1e-6)


@torch.no_grad()
def test_predict_action_chunk_always_finite(patch_vla_jepa_external_models: None) -> None:
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    for seed in [0, 42, 123]:
        set_seed_all(seed)
        chunk = policy.predict_action_chunk(make_inference_batch())
        assert torch.isfinite(chunk).all(), f"non-finite actions with seed={seed}"


# ---------------------------------------------------------------------------
# Action queue behaviour
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_select_action_queue_drains_before_refill(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    batch = make_inference_batch()

    # First call fills the queue (n_action_steps items) and pops one.
    a1 = policy.select_action(batch)
    assert len(policy._queues[ACTION]) == N_ACTION_STEPS - 1

    # Second call pops from the existing queue without calling predict_action_chunk.
    a2 = policy.select_action(batch)
    assert tuple(a1.shape) == EXPECTED_SELECT_ACTION_SHAPE
    assert tuple(a2.shape) == EXPECTED_SELECT_ACTION_SHAPE


@torch.no_grad()
def test_reset_clears_action_queue(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    policy.select_action(make_inference_batch())
    assert len(policy._queues[ACTION]) > 0

    policy.reset()
    assert len(policy._queues[ACTION]) == 0


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def test_lerobot_to_native_training_format(patch_vla_jepa_external_models: None) -> None:
    import numpy as np
    from PIL import Image

    policy = VLAJEPAPolicy(make_config())
    examples = policy._lerobot_to_native(make_train_batch())

    assert len(examples) == BATCH_SIZE
    for ex in examples:
        assert set(ex) >= {"image", "video", "lang", "action", "state"}
        assert len(ex["image"]) == 1 and isinstance(ex["image"][0], Image.Image)
        assert ex["video"].ndim == 5 and ex["video"].dtype == np.uint8  # [V,T,H,W,C]
        assert ex["action"].shape == (ACTION_HORIZON, ACTION_DIM)
        assert ex["state"].shape == (1, STATE_DIM)


def test_lerobot_to_native_inference_omits_action(patch_vla_jepa_external_models: None) -> None:
    policy = VLAJEPAPolicy(make_config())
    for ex in policy._lerobot_to_native(make_inference_batch()):
        assert "action" not in ex
        assert "image" in ex and "video" in ex and "lang" in ex


def test_lerobot_to_native_missing_task_uses_default(patch_vla_jepa_external_models: None) -> None:
    policy = VLAJEPAPolicy(make_config())
    batch = make_inference_batch()
    del batch["task"]
    examples = policy._lerobot_to_native(batch)
    assert all(isinstance(ex["lang"], str) and len(ex["lang"]) > 0 for ex in examples)


def test_lerobot_to_native_string_task_broadcast(patch_vla_jepa_external_models: None) -> None:
    policy = VLAJEPAPolicy(make_config())
    batch = make_inference_batch()
    batch["task"] = "open the drawer"
    assert all(ex["lang"] == "open the drawer" for ex in policy._lerobot_to_native(batch))


def test_lerobot_to_native_no_state_omitted(patch_vla_jepa_external_models: None) -> None:
    from lerobot.utils.constants import OBS_STATE

    policy = VLAJEPAPolicy(make_config())
    batch = make_inference_batch()
    del batch[OBS_STATE]
    assert all("state" not in ex for ex in policy._lerobot_to_native(batch))


def test_native_to_lerobot_both_losses(patch_vla_jepa_external_models: None) -> None:
    policy = VLAJEPAPolicy(make_config())
    loss, logs = policy._native_to_lerobot({"action_loss": torch.tensor(0.5), "wm_loss": torch.tensor(0.1)})
    assert torch.isfinite(loss)
    assert set(logs) == {"action_loss", "wm_loss", "loss"}
    assert logs["action_loss"] == pytest.approx(0.5, abs=1e-5)
    assert logs["wm_loss"] == pytest.approx(0.1, abs=1e-5)


# ---------------------------------------------------------------------------
# Pretrained checkpoint
# Hub tests (opt-in: VLA_JEPA_EXTENDED=1)
# ---------------------------------------------------------------------------


def _make_hub_train_batch(policy: VLAJEPAPolicy, batch_size: int = 1) -> dict:
    """Build a training batch whose keys/shapes match a hub-loaded policy config."""
    cfg = policy.config
    batch: dict = {"task": ["pick up the cube"] * batch_size}
    for key, feat in cfg.image_features.items():
        h, w = feat.shape[-2], feat.shape[-1]
        batch[key] = torch.rand(batch_size, cfg.num_video_frames, 3, h, w)
    if cfg.robot_state_feature is not None:
        batch["observation.state"] = torch.randn(batch_size, 1, cfg.robot_state_feature.shape[0])
    batch[ACTION] = torch.randn(batch_size, cfg.chunk_size, cfg.action_dim)
    return batch


def _make_hub_inference_batch(policy: VLAJEPAPolicy, batch_size: int = 1) -> dict:
    """Build an inference batch whose keys/shapes match a hub-loaded policy config."""
    cfg = policy.config
    batch: dict = {"task": ["pick up the cube"] * batch_size}
    for key, feat in cfg.image_features.items():
        h, w = feat.shape[-2], feat.shape[-1]
        batch[key] = torch.rand(batch_size, 3, h, w)
    if cfg.robot_state_feature is not None:
        batch["observation.state"] = torch.randn(batch_size, cfg.robot_state_feature.shape[0])
    return batch


_CP_ROOT = "lerobot"  # TODO: upload converted checkpoints

# Each tuple: (repo_id, enable_world_model)
_HUB_VARIANTS = [
    (f"{_CP_ROOT}/VLA-JEPA-LIBERO", True),
    (f"{_CP_ROOT}/VLA-JEPA-Pretrain", True),
    (f"{_CP_ROOT}/VLA-JEPA-SimplerEnv", False),
]


@extended_test
@pytest.mark.parametrize("repo_id,enable_world_model", _HUB_VARIANTS)
def test_hub_checkpoint_loads(repo_id: str, enable_world_model: bool) -> None:
    """Policy loads from the converted safetensors checkpoint on the Hub."""
    policy = VLAJEPAPolicy.from_pretrained(repo_id)
    assert policy.config.enable_world_model == enable_world_model
    assert sum(p.numel() for p in policy.parameters()) > 0


@extended_test
@pytest.mark.parametrize("repo_id,enable_world_model", _HUB_VARIANTS)
def test_hub_checkpoint_forward_pass(repo_id: str, enable_world_model: bool) -> None:
    """Policy loaded from hub produces finite losses with a correctly-shaped batch."""
    policy = VLAJEPAPolicy.from_pretrained(repo_id)
    policy.train()

    batch = _make_hub_train_batch(policy)
    loss, logs = policy.forward(batch)
    assert torch.isfinite(loss)
    assert "action_loss" in logs
    if enable_world_model:
        assert "wm_loss" in logs


@extended_test
def test_hub_freeze_qwen_disables_world_model() -> None:
    """freeze_qwen=True (via cli_overrides) freezes qwen and disables the world model."""
    policy = VLAJEPAPolicy.from_pretrained(f"{_CP_ROOT}/VLA-JEPA-LIBERO", cli_overrides=["freeze_qwen=true"])
    assert not policy.config.enable_world_model
    assert policy.model.video_predictor is None
    qwen_params = list(policy.model.qwen.parameters())
    assert all(not p.requires_grad for p in qwen_params)
    assert any(p.requires_grad for p in policy.model.action_model.parameters())


@extended_test
def test_hub_disable_world_model_loads_simpler_env() -> None:
    """SimplerEnv checkpoint (world model disabled) loads cleanly and runs inference."""
    policy = VLAJEPAPolicy.from_pretrained(f"{_CP_ROOT}/VLA-JEPA-SimplerEnv")
    assert not policy.config.enable_world_model
    assert policy.model.video_predictor is None
    assert policy.model.video_encoder is None


@extended_test
def test_hub_libero_inference_shape() -> None:
    """select_action returns the expected shape using the LIBERO hub checkpoint."""
    policy = VLAJEPAPolicy.from_pretrained(f"{_CP_ROOT}/VLA-JEPA-LIBERO")
    policy.eval()
    batch = _make_hub_inference_batch(policy)
    action = policy.select_action(batch)
    assert action.shape[-1] == policy.config.action_dim
