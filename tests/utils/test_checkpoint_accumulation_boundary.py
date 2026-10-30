"""A successful save/load must not silently lose an unfinished optimizer update."""

from types import SimpleNamespace

import pytest

from lerobot.common import train_utils
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.io_utils import write_json


@pytest.fixture
def cfg(tmp_path):
    config = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/boundary"))
    config.accelerator.gradient_accumulation.steps = 2
    config.checkpoint_path = tmp_path / "checkpoint"
    return config


@pytest.mark.parametrize("api", ["checkpoint", "training_state"])
def test_reject_partial_save_before_creating_artifacts(cfg, api):
    accelerator = SimpleNamespace(sync_gradients=False)
    with pytest.raises(ValueError, match="inside a gradient accumulation window"):
        if api == "checkpoint":
            train_utils.save_checkpoint(cfg.checkpoint_path, 3, cfg, None, None, accelerator=accelerator)
        else:
            train_utils.save_training_state(cfg.checkpoint_path, 3, cfg, accelerator=accelerator)
    assert not cfg.checkpoint_path.exists()


def test_observed_boundary_is_not_inferred_from_absolute_step(cfg):
    # A resumed run may deliberately change accumulation. The local completed
    # window can then end at an absolute step not divisible by its new length.
    train_utils._require_accumulation_boundary(7, cfg, SimpleNamespace(sync_gradients=True))


@pytest.mark.parametrize("boundary", [None, False, "true", 1])
def test_bad_resume_rejected_before_rng_changes(cfg, monkeypatch, boundary):
    state = cfg.checkpoint_path / "training_state"
    state.mkdir(parents=True)
    payload = {"step": 3, "grad_accum_steps": 2}
    if boundary is not None:
        payload["accumulation_boundary"] = boundary
    write_json(payload, state / "training_step.json")
    restored = []
    monkeypatch.setattr(train_utils, "load_rng_state", lambda path: restored.append(path))
    with pytest.raises(ValueError, match="accumulation"):
        train_utils.resume_before_prepare(cfg)
    assert restored == []


@pytest.mark.parametrize("step,boundary", [(4, None), (4, True), (7, True)])
def test_completed_or_legacy_aligned_resume(cfg, monkeypatch, step, boundary):
    state = cfg.checkpoint_path / "training_state"
    state.mkdir(parents=True)
    train_utils.save_training_metadata(step, state, cfg, accumulation_boundary=boundary)
    restored = []
    monkeypatch.setattr(train_utils, "load_rng_state", lambda path: restored.append(path))
    assert train_utils.resume_before_prepare(cfg) == step
    assert restored == [state]
