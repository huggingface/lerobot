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

"""Tests for the opt-in EMA shadow maintained by the training pipeline (--ema.enable=true)."""

import draccus
import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import EMAConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR

DUMMY_REPO_ID = "dummy/repo"
DUMMY_STATE_DIM = 6
DUMMY_ACTION_DIM = 6
IMAGE_SIZE = 32
N_EPISODES = 2
EPISODE_LENGTH = 12


def test_ema_config_defaults_match_the_reference():
    cfg = EMAConfig()
    assert not cfg.enable
    assert cfg.inv_gamma == 1.0
    assert cfg.power == 0.75
    assert cfg.update_after_step == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_decay": 0.5, "max_decay": 0.1},
        {"max_decay": 1.5},
        {"min_decay": -0.1},
        {"inv_gamma": 0.0},
        {"power": -1.0},
        {"update_after_step": -1},
        {"decay": 1.5},
        {"decay": -0.1},
        {"decay": 0.99, "min_decay": 0.5},
        {"decay": 0.99, "max_decay": 0.9},
    ],
)
def test_ema_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EMAConfig(**kwargs)


def test_ema_config_cli_parsing():
    cfg = draccus.parse(
        TrainPipelineConfig,
        None,
        args=[
            f"--dataset.repo_id={DUMMY_REPO_ID}",
            "--ema.enable=true",
            "--ema.power=0.8",
            "--ema.update_after_step=10",
        ],
    )
    assert cfg.ema.enable
    assert cfg.ema.power == 0.8
    assert cfg.ema.update_after_step == 10


def test_ema_config_cli_parsing_constant_decay():
    cfg = draccus.parse(
        TrainPipelineConfig,
        None,
        args=[
            f"--dataset.repo_id={DUMMY_REPO_ID}",
            "--ema.enable=true",
            "--ema.decay=0.99",
        ],
    )
    assert cfg.ema.enable
    assert cfg.ema.decay == 0.99


def test_ema_constant_decay_pins_the_schedule():
    """min_decay == max_decay clamps the warmup curve to a constant (how --ema.decay is implemented)."""
    pytest.importorskip("diffusers")
    from diffusers.training_utils import EMAModel

    model = torch.nn.Linear(4, 4)
    ema = EMAModel(
        model.parameters(), decay=0.99, min_decay=0.99, use_ema_warmup=True, inv_gamma=1.0, power=0.75
    )
    # The first update is a hard copy (decay 0); every one after uses the constant decay.
    for step in range(1, 6):
        ema.step(model.parameters())
        if step > 1:
            assert ema.cur_decay_value == 0.99


def test_ema_weights_context_swaps_and_restores():
    pytest.importorskip("diffusers")
    from diffusers.training_utils import EMAModel

    from lerobot.scripts.lerobot_train import _ema_weights

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    ema = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True, inv_gamma=1.0, power=0.75)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        ema.step(model.parameters())

    live = [p.detach().clone() for p in model.parameters()]
    with _ema_weights(ema, model):
        swapped = [p.detach().clone() for p in model.parameters()]
    restored = list(model.parameters())

    assert any(not torch.equal(a, b) for a, b in zip(live, swapped, strict=True))
    assert all(torch.equal(a, b.detach()) for a, b in zip(live, restored, strict=True))


def make_dummy_dataset(tmp_path):
    features = {
        "action": {"dtype": "float32", "shape": (DUMMY_ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (DUMMY_STATE_DIM,), "names": None},
        "observation.images.top": {
            "dtype": "image",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
    }
    root = tmp_path / "_dataset"
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=30, features=features, root=root)
    rng = np.random.default_rng(0)
    for ep_idx in range(N_EPISODES):
        for _ in range(EPISODE_LENGTH):
            dataset.add_frame(
                {
                    "action": rng.standard_normal(DUMMY_ACTION_DIM).astype(np.float32),
                    "observation.state": rng.standard_normal(DUMMY_STATE_DIM).astype(np.float32),
                    "observation.images.top": rng.integers(
                        0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                    ),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def make_train_config(root, output_dir, steps, ema_enable, ema_decay=None):
    from lerobot.configs.default import DatasetConfig
    from lerobot.policies.factory import make_policy_config

    policy_config = make_policy_config(
        "diffusion",
        device="cpu",
        push_to_hub=False,
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        drop_n_last_frames=0,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        num_inference_steps=2,
        pretrained_backbone_weights=None,
        use_group_norm=True,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(root)),
        policy=policy_config,
        output_dir=output_dir,
        steps=steps,
        batch_size=2,
        num_workers=0,
        seed=42,
        log_freq=0,
        env_eval_freq=0,
        save_freq=2,
        ema=EMAConfig(enable=ema_enable, decay=ema_decay),
    )
    cfg.optimizer = policy_config.get_optimizer_preset()
    cfg.scheduler = policy_config.get_scheduler_preset()
    # The config is built in-process, so skip the CLI-oriented validation.
    cfg.validate = lambda: None
    return cfg


def load_safetensors(path):
    from safetensors.torch import load_file

    return load_file(path)


def test_train_diffusion_with_ema_checkpoint_and_resume(tmp_path):
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")
    from lerobot.scripts.lerobot_train import EMA_STATE_FILENAME, train

    root = make_dummy_dataset(tmp_path)
    output_dir = tmp_path / "_output"

    cfg = make_train_config(root, output_dir, steps=4, ema_enable=True)
    train(cfg)

    checkpoint_dir = output_dir / "checkpoints" / "000004"
    ema_state_path = checkpoint_dir / TRAINING_STATE_DIR / EMA_STATE_FILENAME
    ema_model_dir = checkpoint_dir / f"{PRETRAINED_MODEL_DIR}_ema"

    # The shadow state is saved for resume and tracks every optimizer step.
    assert ema_state_path.exists()
    ema_state = torch.load(ema_state_path, weights_only=True)
    assert ema_state["optimization_step"] == 4

    # A directly loadable EMA model is saved next to the live one, with different weights.
    live_weights = load_safetensors(checkpoint_dir / PRETRAINED_MODEL_DIR / "model.safetensors")
    ema_weights = load_safetensors(ema_model_dir / "model.safetensors")
    assert set(live_weights) == set(ema_weights)
    assert any(not torch.equal(live_weights[k], ema_weights[k]) for k in live_weights)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(str(ema_model_dir))
    assert isinstance(policy, DiffusionPolicy)

    # Resuming picks the shadow up where it left off instead of restarting it.
    resume_cfg = make_train_config(root, output_dir, steps=6, ema_enable=True)
    resume_cfg.resume = True
    resume_cfg.checkpoint_path = checkpoint_dir
    train(resume_cfg)

    resumed_state = torch.load(
        output_dir / "checkpoints" / "000006" / TRAINING_STATE_DIR / EMA_STATE_FILENAME,
        weights_only=True,
    )
    assert resumed_state["optimization_step"] == 6


def test_train_with_constant_ema_decay(tmp_path):
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")
    from lerobot.scripts.lerobot_train import EMA_STATE_FILENAME, train

    root = make_dummy_dataset(tmp_path)
    output_dir = tmp_path / "_output"

    cfg = make_train_config(root, output_dir, steps=2, ema_enable=True, ema_decay=0.99)
    train(cfg)

    ema_state = torch.load(
        output_dir / "checkpoints" / "000002" / TRAINING_STATE_DIR / EMA_STATE_FILENAME,
        weights_only=True,
    )
    # The constant decay is implemented by pinning the schedule clamp to that value.
    assert ema_state["decay"] == 0.99
    assert ema_state["min_decay"] == 0.99
    assert ema_state["optimization_step"] == 2


def test_train_with_ema_and_gradient_accumulation(tmp_path):
    """The shadow tracks optimizer steps, not micro-batches, under gradient accumulation."""
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")
    from lerobot.scripts.lerobot_train import EMA_STATE_FILENAME, train

    root = make_dummy_dataset(tmp_path)
    output_dir = tmp_path / "_output"

    cfg = make_train_config(root, output_dir, steps=4, ema_enable=True)
    cfg.accelerator.gradient_accumulation.steps = 2
    train(cfg)

    ema_state = torch.load(
        output_dir / "checkpoints" / "000004" / TRAINING_STATE_DIR / EMA_STATE_FILENAME,
        weights_only=True,
    )
    # 4 micro-batches / 2 accumulation steps = 2 optimizer updates.
    assert ema_state["optimization_step"] == 2


def test_train_without_ema_writes_no_ema_files(tmp_path):
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")
    from lerobot.scripts.lerobot_train import EMA_STATE_FILENAME, train

    root = make_dummy_dataset(tmp_path)
    output_dir = tmp_path / "_output"

    cfg = make_train_config(root, output_dir, steps=2, ema_enable=False)
    train(cfg)

    checkpoint_dir = output_dir / "checkpoints" / "000002"
    assert (checkpoint_dir / PRETRAINED_MODEL_DIR / "model.safetensors").exists()
    assert not (checkpoint_dir / TRAINING_STATE_DIR / EMA_STATE_FILENAME).exists()
    assert not (checkpoint_dir / f"{PRETRAINED_MODEL_DIR}_ema").exists()
