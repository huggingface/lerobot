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

"""Tests for the opt-in EMA of the trainable policy weights maintained by the training pipeline (--ema.type=...)."""

import draccus
import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.train import TrainPipelineConfig, _migrate_legacy_ema_fields
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.optim.ema import EMA, ConstantEMAConfig, EMAConfig, PowerEMAConfig, WarmupEMAConfig
from lerobot.utils.constants import (
    EMA_STATE,
    PRETRAINED_MODEL_DIR,
    PRETRAINED_MODEL_EMA_DIR,
    TRAINING_STATE_DIR,
)

pytestmark = pytest.mark.dataset

DUMMY_REPO_ID = "dummy/repo"
DUMMY_STATE_DIM = 6
DUMMY_ACTION_DIM = 6
IMAGE_SIZE = 32
N_EPISODES = 2
EPISODE_LENGTH = 12


@pytest.fixture(autouse=True)
def _restore_cuda_backends():
    """train() unconditionally sets these global torch.backends flags; several tests in this file
    call train() for real, so restore them at teardown instead of leaking precision/determinism
    settings into whatever test runs next."""
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    yield
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def test_ema_averages_trainable_parameters_in_float32():
    """The first update copies, later ones blend into float32 averages, the swap is undone exactly."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(torch.bfloat16)
    model[0].requires_grad_(False)
    ema = EMA(model, ConstantEMAConfig(decay=0.999))
    assert list(ema.averages) == ["1.weight", "1.bias"]

    for value in (1.0, 2.0):
        with torch.no_grad():
            model[1].weight.fill_(value)
        ema.step(model)
    # 1.001 is not representable in bfloat16.
    torch.testing.assert_close(ema.averages["1.weight"], torch.full((4, 4), 1.001))

    live = [param.detach().clone() for param in model.parameters()]
    with ema.average_parameters(model):
        assert torch.equal(model[1].weight, ema.averages["1.weight"].to(torch.bfloat16))
        assert torch.equal(model[0].weight, live[0])
    assert all(torch.equal(a, b) for a, b in zip(live, model.parameters(), strict=True))


def test_ema_decay_schedules():
    """The diffusers warmup schedule (the default) and the EDM2 power function EMA."""
    assert WarmupEMAConfig().get_decay(11) == 1 - 11**-0.75
    assert WarmupEMAConfig().get_decay(10**9) == 0.9999
    # sigma_rel 0.1 is gamma 6.94 in EDM2.
    assert PowerEMAConfig(sigma_rel=0.1).get_decay(11) == pytest.approx((10 / 11) ** (6.9372 + 1), rel=1e-4)


LEGACY_EMA = {
    "enable": True,
    "decay": None,
    "update_after_step": 0,
    "inv_gamma": 1.0,
    "power": 0.8,
    "min_decay": 0.0,
    "max_decay": 0.9999,
    "use_for_eval": True,
}


@pytest.mark.parametrize(
    "legacy, expected",
    [
        ({**LEGACY_EMA, "enable": False}, None),
        (LEGACY_EMA, WarmupEMAConfig(power=0.8)),
        ({**LEGACY_EMA, "decay": 0.99}, ConstantEMAConfig(decay=0.99)),
    ],
)
def test_legacy_ema_config_is_migrated(legacy, expected):
    migrated = _migrate_legacy_ema_fields({"ema": legacy})
    assert draccus.decode(EMAConfig | None, migrated["ema"]) == expected


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


def make_train_config(root, output_dir, steps):
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
        save_freq=4,
        ema=WarmupEMAConfig(),
    )
    cfg.accelerator.gradient_accumulation.steps = 2
    cfg.optimizer = policy_config.get_optimizer_preset()
    cfg.scheduler = policy_config.get_scheduler_preset()
    # The config is built in-process, so skip the CLI-oriented validation.
    cfg.validate = lambda: None
    return cfg


def load_ema_state(checkpoint_dir):
    with safe_open(checkpoint_dir / TRAINING_STATE_DIR / EMA_STATE, framework="pt") as file:
        averages = {name: file.get_tensor(name) for name in file.keys()}  # noqa: SIM118
        return int(file.metadata()["num_updates"]), averages


@pytest.mark.gpu
def test_train_with_ema_checkpoint_and_resume(tmp_path):
    """The EMA counts optimizer updates, pretrained_model_ema holds the averages, resume is exact."""
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    pytest.importorskip("diffusers", reason="diffusers is required (install lerobot[diffusion])")
    from lerobot.scripts.lerobot_train import train

    root = make_dummy_dataset(tmp_path)
    train(make_train_config(root, tmp_path / "run", steps=4))
    checkpoint_dir = tmp_path / "run" / "checkpoints" / "000004"
    step, averages = load_ema_state(checkpoint_dir)
    # 4 micro-batches / 2 accumulation steps = 2 optimizer updates.
    assert step == 2
    live = load_file(checkpoint_dir / PRETRAINED_MODEL_DIR / "model.safetensors")
    ema = load_file(checkpoint_dir / PRETRAINED_MODEL_EMA_DIR / "model.safetensors")
    assert all(torch.equal(ema[name], average) for name, average in averages.items())
    assert all(torch.equal(ema[name], live[name]) for name in live.keys() - averages.keys())
    assert any(not torch.equal(ema[name], live[name]) for name in averages)

    resume_cfg = make_train_config(root, tmp_path / "run", steps=8)
    resume_cfg.resume = True
    resume_cfg.checkpoint_path = checkpoint_dir
    train(resume_cfg)
    train(make_train_config(root, tmp_path / "uninterrupted", steps=8))
    resumed_step, resumed = load_ema_state(tmp_path / "run" / "checkpoints" / "000008")
    uninterrupted_step, uninterrupted = load_ema_state(tmp_path / "uninterrupted" / "checkpoints" / "000008")
    assert resumed_step == uninterrupted_step == 4
    assert all(torch.equal(average, uninterrupted[name]) for name, average in resumed.items())
