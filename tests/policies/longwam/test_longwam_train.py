# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline native training-loop smoke test with tiny real transformer experts."""

import sys

import numpy as np
import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy
from lerobot.scripts import lerobot_train
from tests.policies.longwam.test_longwam_core import tiny_core
from tests.policies.longwam.test_longwam_policy import config, observation


@pytest.mark.parametrize("mode", ["idm", "cod"])
@pytest.mark.parametrize("streaming", [False, True])
def test_lerobot_train_entrypoint_saves_reloadable_policy(monkeypatch, tmp_path, mode, streaming):
    # No GPU, Hub upload, telemetry, or remote dataset/model download in this test.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr("datasets.config.HF_DATASETS_CACHE", tmp_path / "dataset-cache")

    def build(cfg, **kwargs):
        core = tiny_core(cfg)

        def encode(prompts):
            count = 1 if isinstance(prompts, str) else len(prompts)
            return torch.zeros(count, 4, 16), torch.ones(count, 4, dtype=torch.bool)

        core.encode_prompt = encode
        return core

    monkeypatch.setattr(LongWAMPolicy, "_build_core_model", staticmethod(build))
    cfg = config(load_text_encoder=False)
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
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        "local/longwam-smoke", fps=20, root=root, features=features, use_videos=False
    )
    for step in range(96):
        frame = {key: np.full((16, 16, 3), step, dtype=np.uint8) for key in cfg.camera_keys}
        frame.update(
            {
                "observation.state": np.array([step / 100, 1], dtype=np.float32),
                "action": np.array([step / 100, 0.5, 1], dtype=np.float32),
                "task": "pick up",
            }
        )
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()
    output = tmp_path / "training"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            "--policy.type=longwam",
            "--policy.device=cpu",
            "--policy.dtype=float32",
            "--policy.camera_size=[16,16]",
            "--policy.proprio_dim=2",
            "--policy.action_dim=3",
            "--policy.action_horizon=8",
            "--policy.n_action_steps=6",
            "--policy.load_text_encoder=false",
            f"--policy.denoising={mode}",
            f"--policy.imagine_sigma={0 if mode == 'cod' else 0.9}",
            "--policy.push_to_hub=false",
            "--dataset.repo_id=local/longwam-smoke",
            f"--dataset.root={root}",
            f"--dataset.streaming={str(streaming).lower()}",
            f"--output_dir={output}",
            "--steps=2",
            "--batch_size=2",
            "--num_workers=0",
            "--save_freq=2",
            "--log_freq=1",
            "--env_eval_freq=0",
            "--wandb.enable=false",
        ],
    )
    lerobot_train.main()
    checkpoints = list((output / "checkpoints").glob("[0-9]*/pretrained_model"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "policy_preprocessor.json").exists()
    assert (checkpoint / "policy_postprocessor.json").exists()
    policy = LongWAMPolicy.from_pretrained(checkpoint)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    try:
        result = post(policy.select_action(pre(observation(policy.config, 0))))
        assert result.shape == (1, 3) and torch.isfinite(result).all()
    finally:
        policy.close()

    # Exercise the public resume CLI, including optimizer and processor restore.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            f"--config_path={checkpoint / 'train_config.json'}",
            "--resume=true",
            "--steps=3",
        ],
    )
    lerobot_train.main()
    resumed = output / "checkpoints" / "000003" / "pretrained_model"
    assert (resumed / "model.safetensors").exists()
    restored = LongWAMPolicy.from_pretrained(resumed)
    try:
        assert any(
            not torch.equal(before, after)
            for before, after in zip(policy.parameters(), restored.parameters(), strict=True)
        )
    finally:
        restored.close()
