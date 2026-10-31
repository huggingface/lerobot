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

"""Run the normal lerobot-train loop on a tiny local IMPRINT dataset.

Dry run. Two optimizer steps of stage 1, a memory handoff, then two joint
steps. No paper backbone and no robot success rate.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.imprint.configuration_imprint import ImprintConfig
from lerobot.policies.imprint.modeling_imprint import ImprintPolicy
from lerobot.policies.imprint.processor_imprint import make_imprint_pre_post_processors
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import ACTION, LAST_CHECKPOINT_LINK, OBS_IMAGES, OBS_STATE, PRETRAINED_MODEL_DIR

REPO_ID = "imprint_local/tiny"
IMAGE = 8
STATE_DIM = 4
ACTION_DIM = 4
EPISODES = 2
EPISODE_LENGTH = 8
CAM = f"{OBS_IMAGES}.cam"


def _policy_config(training_stage: str) -> ImprintConfig:
    return ImprintConfig(
        device="cpu",
        push_to_hub=False,
        training_stage=training_stage,
        history_length=4,
        chunk_size=4,
        n_action_steps=1,
        chunk_frames=2,
        memory_dim=32,
        num_latents=4,
        num_heads=4,
        num_cross_blocks=1,
        delta_num_layers=1,
        delta_num_heads=4,
        delta_head_dim=8,
        delta_ffn_inner=32,
        visual_dim=16,
        chunk_feature_dim=16,
        recent_feature_dim=16,
        lang_dim=8,
        tokenizer_hidden=16,
        align_dim=16,
        num_decoder_blocks=1,
        decoder_ffn_inner=32,
        num_query_samples=2,
        expert_layers=1,
        num_inference_steps=2,
    )


def _write_dataset(root) -> None:
    features = {
        ACTION: {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        CAM: {"dtype": "image", "shape": (IMAGE, IMAGE, 3), "names": ["height", "width", "channel"]},
    }
    dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=10, features=features, root=root)
    rng = np.random.default_rng(0)
    for episode in range(EPISODES):
        for _frame in range(EPISODE_LENGTH):
            dataset.add_frame(
                {
                    ACTION: rng.standard_normal(ACTION_DIM).astype(np.float32),
                    OBS_STATE: rng.standard_normal(STATE_DIM).astype(np.float32),
                    CAM: rng.integers(0, 255, size=(IMAGE, IMAGE, 3), dtype=np.uint8),
                    "task": f"task_{episode}",
                }
            )
        dataset.save_episode()
    dataset.finalize()


def _train_config(root, output_dir, policy: ImprintConfig, steps: int) -> TrainPipelineConfig:
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, root=str(root)),
        policy=policy,
        output_dir=output_dir,
        steps=steps,
        batch_size=2,
        num_workers=0,
        seed=0,
        log_freq=1,
        save_freq=steps,
        env_eval_freq=0,
        eval_steps=0,
    )
    config.validate()
    return config


def _pretrained_dir(output_dir):
    matches = [
        path
        for path in output_dir.glob(f"checkpoints/*/{PRETRAINED_MODEL_DIR}")
        if path.parent.name != LAST_CHECKPOINT_LINK
    ]
    assert len(matches) == 1, matches
    return matches[0]


def test_lerobot_train_stage1_then_joint(tmp_path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root)

    stage1_out = tmp_path / "stage1_run"
    train(_train_config(root, stage1_out, _policy_config("memory"), steps=2))
    stage1_dir = _pretrained_dir(stage1_out)
    stage1 = ImprintPolicy.from_pretrained(stage1_dir)
    assert stage1.config.training_stage == "memory"
    assert stage1.decoder is not None
    assert stage1.expert is None

    joint_config = copy.deepcopy(ImprintConfig.from_pretrained(stage1_dir))
    joint_config.training_stage = "joint"
    joint_config.pretrained_path = None
    joint = ImprintPolicy(joint_config)
    report = joint.load_memory_weights(stage1_dir)
    assert "memory.proj_chunk.weight" in report["loaded"]
    assert report["skipped_decoder"]
    transferred = joint.memory.proj_chunk.weight.detach().clone()
    joint_init = tmp_path / "joint_init"
    joint.save_pretrained(joint_init)
    preprocessor, postprocessor = make_imprint_pre_post_processors(joint.config)
    preprocessor.save_pretrained(joint_init)
    postprocessor.save_pretrained(joint_init)

    joint_config = ImprintConfig.from_pretrained(joint_init)
    joint_config.pretrained_path = joint_init
    joint_out = tmp_path / "joint_run"
    train(_train_config(root, joint_out, joint_config, steps=2))
    joint_dir = _pretrained_dir(joint_out)
    trained = ImprintPolicy.from_pretrained(joint_dir)
    assert trained.config.training_stage == "joint"
    assert trained.expert is not None
    assert trained.decoder is None
    assert not torch.equal(trained.memory.proj_chunk.weight, transferred)

    trained.reset()
    step = {
        CAM: torch.rand(1, 3, IMAGE, IMAGE),
        OBS_STATE: torch.randn(1, STATE_DIM),
    }
    action = trained.select_action(step)
    assert tuple(action.shape) == (1, ACTION_DIM)
    assert torch.isfinite(action).all()
    trained.reset()
    assert trained.finalized_chunk_counts().numel() == 0
