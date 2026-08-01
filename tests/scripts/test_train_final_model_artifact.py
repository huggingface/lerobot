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
"""`lerobot_train.train()` calls `WandBLogger.log_final_model` exactly once, at the very end of
training, if and only if `wandb.model_artifact_name` or `wandb.registered_model_name` is set (issue
#5's off switch). `WandBLogger` is mocked one layer up, the same way
`tests/scripts/test_train_dataset_artifact.py` mocks it: no real wandb SDK call happens here, and
the dataset/policy are fully local/synthetic (no HF Hub or torchvision-hub network call: the ACT
backbone is randomly initialized via `pretrained_backbone_weights=None`).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.scripts.lerobot_train as train_module  # noqa: E402
from lerobot.configs.default import (
    DatasetConfig,  # noqa: E402
    WandBConfig,  # noqa: E402
)
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402

_ACTION_DIM = 6
_IMAGE_SIZE = 32
_REPO_ID = "placeholder/unused"


def _build_local_dataset(root: Path) -> None:
    """A tiny, genuinely valid local LeRobot dataset `make_train_eval_datasets` can load offline."""
    features = {
        "action": {"dtype": "float32", "shape": (_ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (_ACTION_DIM,), "names": None},
        "observation.images.cam": {
            "dtype": "image",
            "shape": (_IMAGE_SIZE, _IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=_REPO_ID,
        fps=30,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=False,
        video_backend="pyav",
        metadata_buffer_size=1,
    )
    for frame_index in range(2):
        dataset.add_frame(
            {
                "action": np.full(_ACTION_DIM, frame_index, dtype=np.float32),
                "observation.state": np.full(_ACTION_DIM, frame_index, dtype=np.float32),
                "observation.images.cam": np.random.randint(
                    0, 255, size=(_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8
                ),
                "task": "task-0",
            }
        )
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def _build_cfg(tmp_path: Path, **wandb_kwargs) -> TrainPipelineConfig:
    dataset_root = tmp_path / "dataset"
    _build_local_dataset(dataset_root)

    policy_config = make_policy_config(
        "act",
        push_to_hub=False,
        # Deliberately not pinning `device`: other tests in this process may already have
        # initialized accelerate's process-global `AcceleratorState` on a different device, and a
        # conflicting `Accelerator(cpu=...)` here would raise. Let it match whatever device
        # `auto_select_torch_device` (used elsewhere in this run) already settled on.
        pretrained_backbone_weights=None,  # no torchvision-hub network call
    )
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=_REPO_ID, root=dataset_root),
        policy=policy_config,
        output_dir=tmp_path / "run",
        steps=1,
        save_freq=1,
        batch_size=1,
        num_workers=0,
        eval_steps=0,
        env_eval_freq=0,
        wandb=WandBConfig(enable=True, project="proj", **wandb_kwargs),
    )


def _run_train(monkeypatch, cfg: TrainPipelineConfig) -> MagicMock:
    # `cfg.validate()` (called inside `train()`) re-reads argv for `--policy.path`/`--config_path`;
    # give it an argv with none of those so `_resolve_pretrained_from_cli` is a no-op, matching a
    # fresh (non-resumed, non-`--policy.path`) run built directly via the constructor.
    monkeypatch.setattr(sys, "argv", ["pytest"])

    fake_logger = MagicMock()
    monkeypatch.setattr(train_module, "WandBLogger", lambda cfg: fake_logger)

    train_module.train(cfg)
    return fake_logger


def test_final_model_not_published_when_both_names_unset(monkeypatch, tmp_path):
    cfg = _build_cfg(tmp_path)

    fake_logger = _run_train(monkeypatch, cfg)

    fake_logger.log_final_model.assert_not_called()
    # Byte-for-byte unchanged: the periodic per-checkpoint path still fires.
    fake_logger.log_policy.assert_called_once()


def test_final_model_published_when_model_artifact_name_set(monkeypatch, tmp_path):
    cfg = _build_cfg(tmp_path, model_artifact_name="my-policy")

    fake_logger = _run_train(monkeypatch, cfg)

    fake_logger.log_final_model.assert_called_once()
    _, kwargs = fake_logger.log_final_model.call_args
    assert kwargs["step"] == 1
    assert kwargs["dataset_artifact"] is None
