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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("av")
pytest.importorskip("accelerate")

from lerobot.common.wandb_utils import WandBLogger  # noqa: E402
from lerobot.configs.default import DatasetConfig, WandBConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.scripts import lerobot_train  # noqa: E402


class _LossDataset(torch.utils.data.IterableDataset):
    num_frames = 4
    num_episodes = 1
    meta = SimpleNamespace(camera_keys=[], stats={}, has_language_columns=False)

    def __iter__(self):
        for value in (1.0, 3.0, 5.0, 7.0):
            yield {"value": torch.tensor(value)}


class _LoggingPolicy(torch.nn.Module):
    def __init__(self, metrics):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.metrics = metrics

    def forward(self, batch):
        loss = self.weight * batch["value"].mean()
        if not self.metrics:
            return loss, self.metrics
        metrics = {
            **self.metrics,
            "flow_loss": loss.item() / 4,
            "text_loss": loss.item() / 2,
            "fast_action_loss": loss.item() / 8,
        }
        if "loss" in metrics:
            metrics["loss"] = loss.item()
        return loss, metrics

    def get_optim_params(self):
        return self.parameters()


@pytest.mark.parametrize(
    "policy_metrics",
    [
        None,
        {},
        {"flow_loss": 0.25, "text_loss": 0.5, "fast_action_loss": 0.75},
        {
            "loss": -999.0,
            "grad_norm": -999.0,
            "lr": -999.0,
            "update_s": -999.0,
            "dataloading_s": -999.0,
            "preprocessing_s": -999.0,
            "step_s": -999.0,
            "samples_per_s": -999.0,
            "flow_loss": 0.25,
            "text_loss": 0.5,
            "fast_action_loss": 0.75,
        },
    ],
    ids=["no-metrics", "empty-metrics", "components", "collisions"],
)
def test_train_logs_tracker_averages_and_policy_components(monkeypatch, tmp_path, policy_metrics):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/logging", streaming=True),
        policy=ACTConfig(device="cpu", push_to_hub=False, optimizer_lr=0.0),
        output_dir=tmp_path / "training",
        batch_size=1,
        num_workers=0,
        steps=4,
        log_freq=2,
        env_eval_freq=0,
        save_checkpoint=False,
        wandb=WandBConfig(enable=True, project="test"),
    )
    policy = _LoggingPolicy(policy_metrics)
    # Exercise the real W&B key prefixing without starting a remote run.
    logger = object.__new__(WandBLogger)
    logger._wandb = Mock()
    monkeypatch.setattr(lerobot_train, "WandBLogger", lambda cfg: logger)
    monkeypatch.setattr(lerobot_train, "make_train_eval_datasets", lambda cfg: (_LossDataset(), None))
    monkeypatch.setattr(
        lerobot_train,
        "make_dataloaders",
        lambda cfg, dataset, *args: (torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size), None),
    )
    monkeypatch.setattr(lerobot_train, "make_policy", lambda **kwargs: policy)
    monkeypatch.setattr(
        lerobot_train, "make_pre_post_processors", lambda **kwargs: (torch.nn.Identity(), torch.nn.Identity())
    )
    # Restore the global backend flags that train() sets after each test.
    for backend, name in (
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cuda.matmul, "allow_tf32"),
    ):
        monkeypatch.setattr(backend, name, getattr(backend, name))

    lerobot_train.train(cfg)

    calls = logger._wandb.log.call_args_list
    assert len(calls) == 2
    for call, step, expected_loss in zip(calls, (2, 4), (2.0, 6.0), strict=True):
        assert call.kwargs["step"] == step
        logged = call.kwargs["data"]
        # The last-batch losses are 3 and 7; each two-step window averages to 2 and 6.
        assert logged["train/loss"] == pytest.approx(expected_loss)
        assert logged["train/grad_norm"] == pytest.approx(expected_loss)
        assert logged["train/lr"] == 0.0
        assert logged["train/steps"] == step
        assert logged["train/samples"] == step
        assert logged["train/episodes"] == step / 4
        assert logged["train/epochs"] == step / 4
        for key in ("update_s", "dataloading_s", "preprocessing_s", "step_s", "samples_per_s"):
            assert logged[f"train/{key}"] >= 0
        for key, divisor in (("flow_loss", 4), ("text_loss", 2), ("fast_action_loss", 8)):
            if policy_metrics:
                assert logged[f"train/{key}"] == pytest.approx(expected_loss / divisor)
            else:
                assert f"train/{key}" not in logged
