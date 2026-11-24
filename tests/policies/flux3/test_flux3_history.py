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
"""FLUX.3 history conditioning served from the executor's record, in the training layout."""

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.configs.types import FeatureType
from lerobot.inference import SyncInferenceEngine
from lerobot.inference.history import policy_history
from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from lerobot.processor import make_inference_pipeline
from lerobot.utils.constants import OBS_ACTION, OBS_STATE
from tests.policies.flux3.helpers import TaskVideoVAE, rich_task_config, task_config

pytestmark = pytest.mark.dataset

CAMERA = "observation.images.top"


@pytest.fixture(autouse=True)
def task_vae(monkeypatch, fake_text_encoder):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, cfg: TaskVideoVAE())


def test_history_declaration_matches_the_training_windows():
    cfg = rich_task_config()
    history = policy_history(cfg)
    windows = {window.feature: window.offsets for window in history.observations}
    training = cfg.training_sampler.deltas.observation
    assert windows == {
        CAMERA: training[FeatureType.VISUAL][: cfg.n_obs_steps],
        OBS_STATE: training[FeatureType.STATE],
    }
    assert history.actions == training[FeatureType.ACTION]
    # The PEFT default conditions on the current frame and, for command deltas, the last command.
    history = policy_history(task_config())
    assert all(window.offsets == (0,) for window in history.observations) and history.actions == (-1,)


def training_layout(frames, executed, t, n_obs_steps):
    """The batch a dataset would serve at frame ``t``: past windows, commands and padding."""
    window = range(t + 1 - n_obs_steps, t + 1)
    batch = {"task": frames[t]["task"]}
    for key in (CAMERA, OBS_STATE):
        if n_obs_steps == 1:
            batch[key] = frames[t][key]
        else:
            batch[key] = torch.stack([frames[max(i, 0)][key] for i in window], 1)
            batch[f"{key}_is_pad"] = torch.tensor([[i < 0 for i in window]])
    commands = range(t - n_obs_steps, t)
    batch[OBS_ACTION] = torch.stack(
        [executed[i] if i >= 0 else torch.zeros_like(executed[0]) for i in commands], 1
    )
    batch[f"{OBS_ACTION}_is_pad"] = torch.tensor([[i < 0 for i in commands]])
    return batch


@pytest.mark.parametrize("rich", [False, True])
def test_executor_serves_the_training_layout(rich):
    """Every chunk the executor plays equals the policy's chunk on the hand-built training batch."""
    cfg = (rich_task_config if rich else task_config)(n_action_steps=4)
    policy = Flux3Policy(cfg)
    pre, post = make_flux3_pre_post_processors(cfg)
    generator = torch.Generator().manual_seed(0)
    ticks = 3 * cfg.n_action_steps  # the third chunk sees a full, unpadded window
    frames = [
        {
            CAMERA: torch.rand(1, 3, 64, 96, generator=generator),
            OBS_STATE: torch.rand(1, cfg.action_dim, generator=generator),
            "task": ["pick up box"],
        }
        for _ in range(ticks)
    ]
    executor = SyncInferenceEngine(policy, pre, post, observation_keys=[CAMERA, OBS_STATE])
    executor.reset()
    executed = [executor.step(dict(frame), float(t)) for t, frame in enumerate(frames)]

    pipeline = make_inference_pipeline(pre, policy, post)
    for start in range(0, ticks, cfg.n_action_steps):
        batch = training_layout(frames, executed, start, cfg.n_obs_steps)
        with torch.inference_mode():
            chunk = pipeline(batch)["action"]
        played = torch.stack(executed[start : start + cfg.n_action_steps], 1)
        torch.testing.assert_close(played, chunk[:, : cfg.n_action_steps])
