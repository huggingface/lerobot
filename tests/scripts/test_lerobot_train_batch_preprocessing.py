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

import types

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import (  # noqa: E402
    _dataset_stats_for_processors,
    _preprocess_dataset_batch,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402


def test_preprocess_dataset_batch_normalizes_and_renames_before_processor():
    seen_batch = None

    def preprocessor(batch):
        nonlocal seen_batch
        seen_batch = batch
        return batch

    batch = {
        "image": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "actions": torch.ones(1, 2),
    }
    processed = _preprocess_dataset_batch(
        batch,
        camera_keys=["image"],
        rename_map={"image": f"{OBS_IMAGES}.camera1", "actions": ACTION},
        preprocessor=preprocessor,
    )

    assert seen_batch is processed
    assert set(processed) == {f"{OBS_IMAGES}.camera1", ACTION}
    assert processed[f"{OBS_IMAGES}.camera1"].dtype == torch.float32
    torch.testing.assert_close(processed[f"{OBS_IMAGES}.camera1"], torch.ones(1, 3, 2, 2))


OFFSET = np.array([1.0, -2.0, 0.5], dtype=np.float32)


def _relative_case(use_relative_actions=True, streaming=False):
    """A dataset whose state is held constant within each of its two episodes.

    A chunk anchors on the state at its first frame, so a constant state makes
    ``action - anchor`` exactly OFFSET -- unlike the absolute action distribution, which is
    dominated by the gap between the two episodes.
    """
    state = np.repeat([10.0, -30.0], 20)[:, None].repeat(3, axis=1).astype(np.float32)
    action = state + OFFSET
    dataset = types.SimpleNamespace(
        meta=types.SimpleNamespace(
            stats={ACTION: {"mean": action.mean(0), "std": action.std(0)}, OBS_STATE: {}},
            features={ACTION: {"dtype": "float32", "shape": (3,), "names": ["a", "b", "gripper"]}},
        ),
        hf_dataset={
            ACTION: action.tolist(),
            OBS_STATE: state.tolist(),
            "episode_index": np.repeat([0, 1], 20).tolist(),
        },
    )
    cfg = types.SimpleNamespace(
        trainable_config=types.SimpleNamespace(
            use_relative_actions=use_relative_actions,
            relative_exclude_joints=["gripper"],
            chunk_size=4,
        ),
        dataset=types.SimpleNamespace(streaming=streaming),
    )
    return cfg, dataset


def test_dataset_stats_untouched_when_relative_actions_are_off():
    cfg, dataset = _relative_case(use_relative_actions=False)
    assert _dataset_stats_for_processors(cfg, dataset) is dataset.meta.stats


def test_dataset_stats_describe_the_relative_action_distribution():
    cfg, dataset = _relative_case()
    stats = _dataset_stats_for_processors(cfg, dataset)
    assert stats[OBS_STATE] is dataset.meta.stats[OBS_STATE]  # only the action is relativized
    np.testing.assert_allclose(stats[ACTION]["mean"][:2], OFFSET[:2], atol=1e-4)
    np.testing.assert_allclose(stats[ACTION]["std"][:2], np.zeros(2), atol=1e-4)
    assert stats[ACTION]["std"][2] > 1.0  # the excluded dim keeps its absolute spread


def test_streaming_datasets_warn_and_keep_absolute_stats(caplog):
    cfg, dataset = _relative_case(streaming=True)
    with caplog.at_level("WARNING"):
        stats = _dataset_stats_for_processors(cfg, dataset)
    assert stats is dataset.meta.stats
    assert "streaming" in caplog.text
