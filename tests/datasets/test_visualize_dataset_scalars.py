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

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("datasets")
rr = pytest.importorskip("rerun")

from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.scripts.lerobot_dataset_viz import (  # noqa: E402
    build_blueprint_from_dataset,
    visualize_dataset,
)

pytestmark = [pytest.mark.dataset, pytest.mark.viz]


@pytest.fixture
def scalar_dataset(tmp_path: Path) -> LeRobotDataset:
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["joint", "gripper"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["joint", "gripper"]},
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "q_target": {"dtype": "float32", "shape": (1,), "names": None},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "embedding": {"dtype": "float32", "shape": (2,), "names": None},
        "label": {"dtype": "string", "shape": (1,), "names": None},
    }
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create("local/scalars", root=root, fps=30, features=features, use_videos=False)
    for i in range(2):
        dataset.add_frame(
            {
                "action": np.array([i, 0.5], dtype=np.float32),
                "observation.state": np.array([i, 0.25], dtype=np.float32),
                "state": np.array([100 + i], dtype=np.float32),
                "q_target": np.array([i + 0.25], dtype=np.float32),
                "intervention": np.array([bool(i)]),
                "embedding": np.array([1, 2], dtype=np.float32),
                "label": "demo",
                "task": "pick",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset("local/scalars", root=root)


def test_custom_scalar_views_are_in_the_blueprint(scalar_dataset: LeRobotDataset) -> None:
    blueprint = build_blueprint_from_dataset(scalar_dataset)
    grid = blueprint.root_container.contents[1]
    assert {str(view.origin): view.name for view in grid.contents} == {
        "action": "action",
        "state": "state",
        "custom/state": "state",
        "custom/q_target": "q_target",
        "custom/intervention": "intervention",
    }


def test_custom_scalar_values_are_logged_without_metadata_or_vectors(
    scalar_dataset: LeRobotDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep the real Rerun archetypes, but do not launch a viewer or write a recording.
    monkeypatch.setattr(rr, "init", Mock())
    monkeypatch.setattr(rr, "set_time", Mock())
    log = Mock()
    monkeypatch.setattr(rr, "log", log)

    visualize_dataset(scalar_dataset, episode_index=0, batch_size=1)

    values: dict[str, list[float]] = {}
    for call in log.call_args_list:
        key, entity = call.args
        if isinstance(entity, rr.Scalars):
            values.setdefault(key, []).extend(entity.scalars.as_arrow_array().to_pylist())
    assert set(values) == {"action", "state", "custom/state", "custom/q_target", "custom/intervention"}
    assert values["custom/q_target"] == [0.25, 1.25]
    assert values["custom/intervention"] == [0, 1]
    assert values["action"] == [0, 0.5, 1, 0.5]
    assert values["state"] == [0, 0.25, 1, 0.25]
    assert values["custom/state"] == [100, 101]
