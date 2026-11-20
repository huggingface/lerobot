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

import pickle

import numpy as np
import pytest
import torch

pytest.importorskip("pypaimon")
pytest.importorskip("datasets")

import pypaimon.multimodal as pm

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.paimon_backend import prepare_paimon_dataset


@pytest.fixture(params=["image", "video"])
def dataset_root(tmp_path, request):
    source = tmp_path / "source"
    dataset = LeRobotDataset.create(
        repo_id="test/paimon",
        root=source,
        fps=10,
        use_videos=request.param == "video",
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.images.front": {
                "dtype": request.param,
                "shape": (8, 10, 3),
                "names": ["height", "width", "channels"],
            },
        },
    )
    for episode in range(2):
        for frame in range(3):
            dataset.add_frame(
                {
                    "observation.state": np.array([episode, frame], dtype=np.float32),
                    "action": np.array([frame, episode], dtype=np.float32),
                    "observation.images.front": np.full((8, 10, 3), episode * 30 + frame, dtype=np.uint8),
                    "task": "pick",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    options = {"warehouse": str(tmp_path / "warehouse")}
    conn = pm.connect(options=options)
    conn.load_from_lerobot("frames", source)
    conn.create_lerobot_tag("frames", "train")
    root = prepare_paimon_dataset(
        tmp_path / "paimon", table="frames", catalog_options=options, tag_name="train"
    )
    return source, root


@pytest.mark.parametrize("episodes", [None, [1]])
@pytest.mark.parametrize("return_uint8", [False, True])
def test_matches_default_dataset(dataset_root, episodes, return_uint8):
    source, root = dataset_root
    kwargs = {
        "episodes": episodes,
        "return_uint8": return_uint8,
        "delta_timestamps": {"action": [0, 0.1], "observation.images.front": [-0.1, 0]},
    }
    expected = LeRobotDataset("test/paimon", root=source, **kwargs)
    actual = LeRobotDataset("test/paimon", root=root, **kwargs)
    assert len(actual) == len(expected)
    assert actual.meta.storage_format == "paimon"
    assert actual.absolute_to_relative_idx == expected.absolute_to_relative_idx
    for left, right in zip(
        actual.__getitems__([len(actual) - 1, 0, 0]),
        expected.__getitems__([len(expected) - 1, 0, 0]),
        strict=True,
    ):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key])
            else:
                assert left[key] == right[key]
    restored = pickle.loads(pickle.dumps(actual))
    torch.testing.assert_close(restored[0]["action"], actual[0]["action"])
    actual.reader.close()
    restored.reader.close()


def test_dataloader_training_step(dataset_root):
    _, root = dataset_root
    dataset = LeRobotDataset("test/paimon", root=root, return_uint8=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn"
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    count = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = (model(batch["observation.state"]) - batch["action"]).square().mean()
        loss.backward()
        optimizer.step()
        count += len(batch["action"])
    assert count == 6
    dataset.reader.close()


def test_factory_resolves_policy_windows(dataset_root):
    from types import SimpleNamespace

    from lerobot.configs.default import DatasetConfig
    from lerobot.datasets.factory import make_dataset

    _, root = dataset_root
    cfg = SimpleNamespace(
        dataset=DatasetConfig(repo_id="test/paimon", root=str(root), use_imagenet_stats=False),
        trainable_config=SimpleNamespace(
            observation_delta_indices=None, action_delta_indices=[0, 1], reward_delta_indices=None
        ),
        rename_map={},
        tolerance_s=1e-4,
    )
    dataset = make_dataset(cfg)
    assert dataset.meta.storage_format == "paimon"
    assert dataset[0]["action"].shape == (2, 2)
    dataset.reader.close()


def test_transforms_and_stale_metadata(dataset_root):
    import json

    source, root = dataset_root
    transform = lambda tensor: tensor.flip(-1)  # noqa: E731
    actual = LeRobotDataset("test/paimon", root=root, image_transforms=transform, return_uint8=True)
    expected = LeRobotDataset("test/paimon", root=source, image_transforms=transform, return_uint8=True)
    torch.testing.assert_close(actual[0]["observation.images.front"], expected[0]["observation.images.front"])
    actual.clear_image_transforms()
    expected.clear_image_transforms()
    torch.testing.assert_close(actual[0]["observation.images.front"], expected[0]["observation.images.front"])
    actual.reader.close()
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["fps"] *= 2
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="metadata changed"):
        LeRobotDataset("test/paimon", root=root)


def test_requires_tag_and_preserves_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="tag"):
        prepare_paimon_dataset(tmp_path / "new", table="frames", catalog_options={}, tag_name="")
    with pytest.raises(FileExistsError):
        prepare_paimon_dataset(tmp_path, table="frames", catalog_options={}, tag_name="train")
