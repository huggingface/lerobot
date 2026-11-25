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
"""A policy's training sampler reads the same samples on every dataset reader."""

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.sampling import EpisodeSampler, FrameSampler
from lerobot.configs.types import Deltas, FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_ACTION
from tests.fixtures.constants import DUMMY_REPO_ID

pytestmark = pytest.mark.dataset

LAPTOP, PHONE = "laptop", "phone"
# The fixture's keys are not policy keys: the policy reads them through the rename map.
RENAME = {
    "state": "observation.state",
    LAPTOP: "observation.images.laptop",
    PHONE: "observation.images.phone",
}
SAMPLERS = {
    # A history of one camera, of the state and of the executed commands, the future actions as
    # targets, and no sample anchored on the first frame or the last two.
    "frames": FrameSampler(
        Deltas(
            observation={
                FeatureType.STATE: (-1, 0),
                FeatureType.ACTION: (-2, -1),
                "observation.images.laptop": (-3, 0),
            },
            action=range(3),
        ),
        drop_first=1,
        drop_last=2,
    ),
    # Whole episodes from three starts, one before the episode, with a stride per feature.
    "episodes": EpisodeSampler(
        observation={FeatureType.VISUAL: 4, FeatureType.STATE: 2, FeatureType.ACTION: 3},
        action=1,
        starts=(-3, 0, 5),
    ),
}


def _spec(name: str, length: int, sample: int) -> tuple[int, dict[str, tuple[str, range | tuple]]]:
    """The anchor of a sample and its windows: dataset key -> (column, offsets), written out by hand."""
    if name == "frames":
        return 1 + sample, {
            "state": ("state", (-1, 0)),
            OBS_ACTION: (ACTION, (-2, -1)),
            LAPTOP: (LAPTOP, (-3, 0)),
            ACTION: (ACTION, (0, 1, 2)),
        }
    start = (-3, 0, 5)[sample]
    span = length - start
    return start, {
        LAPTOP: (LAPTOP, range(0, span, 4)),
        PHONE: (PHONE, range(0, span, 4)),
        "state": ("state", range(0, span, 2)),
        OBS_ACTION: (ACTION, range(0, span, 3)),
        ACTION: (ACTION, range(0, span)),
    }


def _expected(plain: LeRobotDataset, name: str, episode: int, sample: int) -> dict:
    """A sample built frame by frame from a dataset without windows."""
    start = int(plain.meta.episodes[episode]["dataset_from_index"])
    length = int(plain.meta.episodes[episode]["length"])
    anchor, windows = _spec(name, length, sample)
    own = plain[start + min(max(anchor, 0), length - 1)]
    item = {"episode_index": episode, "index": int(own["index"]), PHONE: own[PHONE]}
    for key, (column, offsets) in windows.items():
        positions = np.asarray(offsets) + anchor
        item[key] = torch.stack([plain[start + int(p)][column] for p in np.clip(positions, 0, length - 1)])
        item[f"{key}_is_pad"] = torch.from_numpy((positions < 0) | (positions >= length))
    return item


def _assert_sample(actual: dict, expected: dict) -> None:
    assert int(actual["episode_index"]) == expected["episode_index"]
    assert int(actual["index"]) == expected["index"]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert actual[key].shape == value.shape, key
            # Decoders may round video frames slightly differently.
            torch.testing.assert_close(actual[key], value, atol=2 / 255, rtol=0, msg=key)
    windowed = {key.removesuffix("_is_pad") for key in actual if key.endswith("_is_pad")}
    assert windowed == {key.removesuffix("_is_pad") for key in expected if key.endswith("_is_pad")}


@pytest.mark.parametrize("reader", ["map", "streaming", "lance"])
@pytest.mark.parametrize("name", list(SAMPLERS))
def test_every_reader_reads_the_sampler_s_frames(tmp_path, lerobot_dataset_factory, reader, name):
    lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=60, use_videos=True)
    plain = LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "ds", video_backend="pyav")
    sampler = SAMPLERS[name]
    counts = {
        ep: (length - 3 if name == "frames" else sum(start < length for start in (-3, 0, 5)))
        for ep, length in enumerate(plain.meta.episodes["length"])
    }
    keys = list(EpisodeAwareSampler(counts))

    if reader == "streaming":
        stream = StreamingLeRobotDataset(
            DUMMY_REPO_ID, root=tmp_path / "ds", sampler=sampler, rename_map=RENAME, shuffle=False
        )
        assert stream.num_samples == len(keys)
        # The stream visits every sample once, in its own order: match them by sample.
        items = sorted(
            stream, key=lambda item: (int(item["episode_index"]), int(item["index"]), len(item[ACTION]))
        )
        expected = sorted(
            (_expected(plain, name, key.episode, key.sample) for key in keys),
            key=lambda item: (item["episode_index"], item["index"], len(item[ACTION])),
        )
        assert len(items) == len(expected)
        for item, reference in zip(items, expected, strict=True):
            _assert_sample(item, reference)
        return

    root = tmp_path / "ds"
    if reader == "lance":
        pytest.importorskip("lerobot_lancedb", reason="lerobot-lancedb converts the test fixture")
        from lerobot_lancedb.convert import convert

        convert(tmp_path / "lance", root=root)
        root = tmp_path / "lance"
    dataset = LeRobotDataset(DUMMY_REPO_ID, root=root, sampler=sampler, rename_map=RENAME)
    assert dataset.sample_counts == counts and len(dataset) == len(keys)
    for position, key in enumerate(keys):
        reference = _expected(plain, name, key.episode, key.sample)
        _assert_sample(dataset[key], reference)
        # A position addresses the same sample, episodes in ascending order.
        _assert_sample(dataset[position], reference)


def test_the_training_loader_pads_episodes_to_the_longest_of_the_batch(tmp_path, lerobot_dataset_factory):
    lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=60, use_videos=False)
    dataset = LeRobotDataset(
        DUMMY_REPO_ID, root=tmp_path / "ds", sampler=SAMPLERS["episodes"], rename_map=RENAME
    )
    keys = list(EpisodeAwareSampler(dataset.sample_counts))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(keys), sampler=keys, collate_fn=lerobot_collate_fn, num_workers=1
    )
    batch = next(iter(loader))
    items = [dataset[key] for key in keys]
    for key in (ACTION, OBS_ACTION, "state", LAPTOP):
        longest = max(len(item[key]) for item in items)
        assert batch[key].shape[:2] == (len(items), longest)
        for row, item in enumerate(items):
            real = len(item[key])
            # Real rows as read; the padding repeats the last row and is flagged.
            assert torch.equal(batch[key][row, :real], item[key])
            assert torch.equal(batch[f"{key}_is_pad"][row, :real], item[f"{key}_is_pad"])
            assert batch[f"{key}_is_pad"][row, real:].all()
            assert torch.equal(batch[key][row, real:], item[key][-1:].expand_as(batch[key][row, real:]))


@pytest.mark.parametrize(
    "deltas, error",
    [
        (Deltas(observation={"observation.images.top": (-1, 0)}), "no dataset feature maps to"),
        (Deltas(observation={FeatureType.VISUAL: (-1, 0)}), "several frames per camera"),
    ],
)
def test_a_window_the_dataset_cannot_supply_fails_at_construction(
    tmp_path, lerobot_dataset_factory, deltas, error
):
    lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=8, use_videos=False, camera_features={}
    )
    with pytest.raises(ValueError, match=error):
        LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "ds", sampler=FrameSampler(deltas))
