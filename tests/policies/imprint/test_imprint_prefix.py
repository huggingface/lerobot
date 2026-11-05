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

"""Dry run: real dataset prefixes must equal select_action memory inputs."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

pytest.importorskip("datasets")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.imprint.configuration_imprint import ImprintConfig
from lerobot.policies.imprint.dataset_imprint import ImprintDataset
from lerobot.policies.imprint.history import PREFIX_FRAMES, PREFIX_LENGTH, prefix_indices
from lerobot.policies.imprint.memory import RecurrentMemory
from lerobot.policies.imprint.modeling_imprint import ImprintPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.imprint.test_imprint import CAM, tiny_config

pytestmark = pytest.mark.dataset


@pytest.fixture
def episode_root(tmp_path: Path) -> Path:
    root = tmp_path / "prefix_dataset"
    dataset = LeRobotDataset.create(
        repo_id="imprint_local/prefix",
        fps=10,
        root=root,
        features={
            ACTION: {"dtype": "float32", "shape": (4,), "names": None},
            OBS_STATE: {"dtype": "float32", "shape": (4,), "names": None},
            CAM: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
        },
    )
    for episode in range(2):
        for step in range(24):
            dataset.add_frame(
                {
                    ACTION: np.full(4, 10 + step + 100 * episode, dtype=np.float32),
                    OBS_STATE: np.full(4, step + 100 * episode, dtype=np.float32),
                    CAM: np.full((8, 8, 3), step * 10, dtype=np.uint8),
                    "task": "prefix",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def prefix_dataset(root: Path, config: ImprintConfig, episodes: list[int] | None = None) -> ImprintDataset:
    dataset = LeRobotDataset("imprint_local/prefix", root=root, episodes=episodes, video_backend="pyav")
    dataset = LeRobotDataset(
        "imprint_local/prefix",
        root=root,
        episodes=episodes,
        video_backend="pyav",
        delta_timestamps=resolve_delta_timestamps(config, dataset.meta),
    )
    return ImprintDataset(dataset, config)


@pytest.mark.parametrize("encoder", ["recurrent", "full_attention"])
@pytest.mark.parametrize("chunk_frames", [2, 4])
def test_training_prefix_matches_select_action(
    episode_root: Path, encoder: str, chunk_frames: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(29)
    config = tiny_config(
        encoder=encoder,
        history_length=16,
        history_downsample=2,
        chunk_frames=chunk_frames,
        n_action_steps=3,
    )
    dataset = prefix_dataset(episode_root, config, episodes=[1])
    policy = ImprintPolicy(config).eval()
    step = 0

    def teacher_chunk(obs, memory, valid):
        return (
            torch.arange(110 + step, 110 + step + config.chunk_size).float()[None, :, None].expand(1, -1, 4)
        )

    monkeypatch.setattr(policy, "_sample_actions", teacher_chunk)
    reads, positions = [], []
    if isinstance(policy.memory, RecurrentMemory):
        original_read = policy.memory.read
        original_events = policy.memory._events_from_chunks

        def record_events(visual, action, proprio, position):
            positions.append(position.clone())
            return original_events(visual, action, proprio, position)

        def record_read(state, visual, action, proprio, *args, **kwargs):
            reads.append(
                (
                    state.matrices.clone(),
                    state.chunk_count.clone(),
                    visual.clone(),
                    action.clone(),
                    proprio.clone(),
                )
            )
            return original_read(state, visual, action, proprio, *args, **kwargs)

        monkeypatch.setattr(policy.memory, "read", record_read)
        monkeypatch.setattr(policy.memory, "_events_from_chunks", record_events)
    else:
        policy.memory.register_forward_pre_hook(
            lambda module, args: reads.append(tuple(value.clone() for value in args[:4]))
        )
    with torch.no_grad():
        for step in range(24):
            raw = dataset.dataset[step]
            single = {CAM: raw[CAM], OBS_STATE: raw[OBS_STATE]}
            policy.select_action(single)
            rollout, _ = policy._memory_from_rollout(None)
            rollout_read = reads[-1] if reads else None
            # Mix a short and long prefix; batch padding must not move the boundary.
            batch = default_collate(dataset.__getitems__([0, step]))
            visual = policy._encode_visual(batch)
            causal = policy._causal_from_batch(batch, visual.shape[1], 2, torch.device("cpu"), torch.float32)
            training, _ = policy._encode_memory(visual, causal, batch[OBS_STATE], None, batch)
            torch.testing.assert_close(training[1:], rollout, atol=1e-5, rtol=1e-5)
            frames = prefix_indices(step, config.history_downsample)
            assert batch[PREFIX_FRAMES][1, : len(frames)].tolist() == frames
            if rollout_read is not None:
                for trained, deployed in zip(reads[-1], rollout_read, strict=True):
                    torch.testing.assert_close(trained[-1:], deployed, atol=1e-5, rtol=1e-5)
                if isinstance(policy.memory, RecurrentMemory):
                    assert int(rollout_read[1][0]) == len(frames) // chunk_frames
                    if len(frames) >= chunk_frames:
                        assert positions[-1].flatten().tolist() == list(range(len(frames) // chunk_frames))
            obs = policy._current_observation(visual, batch[OBS_STATE], batch)
            torch.testing.assert_close(
                obs[1:], policy._obs_token(policy._encode_visual(single)[:, 0], single[OBS_STATE])
            )
            if step == 23:
                assert len(frames) > 4
                assert causal[1, 0].eq(0).all()
                assert batch["action_is_pad"][1, len(frames) + 1 :].all()
        policy.reset()
        assert policy._episode_step == 0
        assert policy._provisional_step is None
        policy.select_action({CAM: dataset.dataset[0][CAM], OBS_STATE: dataset.dataset[0][OBS_STATE]})
        assert policy._hist_causal.eq(0).all()


def test_prefix_capacity_never_silently_truncates(episode_root: Path) -> None:
    with pytest.raises(ValueError, match="needs 13 prefix frames"):
        prefix_dataset(episode_root, tiny_config(history_downsample=2))


def test_stage1_queries_real_prefix_slots_at_absolute_episode_frames(episode_root: Path) -> None:
    config = tiny_config(training_stage="memory", history_length=16, history_downsample=2, max_query_steps=64)
    dataset = prefix_dataset(episode_root, config)
    batch = default_collate(dataset.__getitems__([0, 23]))
    policy = ImprintPolicy(config)
    steps = policy._query_steps(batch, 13, 2, torch.device("cpu"))
    assert (steps < batch[PREFIX_LENGTH][:, None]).all()
    batch["query_steps"] = torch.tensor([[0, 0], [0, 12]])
    batch["episode_progress"] = torch.tensor([0.0, 1.0])
    decoder_steps = []
    policy.decoder.register_forward_pre_hook(lambda module, args: decoder_steps.append(args[1]))
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert decoder_steps[0].tolist() == [[0, 0], [0, 23]]


def test_prefix_batch_decode_cost_and_worker_pickling(
    episode_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = prefix_dataset(episode_root, tiny_config(history_length=16, history_downsample=2))
    original = dataset.dataset.__getitems__
    decoded = []

    def record(indices):
        decoded.extend(indices)
        return original(indices)

    monkeypatch.setattr(dataset.dataset, "__getitems__", record)
    batch = default_collate(dataset.__getitems__([0, 22, 23]))
    assert batch[CAM].shape[1] == 13
    assert batch[PREFIX_LENGTH].tolist() == [1, 12, 13]
    assert decoded == [*range(0, 23, 2), 23]
    monkeypatch.undo()
    loader = DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")
    first = next(iter(loader))
    assert first[PREFIX_LENGTH].tolist() == [1, 2]
