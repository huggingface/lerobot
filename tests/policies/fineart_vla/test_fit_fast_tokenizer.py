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

"""Exercise FAST fitting's episode sampling with local parquet shards."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import lerobot.policies.fineart_vla.fit_fast_tokenizer as fitting  # noqa: E402

pytestmark = pytest.mark.dataset


class _IdentityTokenizer:
    """Stub only the codec; use real sampling, validation and artifact publication."""

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        return actions.copy()

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        return tokens.copy()

    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True)
        (path / "processor_config.json").write_text("{}")


@pytest.fixture
def fitter(monkeypatch: pytest.MonkeyPatch) -> Mock:
    fit = Mock(return_value=_IdentityTokenizer())
    monkeypatch.setattr(fitting, "_load_fast_fitter", Mock(return_value=SimpleNamespace(fit=fit)))
    monkeypatch.setenv("RANK", "0")
    return fit


def _write_dataset(root: Path, episode_lengths: dict[int, int]) -> Path:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_indices = []
    actions = []
    for episode, length in episode_lengths.items():
        episode_indices.extend([episode] * length)
        actions.extend([[episode * 100 + frame, episode * 100 + frame + 0.5] for frame in range(length)])
    # Each state is half its frame's action, so a chunk's reference state is known from its actions.
    states = (0.5 * np.asarray(actions)).tolist()
    pq.write_table(
        pa.table({"episode_index": episode_indices, "action": actions, "observation.state": states}),
        data_dir / "file-000.parquet",
    )
    return root


def _fit(root: Path, cache: Path, **kwargs) -> Path:
    kwargs.setdefault("normalization_mode", "IDENTITY")
    return Path(
        fitting.fit_fast_tokenizer(
            dataset_repo_id="test/local-actions",
            dataset_root=root,
            cache_dir=cache,
            chunk_size=4,
            n_samples=6,
            validation_samples=3,
            **kwargs,
        )
    )


@pytest.mark.parametrize("lengths", [(3, 8), (1, 3, 8), (4, 4), (4, 8)])
def test_fitting_fills_sample_budget_from_eligible_episodes(
    tmp_path: Path, fitter: Mock, lengths: tuple[int, ...]
) -> None:
    episode_lengths = {index * 3: length for index, length in enumerate(lengths)}
    root = _write_dataset(tmp_path / "dataset", episode_lengths)

    output = _fit(root, tmp_path / "cache")

    fitter.assert_called_once()
    fit_actions = fitter.call_args.args[0]
    assert fit_actions.shape == (6, 4, 2)
    assert fit_actions.dtype == np.float32
    with np.load(output / "reconstruction_examples.npz") as examples:
        validation_actions = examples["original"]
        assert validation_actions.shape == (3, 4, 2)
        all_actions = np.concatenate([fit_actions, validation_actions])
    # A chunk stays inside one episode and retains consecutive frame order.
    np.testing.assert_array_equal(np.diff(all_actions, axis=1), np.ones((9, 3, 2)))
    sampled_episodes = set((all_actions[:, :, 0] // 100).astype(int).flatten())
    assert sampled_episodes == {episode for episode, length in episode_lengths.items() if length >= 4}
    report = json.loads((output / "reconstruction_validation.json").read_text())
    assert report["num_validation_chunks"] == 3


def test_episode_selection_precedes_sample_allocation(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 8, 3: 2, 6: 8, 9: 8})

    _fit(root, tmp_path / "cache", episodes=[0, 3, 6], exclude_episodes=[0])

    actions = fitter.call_args.args[0]
    assert actions.shape == (6, 4, 2)
    assert np.all(actions[:, :, 0] // 100 == 6)


def test_sampling_is_reproducible(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 2, 3: 12, 6: 16})

    first = _fit(root, tmp_path / "cache-first", seed=123)
    second = _fit(root, tmp_path / "cache-second", seed=123)

    assert fitter.call_count == 2
    np.testing.assert_array_equal(fitter.call_args_list[0].args[0], fitter.call_args_list[1].args[0])
    with (
        np.load(first / "reconstruction_examples.npz") as first_examples,
        np.load(second / "reconstruction_examples.npz") as second_examples,
    ):
        np.testing.assert_array_equal(first_examples["original"], second_examples["original"])


def test_all_short_episodes_fail_before_loading_codec(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 1, 3: 3})

    with pytest.raises(RuntimeError, match=r"zero action chunks.*chunk_size=4"):
        _fit(root, tmp_path / "cache")

    fitter.assert_not_called()
    fitting._load_fast_fitter.assert_not_called()


def test_empty_selection_fails_before_loading_codec(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 8})

    with pytest.raises(RuntimeError, match="episode selection is empty"):
        _fit(root, tmp_path / "cache", exclude_episodes=[0])

    fitter.assert_not_called()


def test_fitting_applies_relative_actions_then_normalization(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 8, 3: 8})
    _fit(root, tmp_path / "raw-cache")
    raw = fitter.call_args.args[0]
    # A tiny nonzero std tells runtime's ``std + eps`` apart from dividing by ``std``.
    mean, std = np.array([1.0, -2.0]), np.array([4.0, 1e-7])

    _fit(
        root,
        tmp_path / "cache",
        normalization_mode="MEAN_STD",
        action_stats={"mean": mean, "std": std},
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        action_names=["shoulder", "gripper"],
    )

    # Training subtracts the chunk's first state from every non-excluded dimension, then normalizes.
    first_state = 0.5 * raw[:, :1] * np.array([1.0, 0.0])
    expected = (raw - first_state - mean) / (std + 1e-8)
    np.testing.assert_allclose(fitter.call_args.args[0], expected, rtol=1e-5)


def test_fitting_without_stats_keeps_actions_unnormalized(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 8, 3: 8})
    _fit(root, tmp_path / "raw-cache")
    raw = fitter.call_args.args[0]

    # Training's normalizer passes actions through when it has no statistics for them.
    _fit(root, tmp_path / "cache", normalization_mode="QUANTILES")

    np.testing.assert_array_equal(fitter.call_args.args[0], raw)


def test_cache_key_tracks_relative_action_settings(tmp_path: Path, fitter: Mock) -> None:
    root = _write_dataset(tmp_path / "dataset", {0: 8, 3: 8})
    settings = {"use_relative_actions": True, "action_names": ["shoulder", "gripper"]}

    first = _fit(root, tmp_path / "cache", **settings)
    reused = _fit(root, tmp_path / "cache", **settings)
    excluded = _fit(root, tmp_path / "cache", relative_exclude_joints=["gripper"], **settings)

    assert reused == first
    assert excluded != first
    assert fitter.call_count == 2
