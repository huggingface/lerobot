import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.adapters.libero_safety import (
    LiberoSafetySampleAdapter,
    episode_safe_window,
    load_libero_safety_contract,
    task_split,
)
from lerobot.datasets.adapters.libero_safety_v21 import (
    LiberoSafetyV21Dataset,
    build_episode_safe_action_chunk,
)
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.act.configuration_act import ACTConfig


def test_public_contract_runtime_metadata():
    contract = load_libero_safety_contract()
    assert contract.codebase_version == "v2.1"
    assert (contract.total_episodes, contract.total_frames, contract.total_tasks, contract.fps) == (
        19664,
        3443735,
        15,
        20,
    )
    assert contract.features["observation.state"]["shape"] == [8]
    assert contract.features["actions"]["shape"] == [7]
    assert len(contract.tasks) == 15
    assert all(value > 0 for value in contract.stats["actions"]["std"])
    assert len(task_split(contract)["validation"]) == 2


def test_lazy_v21_episode_decode_and_chunk():
    dataset = LiberoSafetyV21Dataset(episodes=[0], chunk_size=4)
    sample = dataset[0]
    assert sample["observation.image"].shape == (3, 256, 256)
    assert sample["observation.wrist_image"].shape == (3, 256, 256)
    assert sample["observation.state"].shape == (8,)
    assert sample["action"].shape == (4, 7)
    assert sample["task"] == dataset.contract.tasks[0]


def test_key_task_mapping_and_episode_safe_padding():
    contract = load_libero_safety_contract()
    rows = [
        {"actions": torch.ones(7), "task_index": 0, "episode_index": 3},
        {"actions": torch.ones(7) * 2, "task_index": 0, "episode_index": 3},
        {"actions": torch.ones(7) * 9, "task_index": 1, "episode_index": 4},
    ]
    adapted = LiberoSafetySampleAdapter(rows, contract)
    assert adapted[0]["task"] == contract.tasks[0]
    assert "action" in adapted[0] and "actions" not in adapted[0]
    actions, padding = episode_safe_window(adapted, 0, 4)
    assert actions.shape == (4, 7)
    assert padding.tolist() == [False, False, True, True]
    torch.testing.assert_close(actions[2:], torch.zeros(2, 7))


def test_lazy_v21_dataset_is_sampler_compatible():
    """The real training loop drives offline (map-style) datasets through EpisodeAwareSampler,
    which needs num_frames/num_episodes/episodes/absolute_to_relative_idx (see lerobot_train.py)."""
    dataset = LiberoSafetyV21Dataset(episodes=[0], chunk_size=4)
    assert dataset.num_frames == len(dataset)
    assert dataset.num_episodes == 1
    assert dataset.episodes is None
    assert dataset.absolute_to_relative_idx is None
    from_index = dataset.meta.episodes["dataset_from_index"]
    to_index = dataset.meta.episodes["dataset_to_index"]
    assert from_index.tolist() == [0]
    assert to_index.tolist() == [len(dataset)]


class _FakeLiberoSafetyV21Dataset:
    """Stand-in for LiberoSafetyV21Dataset that mimics episode_rows without hitting the
    network, so the eval_split grouping logic in make_train_eval_datasets can be tested
    in isolation from the (large, remote) real dataset."""

    _ALL_ROWS = [{"episode_index": i, "tasks": [f"task-{i % 3}"], "length": 10} for i in range(9)]

    def __init__(self, episodes=None, chunk_size=16, revision="main"):
        self.chunk_size, self.revision = chunk_size, revision
        wanted = None if episodes is None else set(episodes)
        self.episode_rows = [
            row for row in self._ALL_ROWS if wanted is None or row["episode_index"] in wanted
        ]

    def __len__(self):
        return sum(row["length"] for row in self.episode_rows)


def test_make_train_eval_datasets_splits_libero_safety_per_task(monkeypatch):
    import lerobot.datasets.adapters.libero_safety_v21 as libero_safety_v21_module

    monkeypatch.setattr(libero_safety_v21_module, "LiberoSafetyV21Dataset", _FakeLiberoSafetyV21Dataset)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="LIBERO-Safety/libero_safety", eval_split=0.34),
        policy=ACTConfig(chunk_size=4, n_action_steps=4),
    )
    train_dataset, eval_dataset = make_train_eval_datasets(cfg)

    train_episodes = {row["episode_index"] for row in train_dataset.episode_rows}
    eval_episodes = {row["episode_index"] for row in eval_dataset.episode_rows}
    all_episodes = {row["episode_index"] for row in _FakeLiberoSafetyV21Dataset._ALL_ROWS}

    assert train_episodes.isdisjoint(eval_episodes)
    assert train_episodes | eval_episodes == all_episodes
    # 3 episodes per task; eval_split=0.34 -> ceil(3 * 0.34) = 2 held out per task, 1 left for train.
    assert len(eval_episodes) == 6
    assert len(train_episodes) == 3


def _rows(length, value=1.0):
    return [{"actions": [value] * 7} for _ in range(length)]


def test_episode_chunk_boundary_matrix():
    middle, middle_pad = build_episode_safe_action_chunk(_rows(10), 2, 4)
    assert middle.shape == (4, 7) and not middle_pad.any()

    last, last_pad = build_episode_safe_action_chunk(_rows(10), 9, 4)
    assert last_pad.tolist() == [False, True, True, True]
    torch.testing.assert_close(last[1:], torch.zeros(3, 7))

    short, short_pad = build_episode_safe_action_chunk(_rows(2), 0, 4)
    assert short_pad.tolist() == [False, False, True, True]

    exact, exact_pad = build_episode_safe_action_chunk(_rows(4), 0, 4)
    assert not exact_pad.any()

    first_episode, padding = build_episode_safe_action_chunk(_rows(1, 2.0), 0, 3)
    second_episode, _ = build_episode_safe_action_chunk(_rows(1, 9.0), 0, 3)
    assert padding.tolist() == [False, True, True]
    assert first_episode[0, 0] == 2 and first_episode[1:].sum() == 0
    assert second_episode[0, 0] == 9


def test_real_batch_collation_and_physical_target():
    from torch.utils.data import DataLoader

    from lerobot.policies.cig_vla.trajectory_geometry import TrajectoryGeometryTargetBuilder
    from lerobot.utils.collate import lerobot_collate_fn

    dataset = LiberoSafetyV21Dataset(episodes=[0], chunk_size=4)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=lerobot_collate_fn)))
    assert batch["observation.image"].shape == (2, 3, 256, 256)
    assert len(batch["task"]) == 2
    stats = dataset.meta.stats["action"]
    normalized = (batch["action"] - stats["mean"]) / (stats["std"] + 1e-8)
    target = TrajectoryGeometryTargetBuilder().build(
        normalized, batch["observation.state"], dataset.meta.stats, batch["action_is_pad"]
    )
    expected = (batch["action"][..., :3] * (~batch["action_is_pad"]).unsqueeze(-1)).sum(1)
    torch.testing.assert_close(target.translation_goal, expected, atol=1e-7, rtol=1e-5)
    assert torch.isfinite(target.translation_goal).all()
