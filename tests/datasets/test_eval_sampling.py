from collections import Counter

import pytest

from lerobot.datasets.eval_sampling import balanced_eval_indices


def test_one_task_covers_all_episodes_and_late_frames():
    tasks = [0] * 5000
    episodes = [ep for ep in range(5) for _ in range(1000)]
    selected = balanced_eval_indices(tasks, episodes, 250)
    assert Counter(episodes[i] for i in selected) == dict.fromkeys(range(5), 50)
    for ep in range(5):
        frames = [i % 1000 for i in selected if episodes[i] == ep]
        assert frames[0] < 20 and frames[-1] > 980
    assert selected == balanced_eval_indices(tasks, episodes, 250)


def test_short_groups_redistribute_without_duplicates_or_exceeding_cap():
    tasks = [0] * 2 + [1] * 12
    episodes = [0, 0] + [1] * 2 + [2] * 10
    selected = balanced_eval_indices(tasks, episodes, 9)
    assert len(selected) == len(set(selected)) == 9
    assert Counter(tasks[i] for i in selected) == {0: 2, 1: 7}
    assert Counter(episodes[i] for i in selected) == {0: 2, 1: 2, 2: 5}
    assert balanced_eval_indices(tasks, episodes, 1) == [1]
    assert balanced_eval_indices(tasks, episodes, 0) == list(range(14))
    assert balanced_eval_indices([], [], 10) == []


def test_bad_inputs():
    with pytest.raises(ValueError):
        balanced_eval_indices([0], [], 2)
    with pytest.raises(ValueError):
        balanced_eval_indices([], [], -1)
