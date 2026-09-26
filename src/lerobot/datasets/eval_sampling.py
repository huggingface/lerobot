"""Deterministic development frames spread across tasks, episodes and time."""

from collections import defaultdict
from collections.abc import Sequence


def _allocate(capacities: list[int], budget: int) -> list[int]:
    counts = [0] * len(capacities)
    remaining = min(budget, sum(capacities))
    while remaining:
        for i, capacity in enumerate(capacities):
            if counts[i] < capacity and remaining:
                counts[i] += 1
                remaining -= 1
    return counts


def balanced_eval_indices(tasks: Sequence[int], episodes: Sequence[int], max_samples: int) -> list[int]:
    """Allocate the cap across tasks, then episodes; sample temporal bin midpoints.

    Covers every task/episode when the budget permits and never repeats a frame.
    Short groups return unused capacity to other groups. Zero means all frames.
    """
    if len(tasks) != len(episodes) or max_samples < 0:
        raise ValueError("Matching task/episode lengths and a nonnegative cap are required")
    if max_samples == 0 or max_samples >= len(tasks):
        return list(range(len(tasks)))
    groups = defaultdict(lambda: defaultdict(list))
    for index, (task, episode) in enumerate(zip(tasks, episodes, strict=True)):
        groups[int(task)][int(episode)].append(index)
    task_groups = [groups[key] for key in sorted(groups)]
    quotas = _allocate([sum(map(len, group.values())) for group in task_groups], max_samples)
    selected = []
    for group, quota in zip(task_groups, quotas, strict=True):
        frames = [group[key] for key in sorted(group)]
        for values, count in zip(frames, _allocate(list(map(len, frames)), quota), strict=True):
            selected.extend(values[(2 * i + 1) * len(values) // (2 * count)] for i in range(count))
    return sorted(selected)
