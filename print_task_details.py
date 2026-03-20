#!/usr/bin/env python
import argparse
import re

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _normalize_task_text(task_text: str) -> str:
    normalized = task_text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _build_language_to_suites_map() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    try:
        from libero.libero import benchmark
    except ImportError:
        return {}, {}

    bench = benchmark.get_benchmark_dict()
    exact_to_suites: dict[str, set[str]] = {}
    normalized_to_suites: dict[str, set[str]] = {}

    for suite_name, suite_ctor in bench.items():
        try:
            suite = suite_ctor()
        except Exception:
            # Some LIBERO installs expose benchmark keys that are not fully available
            # (e.g. missing task maps for a specific suite). Skip those suites.
            continue
        for task_id in range(len(suite.tasks)):
            task = suite.get_task(task_id)
            language = str(getattr(task, "language", "")).strip()
            if not language:
                continue

            exact_to_suites.setdefault(language, set()).add(str(suite_name))
            normalized_to_suites.setdefault(_normalize_task_text(language), set()).add(str(suite_name))

    return exact_to_suites, normalized_to_suites


def _infer_suite_for_task(
    language_task: str, exact_to_suites: dict[str, set[str]], normalized_to_suites: dict[str, set[str]]
) -> str:
    exact_matches = exact_to_suites.get(language_task, set())
    if exact_matches:
        return ",".join(sorted(exact_matches))

    normalized_matches = normalized_to_suites.get(_normalize_task_text(language_task), set())
    if normalized_matches:
        return ",".join(sorted(normalized_matches))

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=str,
        default="HuggingFaceVLA/libero",
        help="HF dataset repo id, e.g. HuggingFaceVLA/libero",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    ds = LeRobotDataset(
        args.repo,
        force_cache_sync=False,
        download_videos=not args.offline,
    )

    tasks_df = getattr(ds.meta, "tasks", None)
    if tasks_df is None or len(tasks_df) == 0:
        raise ValueError(f"No task metadata found for {args.repo}.")

    if "task_index" not in tasks_df.columns:
        raise ValueError(
            "Task metadata is missing 'task_index' column; cannot map indices to language commands."
        )

    task_names = tasks_df.index.astype(str) if hasattr(tasks_df, "index") else ["<unknown>"] * len(tasks_df)

    exact_to_suites, normalized_to_suites = _build_language_to_suites_map()

    rows = (
        tasks_df.assign(language_task=task_names)
        .assign(
            suite=lambda df: df["language_task"].astype(str).apply(
                lambda task: _infer_suite_for_task(task, exact_to_suites, normalized_to_suites)
            )
        )
        .reset_index(drop=True)
        [["task_index", "suite", "language_task"]]
        .drop_duplicates(subset=["task_index", "suite", "language_task"])
        .sort_values("task_index")
    )

    print("task_index\tsuite\tlanguage_task")
    for task_index, suite, language_task in rows.itertuples(index=False):
        print(f"{int(task_index)}\t{suite}\t{language_task}")


if __name__ == "__main__":
    main()