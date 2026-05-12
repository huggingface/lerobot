#!/usr/bin/env python
"""Build a LIBERO task/object report with representative dataset frames.

The LIBERO task metadata tells us object identities and task language, but it
does not provide object masks or boxes. This script combines that metadata with
successful demonstration frames from a LeRobot dataset so a human can visually
map object names to the rendered objects.

Examples:

```bash
uv run python examples/tree_search/libero_task_object_browser.py \
    --suite libero_object \
    --output_dir outputs/tree_search/libero_object_browser
```

With frames from a LeRobot dataset:

```bash
uv run python examples/tree_search/libero_task_object_browser.py \
    --suite libero_object \
    --dataset_repo_id <dataset-repo-id> \
    --output_dir outputs/tree_search/libero_object_browser \
    --episodes_per_task 2 \
    --camera_keys all
```
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import html
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.image_writer import write_image


@dataclass
class LiberoObject:
    instance: str
    category: str
    label: str
    is_object_of_interest: bool = False


@dataclass
class FrameRecord:
    episode_index: int
    dataset_index: int
    frame_index: int | None
    task: str
    camera_key: str
    image_path: str
    selection_reason: str
    score_key: str | None = None
    score_value: float | bool | None = None


@dataclass
class TaskRecord:
    suite: str
    task_id: int
    name: str
    language: str
    bddl_file: str
    bddl_path: str
    objects: list[LiberoObject]
    objects_of_interest: list[str]
    goal: str
    frames: list[FrameRecord] = field(default_factory=list)
    dataset_episode_count: int = 0


def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def _strip_lisp_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _compact_lisp(text: str) -> str:
    text = _strip_lisp_comments(text)
    return re.sub(r"\s+", " ", text).strip()


def _find_lisp_section(text: str, section_name: str) -> str:
    text = _strip_lisp_comments(text)
    match = re.search(r"\(:" + re.escape(section_name) + r"\b", text)
    if match is None:
        return ""

    depth = 0
    for idx in range(match.start(), len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[match.start() : idx + 1]
    return text[match.start() :]


def _parse_objects(section: str) -> list[LiberoObject]:
    objects: list[LiberoObject] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("(:objects") or line == ")":
            continue
        if " - " not in line:
            continue
        instances_raw, category = line.split(" - ", 1)
        category = category.strip().strip(")")
        for instance in instances_raw.split():
            instance = instance.strip()
            if not instance or instance.startswith("("):
                continue
            objects.append(
                LiberoObject(
                    instance=instance,
                    category=category,
                    label=category.replace("_", " "),
                )
            )
    return objects


def _parse_object_interest(section: str) -> list[str]:
    if not section:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", section)
    return [token for token in tokens if token != "obj_of_interest"]


def _parse_task_ids(value: str | None) -> set[int] | None:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    ids: set[int] = set()
    for token in re.split(r"[, ]+", value.strip()):
        if token:
            ids.add(int(token))
    return ids


def _load_libero_tasks(suite_name: str, task_ids: set[int] | None) -> list[TaskRecord]:
    try:
        from libero.libero import benchmark, get_libero_path
    except ImportError as exc:
        raise SystemExit(
            "Could not import LIBERO. Install the LIBERO extra/dependencies before running this script."
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict))
        raise SystemExit(f"Unknown LIBERO suite '{suite_name}'. Available suites: {available}")

    suite = benchmark_dict[suite_name]()
    n_tasks = getattr(suite, "n_tasks", len(getattr(suite, "tasks", [])))
    bddl_root = Path(get_libero_path("bddl_files"))

    records: list[TaskRecord] = []
    for task_id in range(n_tasks):
        if task_ids is not None and task_id not in task_ids:
            continue

        task = suite.get_task(task_id)
        bddl_path = bddl_root / task.problem_folder / task.bddl_file
        bddl_text = bddl_path.read_text()

        objects = _parse_objects(_find_lisp_section(bddl_text, "objects"))
        objects_of_interest = _parse_object_interest(_find_lisp_section(bddl_text, "obj_of_interest"))
        interest_set = set(objects_of_interest)
        for obj in objects:
            obj.is_object_of_interest = obj.instance in interest_set

        records.append(
            TaskRecord(
                suite=suite_name,
                task_id=task_id,
                name=str(task.name),
                language=str(task.language),
                bddl_file=str(task.bddl_file),
                bddl_path=str(bddl_path),
                objects=objects,
                objects_of_interest=objects_of_interest,
                goal=_compact_lisp(_find_lisp_section(bddl_text, "goal")),
            )
        )
    return records


def _coerce_scalar(value: Any) -> float | bool | int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return None
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return None
    if isinstance(value, list | tuple):
        if len(value) == 1:
            return _coerce_scalar(value[0])
        return None
    if hasattr(value, "item"):
        with contextlib.suppress(Exception):
            return value.item()
    if isinstance(value, bool | int | float):
        return value
    return None


def _episode_tasks(dataset: Any, episode: dict[str, Any]) -> list[str]:
    tasks = episode.get("tasks")
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, Iterable):
        values = [str(task) for task in tasks]
        if values:
            return values

    from_idx = int(episode["dataset_from_index"])
    raw_item = dataset.get_raw_item(from_idx)
    task_index = _coerce_scalar(raw_item.get("task_index"))
    if task_index is None:
        return []
    return [str(dataset.meta.tasks.iloc[int(task_index)].name)]


def _build_episode_index(dataset: Any) -> dict[str, list[int]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode_index in range(dataset.meta.total_episodes):
        episode = dataset.meta.episodes[episode_index]
        for task in _episode_tasks(dataset, episode):
            by_task[_normalize_text(task)].append(episode_index)
    return by_task


def _read_raw_scalar(dataset: Any, dataset_index: int, key: str) -> float | bool | int | None:
    try:
        raw = dataset.select_columns([key])[dataset_index]
    except Exception:
        raw = dataset.get_raw_item(dataset_index)
    return _coerce_scalar(raw.get(key))


def _select_success_like_frame(
    dataset: Any,
    episode: dict[str, Any],
    *,
    prefer_reward_frame: bool,
) -> tuple[int, str, str | None, float | bool | None]:
    from_idx = int(episode["dataset_from_index"])
    to_idx = int(episode["dataset_to_index"])
    final_idx = max(from_idx, to_idx - 1)

    if prefer_reward_frame:
        for key in ("next.success", "success"):
            if key not in dataset.features:
                continue
            for idx in range(final_idx, from_idx - 1, -1):
                value = _read_raw_scalar(dataset, idx, key)
                if bool(value):
                    return idx, f"last_true_{key}", key, bool(value)

        if "next.reward" in dataset.features:
            best_idx = final_idx
            best_reward: float | None = None
            for idx in range(final_idx, from_idx - 1, -1):
                value = _read_raw_scalar(dataset, idx, "next.reward")
                if value is None:
                    continue
                reward = float(value)
                if best_reward is None or reward > best_reward:
                    best_idx = idx
                    best_reward = reward
                if reward > 0:
                    return idx, "last_positive_next.reward", "next.reward", reward
            if best_reward is not None:
                return best_idx, "highest_next.reward", "next.reward", best_reward

    return final_idx, "final_frame_assumed_successful_demo", None, None


def _to_image_payload(value: Any) -> np.ndarray | Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        return value
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _resolve_camera_keys(dataset: Any, requested: str) -> list[str]:
    available = list(dataset.meta.camera_keys)
    if requested.strip().lower() == "all":
        return available
    keys = [key.strip() for key in requested.split(",") if key.strip()]
    missing = [key for key in keys if key not in available]
    if missing:
        raise SystemExit(f"Unknown camera key(s): {missing}. Available camera keys: {available}")
    return keys


def _attach_dataset_frames(
    records: list[TaskRecord],
    *,
    dataset_repo_id: str,
    dataset_root: Path | None,
    revision: str | None,
    video_backend: str | None,
    download_videos: bool,
    camera_keys_arg: str,
    episodes_per_task: int,
    prefer_reward_frame: bool,
    output_dir: Path,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        revision=revision,
        video_backend=video_backend,
        download_videos=download_videos,
    )

    camera_keys = _resolve_camera_keys(dataset, camera_keys_arg)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    episodes_by_task = _build_episode_index(dataset)

    for record in records:
        matching_episodes = episodes_by_task.get(_normalize_text(record.language), [])
        record.dataset_episode_count = len(matching_episodes)
        for episode_index in matching_episodes[:episodes_per_task]:
            episode = dataset.meta.episodes[episode_index]
            dataset_index, reason, score_key, score_value = _select_success_like_frame(
                dataset,
                episode,
                prefer_reward_frame=prefer_reward_frame,
            )
            item = dataset[dataset_index]
            frame_index = _coerce_scalar(item.get("frame_index"))
            task = str(item.get("task", record.language))
            for camera_key in camera_keys:
                if camera_key not in item:
                    continue
                image_name = (
                    f"task_{record.task_id:03d}_ep_{episode_index:06d}_idx_{dataset_index:08d}_"
                    f"{_safe_name(camera_key)}.png"
                )
                image_path = images_dir / image_name
                write_image(_to_image_payload(item[camera_key]), image_path)
                record.frames.append(
                    FrameRecord(
                        episode_index=episode_index,
                        dataset_index=dataset_index,
                        frame_index=int(frame_index) if frame_index is not None else None,
                        task=task,
                        camera_key=camera_key,
                        image_path=image_path.relative_to(output_dir).as_posix(),
                        selection_reason=reason,
                        score_key=score_key,
                        score_value=score_value,
                    )
                )


def _write_json(records: Sequence[TaskRecord], output_dir: Path) -> None:
    payload = [asdict(record) for record in records]
    (output_dir / "task_objects.json").write_text(json.dumps(payload, indent=2))


def _write_csv(records: Sequence[TaskRecord], output_dir: Path) -> None:
    with (output_dir / "task_objects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite",
                "task_id",
                "language",
                "object_instance",
                "object_category",
                "object_label",
                "is_object_of_interest",
                "goal",
                "bddl_file",
            ],
        )
        writer.writeheader()
        for record in records:
            for obj in record.objects:
                writer.writerow(
                    {
                        "suite": record.suite,
                        "task_id": record.task_id,
                        "language": record.language,
                        "object_instance": obj.instance,
                        "object_category": obj.category,
                        "object_label": obj.label,
                        "is_object_of_interest": obj.is_object_of_interest,
                        "goal": record.goal,
                        "bddl_file": record.bddl_file,
                    }
                )


def _render_object_rows(objects: Sequence[LiberoObject]) -> str:
    rows = []
    for obj in objects:
        marker = "yes" if obj.is_object_of_interest else ""
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(obj.instance)}</code></td>"
            f"<td>{html.escape(obj.label)}</td>"
            f"<td><code>{html.escape(obj.category)}</code></td>"
            f"<td>{marker}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_frame_cards(frames: Sequence[FrameRecord]) -> str:
    if not frames:
        return '<p class="muted">No matching dataset frames were exported for this task.</p>'
    cards = []
    for frame in frames:
        score = ""
        if frame.score_key is not None:
            score = f"<span>{html.escape(frame.score_key)}={html.escape(str(frame.score_value))}</span>"
        cards.append(
            '<figure class="frame">'
            f'<img src="{html.escape(frame.image_path)}" loading="lazy" />'
            "<figcaption>"
            f"<strong>{html.escape(frame.camera_key)}</strong>"
            f"<span>episode {frame.episode_index}, dataset index {frame.dataset_index}</span>"
            f"<span>{html.escape(frame.selection_reason)}</span>"
            f"{score}"
            "</figcaption>"
            "</figure>"
        )
    return "\n".join(cards)


def _write_html(records: Sequence[TaskRecord], output_dir: Path) -> None:
    task_sections = []
    for record in records:
        interest = ", ".join(f"<code>{html.escape(name)}</code>" for name in record.objects_of_interest)
        interest_html = interest or '<span class="muted">none</span>'
        task_sections.append(
            '<section class="task">'
            f'<h2><span class="id">Task {record.task_id}</span>{html.escape(record.language)}</h2>'
            '<div class="meta">'
            f"<span>{html.escape(record.suite)}</span>"
            f"<span>{html.escape(record.bddl_file)}</span>"
            f"<span>{record.dataset_episode_count} matching dataset episodes</span>"
            "</div>"
            f"<p><strong>Objects of interest:</strong> {interest_html}</p>"
            f"<pre>{html.escape(record.goal)}</pre>"
            "<table>"
            "<thead><tr><th>Instance</th><th>Readable label</th><th>Category</th><th>Target</th></tr></thead>"
            f"<tbody>{_render_object_rows(record.objects)}</tbody>"
            "</table>"
            '<div class="frames">'
            f"{_render_frame_cards(record.frames)}"
            "</div>"
            "</section>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LIBERO Task Object Browser</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f5;
      --fg: #161616;
      --muted: #686868;
      --line: #d7d7d2;
      --panel: #ffffff;
      --accent: #126c5a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 247, 245, 0.95);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px 24px 40px;
    }}
    .task {{
      margin: 0 0 22px;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .id {{
      display: inline-block;
      margin-right: 10px;
      color: var(--accent);
      font-size: 13px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta span {{
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
    }}
    p {{ margin: 8px 0 10px; }}
    pre {{
      margin: 10px 0 14px;
      padding: 10px;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfbfa;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 14px;
    }}
    th, td {{
      padding: 7px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 12px;
    }}
    .frame {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fbfbfa;
    }}
    .frame img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #e8e8e3;
    }}
    figcaption {{
      display: grid;
      gap: 2px;
      padding: 9px;
      color: var(--muted);
      font-size: 12px;
    }}
    figcaption strong {{ color: var(--fg); }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>LIBERO Task Object Browser</h1>
    <div class="muted">{len(records)} tasks</div>
  </header>
  <main>
    {''.join(task_sections)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object", help="LIBERO suite name, e.g. libero_object.")
    parser.add_argument("--task_ids", default=None, help="Comma-separated task ids, or omit for all tasks.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tree_search/libero_task_object_browser"))
    parser.add_argument("--dataset_repo_id", default=None, help="Optional LeRobot dataset repo id for frames.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Optional local dataset root.")
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument(
        "--download_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/load dataset videos when exporting frames.",
    )
    parser.add_argument("--camera_keys", default="all", help="'all' or a comma-separated camera key list.")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument(
        "--prefer_reward_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use success/reward columns when available; otherwise use each demo's final frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_libero_tasks(args.suite, _parse_task_ids(args.task_ids))
    if args.dataset_repo_id:
        _attach_dataset_frames(
            records,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            revision=args.dataset_revision,
            video_backend=args.video_backend,
            download_videos=args.download_videos,
            camera_keys_arg=args.camera_keys,
            episodes_per_task=args.episodes_per_task,
            prefer_reward_frame=args.prefer_reward_frame,
            output_dir=args.output_dir,
        )

    _write_json(records, args.output_dir)
    _write_csv(records, args.output_dir)
    _write_html(records, args.output_dir)
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {args.output_dir / 'task_objects.json'}")
    print(f"Wrote {args.output_dir / 'task_objects.csv'}")


if __name__ == "__main__":
    main()
