#!/usr/bin/env python3
"""Build a single combined LeRobotDataset from RoboCasa's 16 composite_seen tasks.

RoboCasa 1.0 already ships in LeRobot format (parquet + mp4), distributed as
``lerobot.tar`` archives from Box. This script:

1. Downloads each composite_seen task's ``target/human`` archive via RoboCasa's
   official ``download_datasets`` helper (idempotent — skipped if already on
   disk).
2. Opens each extracted directory as a ``LeRobotDataset``.
3. Merges all 16 into one unified dataset via ``merge_datasets`` (a thin wrapper
   over ``aggregate_datasets`` that revalidates fps / robot_type / features,
   unifies task indices, concatenates videos and parquet, and recomputes stats).
4. Optionally pushes the merged dataset to the Hub.

The result is one ~8,000-trajectory dataset where each episode carries its
source task as the ``task`` field — ready for downstream annotation
(subtasks / memory / VQA / tool calls) without per-task bookkeeping.

Usage::

    uv run python -m lerobot.scripts.build_robocasa_composite_seen \\
        --output-dir=/data/lerobot/robocasa_composite_seen \\
        --hub-repo-id=${HF_USER}/robocasa_composite_seen \\
        --push-to-hub

Prereqs: ``robocasa`` and ``robosuite`` installed (see
``docs/source/benchmarks/robocasa.mdx`` for the editable-install dance — they
are not on PyPI and RoboCasa's own ``setup.py`` pins an old LeRobot version).

The 16 composite_seen tasks are the multi-step subset of the official
RoboCasa365 target benchmark — exactly the slice used to compute the
``Composite-Seen`` column of the leaderboard.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# Canonical 16 composite_seen tasks (RoboCasa365 target benchmark).
# Order matches the leaderboard docs.
COMPOSITE_SEEN_TASKS: list[str] = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]


def _require_robocasa() -> None:
    """Fail fast with an actionable message if robocasa is missing.

    RoboCasa is not on PyPI and is not a LeRobot extra — see the installation
    notes in ``docs/source/benchmarks/robocasa.mdx``.
    """
    try:
        import robocasa  # noqa: F401, PLC0415
        from robocasa.scripts import download_datasets as _dl  # noqa: F401, PLC0415
        from robocasa.utils import dataset_registry as _reg  # noqa: F401, PLC0415
    except ImportError as exc:
        sys.exit(
            "[build_robocasa_composite_seen] robocasa is not importable.\n"
            "Install it (and robosuite) per the LeRobot RoboCasa docs:\n"
            "    git clone https://github.com/robocasa/robocasa.git ~/robocasa\n"
            "    git clone https://github.com/ARISE-Initiative/robosuite.git ~/robosuite\n"
            "    pip install -e ~/robocasa --no-deps\n"
            "    pip install -e ~/robosuite\n"
            f"(original error: {exc})"
        )


def _resolve_task_root(task: str) -> Path:
    """Resolve the local extracted ``LeRobotDataset`` root for a target/human task.

    Uses RoboCasa's own ``dataset_registry`` so we follow whatever directory
    layout RoboCasa picks (currently ``v1.0/target/composite/<task>/<date>/``
    under ``robocasa.macros.DATASET_BASE_DIR``). Falls back to discovering the
    extracted directory if the helper's signature drifted between releases.
    """
    from robocasa.utils import dataset_registry  # noqa: PLC0415

    # ``get_ds_path`` is the canonical helper. RoboCasa 1.0 signature is
    # ``get_ds_path(task, ds_type, return_info=False)`` with ``ds_type`` like
    # ``"human_im"`` (image-observation human demos). We try the common
    # ``split=`` kwarg first (newer registry); if it's rejected, fall back.
    try:
        ds_path = dataset_registry.get_ds_path(
            task=task,
            ds_type="human_im",
            return_info=False,
            split="target",
        )
    except TypeError:
        # Older registry — ds_type alone disambiguates target/human.
        ds_path = dataset_registry.get_ds_path(
            task=task,
            ds_type="human_im",
            return_info=False,
        )

    root = Path(ds_path)
    # ``get_ds_path`` may return either the extracted dir or the .tar; normalize.
    if root.suffix == ".tar":
        root = root.parent
    return root


def _download_task(task: str, *, overwrite: bool = False) -> Path:
    """Download (or locate) a single target/human task and return its extracted root."""
    from robocasa.scripts import download_datasets as dl  # noqa: PLC0415

    # Try the documented programmatic API. The CLI is
    #   python -m robocasa.scripts.download_datasets --tasks <T> --source human --split target
    # which is a thin wrapper over a function of the same name.
    if hasattr(dl, "download_datasets"):
        try:
            dl.download_datasets(
                tasks=[task],
                source="human",
                split="target",
                overwrite=overwrite,
            )
        except TypeError:
            # Older signature — drop the kwargs RoboCasa didn't have yet.
            dl.download_datasets(tasks=[task])
    else:
        # No public function — shell out to the CLI as a last resort. This
        # guarantees we use whatever entrypoint RoboCasa's authors maintain.
        import subprocess  # noqa: PLC0415

        cmd = [
            sys.executable,
            "-m",
            "robocasa.scripts.download_datasets",
            "--tasks",
            task,
            "--source",
            "human",
            "--split",
            "target",
        ]
        if overwrite:
            cmd.append("--overwrite")
        subprocess.run(cmd, check=True)

    root = _resolve_task_root(task)
    if not root.exists():
        raise RuntimeError(
            f"Expected {root} after download, but it doesn't exist. "
            "RoboCasa may have changed its data layout — verify with "
            "`robocasa.utils.dataset_registry.get_ds_path()`."
        )
    return root


def _open_as_lerobot_dataset(task: str, root: Path) -> LeRobotDataset:
    """Open an extracted RoboCasa target/human task as a ``LeRobotDataset``.

    The placeholder ``repo_id`` (``robocasa/<task>_target_human``) is only used
    by the aggregator for logging and for the unified task table — the actual
    data is loaded from ``root``.
    """
    repo_id = f"robocasa/{task}_target_human"
    return LeRobotDataset(repo_id=repo_id, root=root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the 16 RoboCasa composite_seen target tasks into one LeRobotDataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory for the merged dataset (will be created).",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help=(
            "Hub repo_id for the merged dataset (e.g. ``yourname/"
            "robocasa_composite_seen``). Required for ``--push-to-hub``; also "
            "becomes the merged dataset's canonical ``repo_id``."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the merged dataset to the Hub after building. Requires "
        "``--hub-repo-id`` and a prior ``huggingface-cli login``.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When pushing, create the Hub repo as private.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to override the default 16 "
        "composite_seen list (useful for smoke-testing with 1–2 tasks).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step entirely; assume each task is already "
        "extracted on disk at the path ``dataset_registry.get_ds_path`` "
        "returns.",
    )
    parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Force re-download even when a complete local extraction exists.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(COMPOSITE_SEEN_TASKS)
    )
    if not tasks:
        sys.exit("No tasks selected.")

    if args.push_to_hub and not args.hub_repo_id:
        sys.exit("--push-to-hub requires --hub-repo-id.")

    output_repo_id = args.hub_repo_id or "local/robocasa_composite_seen"
    logger.info(
        "Building merged RoboCasa dataset: %d tasks → %s (output dir: %s)",
        len(tasks),
        output_repo_id,
        args.output_dir,
    )

    _require_robocasa()

    # 1. Download (or locate) each task's extracted directory.
    task_roots: list[tuple[str, Path]] = []
    for i, task in enumerate(tasks, 1):
        logger.info("[%d/%d] %s", i, len(tasks), task)
        if args.skip_download:
            root = _resolve_task_root(task)
            if not root.exists():
                sys.exit(
                    f"--skip-download set but extracted directory does not "
                    f"exist for {task}: {root}"
                )
        else:
            root = _download_task(task, overwrite=args.overwrite_download)
        logger.info("  extracted at: %s", root)
        task_roots.append((task, root))

    # 2. Open each as a LeRobotDataset (validation happens inside aggregator).
    datasets: list[LeRobotDataset] = []
    for task, root in task_roots:
        logger.info("Opening %s", task)
        ds = _open_as_lerobot_dataset(task, root)
        logger.info(
            "  %s: %d episodes, %d frames, %d FPS",
            task,
            ds.num_episodes,
            ds.num_frames,
            ds.fps,
        )
        datasets.append(ds)

    # 3. Merge — re-validates features/fps/robot_type, unifies tasks, concats
    #    videos + parquet, recomputes stats.
    logger.info("Merging %d datasets into %s", len(datasets), output_repo_id)
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=args.output_dir,
    )
    logger.info(
        "Merged: %d episodes, %d frames across %d unique task strings",
        merged.num_episodes,
        merged.num_frames,
        len(merged.meta.tasks) if merged.meta.tasks is not None else 0,
    )

    # 4. Push to Hub.
    if args.push_to_hub:
        logger.info("Pushing %s to the Hub (private=%s)", args.hub_repo_id, args.private)
        # ``upload_large_folder=True`` is the right mode for tens-of-GB
        # datasets — uses multipart uploads + resumable transfers.
        merged.push_to_hub(
            private=args.private,
            upload_large_folder=True,
            tags=["lerobot", "robocasa", "composite_seen", "manipulation"],
        )
        logger.info(
            "Push complete: https://huggingface.co/datasets/%s",
            args.hub_repo_id,
        )
    else:
        logger.info(
            "Skipping Hub push (no --push-to-hub). Merged dataset is at %s.",
            args.output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
