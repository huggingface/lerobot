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
"""``lerobot-annotate`` — populate ``language_persistent`` and
``language_events`` columns on a LeRobot dataset.

Annotations live directly in ``data/chunk-*/file-*.parquet``.

Example:

  uv run lerobot-annotate \\
      --root=/path/to/dataset \\
      --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct

Pass ``--job.target=<flavor>`` to run the same command on a Hugging Face
Jobs GPU instead of this machine (see ``lerobot.jobs.annotate``):

  uv run lerobot-annotate \\
      --repo_id=user/dataset \\
      --new_repo_id=user/dataset_annotated \\
      --push_to_hub=true \\
      --job.target=h200
"""

import logging
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.annotations.processing import run_annotation_pipeline
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
from lerobot.configs import parser
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE
from lerobot.utils.import_utils import _datasets_available, require_package

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.datasets.io_utils import load_info
    from lerobot.datasets.utils import create_lerobot_dataset_card

logger = logging.getLogger(__name__)


def _resolve_root(cfg: AnnotationPipelineConfig) -> Path:
    if cfg.root is not None:
        if any(
            Path(cfg.root).resolve().is_relative_to(Path(cache).resolve())
            for cache in (HF_HUB_CACHE, HF_LEROBOT_HUB_CACHE)
        ):
            raise ValueError(
                "Do not annotate an immutable Hub snapshot in place; use --repo_id or a writable copy"
            )
        return Path(cfg.root)
    if cfg.repo_id is not None:
        source = Path(snapshot_download(repo_id=cfg.repo_id, repo_type="dataset", revision=cfg.revision))
        # Videos are immutable shared inputs; only tabular metadata/data need a
        # writable working copy. Never rewrite the revision-safe Hub cache.
        from filelock import FileLock

        base = (
            Path(cfg.runtime.run_uri)
            if cfg.runtime.run_uri and "://" not in cfg.runtime.run_uri
            else HF_LEROBOT_HOME / "processing"
        )
        parent = base / "inputs" / cfg.repo_id.replace("/", "--")
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / source.name
        with FileLock(parent / ".copy.lock"):
            if root.exists():
                return root
            temporary = Path(tempfile.mkdtemp(prefix="copy-", dir=parent))
            _copy_snapshot(source, temporary)
            temporary.rename(root)
        return root
    raise ValueError("Either --root or --repo_id must be provided.")


def _copy_snapshot(source, root):
    for path in source.iterdir():
        if path.name == "videos":
            (root / path.name).symlink_to(path, target_is_directory=True)
        elif path.is_dir():
            shutil.copytree(path, root / path.name)
        else:
            shutil.copy2(path, root / path.name)


@parser.wrap()
def annotate(cfg: AnnotationPipelineConfig) -> None:
    """Run the steerable annotation pipeline against a dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cfg.job.is_remote:
        # Imported lazily: the submitter pulls in LeRobotDataset (the `dataset`
        # extra), which a local annotation run over --root doesn't need.
        from lerobot.jobs.annotate import submit_annotate_to_hf

        return submit_annotate_to_hf(cfg)

    root = _resolve_root(cfg)
    logger.info("annotate: root=%s", root)

    summary = run_annotation_pipeline(cfg, root)
    logger.info("annotate: wrote %d shard(s)", len(summary.written_paths))
    for phase in summary.phases:
        logger.info(
            "annotate: phase=%s processed=%d skipped=%d",
            phase.name,
            phase.episodes_processed,
            phase.episodes_skipped,
        )
    if summary.validation_report.warnings:
        for w in summary.validation_report.warnings:
            logger.warning(w)

    if cfg.push_to_hub and cfg.runtime.mode != "plan":
        if cfg.repo_id is None and cfg.new_repo_id is None:
            raise ValueError(
                "--push_to_hub requires --repo_id or --new_repo_id (the dataset repo to push to)."
            )
        _push_to_hub(root, cfg)


def _push_to_hub(root: Path, cfg: AnnotationPipelineConfig) -> None:
    """Upload the annotated dataset directory to the Hub.

    Pushes to ``cfg.new_repo_id`` when set, otherwise back to ``cfg.repo_id``.
    """
    require_package("datasets", "dataset")

    repo_id = cfg.new_repo_id or cfg.repo_id
    commit_message = cfg.push_commit_message or "Add steerable annotations (lerobot-annotate)"
    api = HfApi()
    logger.info(f"[lerobot-annotate] creating/locating dataset repo {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=cfg.push_private,
        exist_ok=True,
    )
    logger.info(f"[lerobot-annotate] uploading {root} -> {repo_id}...")
    commit_info = api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        # README.md is excluded because when pushing to ``new_repo_id`` the
        # source card's links (e.g. the visualize badge) would keep pointing
        # at the source dataset; a fresh card is generated below instead.
        ignore_patterns=[".annotate_staging/**", "**/.DS_Store", "README.md"],
    )
    logger.info(f"[lerobot-annotate] uploaded to https://huggingface.co/datasets/{repo_id}")

    dataset_info = load_info(root)
    card = create_lerobot_dataset_card(dataset_info=dataset_info, license="apache-2.0", repo_id=repo_id)
    card.push_to_hub(repo_id=repo_id, repo_type="dataset")

    # Tag the upload with the codebase version. ``LeRobotDatasetMetadata``
    # resolves the dataset revision via ``get_safe_version`` which scans
    # for tags like ``v3.0``; without a tag it raises
    # ``RevisionNotFoundError``. Read the version straight from the
    # dataset's own ``meta/info.json`` so we tag whatever the writer
    # actually wrote (no accidental drift if the codebase floor moves).
    version_tag = (
        dataset_info.codebase_version if dataset_info.codebase_version.startswith("v") else CODEBASE_VERSION
    )
    revision = getattr(commit_info, "oid", None)
    tag_kwargs = {
        "repo_id": repo_id,
        "tag": version_tag,
        "repo_type": "dataset",
    }
    if revision is not None:
        tag_kwargs["revision"] = revision

    try:
        with suppress(RevisionNotFoundError):
            api.delete_tag(repo_id, tag=version_tag, repo_type="dataset")
        api.create_tag(**tag_kwargs)
        logger.info(f"[lerobot-annotate] tagged {repo_id} as {version_tag}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"[lerobot-annotate] WARNING: could not create tag {version_tag!r} on {repo_id}: {exc}. "
            "Dataset is uploaded but ``LeRobotDataset`` won't be able to load it until it's tagged. "
            "Run: from huggingface_hub import HfApi; "
            f"HfApi().create_tag({repo_id!r}, tag={version_tag!r}, repo_type='dataset', exist_ok=True)"
        )


def main() -> None:
    annotate()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
