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
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from lerobot.annotations.processing import run_annotation_pipeline
from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
from lerobot.configs import parser
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE

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
        cfg.revision = source.name
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

    if cfg.job.is_remote or cfg.runtime.backend == "hf_jobs":
        # Imported lazily: the submitter pulls in LeRobotDataset (the `dataset`
        # extra), which a local annotation run over --root doesn't need.
        from lerobot.jobs.annotate import submit_annotate_to_hf

        return submit_annotate_to_hf(cfg)

    root = _resolve_root(cfg)
    summary = _run_annotation(cfg, root)
    if cfg.push_to_hub and cfg.runtime.mode != "plan":
        _push_to_hub(root, cfg, changed_paths=_changed_paths(root, summary))


def _run_annotation(cfg, root):
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

    return summary


def _changed_paths(root, summary):
    owners = [
        root / "meta/annotations/ownership" / path.relative_to(root / "data")
        for path in summary.written_paths
    ]
    return list(
        dict.fromkeys(
            [
                *summary.written_paths,
                *getattr(summary, "metadata_paths", []),
                root / "meta/info.json",
                *(path for path in owners if path.exists()),
            ]
        )
    )


def _push_to_hub(root, cfg, *, changed_paths):
    from lerobot.data_processing.sinks.hub import publish_annotation

    return publish_annotation(root, cfg, changed_paths=changed_paths)


def main() -> None:
    annotate()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
