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

For distributed runs, see ``examples/annotations/run_hf_job.py``.
"""

import logging
from pathlib import Path

from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.annotations.steerable_pipeline.frames import make_frame_provider
from lerobot.annotations.steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from lerobot.configs import parser

logger = logging.getLogger(__name__)


def _resolve_root(cfg: AnnotationPipelineConfig) -> Path:
    if cfg.root is not None:
        return Path(cfg.root)
    if cfg.repo_id is not None:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=cfg.repo_id, repo_type="dataset"))
    raise ValueError("Either --root or --repo_id must be provided.")


@parser.wrap()
def annotate(cfg: AnnotationPipelineConfig) -> None:
    """Run the steerable annotation pipeline against a dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = _resolve_root(cfg)
    logger.info("annotate: root=%s", root)

    vlm = make_vlm_client(cfg.vlm)
    frame_provider = make_frame_provider(root, camera_key=cfg.vlm.camera_key, video_backend=cfg.video_backend)
    # Surface the resolved cameras up front so a silent vqa-module no-op
    # is obvious in job output rather than discovered post-hoc by counting
    # parquet rows.
    cam_keys = list(getattr(frame_provider, "camera_keys", []) or [])
    logger.info(
        "annotate: frame_provider default camera=%r, all cameras=%s",
        getattr(frame_provider, "camera_key", None),
        cam_keys,
    )
    if cfg.vqa.enabled and not cam_keys:
        logger.warning(
            "annotate: the vqa module is enabled but no cameras were "
            "resolved — it will produce zero VQA rows. Check "
            "meta/info.json for observation.images.* features, or pass "
            "--vlm.camera_key=<key> to seed the cameras list."
        )
    plan = PlanSubtasksMemoryModule(vlm=vlm, config=cfg.plan, frame_provider=frame_provider)
    interjections = InterjectionsAndSpeechModule(
        vlm=vlm, config=cfg.interjections, seed=cfg.seed, frame_provider=frame_provider
    )
    vqa = GeneralVqaModule(vlm=vlm, config=cfg.vqa, seed=cfg.seed, frame_provider=frame_provider)
    writer = LanguageColumnsWriter()
    validator = StagingValidator(
        dataset_camera_keys=tuple(getattr(frame_provider, "camera_keys", []) or []) or None,
    )

    executor = Executor(
        config=cfg,
        plan=plan,
        interjections=interjections,
        vqa=vqa,
        writer=writer,
        validator=validator,
    )
    summary = executor.run(root)
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

    if cfg.push_to_hub:
        if cfg.repo_id is None and cfg.new_repo_id is None:
            raise ValueError(
                "--push_to_hub requires --repo_id or --new_repo_id (the dataset repo to push to)."
            )
        _push_to_hub(root, cfg)


def _push_to_hub(root: Path, cfg: AnnotationPipelineConfig) -> None:
    """Upload the annotated dataset directory to the Hub.

    Pushes to ``cfg.new_repo_id`` when set, otherwise back to ``cfg.repo_id``.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    repo_id = cfg.new_repo_id or cfg.repo_id
    commit_message = cfg.push_commit_message or "Add steerable annotations (lerobot-annotate)"
    api = HfApi()
    print(f"[lerobot-annotate] creating/locating dataset repo {repo_id}...", flush=True)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=cfg.push_private,
        exist_ok=True,
    )
    print(f"[lerobot-annotate] uploading {root} -> {repo_id}...", flush=True)
    commit_info = api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        ignore_patterns=[".annotate_staging/**", "**/.DS_Store"],
    )
    print(f"[lerobot-annotate] uploaded to https://huggingface.co/datasets/{repo_id}", flush=True)

    # Tag the upload with the codebase version. ``LeRobotDatasetMetadata``
    # resolves the dataset revision via ``get_safe_version`` which scans
    # for tags like ``v3.0``; without a tag it raises
    # ``RevisionNotFoundError``. Read the version straight from the
    # dataset's own ``meta/info.json`` so we tag whatever the writer
    # actually wrote (no accidental drift if the codebase floor moves).
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION  # noqa: PLC0415

    info_path = root / "meta" / "info.json"
    version_tag = CODEBASE_VERSION
    if info_path.exists():
        try:
            from lerobot.utils.io_utils import load_json  # noqa: PLC0415

            info = load_json(info_path)
            ds_version = info.get("codebase_version")
            if isinstance(ds_version, str) and ds_version.startswith("v"):
                version_tag = ds_version
        except Exception as exc:  # noqa: BLE001
            print(
                f"[lerobot-annotate] could not read codebase_version from info.json ({exc}); falling back to {version_tag}",
                flush=True,
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
        from contextlib import suppress  # noqa: PLC0415

        from huggingface_hub.errors import RevisionNotFoundError  # noqa: PLC0415

        with suppress(RevisionNotFoundError):
            api.delete_tag(repo_id, tag=version_tag, repo_type="dataset")
        api.create_tag(**tag_kwargs)
        print(f"[lerobot-annotate] tagged {repo_id} as {version_tag}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[lerobot-annotate] WARNING: could not create tag {version_tag!r} on {repo_id}: {exc}. "
            "Dataset is uploaded but ``LeRobotDataset`` won't be able to load it until it's tagged. "
            "Run: from huggingface_hub import HfApi; "
            f"HfApi().create_tag({repo_id!r}, tag={version_tag!r}, repo_type='dataset', exist_ok=True)",
            flush=True,
        )


def main() -> None:
    annotate()


if __name__ == "__main__":
    main()
