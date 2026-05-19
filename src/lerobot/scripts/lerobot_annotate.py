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

Annotations live directly in ``data/chunk-*/file-*.parquet``: there is no
flavor namespace and no sidecar tree. Multiple revisions of the same dataset
mean multiple dataset copies.

Example:

  uv run lerobot-annotate \\
      --root=/path/to/dataset \\
      --vlm.backend=transformers \\
      --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct
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
    frame_provider = make_frame_provider(root, camera_key=cfg.vlm.camera_key)
    # Surface the resolved cameras up front so silent Module-3-no-op
    # regressions are obvious in job output rather than discovered post-hoc
    # by counting parquet rows.
    cam_keys = list(getattr(frame_provider, "camera_keys", []) or [])
    logger.info(
        "annotate: frame_provider default camera=%r, all cameras=%s",
        getattr(frame_provider, "camera_key", None),
        cam_keys,
    )
    if cfg.module_3.enabled and not cam_keys:
        logger.warning(
            "annotate: Module 3 (VQA) is enabled but no cameras were "
            "resolved — Module 3 will produce zero VQA rows. Check "
            "meta/info.json for observation.images.* features, or pass "
            "--vlm.camera_key=<key> to seed the cameras list."
        )
    module_1 = PlanSubtasksMemoryModule(vlm=vlm, config=cfg.module_1, frame_provider=frame_provider)
    module_2 = InterjectionsAndSpeechModule(
        vlm=vlm, config=cfg.module_2, seed=cfg.seed, frame_provider=frame_provider
    )
    module_3 = GeneralVqaModule(vlm=vlm, config=cfg.module_3, seed=cfg.seed, frame_provider=frame_provider)
    writer = LanguageColumnsWriter()
    validator = StagingValidator(
        dataset_camera_keys=tuple(getattr(frame_provider, "camera_keys", []) or []) or None,
    )

    executor = Executor(
        config=cfg,
        module_1=module_1,
        module_2=module_2,
        module_3=module_3,
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
        _push_to_hub(root, cfg)


def _push_to_hub(root: Path, cfg: AnnotationPipelineConfig) -> None:
    """Upload the annotated dataset directory to the Hugging Face Hub."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    repo_id = cfg.push_to_hub
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
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        ignore_patterns=[".annotate_staging/**", "**/.DS_Store"],
    )
    print(f"[lerobot-annotate] uploaded to https://huggingface.co/datasets/{repo_id}", flush=True)


def main() -> None:
    annotate()


if __name__ == "__main__":
    main()
