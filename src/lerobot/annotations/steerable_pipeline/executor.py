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
"""In-process executor that runs the annotation phases.

The executor runs **six phases** in dependency order:

    phase 1: ``plan`` module (plan + subtasks + memory)
    phase 2: ``interjections`` module (interjections + speech)
    phase 3: ``plan`` plan-update pass — re-runs plan emission at every
             interjection timestamp produced by phase 2
    phase 4: ``vqa`` module (VQA)
    phase 5: validator
    phase 6: writer

Phase 3 is why the ``plan`` module must be re-entered after the
``interjections`` module — to refresh ``plan`` rows at interjection
timestamps.

Distributed execution is provided by Hugging Face Jobs (see
``examples/annotations/run_hf_job.py``); the runner inside the job
invokes ``lerobot-annotate`` which uses this in-process executor.
Episode-level concurrency is controlled by
``ExecutorConfig.episode_parallelism``.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AnnotationPipelineConfig
from .reader import EpisodeRecord, iter_episodes
from .staging import EpisodeStaging
from .validator import StagingValidator
from .writer import LanguageColumnsWriter

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Summary of one pipeline phase across all episodes."""

    name: str
    episodes_processed: int
    episodes_skipped: int


@dataclass
class PipelineRunSummary:
    """Aggregated result returned by :meth:`Executor.run`."""

    phases: list[PhaseResult]
    written_paths: list[Path]
    validation_report: Any  # ValidationReport, kept Any to avoid import cycle


@dataclass
class Executor:
    """Run all six phases over a dataset root in-process.

    Episode-level concurrency comes from ``ExecutorConfig.episode_parallelism``
    (a thread pool); cluster-level concurrency comes from running this
    executor inside a Hugging Face Job. Tests construct the executor
    directly with stub modules.
    """

    config: AnnotationPipelineConfig
    plan: Any  # PlanSubtasksMemoryModule
    interjections: Any  # InterjectionsAndSpeechModule
    vqa: Any  # GeneralVqaModule
    writer: LanguageColumnsWriter
    validator: StagingValidator

    def run(self, root: Path) -> PipelineRunSummary:
        records = list(iter_episodes(root, only_episodes=self.config.only_episodes))
        n = len(records)
        if n == 0:
            raise ValueError(f"No episodes found under {root}/data/")

        print(f"[annotate] {n} episodes total", flush=True)

        staging_dir = self.config.resolved_staging_dir(root)
        staging_dir.mkdir(parents=True, exist_ok=True)

        phases: list[PhaseResult] = []

        # Phase 1: ``plan`` module (plan + subtasks + memory)
        phases.append(self._run_module_phase("plan", records, staging_dir, self.plan))
        # Phase 2: ``interjections`` module (interjections + speech). It
        # reads the ``plan`` module's subtask rows from the same staging
        # tree to ground the interjection prompt in the correct local subtask.
        phases.append(self._run_module_phase("interjections", records, staging_dir, self.interjections))
        # Phase 3: ``plan`` plan-update pass at interjection timestamps.
        phases.append(self._run_plan_update_phase(records, staging_dir))
        # Phase 4: ``vqa`` module (VQA)
        phases.append(self._run_module_phase("vqa", records, staging_dir, self.vqa))

        print("[annotate] running validator...", flush=True)
        report = self.validator.validate(records, staging_dir)
        if not report.ok and not self.config.skip_validation:
            raise RuntimeError(f"Staging validation failed: {report.summary()}")
        print(f"[annotate] validator: {report.summary()}", flush=True)

        print(f"[annotate] writing parquet shards into {root}/data/...", flush=True)
        written = self.writer.write_all(records, staging_dir, root)
        print(f"[annotate] wrote {len(written)} shard(s); pipeline complete", flush=True)

        # Keep meta/info.json aligned with the parquet schema we just wrote.
        # Idempotent and additive: existing user metadata is preserved.
        self._ensure_annotation_metadata_in_info(root)

        return PipelineRunSummary(phases=phases, written_paths=written, validation_report=report)

    @staticmethod
    def _ensure_annotation_metadata_in_info(root: Path) -> None:
        """Write language features and canonical tools to ``meta/info.json``.

        ``LanguageColumnsWriter`` adds ``language_persistent`` and
        ``language_events`` to parquet shards. The metadata must advertise
        those columns too, otherwise non-streaming ``LeRobotDataset`` loads
        cast against the old schema and fail on the extra parquet columns.
        """
        from lerobot.datasets.io_utils import load_info, write_info  # noqa: PLC0415
        from lerobot.datasets.language import SAY_TOOL_SCHEMA, language_feature_info  # noqa: PLC0415

        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            return
        try:
            info = load_info(root)
        except Exception as exc:  # noqa: BLE001
            print(f"[annotate] could not read {info_path}: {exc}", flush=True)
            return

        changed = False

        merged_features = {**info.features, **language_feature_info()}
        if merged_features != info.features:
            info.features = merged_features
            changed = True

        existing = info.tools or []
        names = {(t.get("function") or {}).get("name") for t in existing if isinstance(t, dict)}
        if SAY_TOOL_SCHEMA["function"]["name"] not in names:
            info.tools = [*existing, SAY_TOOL_SCHEMA]
            changed = True

        if changed:
            write_info(info, root)
            print(
                "[annotate] meta/info.json: "
                f"language_features={list(language_feature_info())}, "
                f"tools={[t['function']['name'] for t in (info.tools or [])]}",
                flush=True,
            )

    def _run_module_phase(
        self,
        name: str,
        records: list[EpisodeRecord],
        staging_dir: Path,
        module: Any,
    ) -> PhaseResult:
        if not module.enabled:
            print(f"[annotate] phase={name} skipped (module disabled)", flush=True)
            return PhaseResult(name=name, episodes_processed=0, episodes_skipped=len(records))
        n = len(records)
        parallelism = max(1, min(self.config.executor.episode_parallelism, n))
        print(
            f"[annotate] phase={name} starting on {n} episode(s) (parallelism={parallelism})",
            flush=True,
        )
        t0 = time.time()

        def _do(idx_record: tuple[int, EpisodeRecord]) -> tuple[int, int, float]:
            i, record = idx_record
            ep_start = time.time()
            staging = EpisodeStaging(staging_dir, record.episode_index)
            module.run_episode(record, staging)
            return i, record.episode_index, time.time() - ep_start

        processed = 0
        if parallelism == 1:
            for i, record in enumerate(records, 1):
                _, ep_idx, elapsed = _do((i, record))
                processed += 1
                print(
                    f"[annotate]   {name} episode {i}/{n} (idx={ep_idx}) done in {elapsed:.1f}s",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as pool:
                futures = [pool.submit(_do, (i, r)) for i, r in enumerate(records, 1)]
                for fut in as_completed(futures):
                    i, ep_idx, elapsed = fut.result()
                    processed += 1
                    print(
                        f"[annotate]   {name} episode {processed}/{n} "
                        f"(idx={ep_idx}, submit_order={i}) done in {elapsed:.1f}s",
                        flush=True,
                    )
        total = time.time() - t0
        print(f"[annotate] phase={name} complete: {processed}/{n} in {total:.1f}s", flush=True)
        return PhaseResult(name=name, episodes_processed=processed, episodes_skipped=0)

    def _run_plan_update_phase(  # noqa: PLR0915
        self, records: list[EpisodeRecord], staging_dir: Path
    ) -> PhaseResult:
        """Re-emit ``plan`` rows at each timestamp the ``interjections`` module produced.

        The ``plan`` module owns the prompt; the ``interjections`` module
        produced the timestamps. This phase therefore calls back into the
        ``plan`` module with the interjection timestamps so its existing
        prompt path is reused.
        """
        if not self.plan.enabled or not self.interjections.enabled:
            return PhaseResult(name="plan_update", episodes_processed=0, episodes_skipped=len(records))
        processed = 0
        for record in records:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            interjection_rows = [
                row for row in staging.read("interjections") if row.get("style") == "interjection"
            ]
            interjection_times = [float(row["timestamp"]) for row in interjection_rows]
            interjection_texts = [str(row.get("content") or "") for row in interjection_rows]
            if interjection_times:
                self.plan.run_plan_updates(record, staging, interjection_times, interjection_texts)
                processed += 1
        # Episodes without any interjections are skipped (no plan refresh
        # needed); count them so the summary's processed+skipped == total.
        return PhaseResult(
            name="plan_update",
            episodes_processed=processed,
            episodes_skipped=len(records) - processed,
        )
