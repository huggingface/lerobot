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
"""Executor selection: local vs SLURM via datatrove.

The executor plans **four phases** with the dependency order from the plan:

    phase 1: Module 1 (plan + subtasks + memory)
    phase 2: Module 2 (interjections + speech)
    phase 3: Module 1 plan-update pass — re-runs plan emission at every
             interjection timestamp produced by phase 2
    phase 4: Module 3 (VQA)
    phase 5: validator
    phase 6: writer

Phase 3 is why ``executor.py`` documents the dependency: Module 1 must be
re-entered after Module 2 to refresh ``plan`` rows at interjection times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AnnotationPipelineConfig, ExecutorConfig
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


def select_executor_class(num_episodes: int, config: ExecutorConfig) -> str:
    """Return ``"local"`` or ``"slurm"`` based on the threshold.

    The plan's "executor selection threshold" lives in
    :class:`ExecutorConfig.auto_threshold`. ``force_local`` always wins.
    """
    if config.force_local:
        return "local"
    return "local" if num_episodes <= config.auto_threshold else "slurm"


@dataclass
class Executor:
    """Run all four phases over a dataset root.

    The executor is intentionally framework-agnostic: by default it runs the
    phases inline (suitable for tests, small datasets, and the CLI's
    ``--force-local`` mode). It will optionally hand off to datatrove's
    :class:`LocalPipelineExecutor` or :class:`SlurmPipelineExecutor` when those
    are installed and the dataset is large enough to benefit from them.

    Tests construct the executor directly with stub modules.
    """

    config: AnnotationPipelineConfig
    module_1: Any  # PlanSubtasksMemoryModule
    module_2: Any  # InterjectionsAndSpeechModule
    module_3: Any  # GeneralVqaModule
    writer: LanguageColumnsWriter
    validator: StagingValidator

    def run(self, root: Path) -> PipelineRunSummary:
        records = list(iter_episodes(root, only_episodes=self.config.only_episodes))
        n = len(records)
        if n == 0:
            raise ValueError(f"No episodes found under {root}/data/")

        executor_kind = select_executor_class(n, self.config.executor)
        print(f"[annotate] {n} episodes total; executor={executor_kind}", flush=True)

        staging_dir = self.config.resolved_staging_dir(root)
        staging_dir.mkdir(parents=True, exist_ok=True)

        phases: list[PhaseResult] = []

        # Phase 1: Module 1 (plan + subtasks + memory)
        phases.append(self._run_module_phase("module_1", records, staging_dir, self.module_1))
        # Phase 2: Module 2 (interjections + speech). Module 2 reads
        # Module 1's subtask rows from the same staging tree to ground
        # the interjection prompt in the correct local subtask.
        phases.append(self._run_module_phase("module_2", records, staging_dir, self.module_2))
        # Phase 3: Module 1 plan-update pass at interjection timestamps.
        phases.append(self._run_plan_update_phase(records, staging_dir))
        # Phase 4: Module 3 (VQA)
        phases.append(self._run_module_phase("module_3", records, staging_dir, self.module_3))

        print("[annotate] running validator...", flush=True)
        report = self.validator.validate(records, staging_dir)
        if not report.ok and not self.config.skip_validation:
            raise RuntimeError(f"Staging validation failed: {report.summary()}")
        print(f"[annotate] validator: {report.summary()}", flush=True)

        print(f"[annotate] writing parquet shards into {root}/data/...", flush=True)
        written = self.writer.write_all(records, staging_dir, root)
        print(f"[annotate] wrote {len(written)} shard(s); pipeline complete", flush=True)

        # Persist the tool catalog to meta/info.json so downstream
        # consumers (PI052 / Pi0.5 / dataset visualizer) can read
        # it via ``LeRobotDatasetMetadata.tools`` (PR 1). Idempotent and
        # additive: anything the user pre-populated is preserved; we only
        # ensure the canonical ``say`` schema is present.
        self._ensure_tools_in_info(root)

        return PipelineRunSummary(phases=phases, written_paths=written, validation_report=report)

    def _ensure_tools_in_info(self, root: Path) -> None:
        """Write ``meta/info.json["tools"]`` if missing the canonical ``say``.

        Reads any user-declared tools already in ``info.json`` and merges
        the canonical ``SAY_TOOL_SCHEMA`` into the list (deduped by
        ``function.name``). Writes back to disk only if the list
        changed.
        """
        import json  # noqa: PLC0415

        from lerobot.datasets.language import SAY_TOOL_SCHEMA  # noqa: PLC0415

        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            return
        try:
            info = json.loads(info_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[annotate] could not read {info_path}: {exc}", flush=True)
            return

        existing = info.get("tools")
        if not isinstance(existing, list):
            existing = []
        names = {
            (t.get("function") or {}).get("name")
            for t in existing
            if isinstance(t, dict)
        }
        merged = list(existing)
        if SAY_TOOL_SCHEMA["function"]["name"] not in names:
            merged.append(SAY_TOOL_SCHEMA)
        if merged != existing:
            info["tools"] = merged
            info_path.write_text(json.dumps(info, indent=2))
            print(
                f"[annotate] meta/info.json: tools={[t['function']['name'] for t in merged]}",
                flush=True,
            )

    def _run_module_phase(
        self,
        name: str,
        records: list[EpisodeRecord],
        staging_dir: Path,
        module: Any,
    ) -> PhaseResult:
        import time as _time  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

        if not module.enabled:
            print(f"[annotate] phase={name} skipped (module disabled)", flush=True)
            return PhaseResult(name=name, episodes_processed=0, episodes_skipped=len(records))
        n = len(records)
        parallelism = max(1, min(self.config.executor.episode_parallelism, n))
        print(
            f"[annotate] phase={name} starting on {n} episode(s) "
            f"(parallelism={parallelism})",
            flush=True,
        )
        t0 = _time.time()

        def _do(idx_record: tuple[int, EpisodeRecord]) -> tuple[int, int, float]:
            i, record = idx_record
            ep_start = _time.time()
            staging = EpisodeStaging(staging_dir, record.episode_index)
            module.run_episode(record, staging)
            return i, record.episode_index, _time.time() - ep_start

        processed = 0
        if parallelism == 1:
            for i, record in enumerate(records, 1):
                _, ep_idx, elapsed = _do((i, record))
                processed += 1
                print(
                    f"[annotate]   {name} episode {i}/{n} "
                    f"(idx={ep_idx}) done in {elapsed:.1f}s",
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
        total = _time.time() - t0
        print(f"[annotate] phase={name} complete: {processed}/{n} in {total:.1f}s", flush=True)
        return PhaseResult(name=name, episodes_processed=processed, episodes_skipped=0)

    def _run_plan_update_phase(  # noqa: PLR0915
        self, records: list[EpisodeRecord], staging_dir: Path
    ) -> PhaseResult:
        """Re-emit ``plan`` rows at each interjection timestamp from Module 2.

        Module 1 owns the prompt; Module 2 produced the timestamps. This phase
        therefore calls back into Module 1 with the interjection timestamps so
        Module 1's existing prompt path is reused.
        """
        if not self.module_1.enabled or not self.module_2.enabled:
            return PhaseResult(
                name="module_1_plan_update", episodes_processed=0, episodes_skipped=len(records)
            )
        processed = 0
        for record in records:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            interjection_rows = [
                row
                for row in staging.read("module_2")
                if row.get("style") == "interjection"
            ]
            interjection_times = [float(row["timestamp"]) for row in interjection_rows]
            interjection_texts = [str(row.get("content") or "") for row in interjection_rows]
            if interjection_times:
                self.module_1.run_plan_updates(
                    record, staging, interjection_times, interjection_texts
                )
                processed += 1
        return PhaseResult(name="module_1_plan_update", episodes_processed=processed, episodes_skipped=0)
