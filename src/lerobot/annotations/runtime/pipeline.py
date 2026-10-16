# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Dependency graphs over the same source, planner, worker and finalizer contracts."""

import logging
import os
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace

from .artifacts import ArtifactStore
from .configs import RuntimeConfig, StageConfig
from .planner import StagePlan, load_module, seal_plan
from .runtime import StageSummary, gpu_assignments, run_local


def conditional_items(items, stage):
    """Preserve all source records, with explicit quality-derived masks."""
    for item in items:
        if stage.when:
            value = item.payload
            try:
                for key in stage.when.split("."):
                    value = value[key]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Missing condition {stage.when!r} for {item.key}") from exc
            if not isinstance(value, bool):
                raise ValueError(f"Condition {stage.when!r} must be boolean for {item.key}")
            if not value:
                item = replace(item, mask_reason=item.mask_reason or stage.skip_reason)
        yield item


def ordered_stages(stages: list[StageConfig]) -> list[StageConfig]:
    by_id = {stage.id: stage for stage in stages}
    if len(by_id) != len(stages):
        raise ValueError("Duplicate stage IDs")
    if any(set(stage.depends_on) - by_id.keys() for stage in stages):
        raise ValueError("Unknown stage dependency")
    ordered: list[StageConfig] = []
    remaining = dict(by_id)
    while remaining:
        ready = [stage for stage in remaining.values() if set(stage.depends_on) <= {s.id for s in ordered}]
        if not ready:
            raise ValueError("Stage dependency cycle")
        for stage in ready:
            ordered.append(stage)
            remaining.pop(stage.id)
    return ordered


def run_pipeline(source, stages: list[StageConfig], runtime: RuntimeConfig, *, before_execute=None):
    """A source discovers metadata lazily after declared upstream stages finish.

    `before_execute` owns local services, not scientific configuration. Plans do
    not initialize services. Downstream plans are deferred when upstream results
    do not yet exist; planning never fabricates those results.
    """
    if runtime.backend not in {"local", "slurm", "hf_jobs"}:
        raise ValueError("This backend requires the processing launcher integration")
    if not runtime.run_uri:
        raise ValueError("A persistent runtime.run_uri is required")
    store = ArtifactStore(runtime.run_uri)
    completed: dict[str, tuple[StagePlan, StageSummary]] = {}

    def prepare(stage):
        upstream = {name: completed[name] for name in stage.depends_on}
        return seal_plan(
            store,
            source.dataset_ref,
            stage.factory,
            stage.config,
            conditional_items(source.discover(stage, store, upstream), stage),
            shard_size=runtime.shard_size,
            upstream={name: summary.content_digest for name, (_, summary) in upstream.items()},
        )

    def execute(plan, devices=None):
        start = time.perf_counter()
        try:
            previous_metrics = set(store.list(f"metrics/{plan.plan_id}/workers/*.parquet"))
        except Exception:
            previous_metrics = None
            logging.getLogger(__name__).warning("Could not list worker metrics", exc_info=True)
        if runtime.backend == "hf_jobs":
            from lerobot.jobs.processing import run_hf_stage

            summary = run_hf_stage(store, plan, runtime)
        elif runtime.backend == "slurm":
            from lerobot.jobs.slurm import run_slurm

            summary = run_slurm(store, plan, runtime)
        else:
            summary = run_local(
                store,
                plan,
                workers=runtime.workers,
                batch_size=runtime.batch_size,
                max_retries=runtime.max_retries,
                force_spawn=runtime.max_parallel_stages > 1,
                device_groups=devices,
            )
        if previous_metrics is None:
            return summary  # Telemetry availability must not block scientific work.
        from .metrics import report_stage

        try:
            store.put_json(
                f"metrics/{plan.plan_id}/stages/{uuid.uuid4().hex}.json",
                report_stage(
                    store,
                    plan,
                    summary,
                    wall_seconds=time.perf_counter() - start,
                    worker_paths=set(store.list(f"metrics/{plan.plan_id}/workers/*.parquet"))
                    - previous_metrics,
                ),
            )
        except Exception:
            logging.getLogger(__name__).warning("Could not record stage metrics", exc_info=True)
        return summary

    ordered = ordered_stages(stages)
    if runtime.mode == "plan":
        for stage in ordered:
            if stage.depends_on:
                continue  # No fictional upstream results or service initialization.
            plan = prepare(stage)
            if runtime.backend == "slurm":
                from lerobot.jobs.slurm import render_slurm

                render_slurm(store, plan, runtime)
        return store, completed

    if runtime.max_parallel_stages == 1 and runtime.resource_budget is None:
        for stage in ordered:
            plan = prepare(stage)
            if before_execute:
                before_execute(store, plan)
            completed[stage.id] = (plan, execute(plan))
        return store, completed

    budget = runtime.resource_budget
    if budget is None:
        raise ValueError("A resource budget is required")
    if runtime.backend == "local":
        capacity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        if budget.cpus > capacity:
            raise ValueError("Local resource_budget exceeds available CPU cores")
    available_gpus = (
        [group[0] for group in gpu_assignments(budget.gpus, 1)]
        if runtime.backend == "local" and budget.gpus
        else []
    )
    remaining = {stage.id: stage for stage in ordered}
    prepared = {}
    running: dict[
        Future, tuple[StageConfig, StagePlan, tuple[int, int, float], list[tuple[str, ...]] | None]
    ] = {}
    reserved = [0, 0, 0.0]
    with ThreadPoolExecutor(max_workers=runtime.max_parallel_stages) as executor:
        while remaining or running:
            for name, stage in list(remaining.items()):
                if len(running) >= runtime.max_parallel_stages:
                    break
                if set(stage.depends_on) - completed.keys():
                    continue
                if name not in prepared:
                    prepared[name] = prepare(stage)
                plan = prepared[name]
                # Identical ready nodes share one physical plan. Do not launch
                # competing owners; reuse it after the first node finishes.
                if any(active_plan.plan_id == plan.plan_id for _, active_plan, _, _ in running.values()):
                    continue
                resources = load_module(plan.factory, plan.config).spec.resources
                workers = min(runtime.workers, plan.shards)
                request = (resources.cpus * workers, resources.gpus * workers, resources.memory_gb * workers)
                limits = (budget.cpus, budget.gpus, budget.memory_gb)
                if any(need > limit for need, limit in zip(request, limits, strict=True)):
                    raise ValueError(f"Stage {name} exceeds resource_budget: {request} > {limits}")
                if any(
                    used + need > limit for used, need, limit in zip(reserved, request, limits, strict=True)
                ):
                    continue
                # Source discovery and service startup stay on the controller thread.
                # Only dispatch/wait runs concurrently; no shared-source mutation.
                if before_execute:
                    before_execute(store, plan)
                devices = None
                if runtime.backend == "local" and request[1]:
                    chosen, available_gpus = available_gpus[: request[1]], available_gpus[request[1] :]
                    devices = [
                        tuple(chosen[i : i + resources.gpus]) for i in range(0, len(chosen), resources.gpus)
                    ]
                reserved = [used + need for used, need in zip(reserved, request, strict=True)]
                future = executor.submit(execute, plan, devices)
                running[future] = (stage, plan, request, devices)
                del remaining[name]
            if not running:
                raise RuntimeError("No stage can be admitted")
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                stage, plan, request, devices = running.pop(future)
                completed[stage.id] = (plan, future.result())
                reserved = [used - need for used, need in zip(reserved, request, strict=True)]
                if devices:
                    available_gpus.extend(device for group in devices for device in group)
    return store, completed
