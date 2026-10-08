# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Dependency graphs over the same source, planner, worker and finalizer contracts."""

from .artifacts import ArtifactStore
from .configs import RuntimeConfig, StageConfig
from .planner import StagePlan, seal_plan
from .runtime import StageSummary, run_local


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
    if runtime.backend != "local":
        raise ValueError("This backend requires the processing launcher integration")
    if not runtime.run_uri:
        raise ValueError("A persistent runtime.run_uri is required")
    store = ArtifactStore(runtime.run_uri)
    completed: dict[str, tuple[StagePlan, StageSummary]] = {}
    for stage in ordered_stages(stages):
        if set(stage.depends_on) - completed.keys():
            continue  # plan mode defers discovery which needs real upstream data
        upstream = {name: completed[name] for name in stage.depends_on}
        plan = seal_plan(
            store,
            source.dataset_ref,
            stage.factory,
            stage.config,
            source.discover(stage, store, upstream),
            shard_size=runtime.shard_size,
            upstream={name: summary.accepted_path for name, (_, summary) in upstream.items()},
        )
        if runtime.mode == "plan":
            continue
        if before_execute:
            before_execute(store, plan)
        summary = run_local(
            store,
            plan,
            workers=runtime.workers,
            batch_size=runtime.batch_size,
            max_retries=runtime.max_retries,
        )
        completed[stage.id] = (plan, summary)
    return store, completed
