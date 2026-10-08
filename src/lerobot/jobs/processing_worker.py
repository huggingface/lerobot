# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Internal scheduler command; uses exactly the same worker/finalizer as local runs."""

import argparse
import json
import os
from dataclasses import asdict

from lerobot.data_processing.artifacts import ArtifactStore
from lerobot.data_processing.planner import StagePlan
from lerobot.data_processing.runtime import finalize_stage
from lerobot.data_processing.worker import run_worker_group


def assigned_shards(shards: int, groups: int, index: int) -> list[int]:
    if groups < 1 or not 0 <= index < groups:
        raise ValueError("Invalid worker group/index")
    return list(range(index, shards, groups))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "finalize", "inspect"))
    parser.add_argument("--run-uri", required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args(argv)
    store = ArtifactStore(args.run_uri)
    plan = StagePlan.load(store, args.plan_id, verify_work=args.action != "worker")
    if args.action == "worker":
        index = args.worker_index if args.worker_index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_worker_group(
            store.uri,
            plan.plan_id,
            assigned_shards(plan.shards, args.groups, index),
            args.batch_size,
            args.max_retries,
        )
    elif args.action == "finalize":
        summary = finalize_stage(store, plan)
        print(json.dumps(asdict(summary), sort_keys=True))
    else:
        print(json.dumps(asdict(plan), sort_keys=True))


if __name__ == "__main__":
    main()
