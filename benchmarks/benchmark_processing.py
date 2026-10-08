# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Measure full local conversion or print an explicitly unmeasured scale matrix.

uv run benchmarks/benchmark_processing.py --config source.json --workers 1 2
uv run benchmarks/benchmark_processing.py --plan-matrix --encoder-threads 8

Measurements include acquisition, planning, setup, decoding, encoding, upload,
retries, validation, telemetry and assembly; exclude fixture construction/cleanup.
"""

import argparse
import copy
import json
import platform
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import draccus


def scale_matrix(encoder_threads):
    if encoder_threads < 1:
        raise ValueError("Encoder threads must be positive")
    return {
        "measured": False,
        "cpu": [
            {
                "workers": workers,
                "encoder_threads_per_worker": encoder_threads,
                "requested_cores": workers * encoder_threads,
            }
            for workers in (1, 16, 128, 1024)
        ],
        "gpu": [
            {"workers": workers, "gpus_per_worker": 1, "requested_gpus": workers}
            for workers in (1, 4, 16, 32)
        ],
        "note": "Resource arithmetic only; actual cluster/GPU runs are required.",
    }


def benchmark_conversion(config, workers):
    from lerobot.data_processing.artifacts import ArtifactStore
    from lerobot.data_processing.bundles import check_credentials
    from lerobot.data_processing.conversion import convert_dataset

    check_credentials(draccus.encode(config))
    if not workers or any(count < 1 for count in workers):
        raise ValueError("Positive worker counts are required")
    results = []
    with tempfile.TemporaryDirectory(prefix="lerobot-benchmark-") as directory:
        base = Path(directory)
        for index, count in enumerate(workers):
            cfg = copy.deepcopy(config)
            cfg.output = base / f"release-{index}"
            cfg.runtime.run_uri = str(base / f"run-{index}")
            cfg.runtime.workers, cfg.runtime.backend, cfg.runtime.mode = count, "local", "run"
            start = time.perf_counter()
            root = convert_dataset(cfg)
            wall = time.perf_counter() - start
            store = ArtifactStore(cfg.runtime.run_uri)
            stages = [store.read_json(path) for path in store.list("metrics/*/stages/*.json")]
            input_hours = stages[0]["physical_input_hours"]
            results.append(
                {
                    "workers": count,
                    "encoder_threads_per_worker": cfg.encoder_threads,
                    "wall_seconds": wall,
                    "physical_input_hours": input_hours,
                    "realtime_multiplier": input_hours * 3600 / wall
                    if stages[0]["realtime_multiplier"] is not None
                    else None,
                    "output_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
                    "stages": stages,
                }
            )
    return {
        "scope": "full local conversion; fresh run store per trial, shared input/acquisition cache",
        "order": workers,
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_checkout": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "dependencies": {name: version(name) for name in ("lerobot", "pyarrow", "torch", "av")},
        "config": draccus.encode(config),
        "results": results,
        "limitations": "Sequential trials, not alternating order; no peak memory/utilization/billing measurement. Repeat in reverse order; inspect output parity separately.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-matrix", action="store_true")
    parser.add_argument("--encoder-threads", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    args = parser.parse_args(argv)
    if args.plan_matrix:
        print(json.dumps(scale_matrix(args.encoder_threads), indent=2))
        return
    if not args.config or any(count < 1 for count in args.workers):
        parser.error("A conversion config and positive worker counts are required")
    from lerobot.data_processing.conversion import ConvertConfig

    cfg = draccus.parse(ConvertConfig, args=[f"--config_path={args.config}"])
    print(json.dumps(benchmark_conversion(cfg, args.workers), indent=2))


if __name__ == "__main__":
    main()
