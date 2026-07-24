#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize distributed episode pool benchmark JSON files.")
    parser.add_argument("summaries", nargs="+", help="Rank summary JSON files.")
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    args = parse_args()
    rows = [_load(path) for path in args.summaries]
    rows.sort(key=lambda row: int(row.get("distributed_shard_index", 0)))
    total_bytes = sum(float(row.get("fetch_bytes", 0.0)) for row in rows)
    max_fetch_s = max(float(row.get("fetch_s", 0.0)) for row in rows)
    aggregate_mib_s = total_bytes / max_fetch_s / 1024**2 if max_fetch_s > 0 else float("inf")
    summed_rank_mib_s = sum(float(row.get("fetch_mib_s", 0.0)) for row in rows)
    total_decode_samples_s = sum(float(row.get("pool_decode_training_samples_s", 0.0)) for row in rows)
    total_stream_samples_s = sum(float(row.get("pool_stream_actual_samples_s", 0.0)) for row in rows)
    kept_up = all(bool(row.get("pool_stream_kept_up", 0.0)) for row in rows)

    print("| Aggregate | value |")
    print("|---|---:|")
    print(f"| ranks | {len(rows)} |")
    print(f"| total fetched GiB | {total_bytes / 1024**3:.2f} |")
    print(f"| aggregate fetch MiB/s | {_fmt(aggregate_mib_s)} |")
    print(f"| summed rank fetch MiB/s | {_fmt(summed_rank_mib_s)} |")
    if total_decode_samples_s:
        print(f"| aggregate resident decode samples/s | {_fmt(total_decode_samples_s)} |")
    if total_stream_samples_s:
        print(f"| aggregate stream samples/s | {_fmt(total_stream_samples_s)} |")
        print(f"| all ranks kept up | {'yes' if kept_up else 'no'} |")

    print()
    print("| Rank | host | fetch MiB/s | fetch s | GiB | decode samples/s | stream samples/s | kept up |")
    print("|---:|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        rank = int(row.get("distributed_shard_index", 0))
        print(
            f"| {rank} | {row.get('hostname', '')} | "
            f"{_fmt(float(row.get('fetch_mib_s', 0.0)))} | "
            f"{_fmt(float(row.get('fetch_s', 0.0)))} | "
            f"{float(row.get('fetch_gib', 0.0)):.2f} | "
            f"{_fmt(float(row.get('pool_decode_training_samples_s', 0.0)))} | "
            f"{_fmt(float(row.get('pool_stream_actual_samples_s', 0.0)))} | "
            f"{'yes' if row.get('pool_stream_kept_up', 0.0) else 'no'} |"
        )


if __name__ == "__main__":
    main()
