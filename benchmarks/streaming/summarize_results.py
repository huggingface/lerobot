# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Collapse a directory of benchmark JSON results into one comparison table (and a combined CSV).

python benchmarks/streaming/summarize_results.py benchmarks/streaming/results
"""

import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    ("source", "source"),
    ("mode", "mode"),
    ("video_decode_device", "decode"),
    ("num_workers", "workers"),
    ("batch_size", "bs"),
    ("frames_per_s_node", "frames/s/node"),
    ("first_batch_latency_s", "first_batch_s"),
    ("p50_sample_latency_ms", "p50_ms"),
    ("p95_sample_latency_ms", "p95_ms"),
    ("p99_sample_latency_ms", "p99_ms"),
]


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/streaming/results")
    files = sorted(results_dir.rglob("*.json"))
    if not files:
        print(f"No JSON results under {results_dir}")
        return

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        d["hit_rate"] = d.get("video_decoder_cache", {}).get("hit_rate")
        rows.append(d)

    rows.sort(key=lambda r: (r.get("source", ""), r.get("mode", ""), r.get("video_decode_device", "")))

    headers = [label for _, label in COLUMNS] + ["cache_hit_rate"]
    widths = {h: len(h) for h in headers}
    table = []
    for r in rows:
        row = {label: r.get(key, "") for key, label in COLUMNS}
        row["cache_hit_rate"] = r.get("hit_rate", "")
        table.append(row)
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for row in table:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))

    combined = results_dir / "summary.csv"
    with open(combined, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table)
    print(f"\nWrote {combined}")


if __name__ == "__main__":
    main()
