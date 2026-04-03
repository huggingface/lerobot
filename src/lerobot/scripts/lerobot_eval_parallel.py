#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Run lerobot-eval across N independent subprocesses (shards) for maximum GPU utilization.

Each shard handles a disjoint subset of episodes and writes its own JSON results file.
Results are merged and printed when all shards complete.

Usage:
    lerobot-eval-parallel --num-shards 4 [any lerobot-eval flags]
    lerobot-eval-parallel --num-shards auto [any lerobot-eval flags]
    lerobot-eval-parallel --num-shards auto --render-device cpu [any lerobot-eval flags]

--num-shards auto:
    Calls lerobot-eval-autotune to probe hardware and determine the optimal number of shards.

--render-device gpu|cpu|auto:
    Controls MUJOCO_GL env var. 'gpu' -> EGL (faster, ~3ms/frame, ~200KB VRAM/env).
    'cpu' -> osmesa (slower, ~12ms/frame, 0 VRAM). 'auto' picks based on VRAM headroom.
    Default: auto.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse_known(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--num-shards", default="1")
    p.add_argument("--render-device", choices=["gpu", "cpu", "auto"], default="auto")
    p.add_argument("--output-dir", default=None)
    return p.parse_known_args(argv)


def _resolve_num_shards(num_shards_str: str, passthrough: list[str]) -> int:
    if num_shards_str == "auto":
        from lerobot.scripts.lerobot_eval_autotune import probe_and_recommend

        rec = probe_and_recommend(passthrough)
        print(
            f"[autotune] recommended num_shards={rec.num_shards}, batch_size={rec.batch_size}, MUJOCO_GL={rec.mujoco_gl}"
        )
        return rec.num_shards
    return int(num_shards_str)


def _resolve_mujoco_gl(render_device: str, num_shards: int, passthrough: list[str]) -> str:
    if render_device == "gpu":
        return "egl"
    if render_device == "cpu":
        return "osmesa"
    # auto: use EGL for single shard; for multiple shards check VRAM headroom
    if num_shards == 1:
        return "egl"
    try:
        from lerobot.scripts.lerobot_eval_autotune import probe_and_recommend

        rec = probe_and_recommend(passthrough, skip_timing=True)
        return rec.mujoco_gl
    except Exception:
        # Conservative fallback: osmesa avoids EGL VRAM contention
        return "osmesa"


def _extract_output_dir(passthrough: list[str]) -> str | None:
    for tok in passthrough:
        if tok.startswith("--output-dir="):
            return tok.split("=", 1)[1]
        if tok == "--output-dir":
            idx = passthrough.index(tok)
            if idx + 1 < len(passthrough):
                return passthrough[idx + 1]
    return None


def _merge_shards(output_dir: str, num_shards: int) -> dict:
    """Merge per-shard JSON files into a single result dict and write eval_info.json."""
    all_per_task: list[dict] = []
    per_group: dict[str, dict] = {}

    for k in range(num_shards):
        shard_path = Path(output_dir) / f"shard_{k}_of_{num_shards}.json"
        if not shard_path.exists():
            print(f"[warning] shard file not found: {shard_path}", file=sys.stderr)
            continue
        with open(shard_path) as f:
            shard = json.load(f)
        all_per_task.extend(shard.get("per_task", []))
        for group, metrics in shard.get("per_group", {}).items():
            if group not in per_group:
                per_group[group] = {"sum_rewards": [], "max_rewards": [], "successes": []}
            for key in ("sum_rewards", "max_rewards", "successes"):
                # metrics may store aggregates; reconstruct lists if possible
                per_group[group][key].extend(metrics.get(key, []))

    # Re-aggregate
    import numpy as np

    def _nanmean(xs: list) -> float:
        return float(np.nanmean(xs)) if xs else float("nan")

    groups_out = {}
    all_sr, all_mr, all_succ = [], [], []
    for group, acc in per_group.items():
        groups_out[group] = {
            "avg_sum_reward": _nanmean(acc["sum_rewards"]),
            "avg_max_reward": _nanmean(acc["max_rewards"]),
            "pc_success": _nanmean(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
        }
        all_sr.extend(acc["sum_rewards"])
        all_mr.extend(acc["max_rewards"])
        all_succ.extend(acc["successes"])

    overall = {
        "avg_sum_reward": _nanmean(all_sr),
        "avg_max_reward": _nanmean(all_mr),
        "pc_success": _nanmean(all_succ) * 100 if all_succ else float("nan"),
        "n_episodes": len(all_sr),
    }

    merged = {"per_task": all_per_task, "per_group": groups_out, "overall": overall}
    out_path = Path(output_dir) / "eval_info.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def main(argv: list[str] | None = None) -> None:
    args, passthrough = _parse_known(argv if argv is not None else sys.argv[1:])

    num_shards = _resolve_num_shards(args.num_shards, passthrough)
    mujoco_gl = _resolve_mujoco_gl(args.render_device, num_shards, passthrough)

    output_dir = args.output_dir or _extract_output_dir(passthrough)

    print(f"[lerobot-eval-parallel] launching {num_shards} shard(s), MUJOCO_GL={mujoco_gl}")

    child_env = {**os.environ, "MUJOCO_GL": mujoco_gl, "OMP_NUM_THREADS": "1"}

    procs = []
    for k in range(num_shards):
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            f"eval.shard_id={k}",
            f"eval.num_shards={num_shards}",
            *passthrough,
        ]
        if output_dir:
            # Each shard shares the same output_dir; shard files are named shard_K_of_N.json
            cmd.append(f"output_dir={output_dir}")
        procs.append(subprocess.Popen(cmd, env=child_env))

    return_codes = [p.wait() for p in procs]
    if any(rc != 0 for rc in return_codes):
        failed = [k for k, rc in enumerate(return_codes) if rc != 0]
        print(f"[lerobot-eval-parallel] shards {failed} failed with non-zero exit codes.", file=sys.stderr)
        sys.exit(1)

    if output_dir and num_shards > 1:
        merged = _merge_shards(output_dir, num_shards)
        print("\n=== Merged Results ===")
        print(json.dumps(merged["overall"], indent=2))


if __name__ == "__main__":
    main()
