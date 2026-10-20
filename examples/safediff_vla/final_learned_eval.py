#!/usr/bin/env python
"""PRIMARY learned end-to-end evaluation (frozen `grounded_grasp_v2_20k/checkpoints/020000`), ONE task x ONE seed per
OS process. Wraps the existing `eval_grounded_grasp_v2_monitor_baseline.run_task` (monitor-only logging; the policy's
actions are never modified). NO oracle input to the policy: privileged simulator reads (target pose, is_grasped,
finger/contact snapshot, distractor poses) are DIAGNOSTIC LOGGING ONLY. Adds: resolved checkpoint/config provenance,
the language string + provenance the env actually fed the policy, and a per-step physics snapshot.

Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/final_learned_eval.py --task select_book --seed 3000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_grounded_grasp_v2_monitor_baseline as H  # noqa: E402
import g25_common as C25  # noqa: E402

from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT_ROOT = Path("outputs/eval/safediff_final_A")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args()
    out = Path(a.out) / a.task / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    ckpt = Path(H.CHECKPOINT)
    cfg = json.loads((ckpt / "config.json").read_text())
    prov = {
        "resolved_checkpoint": str(ckpt.resolve()),
        "checkpoint_step_dir": ckpt.parent.name,
        "architecture": cfg["architecture"],
        "grounded_grasp_close_threshold_m": cfg.get("grounded_grasp_close_threshold_m"),
        "use_local_grasp_refiner_flags": {k: v for k, v in cfg.items() if k.startswith("local_grasp_use")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    lang: list[dict] = []
    snaps: list[dict] = []
    all_snaps: list[list[dict]] = []
    orig_reset, orig_step = VLABenchEnvImpl.reset, VLABenchEnvImpl.step

    def _others(env) -> dict:
        t = env._env.task
        d = {}
        for k, ent in t.entities.items():
            if k == t.target_entity:
                continue
            try:
                d[k] = np.asarray(ent.get_xpos(env._env.physics), float).tolist()
            except Exception:  # noqa: BLE001
                pass
        return d

    def reset(self, *args, **kw):
        r = orig_reset(self, *args, **kw)
        lang.append({"text": self.task_description, "source": self.task_description_source, "target": self._env.task.target_entity})
        all_snaps.append([])
        all_snaps[-1].append({"others": _others(self), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in C25.snapshot(self._env).items()}})
        return r

    def step(self, action):
        r = orig_step(self, action)
        s = C25.snapshot(self._env)
        all_snaps[-1].append({"others": _others(self), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()}})
        return r

    VLABenchEnvImpl.reset, VLABenchEnvImpl.step = reset, step
    policy = SafeDiffVLAPolicy.from_pretrained(H.CHECKPOINT).to("cuda").eval()
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2"
    assert not getattr(policy.config, "local_grasp_use_correction_head", False)
    try:
        H.run_task(policy, "cuda", a.task, [a.seed], out)
    finally:
        VLABenchEnvImpl.reset, VLABenchEnvImpl.step = orig_reset, orig_step
    (out / "provenance.json").write_text(json.dumps({**prov, "language": lang}, indent=1))
    (out / "snapshots.json").write_text(json.dumps(all_snaps))
    print(f"[A] {a.task} seed {a.seed} done; language={lang}")


if __name__ == "__main__":
    main()
