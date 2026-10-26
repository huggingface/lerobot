#!/usr/bin/env python
"""PHASE E0 step 4: SMALL closed-loop smoke test of corrected language, 3 models x 3 tasks x 3 seeds,
each under both language conditions, through the unchanged `final_eval_paper_multivariant.run_model_task`
pipeline (same policies, same eval_policy, same instrumentation; only seeds/output root are set here).

    OLD-EVAL        language = task name ("select_book")      -- forced by patching `resolve_task_language`
    CORRECTED-EVAL  language = "primitive: <native instruction>" -- the fixed env wrapper as committed

Extra per-step, per-candidate trace (ee-to-each-same-type-candidate distance, per-candidate is_grasped,
gripper) is recorded so wrong-object approach/grasp can be told apart from success.

Worker:   MUJOCO_GL=egl uv run python .../phase_e0_smoke_closed_loop.py --model M --task T --language old|corrected
Launcher: MUJOCO_GL=egl uv run python .../phase_e0_smoke_closed_loop.py --launch
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path("outputs/eval/safediff_vla_e0_corrected_language_smoke")
MODELS = ["SmolVLA", "TemporalDecoder-SinCos-PadFix", "GroundedGrasp-v2"]
TASKS = ["select_book", "select_poker", "select_fruit"]
SEEDS = [1000, 1001, 1002]


def worker(model: str, task: str, language: str) -> None:
    import final_eval_paper_multivariant as fe
    import lerobot.envs.vlabench as vb

    if language == "old":
        vb.resolve_task_language = lambda task_obj, task_name: (task_name, "OLD_EVAL_task_name_forced")
    fe.SEEDS = SEEDS
    fe.MACHINE = "smoke"
    root = ROOT / {"old": "OLD_EVAL", "corrected": "CORRECTED_EVAL"}[language] / f"{model}__{task}"

    # per-candidate trace, patched BEFORE run_model_task installs its own env hooks (they wrap ours)
    cls = vb.VLABenchEnv
    orig_reset, orig_step = cls.reset, cls.step
    eps: list[dict] = []

    def snap(self, t):
        e = self._env; ph = e.physics; tk = e.task
        ee = np.asarray(tk.robot.get_end_effector_pos(ph), dtype=float)
        rec = {"t": t, "ee": ee.tolist(), "cand": {}}
        for n, ent in tk.entities.items():
            if type(ent).__name__ == eps[-1]["cand_type"]:
                p = np.asarray(ent.get_xpos(ph), dtype=float)
                try:
                    g = bool(ent.is_grasped(ph, tk.robot))
                except Exception:  # noqa: BLE001
                    g = False
                rec["cand"][n] = {"d": float(np.linalg.norm(ee - p)), "grasped": g, "pos": p.tolist()}
        return rec

    def reset(self, seed=None, **kw):
        out = orig_reset(self, seed=seed, **kw)
        if seed is not None:
            tk = self._env.task
            eps.append({"seed": int(seed) if not isinstance(seed, (list, tuple)) else int(seed[0]),
                        "language_text": self.task_description, "language_source": self.task_description_source,
                        "target": tk.target_entity, "native_instruction": tk.instructions[0],
                        "cand_type": type(tk.entities[tk.target_entity]).__name__, "trace": []})
            eps[-1]["trace"].append(snap(self, 0))
        return out

    def step(self, action):
        out = orig_step(self, action)
        if eps and not out[2]:
            eps[-1]["trace"].append(snap(self, len(eps[-1]["trace"])))
        return out

    cls.reset, cls.step = reset, step
    try:
        fe.run_model_task(model, task, root)
    finally:
        cls.reset, cls.step = orig_reset, orig_step
    (root / "candidate_traces.json").write_text(json.dumps(eps))


def launch(workers: int) -> None:
    jobs = [(m, t, l) for l in ("corrected", "old") for m in MODELS for t in TASKS]

    def run(j):
        m, t, l = j
        log = ROOT / "logs" / f"{l}__{m}__{t}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, __file__, "--model", m, "--task", t, "--language", l], stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"done {l} {m} {t} rc={rc}", flush=True)
        return rc

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model"); ap.add_argument("--task"); ap.add_argument("--language", choices=["old", "corrected"])
    a = ap.parse_args()
    launch(a.workers) if a.launch else worker(a.model, a.task, a.language)
