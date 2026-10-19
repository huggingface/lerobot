#!/usr/bin/env python
"""G25 STEP 2 reference: VLABench's own scripted EXPERT (the data generator: env.get_expert_skill_sequence()) run in simulation, with the same per-step physics snapshot (g25_common) recorded around every env.step. The expert scenes are VLABench's random scenes (not our policy seeds): this gives the reference distribution of the object-in-gripper transform and of hold dynamics during a successful grasp+transport. The expert gripper command is the commanded finger value in the env action (last 2 entries); t_close = first step with it <= 0.02 (closing)."""
from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import g25_common as C25  # noqa: E402
from VLABench.robots import Franka  # noqa: E402,F401
from VLABench.envs import load_env  # noqa: E402
from VLABench.tasks import *  # noqa: E402,F401,F403

CONTAINER = {"select_toy": "giftbox_seen", "select_fruit": "plate_seen", "select_mahjong": "placemat"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default="outputs/eval/safediff_vla_g25"); a = ap.parse_args(); out = Path(a.out) / a.task; out.mkdir(parents=True, exist_ok=True); eps = []
    for k in range(a.n):
        t0 = time.time(); env = load_env(a.task, robot="franka", eval=False); env.reset(); steps = []; orig = env.step
        def step(action, _o=orig):
            ts = _o(action); s = C25.snapshot(env, CONTAINER.get(a.task)); s["grip_cmd"] = float(np.mean(np.asarray(action)[-2:])); s["t"] = len(steps); steps.append(s); return ts
        env.step = step; success = False; err = None
        try:
            for skill in env.get_expert_skill_sequence():
                obs, wp, stage_ok, task_ok = skill(env)
                if task_ok: success = True; break
        except Exception as ex: err = repr(ex)[:200]
        eps.append({"k": k, "success": success, "n": len(steps), "steps": steps, "err": err, "target": env.task.target_entity}); print(f"[{a.task}] expert ep {k}: steps {len(steps)} success {success} err {err} ({time.time() - t0:.0f}s)", flush=True)
        try: env.close()
        except Exception: pass
    pickle.dump(eps, open(out / "expert_trace.pkl", "wb"))


if __name__ == "__main__":
    main()
