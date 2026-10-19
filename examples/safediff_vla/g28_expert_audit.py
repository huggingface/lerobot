#!/usr/bin/env python
"""G28 STEP 1: VLABench scripted-EXPERT view audit for select_book / select_drink (successful episodes). Per env step: EE pose, target keypoint (world), target projection + visible-pixel count in the WRIST and FRONT cameras, finger command; RGB frames (wrist+front, 240px) every 5 steps for the 40 steps before the close. Expert scenes are VLABench's own random scenes."""
from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import g25_common as C25  # noqa: E402
import g28_common as C28  # noqa: E402
from VLABench.robots import Franka  # noqa: E402,F401
from VLABench.envs import load_env  # noqa: E402
from VLABench.tasks import *  # noqa: E402,F401,F403


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default="outputs/eval/safediff_vla_g28"); a = ap.parse_args(); out = Path(a.out) / a.task / "expert"; out.mkdir(parents=True, exist_ok=True); eps = []
    for k in range(a.n):
        t0 = time.time(); env = load_env(a.task, robot="franka", eval=False); env.reset(); steps = []; frames = {}; orig = env.step; state = {"close": None}
        def step(action, _o=orig):
            ts = _o(action); s = C25.snapshot(env); ph = env.physics; kp = np.array(env.task.entities[env.task.target_entity].get_grasped_keypoints(ph)[0], float, copy=True); s["kp"] = kp; s["grip_cmd"] = float(np.mean(np.asarray(action)[-2:])); s["t"] = len(steps)
            for nm, cid in (("wrist", C28.WRIST), ("front", C28.FRONT)): s[f"{nm}_uv"] = C28.cam_proj(ph, cid, kp); s[f"{nm}_vis"] = C28.target_pixels(env, cid)[0]
            if state["close"] is None and s["grip_cmd"] < 0.03: state["close"] = s["t"]
            if len(steps) % 5 == 0: frames[s["t"]] = (C28.frame(env, C28.WRIST), C28.frame(env, C28.FRONT))
            steps.append(s); return ts
        env.step = step; success = False; err = None; EP = None
        try:
            for skill in env.get_expert_skill_sequence():
                obs, wp, stage_ok, task_ok = skill(env)
                if task_ok: success = True; break
        except Exception as ex: err = repr(ex)[:200]
        tc = state["close"]; keep = {t: f for t, f in frames.items() if tc is not None and tc - 40 <= t <= tc + 5}
        eps.append({"k": k, "success": success, "n": len(steps), "t_close": tc, "steps": steps, "frames": keep, "target": env.task.target_entity, "err": err}); print(f"[{a.task}] expert ep {k}: steps {len(steps)} close {tc} success {success} err {err} ({time.time() - t0:.0f}s)", flush=True)
        try: env.close()
        except Exception: pass
    pickle.dump(eps, open(out / "expert_audit.pkl", "wb"))


if __name__ == "__main__":
    main()
