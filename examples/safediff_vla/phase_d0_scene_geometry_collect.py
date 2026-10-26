#!/usr/bin/env python
"""PHASE D0 step E (auxiliary, simulator-only): the dataset stores no object positions, so scene geometry is
sampled from the SAME scene generator (`env.reset` only -- no env.step, no policy, no rollout). Per scene:
candidate (same-type) object world positions, the instructed target, its referent, robot base. Seeds 5000+
(disjoint from every earlier audit). Used ONLY for language-free candidate-selection diagnostics."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402
from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    a = ap.parse_args(); rows = []
    for task in a.tasks:
        env = make_env(VLABenchEnv(task=task), n_envs=1, use_async_envs=False)[task][0]; t0 = time.time()
        for seed in range(5000, 5000 + a.n):
            try:
                set_seed(seed); env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
                impl = env.envs[0]
                while not hasattr(impl, "_env"):
                    impl = impl.env
                e = impl._env; ph, tk = e.physics, e.task
                tcls = type(tk.entities[tk.target_entity]).__name__
                cands = {n: np.asarray(en.get_xpos(ph), dtype=float).tolist() for n, en in tk.entities.items() if type(en).__name__ == tcls}
                rows.append({"task": task, "seed": seed, "target": tk.target_entity, "instruction": impl.task_description,
                             "cands": cands, "base": np.asarray(impl._robot_base_xyz, dtype=float).tolist(),
                             "ee": np.asarray(tk.robot.get_end_effector_pos(ph), dtype=float).tolist()})
            except Exception as ex:  # noqa: BLE001
                print("skip", task, seed, repr(ex)[:80], flush=True)
        print(f"{task}: {sum(r['task'] == task for r in rows)} scenes in {time.time() - t0:.0f}s", flush=True)
        env.close()
        (OUT / f"sim_scenes_{task}.json").write_text(json.dumps([r for r in rows if r["task"] == task]))


if __name__ == "__main__":
    main()
