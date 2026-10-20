#!/usr/bin/env python
"""ORACLE-GROUNDED stage-wise execution diagnostic (NOT the learned policy, never merged into learned SR). One fresh OS
process per task x seed, using ONLY previously validated harnesses (privileged get_grasped_keypoints oracle in all;
expert-derived pose/waypoint for book/drink):
  poker, mahjong : g21_closed_loop --arm geo   (oracle keypoint + geometric approach, 3 cm latched close)
  drink          : g28_arm --arm C             (oracle keypoint + EXPERT-derived waypoint + expert pose, 3 cm close)
  book           : g29_arm --arm B             (oracle keypoint + EXPERT grasp pose, pose-completion (1 cm) close)
  fruit, toy     : g24_release_gate            (oracle keypoint + geometry + instruction-conditioned release gate)
"""
import shutil, subprocess, sys
from pathlib import Path

ROOT = Path("outputs/eval/safediff_final_B"); H = Path(__file__).parent
SPEC = {"select_poker": ("g21", "g21_closed_loop.py", ["--arm", "geo"]), "select_mahjong": ("g21", "g21_closed_loop.py", ["--arm", "geo"]),
        "select_drink": ("g28C", "g28_arm.py", ["--arm", "C"]), "select_book": ("g29B", "g29_arm.py", ["--arm", "B"]),
        "select_fruit": ("g24", "g24_release_gate.py", []), "select_toy": ("g24", "g24_release_gate.py", [])}
for task, (tag, script, extra) in SPEC.items():
    for seed in range(3000, 3020):
        out = ROOT / tag / f"seed{seed}"; done = out / f"{task}.done"
        if done.exists(): continue
        out.mkdir(parents=True, exist_ok=True)
        if tag == "g28C": shutil.copy("outputs/eval/safediff_vla_g28/expert_refs.json", out / "expert_refs.json")
        with open(out / f"{task}.log", "w") as f:
            rc = subprocess.run([sys.executable, str(H / script), "--task", task, *extra, "--seeds", str(seed), "--out", str(out)], stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0: done.write_text("ok")
        print(task, seed, "rc", rc, flush=True)
