#!/usr/bin/env python
"""StagePlan-G TRAINING-scene generator (simulator). One env per process; for every seed (>= 100000, disjoint from eval seeds
3000-3019 and dev seeds 1000-1004) ONE reset renders the scene once and ALL valid selectable referents are enumerated, giving
same-scene / different-referent counterfactual supervision (privileged anchors are TRAINING labels only, never inference input).

Candidate instruction = the task's NATIVE config-manager `get_instruction` template evaluated for the candidate (no string
replace). A candidate is valid iff its native instruction (with the dataset's 'primitive: ' prefix) is VERBATIM one of the
training task strings. Adaptations of the native call (documented, validated by the same membership test):
  - templates that index `target_entity` (e.g. mahjong value/type) are tried with the entity name as given, then as a
    (head, tail) split on the first '_' ; the first form whose instruction is a training string is accepted;
  - templates that read `self.target_card` (poker) get it set from the entity name `<value>_of_<suit>`.
The native instruction of the ACTUAL target must equal the live `task_description` (per-scene consistency check).
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/sg_gen_scenes.py --task select_poker --seed0 100000 --n 50 --out outputs/stageplan/pilot
"""

from __future__ import annotations

import argparse
import glob
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402

from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402


def native_instruction(cm, name: str, vocab: set[str]) -> tuple[str | None, str]:
    cfg = cm.config["task"]; saved = (cfg.get("instructions"), getattr(cm, "target_entity", None), getattr(cm, "target_card", None))
    forms = [("name", name)]
    if "_" in name: forms.append(("split", tuple(name.split("_", 1))))
    m = re.match(r"^(.+)_of_(.+)$", name)
    try:
        for tag, X in forms:
            if hasattr(cm, "target_card"): cm.target_card = (m.group(1), m.group(2)) if m else None  # never inherit the real target's card
            cm.target_entity = X; cfg["instructions"] = None
            try: cm.get_instruction(target_entity=X, target_container=getattr(cm, "target_container", None), init_container=getattr(cm, "init_container", None))
            except Exception:  # noqa: BLE001
                continue
            ins = cfg.get("instructions")
            if ins:
                t = ins[0].strip(); t = t if t.startswith(("primitive:", "composite:")) else "primitive: " + t
                if t in vocab: return t, tag
    finally:
        cfg["instructions"], cm.target_entity = saved[0], saved[1]
        if saved[2] is not None: cm.target_card = saved[2]
    return None, "no_valid_form"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seed0", type=int, default=100000); ap.add_argument("--n", type=int, default=50); ap.add_argument("--out", default="outputs/stageplan/pilot"); a = ap.parse_args()
    assert a.seed0 >= 100000, "generation seeds must be disjoint from eval (3000-3019) / dev (1000-1004) seeds"
    vocab = set(pd.read_parquet(glob.glob("/home/user_lerobot/.cache/huggingface/hub/datasets--lerobot--vlabench_unified/snapshots/*/meta/tasks.parquet")[0]).reset_index()["task"])
    env = make_env(VLABenchEnv(task=a.task), n_envs=1, use_async_envs=False)[a.task][0]; scenes = []
    for s in range(a.seed0, a.seed0 + a.n):
        obs, _ = env.reset(seed=[s]); e = env.unwrapped.envs[0]; t = e._env.task; ph = e._env.physics; cm = t.config_manager; base = robot_base(e)
        px = {k: np.asarray(v)[0] for k, v in obs["pixels"].items()}; i480 = np.stack([px[k] for k in ("image", "second_image", "wrist_image")])
        i224 = np.stack([cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA) for x in i480])
        cands, non = [], []
        for name, ent in t.entities.items():
            if name == "table": continue
            try: kps = np.asarray(ent.get_grasped_keypoints(ph), float).reshape(-1, 3) - base; pos = np.asarray(ent.get_xpos(ph), float) - base
            except Exception:  # noqa: BLE001
                non.append({"name": name, "reason": "no keypoints"}); continue
            ins, how = native_instruction(cm, name, vocab)
            if ins is None: non.append({"name": name, "reason": "native instruction not a training string"}); continue
            cands.append({"name": name, "instruction": ins, "form": how, "kps": kps, "n_kp": len(kps), "anchor": kps[0], "obj_center": pos})
        base_ins = e.task_description; tgt = t.target_entity
        chk = {"target_in_candidates": any(c["name"] == tgt for c in cands), "native_matches_live": any(c["name"] == tgt and c["instruction"] == base_ins for c in cands),
               "unique_instructions": len({c["instruction"] for c in cands}) == len(cands), "n_cand": len(cands)}
        scenes.append({"task": a.task, "seed": s, "imgs480": i480, "imgs224": i224, "state": np.asarray(obs["agent_pos"])[0], "base_instruction": base_ins, "target": tgt, "candidates": cands, "non_referents": non, "base": base, "checks": chk})
        if (s - a.seed0) % 10 == 0: print(f"[{a.task}] seed {s} cands={len(cands)} checks={chk}", flush=True)
    env.close(); Path(a.out).mkdir(parents=True, exist_ok=True)
    pickle.dump(scenes, open(Path(a.out) / f"scenes_{a.task}.pkl", "wb")); print("saved", a.task, len(scenes))


if __name__ == "__main__":
    main()
