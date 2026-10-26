#!/usr/bin/env python
"""PHASE G1-A: counterfactual referent dataset generator (simulator `env.reset` only -- no env.step, no policy, no rollout).

One SCENE = one frozen observation (the exact RGB x3 cameras + robot state the policy receives from the LeRobot VLABench
wrapper) + EVERY same-type candidate object as a valid referent. A sample = (scene, referent): same pixels/state, the
instruction names a different object, and the label moves with it.

LABEL (audited, see G1 report): anchor(c) = VLABench's own expert grasp keypoint of candidate c
(`entity.get_grasped_keypoints(physics)[-1]`, the point `SkillLib.pick` drives the gripper to before closing -- the dataset's
"commanded xyz at the open->close transition"), expressed in the robot frame (minus robot base) exactly like the dataset.
It is NOT the object origin (poker: +2 cm in z) and NOT copied from another referent's demonstration.
Validity: `find_keypoint_and_prepare_grasp` (the expert's own collision check, gripper vs scene point cloud) is run for every
candidate; a candidate for which it would fall back to its default "cant find a valid grasp point" is flagged invalid.
Per-camera 8x8 token-cell coverage of every candidate (simulator segmentation) is stored for TRAINING-ONLY referent supervision.
Seeds 10000+ (disjoint from every earlier audit).
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/phase_g1a_counterfactual_dataset.py --task select_poker --n 40
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase_g05_clean_grounding_collect import cell_fractions, project, region_masks  # noqa: E402
from phase_g06_counterfactual_instruction import fill, make_template  # noqa: E402
from phase_e0_instruction_plumbing_check import TASKS_PARQUET  # noqa: E402

from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

OUT = Path("outputs/data/safediff_vla_g1_counterfactual")
SEED0 = 10000


def expert_validity(env_vl, entity) -> tuple[bool, bool]:
    """Run the expert's own keypoint/collision check (consumes RNG -> state saved/restored). Returns (found_valid, used_default_fallback)."""
    from VLABench.utils.utils import find_keypoint_and_prepare_grasp  # noqa: PLC0415

    st_py, st_np = random.getstate(), np.random.get_state()
    try:
        kp, prep, quat = find_keypoint_and_prepare_grasp(env_vl, entity, [np.array([-np.pi, 0.0, 0.0])], specific_keypoint_id=0)
        fallback = bool(np.allclose(prep, np.asarray(kp) + np.array([0, -0.1, 0])))
        return (not fallback), fallback
    except Exception:  # noqa: BLE001
        return False, True
    finally:
        random.setstate(st_py); np.random.set_state(st_np)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--start", type=int, default=0); a = ap.parse_args()
    train_strings = set(pd.read_parquet(TASKS_PARQUET).index)
    out = OUT / a.task; out.mkdir(parents=True, exist_ok=True)
    env = make_env(VLABenchEnv(task=a.task), n_envs=1, use_async_envs=False)[a.task][0]
    index = []; t0 = time.time()
    for i in range(a.start, a.start + a.n):
        seed = SEED0 + i
        try:
            set_seed(seed); raw, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
            impl = env.envs[0]
            while not hasattr(impl, "_env"):
                impl = impl.env
            e = impl._env; ph, tk = e.physics, e.task
            tname = tk.target_entity; tcls = type(tk.entities[tname]).__name__
            base = np.asarray(impl._robot_base_xyz, dtype=float)
            wrapper_text = list(env.call("task_description"))[0]
            tpl, form = make_template(wrapper_text, tname); assert fill(tpl, form, tname) == wrapper_text
            cands = [n for n, en in tk.entities.items() if type(en).__name__ == tcls]
            masks = region_masks(impl)
            rec = {"scene_id": f"{a.task}_{seed}", "task": a.task, "seed": seed, "original_instruction": wrapper_text, "original_target": tname,
                   "robot_base_world": base.tolist(), "candidates": []}
            arrays = {"cellfrac": np.zeros((len(cands), 3, 64), np.float32)}
            for ci, n in enumerate(cands):
                en = tk.entities[n]; kps = [np.asarray(k, dtype=float) for k in en.get_grasped_keypoints(ph)]
                valid, fb = expert_validity(e, en)
                xp = np.asarray(en.get_xpos(ph), dtype=float); text = fill(tpl, form, n)
                arrays["cellfrac"][ci] = np.stack([cell_fractions(masks["per_camera"][c][n]) for c in range(3)])
                rec["candidates"].append({
                    "name": n, "instruction": text, "instruction_in_training_set": text in train_strings,
                    "object_pose_world": {"xpos": xp.tolist(), "xquat": np.asarray(en.get_xquat(ph), dtype=float).tolist() if hasattr(en, "get_xquat") else None},
                    "n_grasp_keypoints": len(kps), "keypoint_world": kps[-1].tolist(), "anchor_robot": (kps[-1] - base).tolist(),
                    "expert_collision_check_valid": valid, "expert_used_default_fallback": fb,
                    "visible_px_per_camera": [int(masks["per_camera"][c][n].sum()) for c in range(3)],
                    "proj_px_per_camera": [project(impl, impl._camera_order[c], kps[-1]) for c in range(3)]})
            px = raw["pixels"]
            arrays["rgb"] = np.stack([np.asarray(px[k])[0] for k in ("image", "second_image", "wrist_image")]).astype(np.uint8)  # [3,480,480,3]
            arrays["state"] = np.asarray(raw["agent_pos"])[0].astype(np.float32)
            np.savez_compressed(out / f"{rec['scene_id']}.npz", **arrays)
            (out / f"{rec['scene_id']}.json").write_text(json.dumps(rec))
            index.append(rec["scene_id"])
        except Exception as ex:  # noqa: BLE001
            print("skip", seed, repr(ex)[:100], flush=True)
        if (i + 1) % 10 == 0:
            print(f"{a.task}: {len(index)} scenes ({time.time() - t0:.0f}s)", flush=True)
    (out / f"index_{a.start}_{a.start + a.n}.json").write_text(json.dumps(index)); env.close()


if __name__ == "__main__":
    main()
