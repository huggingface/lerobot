#!/usr/bin/env python
"""G13 pass 1: OBSERVATION BANK. Deterministic re-execution of the existing G5 raw-G4d / G7 oracle episodes (policy behaviour untouched), storing at the same 797 frames (every 10th step) everything pass 2 needs to rebuild the policy's batch offline:
policy-input images (3 x 480x480 uint8; camera1 = front, camera2 = right, camera3 = wrist), normalised state, language tokens/mask, plus DIAGNOSTIC-ONLY simulator segmentation (robot = root body 'franka/', cards = card + holder bodies; rendered with the same cameras, verified pixel-identical to the policy images)
and geometry (EE, GT anchor, card positions / displacement, gripper state, phase). Segmentation is never a policy input."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g8_capture_audit import STATE  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402

KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"); CAMS = (2, 0, 3); BANK: list[dict] = []; IMGS: list[np.ndarray] = []; MASKS: list[np.ndarray] = []


def seg_masks(physics, task, card_names):
    model = physics.model; root_cache = {}
    def root(gid):
        if gid not in root_cache:
            b = int(model.geom_bodyid[gid]); name = model.body(b).name
            while b > 0: name = model.body(b).name; b = int(model.body_parentid[b])
            root_cache[gid] = model.body(int(model.geom_bodyid[gid])).name if False else name
        return root_cache[gid]
    out = np.zeros((3, 2, 480, 480), bool)
    for ci, cid in enumerate(CAMS):
        seg = physics.render(height=480, width=480, camera_id=cid, segmentation=True); ids, typ = seg[..., 0], seg[..., 1]
        for gid in np.unique(ids[typ == 5]):
            if gid < 0: continue
            r = root(int(gid)); m = (ids == gid) & (typ == 5)
            if r.startswith("franka/"): out[ci, 0] |= m
            elif r.startswith("card_holder") or any(r.startswith(n + "/") for n in card_names): out[ci, 1] |= m
    return out


@torch.no_grad()
def audit(policy, batch, t, orig_enc):
    if t % 10 != 0: return
    env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; ents = {n: e for n, e in task.entities.items() if type(e).__name__ == type(task.entities[tgt]).__name__}
    kp = {n: np.asarray(e.get_grasped_keypoints(physics)[-1], dtype=float) for n, e in ents.items()}; pos = {n: np.asarray(e.get_xpos(physics), dtype=float) for n, e in ents.items()}
    if "init" not in STATE or STATE["ep"] != G5.ORACLE_ENV["ep"]: STATE["init"] = {n: pos[n].copy() for n in pos}; STATE["ep"] = G5.ORACLE_ENV["ep"]
    imgs = np.stack([(batch[k][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8) for k in KEYS]); masks = seg_masks(physics, task, list(ents.keys()))
    robot = task.robot; ee = np.asarray(robot.get_end_effector_pos(physics), dtype=float); gm, gs = float(policy.state_grip_mean), float(policy.state_grip_std); st = policy._current_state(batch)[0].cpu().numpy(); disp = {n: float(np.linalg.norm(pos[n][:2] - STATE["init"][n][:2])) for n in pos}
    BANK.append({"ep": G5.ORACLE_ENV["ep"], "t": t, "phase": int(policy._v2_grasp_phase), "state_norm": st, "grip_raw": float(st[6] * gs + gm), "ee_robot": ee - base, "gt_anchor_robot": kp[tgt] - base, "target_name": tgt, "ee_gt_xy": float(np.linalg.norm(ee[:2] - kp[tgt][:2])), "ee_gt_z": float(ee[2] - kp[tgt][2]),
                 "n_displaced": int(sum(v > 0.015 for v in disp.values())), "target_displaced": disp[tgt] > 0.015, "any_card_displaced_cm": max(disp.values()) * 100, "cards_xyz_robot": {n: pos[n] - base for n in pos},
                 "min_ee_card_xy": float(min(np.linalg.norm(ee[:2] - pos[n][:2]) for n in pos)), "lang_tokens": batch["observation.language.tokens"][0].cpu().numpy(), "lang_mask": batch["observation.language.attention_mask"][0].cpu().numpy()})
    IMGS.append(imgs); MASKS.append(np.packbits(masks.reshape(-1)))


def main():
    model = sys.argv[sys.argv.index("--model") + 1]; out = Path(sys.argv[sys.argv.index("--out") + 1]); G5.AUDIT["fn"] = audit
    try: G5.main()
    finally:
        d = out / model; d.mkdir(parents=True, exist_ok=True); np.save(d / "imgs.npy", np.stack(IMGS)); np.save(d / "masks_packed.npy", np.stack(MASKS)); pickle.dump(BANK, open(d / "bank.pkl", "wb")); print("bank frames:", len(BANK))


if __name__ == "__main__":
    main()
