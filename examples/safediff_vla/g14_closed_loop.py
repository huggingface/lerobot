#!/usr/bin/env python
"""G14 closed-loop VISUAL FEEDBACK-LOOP diagnostic (PRIVILEGED, diagnostic only; not a method). Raw G4d policy, seeds 3000-3009, everything unchanged except the WRIST camera image of the policy batch:
 trigger (every env step): physical gripper CLOSED (observation.state gripper > 0.5) AND EE 3D distance to the GT grasp keypoint > 6 cm  (GT distance = privileged gating, single threshold)
 --g14-mode log    : no intervention (logging only; must reproduce the G5 raw-G4d rollouts exactly)
 --g14-mode open   : replace the wrist ROBOT/gripper pixels by those of a DONOR frame whose gripper is visibly OPEN, matched on EE pose (<=5cm) and target-relative geometry (<=10cm) (G13 swapR_diff_wrist); scene/card/table pixels untouched; the state gripper bit stays CLOSED (state says closed, image looks open)
 --g14-mode closed : CONTROL, same procedure with a donor whose gripper is CLOSED (procedure-artefact control)
If no donor exists the frame is left unchanged. Donor bank = the G13 observation bank (wrist images + robot segmentation); fill for uncovered receiver robot pixels = same-camera dataset-mean image (the G13 rule). Robot segmentation (simulator) is diagnostic-only.
At every replan the hook also records the chunk actually planned and, for intervened replans, the counterfactual chunk the UNPATCHED image would have produced (decoder + G4d anchor re-run on the original batch, all policy state restored)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

MODE = sys.argv[sys.argv.index("--g14-mode") + 1]; del sys.argv[sys.argv.index("--g14-mode"):sys.argv.index("--g14-mode") + 2]
sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

BANK = Path("outputs/eval/safediff_vla_g13_bank"); MEAN = np.load("outputs/eval/safediff_vla_g13_audit/mean_images.npy")[2]; CAM = "observation.images.camera3"; FAR = 0.06
DB: dict = {}; LOG: list[dict] = []; EVENTS: list[dict] = []; ST = {"ep": -1, "t": 0, "pending": None}; ROOT: dict = {}


def load_donors():
    ims, mks, info = [], [], []
    for src in ("g4d", "oracle"):
        b = pickle.load(open(BANK / src / "bank.pkl", "rb")); im = np.load(BANK / src / "imgs.npy", mmap_mode="r"); mk = np.load(BANK / src / "masks_packed.npy")
        for i, e in enumerate(b): ims.append(np.asarray(im[i][2])); mks.append(np.unpackbits(mk[i])[:3 * 2 * 480 * 480].reshape(3, 2, 480, 480)[2, 0].astype(bool)); info.append((e["grip_raw"], e["ee_robot"], e["gt_anchor_robot"] - e["ee_robot"], src, e["ep"], e["t"]))
    DB.update({"im": ims, "mk": mks, "grip": np.array([x[0] for x in info]), "ee": np.array([x[1] for x in info]), "ev": np.array([x[2] for x in info]), "info": info})


def wrist_robot_mask(physics):
    model = physics.model; seg = physics.render(height=480, width=480, camera_id=3, segmentation=True); ids, typ = seg[..., 0], seg[..., 1]; out = np.zeros((480, 480), bool)
    for gid in np.unique(ids[typ == 5]):
        if gid < 0: continue
        if int(gid) not in ROOT:
            b = int(model.geom_bodyid[gid]); name = model.body(b).name
            while b > 0: name = model.body(b).name; b = int(model.body_parentid[b])
            ROOT[int(gid)] = name
        if ROOT[int(gid)].startswith("franka/"): out |= (ids == gid) & (typ == 5)
    return out


def find_donor(ee, ev, want_closed):
    pool = (DB["grip"] > 0.5) if want_closed else (DB["grip"] <= 0.5); d_ee = np.linalg.norm(DB["ee"] - ee, axis=1); d_ev = np.linalg.norm(DB["ev"] - ev, axis=1); ok = pool & (d_ee <= 0.05) & (d_ev <= 0.10)
    return int(np.argmin(np.where(ok, d_ee + 0.5 * d_ev, 1e9))) if ok.any() else -1


def chunk_np(policy, act): return {"xyz": (act[:, :3] * policy.action_pos_std + policy.action_pos_mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}


@torch.no_grad()
def pre(policy, batch):
    env = G5.ORACLE_ENV["env"]
    if ST["ep"] != G5.ORACLE_ENV["ep"]: ST.update(ep=G5.ORACLE_ENV["ep"], t=0)
    physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; kp = np.asarray(task.entities[tgt].get_grasped_keypoints(physics)[-1], dtype=float); ee_w = np.asarray(task.robot.get_end_effector_pos(physics), dtype=float)
    gm, gs = float(policy.state_grip_mean), float(policy.state_grip_std); grip = float(policy._current_state(batch)[0, 6] * gs + gm); d3 = float(np.linalg.norm(ee_w - kp)); ee = ee_w - base; ev = (kp - base) - ee; trig = MODE != "log" and grip > 0.5 and d3 > FAR
    rec = {"ep": ST["ep"], "t": ST["t"], "grip": grip, "d3": d3, "dxy": float(np.linalg.norm(ee_w[:2] - kp[:2])), "ee": ee, "trig": bool(trig), "donor": -1, "patched": False, "chunk": None, "raw_chunk": None}; ST["t"] += 1; CHUNK = G5.CHUNK; CHUNK["last"] = None; ST["pending"] = None
    if trig:
        j = find_donor(ee, ev, MODE == "closed"); rec["donor"] = j
        if j >= 0:
            orig = (batch[CAM][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8); rm = wrist_robot_mask(physics); x = orig.copy(); left = rm & ~DB["mk"][j]; x[left] = MEAN[left]; x[DB["mk"][j]] = DB["im"][j][DB["mk"][j]]
            ST["pending"] = {"orig_tensor": batch[CAM].clone(), "orig_img": orig, "patched_img": x, "donor_img": DB["im"][j]}; batch[CAM] = torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(batch[CAM].device); rec["patched"] = True
    LOG.append(rec)


@torch.no_grad()
def post(policy, batch, t, orig_enc):
    rec = LOG[-1]; ch = G5.CHUNK["last"]
    if ch is not None:
        rec["chunk"] = chunk_np(policy, ch)
        if rec["patched"]:
            p = ST["pending"]; g = G5.GROUNDER["g"]; saved_cache = g.cache; saved_last = policy._v2_last_target_xyz; rng = (torch.get_rng_state(), torch.cuda.get_rng_state()); patched = batch[CAM]
            try:
                batch[CAM] = p["orig_tensor"]; g.cache = None; act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, batch)[0][0]; rec["raw_chunk"] = chunk_np(policy, act)
            finally: batch[CAM] = patched; g.cache = saved_cache; policy._v2_last_target_xyz = saved_last; torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
            if sum(1 for e in EVENTS if e["ep"] == rec["ep"]) < 3: EVENTS.append({"ep": rec["ep"], "t": rec["t"], "donor": rec["donor"], "orig": p["orig_img"], "patched": p["patched_img"], "donor_img": p["donor_img"]})


def main():
    out = Path(sys.argv[sys.argv.index("--out") + 1]); load_donors(); G5.PRE["fn"] = pre; G5.AUDIT["fn"] = post
    try: G5.main()
    finally: pickle.dump({"log": LOG, "events": EVENTS, "mode": MODE, "donor_info": DB["info"]}, open(out / "g4d" / "g14_log.pkl", "wb")); print("g14 log steps:", len(LOG), "patched:", sum(r["patched"] for r in LOG))


if __name__ == "__main__":
    main()
