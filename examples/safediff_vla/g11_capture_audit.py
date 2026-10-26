#!/usr/bin/env python
"""G11-D capture-only audit (same deterministic re-execution of the existing G5 raw-G4d / G7 oracle episodes, same frames = every 10th step). The residual conditioner is attached but kept OFF during the real episode; inside the hook each frame queries the SAME frozen decoder twice:
mode V2 (residual OFF) and mode RES (residual ON, trained conditioner), for anchors A0 (G4d pred) | A1 GT | A2 nearest wrong card | A3..A8 GT +-5cm x/y/z (grounded feature = V2's own) and for the relative-error intervention S: EE-z state +-2cm with the GT target fixed (changes e_z by -+2cm; decoder state input only)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
import g11_residual as R  # noqa: E402
from g8_capture_audit import FixedHead, FixedFeat, STATE  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

D5 = 0.05; AUD: list[dict] = []; CKPT = "outputs/train/safediff_vla_g11/residual_conditioner.pt"


@torch.no_grad()
def audit(policy, batch, t, orig_enc):
    if t % 10 != 0: return
    dec = policy.decoder
    if getattr(dec, "residual", None) is None: res = R.install(policy); res.load_state_dict(torch.load(CKPT)); res.eval(); dec.residual_on = False
    env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; ents = {n: e for n, e in task.entities.items() if type(e).__name__ == type(task.entities[tgt]).__name__}
    kp = {n: np.asarray(e.get_grasped_keypoints(physics)[-1], dtype=float) for n, e in ents.items()}; pos = {n: np.asarray(e.get_xpos(physics), dtype=float) for n, e in ents.items()}
    if "init" not in STATE or STATE["ep"] != G5.ORACLE_ENV["ep"]: STATE["init"] = {n: pos[n].copy() for n in pos}; STATE["ep"] = G5.ORACLE_ENV["ep"]
    wrong = min((n for n in ents if n != tgt), key=lambda n: np.linalg.norm(kp[n][:2] - kp[tgt][:2])); gt = kp[tgt] - base; wr = kp[wrong] - base
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state()); saved_last = policy._v2_last_target_xyz; saved = (policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent)
    tok, pad, mod = orig_enc(batch); dev = tok.device; mean, std = policy.action_pos_mean, policy.action_pos_std
    if "g4d" not in STATE: STATE["g4d"] = G5.Grounder("g4d", policy)
    pred = STATE["g4d"].compute(tok, pad, mod, batch)["xyz_n"]; tq = policy.target_query_extractor(tok, pad, mod); own = policy.visual_grounding_cross_attn(tq, tok, pad, mod); nrm = lambda a: ((torch.as_tensor(a, dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0)
    conds = {"A0": pred, "A1": nrm(gt), "A2": nrm(wr), "A3": nrm(gt + [D5, 0, 0]), "A4": nrm(gt - [D5, 0, 0]), "A5": nrm(gt + [0, D5, 0]), "A6": nrm(gt - [0, D5, 0]), "A7": nrm(gt + [0, 0, D5]), "A8": nrm(gt - [0, 0, D5])}; out = {}; sz = float(policy.state_pos_std[2]); cur = policy._current_state(batch)
    try:
        policy._encode_multimodal_latent = lambda b: (tok, pad, mod); policy.visual_grounding_cross_attn = FixedFeat(own)
        for mode in ("V2", "RES"):
            dec.residual_on = mode == "RES"
            for k, a in conds.items():
                policy.target_point_head = FixedHead(a); act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, batch)[0][0]
                out[(mode, k)] = {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy(), "anchor_robot": (a[0] * std + mean).cpu().numpy()}
            for lab, sgn in (("Sp2", +1), ("Sm2", -1)):
                so = cur.clone(); so[:, 2] += sgn * 0.02 / sz; policy._current_state = lambda bb, s_=so: s_; policy.target_point_head = FixedHead(conds["A1"])
                try: act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, batch)[0][0]
                finally: del policy.__dict__["_current_state"]
                out[(mode, lab)] = {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}
    finally:
        dec.residual_on = False; policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent = saved; policy._v2_last_target_xyz = saved_last; torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
        if "_current_state" in policy.__dict__: del policy.__dict__["_current_state"]
    robot = task.robot; ee = np.asarray(robot.get_end_effector_pos(physics), dtype=float); cur7 = policy._current_state(batch)[0, :7].cpu().numpy(); disp = max(float(np.linalg.norm(pos[n][:2] - STATE["init"][n][:2])) for n in pos)
    AUD.append({"ep": G5.ORACLE_ENV["ep"], "t": t, "phase": int(policy._v2_grasp_phase), "ee_robot": ee - base, "state7": cur7, "any_card_displaced_cm": disp * 100, "ee_gt_xy": float(np.linalg.norm(ee[:2] - kp[tgt][:2])), "ee_gt_z": float(ee[2] - kp[tgt][2]), "target_name": tgt, "wrong_name": wrong, "cond": out})


def main():
    model = sys.argv[sys.argv.index("--model") + 1]; out = Path(sys.argv[sys.argv.index("--out") + 1]); G5.AUDIT["fn"] = audit
    try: G5.main()
    finally: pickle.dump(AUD, open(out / model / "audit11.pkl", "wb")); print("audit frames:", len(AUD))


if __name__ == "__main__":
    main()
