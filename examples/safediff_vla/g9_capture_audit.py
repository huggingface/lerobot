#!/usr/bin/env python
"""G9-A capture-only audit: identical re-execution of the existing G5 raw-G4d / G7 oracle episodes as G8 (same seeds, same frames: every 10th step), but each frame queries the frozen V2 decoder at TARGET-BRANCH GAIN gamma in {1.0, 4.0} for anchor conditions A0..A8
(A0 G4d prediction, A1 GT keypoint, A2 nearest wrong card, A3/A4 GT +-5cm x, A5/A6 +-5cm y, A7/A8 +-5cm z; grounded_target_feature = V2's own, unchanged).
gamma scales ONLY the target_xyz branch's additive contribution to the decoder queries: queries += gamma * target_xyz_encoder(target_xyz). Coordinates are not scaled; gripper/phase/feature branches untouched; no weights change. gamma is restored to 1.0 after every query (the real episode is unaffected)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g8_capture_audit import FixedHead, FixedFeat, STATE  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

D5 = 0.05; GAMMAS = (1.0, 4.0); AUD: list[dict] = []


class GammaWrap(nn.Module):
    def __init__(self, orig): super().__init__(); self.orig = orig; self.gamma = 1.0
    def forward(self, x): return self.gamma * self.orig(x)


@torch.no_grad()
def audit(policy, batch, t, orig_enc):
    if t % 10 != 0: return
    env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; ents = {n: e for n, e in task.entities.items() if type(e).__name__ == type(task.entities[tgt]).__name__}
    kp = {n: np.asarray(e.get_grasped_keypoints(physics)[-1], dtype=float) for n, e in ents.items()}; pos = {n: np.asarray(e.get_xpos(physics), dtype=float) for n, e in ents.items()}
    if "init" not in STATE or STATE["ep"] != G5.ORACLE_ENV["ep"]: STATE["init"] = {n: pos[n].copy() for n in pos}; STATE["ep"] = G5.ORACLE_ENV["ep"]
    wrong_name = min((n for n in ents if n != tgt), key=lambda n: np.linalg.norm(kp[n][:2] - kp[tgt][:2])); gt = kp[tgt] - base; wr = kp[wrong_name] - base
    if not isinstance(policy.decoder.target_xyz_encoder, GammaWrap): policy.decoder.target_xyz_encoder = GammaWrap(policy.decoder.target_xyz_encoder)
    gw = policy.decoder.target_xyz_encoder; rng = (torch.get_rng_state(), torch.cuda.get_rng_state()); saved_last = policy._v2_last_target_xyz; saved = (policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent)
    tok, pad, mod = orig_enc(batch); dev = tok.device; mean, std = policy.action_pos_mean, policy.action_pos_std
    if "g4d" not in STATE: STATE["g4d"] = G5.Grounder("g4d", policy)
    pred = STATE["g4d"].compute(tok, pad, mod, batch)["xyz_n"]; tq = policy.target_query_extractor(tok, pad, mod); own = policy.visual_grounding_cross_attn(tq, tok, pad, mod); nrm = lambda a: ((torch.as_tensor(a, dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0)
    conds = {"A0": pred, "A1": nrm(gt), "A2": nrm(wr), "A3": nrm(gt + [D5, 0, 0]), "A4": nrm(gt - [D5, 0, 0]), "A5": nrm(gt + [0, D5, 0]), "A6": nrm(gt - [0, D5, 0]), "A7": nrm(gt + [0, 0, D5]), "A8": nrm(gt - [0, 0, D5])}; out = {}
    try:
        policy._encode_multimodal_latent = lambda b: (tok, pad, mod); policy.visual_grounding_cross_attn = FixedFeat(own)
        for g in GAMMAS:
            gw.gamma = g
            for k, a in conds.items():
                policy.target_point_head = FixedHead(a); act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, batch)[0][0]
                out[(g, k)] = {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy(), "anchor_robot": (a[0] * std + mean).cpu().numpy()}
    finally:
        gw.gamma = 1.0; policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent = saved; policy._v2_last_target_xyz = saved_last; torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    robot = task.robot; ee = np.asarray(robot.get_end_effector_pos(physics), dtype=float); cur = policy._current_state(batch)[0, :7].cpu().numpy(); disp = max(float(np.linalg.norm(pos[n][:2] - STATE["init"][n][:2])) for n in pos)
    AUD.append({"ep": G5.ORACLE_ENV["ep"], "t": t, "phase": int(policy._v2_grasp_phase), "ee_robot": ee - base, "state7": cur, "any_card_displaced_cm": disp * 100, "ee_gt_xy": float(np.linalg.norm(ee[:2] - kp[tgt][:2])), "ee_gt_z": float(ee[2] - kp[tgt][2]), "target_name": tgt, "wrong_name": wrong_name, "cond": out})


def main():
    model = sys.argv[sys.argv.index("--model") + 1]; out = Path(sys.argv[sys.argv.index("--out") + 1]); G5.AUDIT["fn"] = audit
    try: G5.main()
    finally: pickle.dump(AUD, open(out / model / "audit9.pkl", "wb")); print("audit frames:", len(AUD))


if __name__ == "__main__":
    main()
