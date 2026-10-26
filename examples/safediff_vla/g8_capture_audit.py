#!/usr/bin/env python
"""G8 capture-only audit. Re-executes the EXISTING G5 raw-G4d and G7 oracle episodes (same seeds 3000-3009, identical code path; the audit hook runs AFTER the real action has been chosen and restores all policy state / RNG),
and at every 10th step queries the frozen V2 decoder on the SAME observation with counterfactual conditioning:
  anchor conditions (target_xyz only; grounded_target_feature = V2's own):  A0 current G4d prediction | A1 GT keypoint | A2 nearest wrong card | A3/A4 GT +-5cm x | A5/A6 GT +-5cm y | A7/A8 GT +-5cm z
  feature-pathway conditions (extension; the decoder ALSO receives V2's grounded_target_feature):  F1 GT anchor + feature zeroed | F2 GT anchor + oracle-attention feature (GT card cells) | F3 wrong anchor + oracle-attention feature (wrong card cells) | F4 GT anchor + wrong-card feature | F5 wrong anchor + GT-card feature
No weights change; GT is used only to build counterfactual conditions and strata."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g4b_geometry import CAL, proj  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

D5 = 0.05; AUD: list[dict] = []; STATE: dict = {}


class FixedHead(nn.Module):
    def __init__(self, v): super().__init__(); self.v = v
    def forward(self, x): return self.v.to(device=x.device, dtype=x.dtype)


class FixedFeat(nn.Module):
    def __init__(self, v): super().__init__(); self.v = v
    def forward(self, *a, **k): return self.v


def cell_tokens(world_xyz):
    idx = []
    for cam in (0, 1):
        u, v = proj(CAL[cam], world_xyz); idx.append(cam * 64 + int(np.clip(v // 60, 0, 7)) * 8 + int(np.clip(u // 60, 0, 7)))
    return idx


@torch.no_grad()
def oracle_feature(policy, tok, pad, mod, card_world):
    tq = policy.target_query_extractor(tok, pad, mod); valid = pad.to(torch.bool).clone(); keep = torch.zeros_like(valid)
    for i in cell_tokens(card_world): keep[:, i] = True
    keep &= valid; grounded, _ = policy.visual_grounding_cross_attn.attn(query=tq.unsqueeze(1), key=tok, value=tok, key_padding_mask=~keep, need_weights=False); return grounded.squeeze(1)


@torch.no_grad()
def audit(policy, batch, t, orig_enc):
    if t % 10 != 0: return
    env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; ents = {n: e for n, e in task.entities.items() if type(e).__name__ == type(task.entities[tgt]).__name__}
    kp = {n: np.asarray(e.get_grasped_keypoints(physics)[-1], dtype=float) for n, e in ents.items()}; pos = {n: np.asarray(e.get_xpos(physics), dtype=float) for n, e in ents.items()}
    if "init" not in STATE or STATE["ep"] != G5.ORACLE_ENV["ep"]: STATE["init"] = {n: pos[n].copy() for n in pos}; STATE["ep"] = G5.ORACLE_ENV["ep"]
    wrong_name = min((n for n in ents if n != tgt), key=lambda n: np.linalg.norm(kp[n][:2] - kp[tgt][:2])); gt = kp[tgt] - base; wr = kp[wrong_name] - base
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state()); saved_last = policy._v2_last_target_xyz; saved = (policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent)
    tok, pad, mod = orig_enc(batch); dev = tok.device; mean, std = policy.action_pos_mean, policy.action_pos_std
    if "g4d" not in STATE: STATE["g4d"] = G5.Grounder("g4d", policy)
    pred = STATE["g4d"].compute(tok, pad, mod, batch)["xyz_n"]; tq = policy.target_query_extractor(tok, pad, mod); own = policy.visual_grounding_cross_attn(tq, tok, pad, mod); zero = torch.zeros_like(own)
    f_gt = oracle_feature(policy, tok, pad, mod, pos[tgt]); f_wr = oracle_feature(policy, tok, pad, mod, pos[wrong_name]); nrm = lambda a: ((torch.as_tensor(a, dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0)
    conds = {"A0": (pred, own), "A1": (nrm(gt), own), "A2": (nrm(wr), own), "A3": (nrm(gt + [D5, 0, 0]), own), "A4": (nrm(gt - [D5, 0, 0]), own), "A5": (nrm(gt + [0, D5, 0]), own), "A6": (nrm(gt - [0, D5, 0]), own), "A7": (nrm(gt + [0, 0, D5]), own), "A8": (nrm(gt - [0, 0, D5]), own),
             "F1": (nrm(gt), zero), "F2": (nrm(gt), f_gt), "F3": (nrm(wr), f_wr), "F4": (nrm(gt), f_wr), "F5": (nrm(wr), f_gt)}; out = {}
    try:
        policy._encode_multimodal_latent = lambda b: (tok, pad, mod)
        for k, (a, f) in conds.items():
            policy.target_point_head = FixedHead(a); policy.visual_grounding_cross_attn = FixedFeat(f); act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, batch)[0][0]
            out[k] = {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy(), "anchor_robot": (a[0] * std + mean).cpu().numpy()}
    finally:
        policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent = saved; policy._v2_last_target_xyz = saved_last; torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    robot = task.robot; ee = np.asarray(robot.get_end_effector_pos(physics), dtype=float); cur = policy._current_state(batch)[0, :7].cpu().numpy()
    disp = max(float(np.linalg.norm(pos[n][:2] - STATE["init"][n][:2])) for n in pos)
    AUD.append({"ep": G5.ORACLE_ENV["ep"], "t": t, "phase": int(policy._v2_grasp_phase), "ee_world": ee, "ee_robot": ee - base, "gt_world_kp": kp[tgt], "target_pos": pos[tgt], "wrong_name": wrong_name, "target_name": tgt, "state7": cur, "any_card_displaced_cm": disp * 100,
                "target_displaced_cm": float(np.linalg.norm(pos[tgt][:2] - STATE["init"][tgt][:2])) * 100, "is_grasped": bool(ents[tgt].is_grasped(physics, robot)), "ee_gt_xy": float(np.linalg.norm(ee[:2] - kp[tgt][:2])), "ee_gt_z": float(ee[2] - kp[tgt][2]),
                "ee_wrong_xy": float(np.linalg.norm(ee[:2] - kp[wrong_name][:2])), "cond": out})


def main():
    model = sys.argv[sys.argv.index("--model") + 1]; out = Path(sys.argv[sys.argv.index("--out") + 1]); G5.AUDIT["fn"] = audit
    try: G5.main()
    finally: out.mkdir(parents=True, exist_ok=True); pickle.dump(AUD, open(out / model / "audit.pkl", "wb")); print("audit frames:", len(AUD))


if __name__ == "__main__":
    main()
