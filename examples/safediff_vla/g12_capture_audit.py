#!/usr/bin/env python
"""G12 capture-only GRIPPER-STATE causal intervention audit. Same deterministic re-execution of the existing G5 raw-G4d / G7 oracle episodes, same 797 frames (every 10th step), policy behaviour untouched.
On the SAME observation (image, language, EE pose, rotation, GT target anchor A1, phase flag, V2's own grounded feature) only the gripper STATE input is changed (state convention: 0 = open, 1 = closed):
  C0       original state
  A_open / A_closed : decoder state vector only (path P1: decoder.state_encoder input, encoded dim 9)
  B_open / B_closed : every gripper-state representation we can change consistently = P1 + the VLM prefix state token (batch observation.state gripper dim; VLM latents AND V2's grounded feature recomputed from them) (P1+P2)
Not changed (stated limits): the images still show the real gripper (P3); the phase flag (P4) is the policy's own internal phase and is held fixed."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g8_capture_audit import FixedHead, FixedFeat, STATE  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402

AUD: list[dict] = []


@torch.no_grad()
def audit(policy, batch, t, orig_enc):
    if t % 10 != 0: return
    env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; ents = {n: e for n, e in task.entities.items() if type(e).__name__ == type(task.entities[tgt]).__name__}
    kp = {n: np.asarray(e.get_grasped_keypoints(physics)[-1], dtype=float) for n, e in ents.items()}; pos = {n: np.asarray(e.get_xpos(physics), dtype=float) for n, e in ents.items()}
    if "init" not in STATE or STATE["ep"] != G5.ORACLE_ENV["ep"]: STATE["init"] = {n: pos[n].copy() for n in pos}; STATE["ep"] = G5.ORACLE_ENV["ep"]
    gt = kp[tgt] - base; rng = (torch.get_rng_state(), torch.cuda.get_rng_state()); saved_last = policy._v2_last_target_xyz; saved = (policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent)
    tok, pad, mod = orig_enc(batch); dev = tok.device; mean, std = policy.action_pos_mean, policy.action_pos_std; tq = policy.target_query_extractor(tok, pad, mod); own = policy.visual_grounding_cross_attn(tq, tok, pad, mod)
    a_gt = ((torch.as_tensor(gt, dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0); gm, gs = float(policy.state_grip_mean), float(policy.state_grip_std); norm_g = lambda raw: (raw - gm) / gs; cur = policy._current_state(batch); orig_state = batch["observation.state"]; out = {}
    def run(key, feat, enc=None, b=None, state_override=None):
        policy.target_point_head = FixedHead(a_gt); policy.visual_grounding_cross_attn = FixedFeat(feat); policy._encode_multimodal_latent = enc if enc is not None else (lambda bb: (tok, pad, mod))
        if state_override is not None: policy._current_state = lambda bb: state_override
        try: act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, b if b is not None else batch)[0][0]
        finally:
            if "_current_state" in policy.__dict__: del policy.__dict__["_current_state"]
        out[key] = {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}
    try:
        run("C0", own)
        for lab, raw in (("open", 0.0), ("closed", 1.0)):
            so = cur.clone(); so[:, GRIPPER_INDEX_RAW] = norm_g(raw); run(f"A_{lab}", own, state_override=so)
            b2 = dict(batch); st2 = orig_state.clone(); (st2[:, 0] if st2.ndim > 2 else st2)[:, GRIPPER_INDEX_RAW] = norm_g(raw); b2["observation.state"] = st2; tok2, pad2, mod2 = orig_enc(b2); tq2 = policy.target_query_extractor(tok2, pad2, mod2); own2 = policy.visual_grounding_cross_attn(tq2, tok2, pad2, mod2)
            run(f"B_{lab}", own2, enc=lambda bb, T=(tok2, pad2, mod2): T, b=b2)
    finally:
        policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent = saved; policy._v2_last_target_xyz = saved_last; torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
        if "_current_state" in policy.__dict__: del policy.__dict__["_current_state"]
    robot = task.robot; ee = np.asarray(robot.get_end_effector_pos(physics), dtype=float); cur7 = cur[0, :7].cpu().numpy(); disp = max(float(np.linalg.norm(pos[n][:2] - STATE["init"][n][:2])) for n in pos)
    AUD.append({"ep": G5.ORACLE_ENV["ep"], "t": t, "phase": int(policy._v2_grasp_phase), "ee_robot": ee - base, "state7": cur7, "grip_raw": float(cur7[GRIPPER_INDEX_RAW] * gs + gm), "any_card_displaced_cm": disp * 100, "ee_gt_xy": float(np.linalg.norm(ee[:2] - kp[tgt][:2])), "ee_gt_z": float(ee[2] - kp[tgt][2]), "target_name": tgt, "gt_anchor_robot": gt, "cond": out})


def main():
    model = sys.argv[sys.argv.index("--model") + 1]; out = Path(sys.argv[sys.argv.index("--out") + 1]); G5.AUDIT["fn"] = audit
    try: G5.main()
    finally: pickle.dump(AUD, open(out / model / "audit12.pkl", "wb")); print("audit frames:", len(AUD))


if __name__ == "__main__":
    main()
