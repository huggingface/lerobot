#!/usr/bin/env python
"""G10-Z capture-only audit (same deterministic re-execution of the existing G5 raw-G4d / G7 oracle episodes as G8/G9, same frames = every 10th step; policy behaviour untouched, state/RNG restored).
Reference (Z0) = V2 decoder with the GT anchor (A1) and V2's own grounded feature. Conditions on the SAME observation:
  target branch ablations : Z1 target_xyz branch contribution = 0 | Z2a target z := poker-demo mean z (0.0695) | Z2p/Z2m target z +-0.5mm (in-distribution, ~2.5 sigma) | Z3 grounded_target_feature = 0 | Z4 both zeroed
  visual/text memory ablations (decoder cross-attention memory only; anchor and grounded feature held at Z0): V1 all IMAGE tokens masked | V2 all TEXT tokens masked
  state perturbations (observed range, +-1cm and +-2cm of EE z; normalised state dim 2): S1 decoder state input only | S2 observation.state of the whole batch (decoder input AND the VLM prefix state token / latents recomputed)
  branch norms recorded at Z0 (query components and per-layer self-attn / cross-attn / FFN output norms).
No weights change, no gripper logic change."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g8_capture_audit import FixedHead, FixedFeat, STATE  # noqa: E402
from g9_capture_audit import GammaWrap  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.utils import IMAGE_MODALITY, TEXT_MODALITY  # noqa: E402

AUD: list[dict] = []; POKER_Z = 0.0695


def tok_norm(x): return float(x.float().norm(dim=-1).mean())


@torch.no_grad()
def audit(policy, batch, t, orig_enc):
    if t % 10 != 0: return
    env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; base = robot_base(env); tgt = task.target_entity; ents = {n: e for n, e in task.entities.items() if type(e).__name__ == type(task.entities[tgt]).__name__}
    kp = {n: np.asarray(e.get_grasped_keypoints(physics)[-1], dtype=float) for n, e in ents.items()}; pos = {n: np.asarray(e.get_xpos(physics), dtype=float) for n, e in ents.items()}
    if "init" not in STATE or STATE["ep"] != G5.ORACLE_ENV["ep"]: STATE["init"] = {n: pos[n].copy() for n in pos}; STATE["ep"] = G5.ORACLE_ENV["ep"]
    gt = kp[tgt] - base; dec = policy.decoder
    if not isinstance(dec.target_xyz_encoder, GammaWrap): dec.target_xyz_encoder = GammaWrap(dec.target_xyz_encoder)
    gw = dec.target_xyz_encoder; rng = (torch.get_rng_state(), torch.cuda.get_rng_state()); saved_last = policy._v2_last_target_xyz; saved = (policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent)
    tok, pad, mod = orig_enc(batch); dev = tok.device; mean, std = policy.action_pos_mean, policy.action_pos_std; tq = policy.target_query_extractor(tok, pad, mod); own = policy.visual_grounding_cross_attn(tq, tok, pad, mod); zero = torch.zeros_like(own)
    nrm = lambda a: ((torch.as_tensor(a, dtype=std.dtype, device=dev) - mean) / std).unsqueeze(0); a_gt = nrm(gt); sz = float(policy.state_pos_std[2]); orig_state = batch["observation.state"]; out = {}; norms = {}

    def run(key, anchor, feat, enc=None, b=None, gamma=1.0, state_override=None):
        policy.target_point_head = FixedHead(anchor); policy.visual_grounding_cross_attn = FixedFeat(feat); policy._encode_multimodal_latent = enc if enc is not None else (lambda bb: (tok, pad, mod)); gw.gamma = gamma
        if state_override is not None: policy._current_state = lambda bb: state_override
        try: act = SafeDiffVLAPolicy._plan_temporal_decoder(policy, b if b is not None else batch)[0][0]
        finally:
            gw.gamma = 1.0
            if "_current_state" in policy.__dict__: del policy.__dict__["_current_state"]
        out[key] = {"xyz": (act[:, :3] * std + mean).cpu().numpy(), "grip": (act[:, 6] * policy.action_grip_std + policy.action_grip_mean).cpu().numpy()}

    try:
        hooks, store = [], {}
        def hk(name, sel=None): return lambda m, i, o: store.__setitem__(name, tok_norm((o[0] if isinstance(o, tuple) else o) if sel is None else sel(o)))
        hooks += [dec.state_encoder.register_forward_hook(hk("q_state")), dec.target_xyz_encoder.register_forward_hook(hk("q_target")), dec.grounded_feature_encoder.register_forward_hook(hk("q_feature")), dec.phase_encoder.register_forward_hook(hk("q_phase"))]
        for li, layer in enumerate(dec.transformer.layers): hooks += [layer.self_attn.register_forward_hook(hk(f"L{li}_selfattn")), layer.multihead_attn.register_forward_hook(hk(f"L{li}_crossattn")), layer.linear2.register_forward_hook(hk(f"L{li}_ffn"))]
        hooks.append(dec.action_head.register_forward_pre_hook(lambda m, i: store.__setitem__("final_hidden", tok_norm(i[0]))))
        run("Z0", a_gt, own)
        for h in hooks: h.remove()
        store["q_base"] = tok_norm(dec.action_queries + dec.horizon_position_embedding.to(dev)); norms = dict(store)
        run("Z1", a_gt, own, gamma=0.0); zc = gt.copy(); zc[2] = POKER_Z; run("Z2a", nrm(zc), own); zp = gt.copy(); zp[2] += 0.0005; zm = gt.copy(); zm[2] -= 0.0005; run("Z2p", nrm(zp), own); run("Z2m", nrm(zm), own)
        run("Z3", a_gt, zero); run("Z4", a_gt, zero, gamma=0.0)
        pm = pad.clone(); pm[mod == IMAGE_MODALITY] = False; run("V1", a_gt, own, enc=lambda bb: (tok, pm, mod)); pt = pad.clone(); pt[mod == TEXT_MODALITY] = False; run("V2", a_gt, own, enc=lambda bb: (tok, pt, mod))
        cur = policy._current_state(batch)
        for cm in (1, 2):
            for sgn, lab in ((+1, "p"), (-1, "m")):
                d = sgn * cm / 100.0 / sz; so = cur.clone(); so[:, 2] += d; run(f"S1_{lab}{cm}", a_gt, own, state_override=so)
                b2 = dict(batch); st2 = orig_state.clone(); (st2[:, 0] if st2.ndim > 2 else st2)[:, 2] += d; b2["observation.state"] = st2; tok2, pad2, mod2 = orig_enc(b2); tq2 = policy.target_query_extractor(tok2, pad2, mod2); own2 = policy.visual_grounding_cross_attn(tq2, tok2, pad2, mod2)
                run(f"S2_{lab}{cm}", a_gt, own2, enc=lambda bb, T=(tok2, pad2, mod2): T, b=b2)
    finally:
        gw.gamma = 1.0; policy.target_point_head, policy.visual_grounding_cross_attn, policy._encode_multimodal_latent = saved; policy._v2_last_target_xyz = saved_last; torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
        if "_current_state" in policy.__dict__: del policy.__dict__["_current_state"]
    robot = task.robot; ee = np.asarray(robot.get_end_effector_pos(physics), dtype=float); cur7 = policy._current_state(batch)[0, :7].cpu().numpy(); disp = max(float(np.linalg.norm(pos[n][:2] - STATE["init"][n][:2])) for n in pos)
    AUD.append({"ep": G5.ORACLE_ENV["ep"], "t": t, "phase": int(policy._v2_grasp_phase), "ee_robot": ee - base, "state7": cur7, "any_card_displaced_cm": disp * 100, "ee_gt_xy": float(np.linalg.norm(ee[:2] - kp[tgt][:2])), "ee_gt_z": float(ee[2] - kp[tgt][2]), "target_name": tgt, "gt_anchor_robot": gt, "norms": norms, "cond": out})


def main():
    model = sys.argv[sys.argv.index("--model") + 1]; out = Path(sys.argv[sys.argv.index("--out") + 1]); G5.AUDIT["fn"] = audit
    try: G5.main()
    finally: pickle.dump(AUD, open(out / model / "audit10.pkl", "wb")); print("audit frames:", len(AUD))


if __name__ == "__main__":
    main()
