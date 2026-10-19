#!/usr/bin/env python
"""G5 minimal closed-loop grounding-transfer smoke (select_poker). Base policy = V2 GroundedGrasp 20k (decoder, reactive close, phase logic UNCHANGED, CORRECTED-EVAL language via the env wrapper's task_description).
--model v2   : untouched.
--model g3   : G3-M1 referent-bottleneck head (seed 0) replaces v2's target_point_head OUTPUT at every replan (same normalized action-position space).
--model g4d  : G4d (seed 0, beta=0.2 fine-dominant fusion; OOF-trained matcher) replaces it: coarse M1 map -> proposals (front top-2 + right centroid->table plane->front, 144px) -> frozen-VLM crop tokens -> SpatialMatcher -> fused map -> anchor head.
Nothing else is touched; instruction is NEVER swapped; GT (entities/segmentation) is logged for diagnostics only.
Per env step we log: physics (EE, GT target, all same-type candidates, is_grasped), phase / replan / reactive-close, commanded action, planner target (what the decoder was conditioned on) and a MONITOR target recomputed on every frame (monitor-only, never fed back),
plus the grounder internals (crop boxes, fine p, selected crop, alpha-weighted fine location, final-map argmax cells)."""
from __future__ import annotations

import json, logging, math, sys
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd, torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, extract_physics_record, robot_base  # noqa: E402
from phase_e0_instruction_plumbing_check import TASKS_PARQUET  # noqa: E402
from g1_train import Bottleneck  # noqa: E402
import g4c_matcher_train as gc  # noqa: E402
import g4b_proposals as P  # noqa: E402
from g4c_build_cache import proj_matrix  # noqa: E402
from g6_stabilizer import Stabilizer  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)
COARSE = "outputs/train/safediff_vla_g3/M1_s0.pt"; OD = Path("outputs/eval/safediff_vla_g4d_offline"); OUT = Path("outputs/eval/safediff_vla_g5_closed_loop"); FRONT = "observation.images.camera1"


class Grounder:
    def __init__(self, kind: str, policy: SafeDiffVLAPolicy):
        self.kind = kind; dev = "cuda"; self.coarse = Bottleneck().to(dev); self.coarse.load_state_dict(torch.load(COARSE)); self.coarse.eval(); self.cache: dict | None = None
        if kind == "g4d": self.m = gc.G4cModel("add", 0.2).to(dev); self.m.load_state_dict(torch.load(OD / "g4d_matcher_s0.pt")); self.m.eval()
        self.vlm = policy.backbone.model.vlm_with_expert; self.scale = math.sqrt(960)

    @torch.no_grad()
    def compute(self, tok, pad, mod, batch) -> dict:
        mod = mod.clone().long(); mod[~pad.to(torch.bool)] = 3; tokf = tok.float(); dev = tok.device
        if self.kind == "g3":
            xyz, w = self.coarse(tokf, mod); wn = w[0].cpu().numpy(); return {"xyz_n": xyz, "info": {"coarse_front_argmax": int(wn[:64].argmax()), "coarse_right_argmax": int(wn[64:128].argmax()), "coarse_front_max": float(wn[:64].max())}}
        w, xx = self.coarse.weights(tokf, mod); v = self.coarse.Wv(xx); wn = w[0].double().cpu().numpy(); cands = P.make(wn)
        img = batch[FRONT][0]; assert img.shape[-1] == 480 and img.shape[-2] == 480, f"front image is {tuple(img.shape)}, expected 480x480"; rgb = (img.float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        crops = [rgb[b[1]:b[1] + b[2], b[0]:b[0] + b[2]] for b in (c["box"] for c in cands)]
        x = torch.cat([resize_with_pad(torch.from_numpy(c).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0, 512, 512, pad_value=0) * 2.0 - 1.0 for c in crops]); H = self.vlm.embed_image(x).float() * self.scale  # [n,64,960]
        J = P.J; Hp = torch.zeros(1, J, 64, 960, device=dev); Hp[0, :len(cands)] = H; proj = torch.zeros(1, J, 64, 192, device=dev); cv = torch.zeros(1, J, dtype=torch.bool, device=dev)
        for j, c in enumerate(cands): proj[0, j] = torch.from_numpy(proj_matrix(tuple(c["box"]))).to(dev); cv[0, j] = True
        ids = batch["observation.language.tokens"][0]; am = batch["observation.language.attention_mask"][0].bool(); te = (self.vlm.embed_language_tokens(ids[am].unsqueeze(0))[0].float() * self.scale).unsqueeze(0); tm = torch.ones(1, te.shape[1], dtype=torch.bool, device=dev)
        xyz, w2, p, s, al = self.m(w, v, proj, cv, Hp, te, tm); p = p[0].cpu().numpy(); al = al[0].cpu().numpy(); w2n = w2[0].cpu().numpy(); top = int(np.where(cv[0].cpu().numpy(), s[0].cpu().numpy(), -1e9).argmax()); bx, by, sd = cands[top]["box"]
        fine_px = (al[top][:, None] * np.array([[bx + (t % 8 + .5) * sd / 8, by + (t // 8 + .5) * sd / 8] for t in range(64)])).sum(0)
        info = {"boxes": [list(map(int, c["box"])) for c in cands], "src": [c["src"] for c in cands], "p": p[:len(cands)].tolist(), "selected": top, "fine_px_front": fine_px.tolist(), "coarse_front_argmax": int(wn[:64].argmax()), "coarse_right_argmax": int(wn[64:128].argmax()),
                "final_front_argmax": int(w2n[:64].argmax()), "final_right_argmax": int(w2n[64:128].argmax()), "w2": np.round(w2n, 3).tolist(), "alpha_selected_max": float(al[top].max())}
        return {"xyz_n": xyz, "info": info}


class StabGrounder(Grounder):
    """G4d + G6 temporal referent stabilisation. The model is run normally; the Stabilizer (spatial lock on the selected card's front-image location, hysteresis, 1-replan grace) decides which candidate's fine map the FINAL map / anchor head should use.
    The anchor xyz is always recomputed from the current frame (candidate probability forced onto the locked candidate); nothing but the candidate identity persists."""
    def __init__(self, policy):
        super().__init__("g4d", policy); self.stab = Stabilizer(); self.in_plan = False; self.last_xyz_n = None

    @torch.no_grad()
    def compute(self, tok, pad, mod, batch) -> dict:
        mod = mod.clone().long(); mod[~pad.to(torch.bool)] = 3; tokf = tok.float(); dev = tok.device; w, xx = self.coarse.weights(tokf, mod); v = self.coarse.Wv(xx); wn = w[0].double().cpu().numpy(); cands = P.make(wn)
        img = batch[FRONT][0]; assert img.shape[-1] == 480 and img.shape[-2] == 480; rgb = (img.float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        crops = [rgb[b[1]:b[1] + b[2], b[0]:b[0] + b[2]] for b in (c["box"] for c in cands)]
        x = torch.cat([resize_with_pad(torch.from_numpy(c).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0, 512, 512, pad_value=0) * 2.0 - 1.0 for c in crops]); H = self.vlm.embed_image(x).float() * self.scale
        J = P.J; Hp = torch.zeros(1, J, 64, 960, device=dev); Hp[0, :len(cands)] = H; proj = torch.zeros(1, J, 64, 192, device=dev); cv = torch.zeros(1, J, dtype=torch.bool, device=dev)
        for j, c in enumerate(cands): proj[0, j] = torch.from_numpy(proj_matrix(tuple(c["box"]))).to(dev); cv[0, j] = True
        ids = batch["observation.language.tokens"][0]; am = batch["observation.language.attention_mask"][0].bool(); te = (self.vlm.embed_language_tokens(ids[am].unsqueeze(0))[0].float() * self.scale).unsqueeze(0); tm = torch.ones(1, te.shape[1], dtype=torch.bool, device=dev)
        xyz, w2, p, s, al = self.m(w, v, proj, cv, Hp, te, tm); pn = p[0].cpu().numpy(); aln = al[0].cpu().numpy(); w2n = w2[0].cpu().numpy(); top = int(np.where(cv[0].cpu().numpy(), s[0].cpu().numpy(), -1e9).argmax())
        fine = []
        for j, c in enumerate(cands): bx, by, sd = c["box"]; fine.append((aln[j][:, None] * np.array([[bx + (t % 8 + .5) * sd / 8, by + (t // 8 + .5) * sd / 8] for t in range(64)])).sum(0))
        fc = int(w2n[:64].argmax()); mc = np.array([(fc % 8 + .5) * 60, (fc // 8 + .5) * 60]); sc = [{"box": tuple(c["box"]), "p": float(pn[j]), "fine": tuple(fine[j]), "map_ok": bool(j == top and c["box"][0] <= mc[0] <= c["box"][0] + c["box"][2] and c["box"][1] <= mc[1] <= c["box"][1] + c["box"][2])} for j, c in enumerate(cands)]
        if self.in_plan: dec = self.stab.step(sc)
        else: a = self.stab.peek(sc); dec = {"action": "peek", "idx": a}
        out_xyz, out_w2, used = xyz, w2n, "raw"
        if dec["action"] == "grace" and self.last_xyz_n is not None: out_xyz, used = self.last_xyz_n, "grace_prev_anchor"
        elif dec["idx"] is not None and dec["idx"] != top:
            pf = torch.zeros(1, J, device=dev); pf[0, dec["idx"]] = 1.0; out_xyz, w2f, *_ = self.m(w, v, proj, cv, Hp, te, tm, p_force=pf); out_w2 = w2f[0].cpu().numpy(); used = "forced_candidate"
        if self.in_plan: self.last_xyz_n = out_xyz.detach().clone()
        info = {"boxes": [list(map(int, c["box"])) for c in cands], "src": [c["src"] for c in cands], "p": pn[:len(cands)].tolist(), "selected": top, "fine_px_front": fine[top].tolist(), "coarse_front_argmax": int(wn[:64].argmax()), "coarse_right_argmax": int(wn[64:128].argmax()),
                "final_front_argmax": int(out_w2[:64].argmax()), "final_right_argmax": int(out_w2[64:128].argmax()), "w2": np.round(out_w2, 3).tolist(), "alpha_selected_max": float(aln[top].max()),
                "stab": {"action": dec["action"], "idx": dec["idx"], "used": used, "locked": self.stab.locked, "L": (self.stab.L.tolist() if self.stab.L is not None else None), "missing": self.stab.missing, "n_switch": self.stab.n_switch, "in_plan": self.in_plan, "raw_xyz_n": xyz[0].cpu().tolist()}}
        return {"xyz_n": out_xyz, "info": info}


ORACLE_ENV: dict = {"env": None, "ep": -1}
PRE: dict = {"fn": None}  # G14: optional hook fn(policy, batch) run at the START of every select_action; may patch `batch` in place (privileged diagnostics only)
GROUNDER: dict = {"g": None}
CHUNK: dict = {"last": None}  # G14: last planned chunk (actions [H,7], normalised) recorded by a plan_action_chunk wrapper
AUDIT: dict = {"fn": None}  # G8: optional capture-only hook fn(policy, batch, t, orig_encode) called after the real action was chosen


class OracleGrounder(Grounder):
    """G7 DIAGNOSTIC ONLY (never a method): the decoder conditioning / planner target is the GT instructed-object grasp anchor `entity.get_grasped_keypoints(physics)[-1]` in robot-base frame (the exact G1/G4 supervision definition), converted to the same normalised
    action-position space. G4d still runs (logging only) so its prediction error is recorded at every replan. In PRE_GRASP the live keypoint is used (it follows a card moved by contact); after the reactive close (POST_GRASP) the last pre-grasp value is frozen,
    mirroring the training target (the fixed demonstrated grasp point) instead of a keypoint that lifts with the card. Everything else (decoder, reactive close, phase, queue, horizon) is untouched."""
    def __init__(self, policy):
        super().__init__("g4d", policy); self.policy = policy; self.last = None

    def gt_anchor(self) -> np.ndarray:
        if self.policy._v2_grasp_phase == 1 and self.last is not None: return self.last
        env = ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; ent = task.entities[task.target_entity]; kp = np.asarray(ent.get_grasped_keypoints(physics)[-1], dtype=float); self.last = kp - robot_base(env); return self.last

    @torch.no_grad()
    def compute(self, tok, pad, mod, batch) -> dict:
        out = super().compute(tok, pad, mod, batch); mean, std = self.policy.action_pos_mean, self.policy.action_pos_std; gt = self.gt_anchor(); pred = (out["xyz_n"] * std + mean)[0].cpu().numpy()
        out["info"]["oracle_anchor_robot"] = gt.tolist(); out["info"]["g4d_pred_robot"] = pred.tolist(); out["info"]["g4d_err_xy"] = float(np.linalg.norm(pred[:2] - gt[:2])); out["info"]["g4d_err_3d"] = float(np.linalg.norm(pred - gt))
        out["xyz_n"] = ((torch.as_tensor(gt, dtype=std.dtype, device=std.device) - mean) / std).unsqueeze(0); return out


class OverrideHead(nn.Module):
    def __init__(self, orig: nn.Module, g: Grounder): super().__init__(); self.orig = orig; self.g = g

    def forward(self, x): assert self.g.cache is not None; return self.g.cache["xyz_n"].to(device=x.device, dtype=x.dtype)


def install_policy_instrumentation(policy: SafeDiffVLAPolicy, grounder: Grounder | None, preprocessor) -> dict:
    orig_select = SafeDiffVLAPolicy.select_action; orig_plan = SafeDiffVLAPolicy._plan_temporal_decoder; orig_enc = policy._encode_multimodal_latent; log: list[dict] = []; flag = {"replan": False}; lang: list[dict] = []
    tokenizer = next((getattr(s, "input_tokenizer", None) for s in getattr(preprocessor, "steps", []) if getattr(s, "input_tokenizer", None) is not None), None)

    def enc(batch):
        out = orig_enc(batch)
        if grounder is not None and grounder.cache is None: grounder.cache = grounder.compute(*out, batch)
        return out

    def plan(self, batch):
        flag["replan"] = True
        if grounder is not None and hasattr(grounder, "in_plan"): grounder.in_plan = True
        try: return orig_plan(self, batch)
        finally:
            if grounder is not None and hasattr(grounder, "in_plan"): grounder.in_plan = False

    def select(self, batch):
        if PRE["fn"] is not None: PRE["fn"](self, batch)
        if grounder is not None: grounder.cache = None
        flag["replan"] = False; close_before = self._v2_close_triggered
        if not log and not lang:
            ids = batch["observation.language.tokens"][0].cpu().tolist(); am = batch["observation.language.attention_mask"][0].cpu().tolist(); lang.append({"token_ids": ids, "attention_mask": am, "decoded": tokenizer.decode([i for i, m in zip(ids, am) if m], skip_special_tokens=False) if tokenizer is not None else None,
                                                                         "front_image_shape": list(batch[FRONT].shape)})
        action = orig_select(self, batch); replanned = flag["replan"]; closed_now = (not close_before) and self._v2_close_triggered
        if AUDIT["fn"] is not None: AUDIT["fn"](self, batch, len(log), orig_enc)
        with torch.no_grad():
            tok, pad, mod = self._encode_multimodal_latent(batch)  # (cached grounder output reused if this step already encoded)
            if grounder is None:
                q = self.target_query_extractor(tok, pad, mod); gf = self.visual_grounding_cross_attn(q, tok, pad, mod); mxyz = self.target_point_head(gf); ginfo = None
            else: mxyz = grounder.cache["xyz_n"]; ginfo = grounder.cache["info"]
        mon = (mxyz * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy(); pl = None
        if self._v2_last_target_xyz is not None: pl = ((self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean)[0]).detach().cpu().numpy()
        cmd = (action[..., :3] * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy(); rot = (action[..., ROTATION_SLICE_RAW] * self.action_rot_std + self.action_rot_mean)[0].detach().cpu().numpy(); grip = float((action[0, GRIPPER_INDEX_RAW] * self.action_grip_std + self.action_grip_mean).item())
        log.append({"phase": int(self._v2_grasp_phase), "reactive_close_event": bool(closed_now), "planner_replanned_this_step": bool(replanned), "monitor_target_robot_frame": mon.tolist(), "planner_target_robot_frame": pl.tolist() if pl is not None else None,
                    "commanded_xyz_robot_frame": cmd.tolist(), "commanded_rot_euler_raw": rot.tolist(), "commanded_gripper_raw": grip, "grounder": ginfo})
        return action

    orig_pac = policy.plan_action_chunk
    def pac(batch):
        out = orig_pac(batch); CHUNK["last"] = out[0][0].detach().clone(); return out
    policy.plan_action_chunk = pac
    policy._encode_multimodal_latent = enc; policy._plan_temporal_decoder = plan.__get__(policy, SafeDiffVLAPolicy); policy.select_action = select.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log, "lang": lang}


def install_env_instrumentation(policy_log: list[dict], lang: list[dict], on_reset=None) -> dict:
    cls = VLABenchEnvImpl; orig_step, orig_reset = cls.step, cls.reset; state: dict[str, Any] = {"steps": [], "episode_meta": {}}

    def candidates(env):
        physics = env._env.physics; task = env._env.task; tn = task.target_entity; tt = type(task.entities[tn]).__name__ if tn in task.entities else None; out = {}
        for n, e in task.entities.items():
            if tt is not None and type(e).__name__ == tt:
                try: out[n] = np.asarray(e.get_xpos(physics), dtype=float).tolist()
                except Exception: pass  # noqa: BLE001
        return out

    def merge(rec, env):
        base = robot_base(env); rec["candidates_world"] = candidates(env)
        for k in ("phase", "reactive_close_event", "planner_replanned_this_step", "monitor_target_world", "planner_target_world", "commanded_xyz_world", "commanded_gripper_raw", "grounder"): rec[k] = None
        if policy_log:
            p = policy_log[-1]; rec["phase"] = p["phase"]; rec["reactive_close_event"] = p["reactive_close_event"]; rec["planner_replanned_this_step"] = p["planner_replanned_this_step"]; rec["commanded_gripper_raw"] = p["commanded_gripper_raw"]; rec["grounder"] = p["grounder"]
            rec["monitor_target_world"] = (np.asarray(p["monitor_target_robot_frame"]) + base).tolist()
            if p["planner_target_robot_frame"] is not None: rec["planner_target_world"] = (np.asarray(p["planner_target_robot_frame"]) + base).tolist()
            rec["commanded_xyz_world"] = (np.asarray(p["commanded_xyz_robot_frame"]) + base).tolist()

    def reset(self, seed=None, **kw):
        res = orig_reset(self, seed=seed, **kw)
        if seed is not None: ORACLE_ENV["env"] = self; ORACLE_ENV["ep"] += 1
        if seed is not None and on_reset is not None: on_reset()
        if seed is not None:
            state["episode_meta"] = {"task_description": getattr(self, "task_description", None), "task_description_source": getattr(self, "task_description_source", None), "target_entity": self._env.task.target_entity, "robot_base": robot_base(self).tolist()}
            r = extract_physics_record(self, 0); merge(r, self); state["steps"] = [r]
        return res

    def step(self, action):
        res = orig_step(self, action)
        if not res[2]: r = extract_physics_record(self, len(state["steps"])); merge(r, self); state["steps"].append(r)
        return res

    cls.step, cls.reset = step, reset; state["_orig"] = (cls, orig_step, orig_reset); return state


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["v2", "g3", "g4d", "g4d_stab", "oracle"]); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3010))); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    out_dir = Path(a.out) / a.model; out_dir.mkdir(parents=True, exist_ok=True); train_strings = set(pd.read_parquet(TASKS_PARQUET).index)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2"
    consts = json.load(open("outputs/data/safediff_vla_g1_counterfactual/policy_constants.json")); assert np.allclose(policy.action_pos_mean.cpu().numpy(), consts["pos_mean"]) and np.allclose(policy.action_pos_std.cpu().numpy(), consts["pos_std"]), "normalisation mismatch"
    env_cfg = VLABenchEnv(task="select_poker"); env = make_env(env_cfg, n_envs=1, use_async_envs=False)["select_poker"][0]
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    grounder = None if a.model == "v2" else (StabGrounder(policy) if a.model == "g4d_stab" else (OracleGrounder(policy) if a.model == "oracle" else Grounder(a.model, policy)))
    GROUNDER["g"] = grounder
    if grounder is not None: policy.target_point_head = OverrideHead(policy.target_point_head, grounder)
    inst = install_policy_instrumentation(policy, grounder, pre); env_state = install_env_instrumentation(inst["log"], inst["lang"], on_reset=(grounder.stab.reset if isinstance(grounder, StabGrounder) else ((lambda: setattr(grounder, 'last', None)) if isinstance(grounder, OracleGrounder) else None))); records: list[dict] = []

    def cb(episode_ix, rollout_data, env_idx, done_index):
        meta = dict(env_state["episode_meta"]); lang = inst["lang"][0] if inst["lang"] else {}; meta["in_training_strings"] = (meta.get("task_description") in train_strings) if meta.get("task_description") else None
        records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "meta": meta, "language": lang, "steps": list(env_state["steps"])}); inst["log"].clear(); inst["lang"].clear()
        logger.info("  [%s] episode %d len=%d target=%s desc=%r", a.model, episode_ix, done_index + 1, meta.get("target_entity"), meta.get("task_description"))

    set_seed(a.seeds[0])
    try:
        info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre, postprocessor=post, n_episodes=len(a.seeds), max_episodes_rendered=len(a.seeds), videos_dir=out_dir / "videos", start_seed=a.seeds[0], episode_callback=cb)
    finally:
        env.close(); cls, os_, or_ = env_state["_orig"]; cls.step, cls.reset = os_, or_
    for ep, r in zip(info["per_episode"], records, strict=True): r["seed"] = ep["seed"]; r["success"] = ep["success"]
    (out_dir / "episodes_raw.json").write_text(json.dumps(records)); logger.info("[%s] successes: %s", a.model, [ep["success"] for ep in info["per_episode"]])


if __name__ == "__main__":
    main()
