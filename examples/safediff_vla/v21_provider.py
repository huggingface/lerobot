"""[StagePlan-v2.1-grounded] Mixed stage-variable provider for the UNCHANGED StagePlan-v2 executor (`SafeDiffVLAStagePlanV2Policy.install(provider)`).

  grasp_xyz              <- NEW: RGB (front+right, original 480 px) -> frozen proposals (front g64 thr 0.10, right g64) -> calibrated triangulation -> frozen HR referent scorer
                            (language = frozen VLM text-token states from the policy's own batch) -> shared learned local-offset head -> robot-base-frame metres
  grasp_rot, approach_dir, pre_dist, wp_rel  <- RETAINED from the frozen StagePlan-v2 head (outputs/train/safediff_stageplan_v2); waypoint = grasp_xyz + wp_rel * WP_SCALE_M as in v2
  fallback               : no proposals / selected proposal not triangulable -> the frozen StagePlan-v2 grasp_xyz (logged; never an oracle)
Inference inputs: raw RGB (supplied by the harness through `raw_fn`), the policy batch (language/proprio). No simulator state."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).parent))
import v21hr_candidate as H  # noqa: E402
import v21q_prop_iter as Q  # noqa: E402
import v21g_pipeline as GP  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import PRE_DIST_SCALE_M, WP_SCALE_M, StagePlanV2Head, StageVars  # noqa: E402

X = Path("outputs/stageplan_v21/x"); V2 = Path("outputs/train/safediff_stageplan_v2")


class GroundedProvider:
    def __init__(self, policy, raw_fn, anchor_head_path=X / "anchor_head.pt", dev="cuda"):
        self.policy, self.raw_fn, self.dev = policy, raw_fn, dev
        cfg = json.load(open(V2 / "stageplan_config.json")); self.v2 = StagePlanV2Head(policy._multimodal_latent_dim(), **cfg["head"]); self.v2.load_state_dict(load_file(V2 / "stageplan_head.safetensors")); self.v2.to(dev).eval()
        self.fm = Q.Head(64).to(dev); self.fm.load_state_dict(torch.load(X / "proposal_head_v21q_g64.pt")); self.fm.eval(); self.rm = Q.Head(64).to(dev); self.rm.load_state_dict(torch.load(X / "proposal_head_right_g64.pt")); self.rm.eval()
        self.sc = H.Scorer("B").to(dev); self.sc.load_state_dict(torch.load(X / "scorer_expanded.pt")); self.sc.eval(); self.ah = GP.AnchorHead().to(dev); self.ah.load_state_dict(torch.load(anchor_head_path)); self.ah.eval()
        self.vm = policy.backbone.model.vlm_with_expert.get_vlm_model().vision_model; self.ve = policy.backbone.model.vlm_with_expert; self.info = {}

    @torch.no_grad()
    def _feat(self, img):
        x = torch.from_numpy(cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].to(self.dev) * 2 - 1
        return self.vm(pixel_values=x.to(self.vm.dtype)).last_hidden_state[0].float().reshape(32, 32, -1).permute(2, 0, 1)[None]

    @torch.no_grad()
    def _crop(self, img, u, v):
        u, v = int(round(min(max(u, 0), 479))), int(round(min(max(v, 0), 479))); h = H.CROP_PX // 2; pad = np.pad(img, ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[v: v + H.CROP_PX, u: u + H.CROP_PX]
        return self.ve.embed_image(torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].to(self.dev) * 2 - 1)[0].float()

    @torch.no_grad()
    def __call__(self, policy, batch) -> StageVars:
        tokens, pad, mod = policy._encode_multimodal_latent(batch); state = policy._current_state(batch); o = self.v2(tokens, pad, mod, state)
        g2 = (o["grasp_xyz"] * policy.action_pos_std + policy.action_pos_mean)[0].float().cpu().numpy().astype(np.float64); eul = torch.atan2(o["grasp_rot_sc"][0, :, 0], o["grasp_rot_sc"][0, :, 1]).float().cpu().numpy().astype(np.float64)
        r = float(np.clip(o["pre_dist"][0].item() * PRE_DIST_SCALE_M, 0.05, 0.15)); wp_rel = o["wp_rel"][0].float().cpu().numpy().astype(np.float64) * WP_SCALE_M; ad = o["approach_dir"][0].float().cpu().numpy().astype(np.float64)
        info = {"source": "v2_fallback", "g_v2": g2.tolist()}; g = g2
        raw = self.raw_fn(); front, right = raw["image"], raw["second_image"]
        Pf = Q.decode(self.fm(self._feat(front))[0].cpu(), 64, GP.THR); Pr = Q.decode(self.rm(self._feat(right))[0].cpu(), 64, GP.THR); info.update({"n_front": len(Pf), "n_right": len(Pr), "front_props": [list(map(float, p)) for p in Pf]})
        if Pf:
            m = (mod[0] == 1) & pad[0].bool(); txt = tokens[0][m].float()[None]; tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device=self.dev)
            ft = [self._crop(front, p[0], p[1]) for p in Pf]; lg = self.sc(txt, tm, torch.stack(ft)[None])[0]; sel = int(lg.argmax()); info.update({"scores": lg.cpu().tolist(), "selected": sel, "selected_uv": list(map(float, Pf[sel][:2]))})
            assoc = GP.associate(Pf, Pr)
            if sel in assoc:
                ref, j, gap = assoc[sel]; off = self.ah(ft[sel][None], self._crop(right, Pr[j][0], Pr[j][1])[None], torch.tensor(ref, dtype=torch.float32, device=self.dev)[None])[0].cpu().numpy() / 100
                g = ref + off; info.update({"source": "grounded", "ref": ref.tolist(), "offset_cm": (off * 100).tolist(), "ray_gap_m": float(gap)})
            else: info["source"] = "v2_fallback_no_triangulation"
        else: info["source"] = "v2_fallback_no_proposals"
        info["g_final"] = np.asarray(g).tolist(); self.info = info
        return StageVars(np.asarray(g, float), eul, ad, r, True, np.asarray(g, float) + wp_rel)
