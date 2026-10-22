"""[v2.1 integration gate 3] Numerical round trip of a grasp_xyz through the UNCHANGED StagePlan-v2 executor path, 20 TRAIN labels/task:
 label grasp_xyz (robot frame, m) -> StageVars -> (a) `_StageTargetHead` normalised planner-target slot -> `* action_pos_std + action_pos_mean` (what `_grounded_grasp_v2_reactive_close` reads as CLOSE target)
                                  -> (b) `build_pregrasp_chunk` final polyline point -> normalised action slot (plan_action_chunk) -> the real POSTPROCESSOR (unnormalise) -> env wrapper ctrl conversion (`+ _robot_base_xyz`, IK target)
 Compared with the original label; also reports the policy buffers (mean/std), checks no axis swap / sign flip / base offset / rotation / unit change."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import StageVars, build_pregrasp_chunk, SafeDiffVLAStagePlanV2Policy, _StageTargetHead  # noqa: E402
from v21_common import CKPT  # noqa: E402

pol = SafeDiffVLAStagePlanV2Policy.from_pretrained(CKPT).to("cuda").eval(); pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
L = pickle.load(open("outputs/stageplan/labels_v2.pkl", "rb")); train = set(json.load(open("outputs/stageplan/split_constants.json"))["train"]); BASE = np.array([0, -0.4, 0.78]); res = {"action_pos_mean": pol.action_pos_mean.cpu().tolist(), "action_pos_std": pol.action_pos_std.cpu().tolist()}
pol.reset(); pol._sv = None; head = _StageTargetHead(pol)
for fam in ("select_poker", "select_mahjong"):
    eps = [e for e in sorted(L) if L[e]["fam"] == fam and e in train][:20]; d_slot, d_chunk, d_post, d_world = [], [], [], []
    for e in eps:
        l = L[e]; g = l["grasp_xyz"].astype(np.float64); sv = StageVars(g, l["grasp_euler"].astype(np.float64), l["approach_dir"].astype(np.float64), float(l["pre_dist"]), False, None); pol._sv = sv
        slot = head(torch.zeros(1, 4, device="cuda"))[0]; d_slot.append(np.abs((slot * pol.action_pos_std + pol.action_pos_mean).double().cpu().numpy() - g).max())  # (a)
        ee0 = g + np.array([0.15, -0.1, 0.2]); pos, rot = build_pregrasp_chunk(ee0, l["grasp_euler"].astype(np.float64), sv); d_chunk.append(np.abs(pos[-1] - g).max())  # chunk ends exactly at the grasp pose
        a = np.zeros((1, 7), np.float32); a[0, :3] = (pos[-1] - pol.action_pos_mean.cpu().numpy()) / pol.action_pos_std.cpu().numpy(); a[0, 3:6] = (rot[-1] - pol.action_rot_mean.cpu().numpy()) / pol.action_rot_std.cpu().numpy(); a[0, 6] = float(((1.0 - pol.action_grip_mean) / pol.action_grip_std).item())
        out = post(torch.from_numpy(a).cuda())  # real postprocessor
        out = out.detach().cpu().numpy().reshape(-1)[:7]; d_post.append(np.abs(out[:3] - g).max()); d_world.append(np.abs((out[:3] + BASE) - (g + BASE)).max())  # wrapper adds base to get the world IK target
    res[fam] = {"max |slot->phys - label| (m)": float(np.max(d_slot)), "max |chunk end - label| (m)": float(np.max(d_chunk)), "max |postprocessor out - label| (m)": float(np.max(d_post)), "max |world IK target - (label+base)| (m)": float(np.max(d_world)), "n": len(eps)}
json.dump(res, open("outputs/stageplan_v21/x/roundtrip.json", "w"), indent=1); print(json.dumps(res, indent=1))
