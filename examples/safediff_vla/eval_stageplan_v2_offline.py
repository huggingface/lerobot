#!/usr/bin/env python
"""StagePlan-v2 offline validation on the episode-DISJOINT validation split: per task, grasp xyz error (median/mean, <2/<3/<5 cm), grasp
rotation geodesic error, approach direction cosine, pre-grasp distance error, waypoint-offset error; plus SAME-OBSERVATION instruction-swap
sensitivity (same image/state, another same-family instruction): mean predicted grasp shift and the error with own vs swapped instruction.
Usage: uv run python examples/safediff_vla/eval_stageplan_v2_offline.py [--head outputs/train/safediff_stageplan_v2]
"""

from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import PRE_DIST_SCALE_M, WP_SCALE_M, SafeDiffVLAStagePlanV2Policy, StagePlanV2Head  # noqa: E402

CKPT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model"; ROOT = Path("outputs/stageplan")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="outputs/train/safediff_stageplan_v2"); a = ap.parse_args(); dev = "cuda"
    labels = pickle.load(open(ROOT / "labels_v2.pkl", "rb")); sp = json.load(open(ROOT / "split_constants.json")); val = sp["val"]
    pol = SafeDiffVLAStagePlanV2Policy.from_pretrained(CKPT).to(dev).eval(); cfg = json.load(open(Path(a.head) / "stageplan_config.json"))
    head = StagePlanV2Head(pol._multimodal_latent_dim(), **cfg["head"]); head.load_state_dict(load_file(Path(a.head) / "stageplan_head.safetensors")); head.to(dev).eval()
    pre, _ = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": dev}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    ds = LeRobotDataset("lerobot/vlabench_unified", episodes=val, image_transforms=None, delta_timestamps=None, video_backend="torchcodec", return_uint8=True)
    ec = np.array(ds.hf_dataset["episode_index"]); fc = np.array(ds.hf_dataset["frame_index"]); idx0 = [int(np.where((ec == e) & (fc == 0))[0][0]) for e in val]
    fam = {e: labels[e]["fam"] for e in val}; byfam = {}
    for e in val: byfam.setdefault(fam[e], []).append(e)
    tasks = {}  # episode -> instruction string (dataset task)
    rows = []
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx0), batch_size=16, num_workers=4)
    def predict(b, text=None):
        for k in [k for k in b if k.startswith("observation.images.") and b[k].dtype == torch.uint8]: b[k] = b[k].float() / 255.0
        if text is not None: b = {**b, "task": text}
        bb = pre(b)
        with torch.no_grad():
            tok, pad, mod = pol._encode_multimodal_latent(bb); o = head(tok, pad, mod, pol._current_state(bb))
        g = (o["grasp_xyz"] * pol.action_pos_std + pol.action_pos_mean).cpu().numpy(); eul = torch.atan2(o["grasp_rot_sc"][..., 0], o["grasp_rot_sc"][..., 1]).cpu().numpy()
        return {"g": g, "eul": eul, "dir": o["approach_dir"].cpu().numpy(), "pre": o["pre_dist"].cpu().numpy() * PRE_DIST_SCALE_M, "wp": o["wp_rel"].cpu().numpy() * WP_SCALE_M}
    for b in dl:
        eps = b["episode_index"].tolist(); texts = list(b["task"]); P = predict(dict(b))
        # swapped instruction: instruction of the next val episode of the same family (cyclic)
        sw = []
        for e, t in zip(eps, texts):
            lst = byfam[fam[e]]; o_e = lst[(lst.index(e) + 1) % len(lst)]; sw.append(ds.meta.episodes["tasks"][o_e][0])
        Ps = predict(dict(b), text=sw)
        for i, e in enumerate(eps):
            l = labels[e]; gt_g = l["grasp_xyz"]
            rows.append({"fam": fam[e], "err": np.linalg.norm(P["g"][i] - gt_g) * 100, "err_sw": np.linalg.norm(Ps["g"][i] - gt_g) * 100, "shift": np.linalg.norm(Ps["g"][i] - P["g"][i]) * 100,
                         "rot": float(np.degrees((Rotation.from_euler("xyz", P["eul"][i]).inv() * Rotation.from_euler("xyz", l["grasp_euler"])).magnitude())), "dircos": float(P["dir"][i] @ l["approach_dir"]),
                         "pre_err": abs(P["pre"][i] - l["pre_dist"]) * 100, "wp_err": np.linalg.norm(P["wp"][i] - l["wp_rel"]) * 100})
    import pandas as pd; D = pd.DataFrame(rows); pd.set_option("display.width", 250)
    g = D.groupby("fam")
    T = g.agg(n=("err", "size"), grasp_med=("err", "median"), grasp_mean=("err", "mean"), lt2=("err", lambda x: (x < 2).mean()), lt3=("err", lambda x: (x < 3).mean()), lt5=("err", lambda x: (x < 5).mean()), rot_med_deg=("rot", "median"), dir_cos=("dircos", "mean"), pre_err_cm=("pre_err", "median"), wp_err_cm=("wp_err", "median")).round(2)
    L = g.agg(grasp_err_own=("err", "mean"), grasp_err_swapped=("err_sw", "mean"), mean_shift_cm=("shift", "mean")).round(2)
    print("== offline validation (frame 0, episode-disjoint val)"); print(T.to_string()); print("\n== same-observation instruction swap (cm)"); print(L.to_string()); D.to_pickle(Path(a.head) / "offline_val_v2.pkl")


if __name__ == "__main__":
    main()
