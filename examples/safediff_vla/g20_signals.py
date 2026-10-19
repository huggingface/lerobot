#!/usr/bin/env python
"""G20 helper (no training): re-run the frozen G4d grounder on the 797 fixed-observation bank frames and store its INTERNAL 2D signals (not only the xyz anchor): candidate boxes, selected candidate, fine alpha-weighted centroid (front px), final-map argmax cells (front/right), the 128-cell final map w2, the xyz anchor. Same call as g16/g19 (original batch). Also stores the saved first-replan record of each closed-loop G4d episode (cl_geo/g4d, seeds 3000-3009) for a cross-check."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g13_interventions import KEYS  # noqa: E402
from g17_capture import load_bank  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g20")


def main():
    OUT.mkdir(parents=True, exist_ok=True); B, IM, SRC = load_bank(); dev = "cuda"; policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to(dev).eval(); mean, std = policy.action_pos_mean, policy.action_pos_std; enc = SafeDiffVLAPolicy._encode_multimodal_latent; grounder = G5.Grounder("g4d", policy); R = []
    for i, e in enumerate(B):
        b = {k: torch.from_numpy(IM[i][c]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev) for c, k in enumerate(KEYS)}; b["observation.state"] = torch.from_numpy(e["state_norm"]).float().unsqueeze(0).to(dev); b["observation.language.tokens"] = torch.from_numpy(e["lang_tokens"]).long().unsqueeze(0).to(dev); b["observation.language.attention_mask"] = torch.from_numpy(e["lang_mask"]).bool().unsqueeze(0).to(dev)
        with torch.no_grad(): lat = enc(policy, b); out = grounder.compute(*lat, b)
        R.append({"i": i, "src": SRC[i], "ep": e["ep"], "t": e["t"], "anchor": (out["xyz_n"].reshape(-1)[:3] * std + mean).cpu().numpy(), "info": out["info"]})
        if (i + 1) % 200 == 0: print(i + 1, flush=True)
    cl = {}
    for m in ("g4d", "oracle"):
        E = json.load(open(Path("outputs/eval/safediff_vla_g19/cl_geo") / m / "episodes_raw.json")); cl[m] = [{"seed": e["seed"], "meta": e["meta"], "step0": e["steps"][0], "step1": e["steps"][1]} for e in E]
    pickle.dump({"R": R, "cl": cl}, open(OUT / "g20_signals.pkl", "wb")); print("saved", len(R))


if __name__ == "__main__":
    main()
