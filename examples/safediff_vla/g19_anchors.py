#!/usr/bin/env python
"""G19 helper: the raw G4d predicted anchor (robot frame, m) for each of the 797 fixed-observation bank frames (same grounder call as g16/g17: computed from the ORIGINAL batch)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g13_interventions import KEYS  # noqa: E402
from g17_capture import load_bank  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g19")


def main():
    OUT.mkdir(parents=True, exist_ok=True); B, IM, SRC = load_bank(); dev = "cuda"; policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to(dev).eval(); mean, std = policy.action_pos_mean, policy.action_pos_std; enc = SafeDiffVLAPolicy._encode_multimodal_latent; grounder = G5.Grounder("g4d", policy); A = []
    for i, e in enumerate(B):
        b = {k: torch.from_numpy(IM[i][c]).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev) for c, k in enumerate(KEYS)}; b["observation.state"] = torch.from_numpy(e["state_norm"]).float().unsqueeze(0).to(dev); b["observation.language.tokens"] = torch.from_numpy(e["lang_tokens"]).long().unsqueeze(0).to(dev); b["observation.language.attention_mask"] = torch.from_numpy(e["lang_mask"]).bool().unsqueeze(0).to(dev)
        with torch.no_grad(): lat = enc(policy, b); A.append((grounder.compute(*lat, b)["xyz_n"].reshape(1, -1)[0, :3] * std + mean).cpu().numpy())
    A = np.array(A); np.save(OUT / "g4d_anchors.npy", A); gt = np.array([e["gt_anchor_robot"] for e in B]); print("saved", A.shape, "mean |G4d-GT| xy/z/3d cm:", np.linalg.norm(A[:, :2] - gt[:, :2], axis=1).mean() * 100, abs(A[:, 2] - gt[:, 2]).mean() * 100, np.linalg.norm(A - gt, axis=1).mean() * 100)


if __name__ == "__main__":
    main()
