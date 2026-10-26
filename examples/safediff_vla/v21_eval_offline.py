"""[StagePlan-v2.1] Same-scene counterfactual offline gate on the pilot VAL scenes (cached frozen-backbone tokens).
Usage: uv run python examples/safediff_vla/v21_eval_offline.py --head v2|<v2.1 dir> [--which val|train]"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).parent))
from v21_common import OUT, cf_metrics, load_policy, scene_split  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import StagePlanV2Head  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v21 import StagePlanV21Head  # noqa: E402


def load_head(spec, dim):
    if spec == "v2":
        d = Path("outputs/train/safediff_stageplan_v2"); h = StagePlanV2Head(dim, **json.load(open(d / "stageplan_config.json"))["head"])
    else:
        d = Path(spec); h = StagePlanV21Head(dim, **json.load(open(d / "stageplan_config.json"))["head"])
    h.load_state_dict(load_file(d / "stageplan_head.safetensors")); return h.cuda().eval()


@torch.no_grad()
def predict_all(head, pol, cache, bs=64, aux=False):
    mean, std = pol.action_pos_mean.cpu(), pol.action_pos_std.cpu(); G, R = {}, {}
    for i in range(0, len(cache), bs):
        E = cache[i:i + bs]; tok = torch.stack([e["tok"] for e in E]).float().cuda(); pad = torch.stack([e["pad"] for e in E]).cuda(); mod = torch.stack([e["mod"] for e in E]).cuda(); st = torch.stack([e["state"] for e in E]).cuda()
        o = head(tok, pad, mod, st)
        g = (o["grasp_xyz"].cpu() * std + mean).numpy().astype(np.float64); r = (o["ref_xyz"].cpu() * std + mean).numpy().astype(np.float64) if "ref_xyz" in o else g
        for e, gg, rr in zip(E, g, r): G[(e["scene_id"], e["ci"])] = gg; R[(e["scene_id"], e["ci"])] = rr
    return G, R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="v2"); ap.add_argument("--which", default="val"); ap.add_argument("--out", default=None); a = ap.parse_args()
    rows = pickle.load(open(OUT / "referent_table.pkl", "rb")); split = json.load(open(OUT / "pilot_scene_split.json")); cache = torch.load(OUT / "pilot_token_cache.pt", weights_only=False)
    pol, _ = load_policy(); head = load_head(a.head, pol._multimodal_latent_dim()); G, R = predict_all(head, pol, cache)
    res = {"grasp": cf_metrics(G, rows, split, a.which)}
    if a.head != "v2": res["ref"] = cf_metrics(R, rows, split, a.which, key="ref")
    for k, v in res.items():
        print(f"== {k} ({a.which} scenes, same-scene instruction swaps)")
        for t, m in v.items(): print(f"  {t:15s}", {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in m.items()})
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()
