#!/usr/bin/env python
"""G4: per-camera pixel bounding boxes (largest connected component of the segmentation mask) of every poker candidate (simulator segmentation; DIAGNOSTIC / TRAINING-SIDE ONLY, never an inference input).
The poker scene generator is process-deterministic, so `env.reset(seed)` reproduces each stored G1 scene; reproduction is VERIFIED against the stored RGB."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from phase_g05_clean_grounding_collect import region_masks  # noqa: E402

from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

D = Path("outputs/data/safediff_vla_g1_counterfactual/select_poker")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", type=int, required=True); ap.add_argument("--n", type=int, required=True); a = ap.parse_args()
    env = make_env(VLABenchEnv(task="select_poker"), n_envs=1, use_async_envs=False)["select_poker"][0]; out = {}
    for i in range(a.start, a.start + a.n):
        seed = 10000 + i; sid = f"select_poker_{seed}"
        try:
            set_seed(seed); raw, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True}); impl = env.envs[0]
            while not hasattr(impl, "_env"): impl = impl.env
            z = np.load(D / f"{sid}.npz"); rec = json.load(open(D / f"{sid}.json"))
            px = raw["pixels"]; diff = [float(np.abs(np.asarray(px[k])[0].astype(int) - z["rgb"][c].astype(int)).mean()) for c, k in enumerate(("image", "second_image", "wrist_image"))]
            masks = region_masks(impl); boxes = {}
            for c in rec["candidates"]:
                b = []
                for cam in range(3):
                    m = masks["per_camera"][cam][c["name"]]; lab, k = ndimage.label(m)
                    if k == 0: b.append(None); continue
                    big = int(np.argmax(ndimage.sum(m, lab, range(1, k + 1)))) + 1  # largest connected component: drops stray 1-px segmentation specks
                    ys, xs = np.nonzero(lab == big); b.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
                boxes[c["name"]] = b
            out[sid] = {"boxes_xyxy_per_camera": boxes, "rgb_mean_abs_diff_vs_stored": diff}
        except Exception as ex:  # noqa: BLE001
            print("skip", sid, repr(ex)[:80], flush=True)
        if (i + 1 - a.start) % 10 == 0: print(a.start, i + 1 - a.start, flush=True)
    (D / f"bboxes_{a.start}_{a.n}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
