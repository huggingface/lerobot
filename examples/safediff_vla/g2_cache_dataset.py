#!/usr/bin/env python
"""G2: cache FROZEN-v2 VLM latents of ORIGINAL vlabench_unified `select_fruit` frames (forward only; dataset language = the
referent-bearing training instruction, exactly as v2 saw it). Episode-level split (v2's own held-out rule first):
    HELD-OUT  = last ceil(10%) episodes of every task string          -> retention TEST (never trained/selected on)
    VAL-D     = last ceil(10%) of the remaining episodes per string   -> checkpoint selection (retention guard)
    TRAIN-D   = everything else                                       -> dataset-domain training
Label = v2's own anchor (first open->close commanded xyz, robot frame), supervised only while the transition is within the 50-step
window (k = steps to anchor <= 49), EXACTLY as v2 was supervised.
TRAIN-D frames are stratified by k: EARLY 31<=k<=49, MID 11<=k<=30, NEAR 0<=k<=10, `PER_BUCKET` random frames per episode per bucket.
HELD-OUT / VAL-D frames: f0 (k = t_close, unsupervised by v2), k=45, 30, 10, with the true text; HELD-OUT also with 2 swapped referents."""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, REPO_ID  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402

import argparse
D0 = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit"); PER_BUCKET = 2
BUCKETS = {"EARLY": (31, 49), "MID": (11, 30), "NEAR": (0, 10)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="select_fruit"); ap.add_argument("--prefix", default="g2"); ap.add_argument("--out", default="outputs/eval/safediff_vla_g2_offline"); a = ap.parse_args()
    OUT = Path(a.out); TASK = a.task; P = a.prefix
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
    from lerobot.processor.rename_processor import rename_batch_keys
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(D0 / "episodes.json"); df = df[df.has_anchor & (df.task == TASK)]
    held, vald = [], []
    for _, g in df.groupby("instruction"):
        g = g.sort_values("episode"); eps = g.episode.tolist(); h = eps[len(eps) - math.ceil(len(eps) * 0.1):]; held += h
        rest = [e for e in eps if e not in h]; vald += rest[len(rest) - math.ceil(len(rest) * 0.1):]
    role = {e: "held" for e in held} | {e: "vald" for e in vald}; df["role"] = df.episode.map(lambda e: role.get(e, "train"))
    print(df.role.value_counts().to_dict(), flush=True)
    instr = sorted(df.instruction.unique()); ds = LeRobotDataset(REPO_ID); from_idx = ds.meta.episodes["dataset_from_index"]; cams = ds.meta.camera_keys
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    rng = np.random.default_rng(0)

    def enc(ep, f, text):
        item = ds[int(from_idx[int(ep)]) + f]; it = dict(item); it["task"] = text
        b = {kk: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for kk, v in it.items()}
        for c in cams:
            if c in b and b[c].dtype == torch.uint8: b[c] = b[c].to(torch.float32) / 255.0
        with torch.no_grad(): tok, pad, mod = policy._encode_multimodal_latent(pre(rename_batch_keys(b, RENAME_MAP)))
        mod = mod.clone(); mod[~pad.to(torch.bool)] = 3; return tok[0].half().cpu(), mod[0].to(torch.int8).cpu()

    def pack(T, M):
        nmax = max(t.shape[0] for tt in T for t in tt); S, V = len(T), len(T[0])
        tok = torch.zeros(S, V, nmax, 960, dtype=torch.float16); mod = torch.full((S, V, nmax), 3, dtype=torch.int8)
        for i in range(S):
            for j in range(V): tok[i, j, :T[i][j].shape[0]] = T[i][j]; mod[i, j, :M[i][j].shape[0]] = M[i][j]
        return tok, mod

    # ---- TRAIN-D (stratified by k) ----
    T, M, A, meta = [], [], [], []
    for _, e in df[df.role == "train"].iterrows():
        tc = int(e.t_close)
        for bname, (lo, hi) in BUCKETS.items():
            ks = [k for k in range(lo, hi + 1) if tc - k >= 0]
            if not ks: continue
            for k in rng.choice(ks, size=min(PER_BUCKET, len(ks)), replace=False):
                t, m = enc(e.episode, tc - int(k), e.instruction); T.append([t]); M.append([m]); A.append(e.anchor); meta.append({"episode": int(e.episode), "k": int(k), "bucket": bname, "text": e.instruction})
    tok, mod = pack(T, M); torch.save({"tokens": tok, "modality": mod, "anchors": torch.tensor(np.array(A), dtype=torch.float32), "meta": meta}, OUT / f"{P}_train_dataset_cache.pt")
    print("TRAIN-D", tuple(tok.shape), {b: sum(m["bucket"] == b for m in meta) for b in BUCKETS}, flush=True)

    # ---- VAL-D / HELD-OUT (fixed k) ----
    for role_name, nsw in (("vald", 0), ("held", 2)):
        T, M, A, meta = [], [], [], []
        for _, e in df[df.role == role_name].iterrows():
            tc = int(e.t_close); others = [s for s in instr if s != e.instruction]; texts = [e.instruction, *[rng.choice(others) for _ in range(nsw)]]
            for name, k in (("f0", tc), ("k45", 45), ("k30", 30), ("k10", 10)):
                f = tc - k
                if f < 0: continue
                tt, mm = zip(*[enc(e.episode, f, x) for x in texts]); T.append(list(tt)); M.append(list(mm)); A.append(e.anchor)
                meta.append({"episode": int(e.episode), "frame": name, "k": int(k), "texts": list(texts), "image_frame": int(f)})
        tok, mod = pack(T, M); torch.save({"tokens": tok, "modality": mod, "anchors": torch.tensor(np.array(A), dtype=torch.float32), "meta": meta}, OUT / f"{P}_{role_name}_cache.pt")
        print(role_name, tuple(tok.shape), {k: sum(m["frame"] == k for m in meta) for k in ("f0", "k45", "k30", "k10")}, flush=True)
    (OUT / f"{P}_dataset_roles.json").write_text(json.dumps({"counts": df.role.value_counts().to_dict(), "per_bucket_per_episode": PER_BUCKET}))


if __name__ == "__main__":
    main()
