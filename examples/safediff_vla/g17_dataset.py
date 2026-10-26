#!/usr/bin/env python
"""G17-E/F on the 6-task dataset (offline, NO training of any policy part). Frames: PRE_GRASP, demonstration-anchor-valid, from the G11/G16 held-out episodes (E) and train episodes (F probe fit).
E: anchor sensitivity (gain = d chunk / d anchor, +-5cm x, +-5cm y, +-3cm z around the demonstration anchor) per task, at each injection location (same conditions as g17_capture, GT anchor); natural target-z spread per task is reported so within-task Z perturbations can be judged against it.
F: ridge probes (diagnostic only): residual r_k = action_k - EE (k = 10, 30; x,y,z) from [final hidden token k] (H), [geometry features of e = anchor - EE] (G), or both (H+G); fit on train-episode frames, R^2 on held-out-episode frames. Final hidden = action-head input of the frozen V2 (anchor = demonstration anchor)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
import g17_decoder as D  # noqa: E402
from g16_geometry import GeometryConditioner  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy, pad_or_crop_horizon  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g17"); TASKS = ["select_poker", "select_mahjong", "select_fruit", "select_drink", "select_toy", "select_book"]; DX, DZ = 0.05, 0.03; NB_E, NB_TR = 60, 100


def main():
    rng = np.random.default_rng(0); E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); E = E[E.task.isin(TASKS)]; eps = sorted(E.episode.astype(int).tolist()); perm = np.random.default_rng(0).permutation(eps); hold = set(perm[:int(0.1 * len(eps))].tolist()); tr_eps = [e for e in eps if e not in hold]; ho_eps = sorted(hold); ep2task = dict(zip(E.episode.astype(int), E.task))
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); dec = policy.decoder; D.capture_install(policy); mean, std = policy.action_pos_mean, policy.action_pos_std; s_mean, s_std = policy.state_pos_mean, policy.state_pos_std
    ck = torch.load("outputs/train/safediff_vla_g16/geometry_conditioner.pt", weights_only=False); geo = GeometryConditioner(float(ck["alpha_max"]), out_dim=dec.action_head.in_features).to("cuda"); geo.load_state_dict(ck["state"]); geo.eval()
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    def loader(ep_list, bs=16):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", episodes=ep_list), policy=policy.config, rename_map=RENAME_MAP, batch_size=bs); return torch.utils.data.DataLoader(make_dataset(cfg), batch_size=bs, shuffle=True, num_workers=6, drop_last=True)
    def prep(b):
        for k in list(b):
            if k.startswith("observation.images") and b[k].dtype == torch.uint8: b[k] = b[k].float() / 255.0
        return pre(b)
    def frames(ld, nb):
        it = iter(ld); n = 0
        while n < nb:
            try: b = prep(next(it))
            except StopIteration: it = iter(ld); continue
            n += 1
            with torch.no_grad(): policy.forward(b)
            ctx = dec._ctx; gt, valid = ctx["gt"]; m = valid & (ctx["phase"] == 0)
            if not m.any(): continue
            clean = policy._encode_action_target(pad_or_crop_horizon(b["action"], policy.config.action_horizon))[..., :3] * std + mean; ee = ctx["state"][:, :3] * s_std + s_mean; epi = b["episode_index"].cpu().numpy() if "episode_index" in b else None
            yield {k: (v[m] if torch.is_tensor(v) else v) for k, v in ctx.items() if k != "gt"}, gt[m], clean[m], ee[m], [ep2task[int(x)] for x in epi[m.cpu().numpy()]]
    # ---------------- E ----------------
    conds = [("V2", None)] + [("G4", None)] + [("move", l) for l in D.LOCS[1:]] + [("extra", l) for l in D.LOCS] + [("geo", l) for l in D.LOCS]
    rows = []; zstat = []
    for ctx, gt, clean, ee, tasks in frames(loader(ho_eps), NB_E):
        T = {"gt": gt}; pert = [(0, DX, "x"), (1, DX, "y"), (2, DZ, "z")]
        for ax, d, nm in pert:
            for sg, sn in ((1, "p"), (-1, "m")): v = gt.clone(); v[:, ax] += sg * d / std[ax]; T[nm + sn] = v
        zstat += list(zip(tasks, (gt[:, 2] * std[2] + mean[2]).cpu().numpy()))
        def xyz(T_, name, loc):
            if name == "V2": out = D.run(dec, ctx, T_)[0]
            elif name == "G4": out = D.run(dec, ctx, T_, in_scale=4.0)[0]
            elif name == "move": out = D.place(dec, ctx, T_, loc, "move")
            elif name == "extra": out = D.place(dec, ctx, T_, loc, "extra")
            else: out = D.place(dec, ctx, T_, loc, "extra", vec=geo(T_ * std + mean - ee, std))
            return (policy._decode_action_prediction(out)[..., :3] * std + mean)
        for name, loc in conds:
            r = {}
            for ax, d, nm in pert: dif = xyz(T[nm + "p"], name, loc) - xyz(T[nm + "m"], name, loc); r[nm] = {"first10": (dif[:, :10, ax].mean(1) / (2 * d)).cpu().numpy(), "late": (dif[:, 30:, ax].mean(1) / (2 * d)).cpu().numpy()}
            rows.append({"cond": (name, loc), "tasks": tasks, "gain": r})
    pickle.dump({"rows": rows, "zstat": zstat}, open(OUT / "g17_dataset_E.pkl", "wb")); print("E done", len(rows), flush=True)
    # ---------------- F ----------------
    def collect(ld, nb):
        H, G, Y = [], [], []
        for ctx, gt, clean, ee, tasks in frames(ld, nb):
            _, rec = D.run(dec, ctx, gt, record=True); H.append(torch.stack([rec["x_final"][:, 9], rec["x_final"][:, 29]], 1).cpu().numpy()); G.append(GeometryConditioner.features(gt * std + mean - ee, std).cpu().numpy()); Y.append(torch.stack([clean[:, 9] - ee, clean[:, 29] - ee], 1).cpu().numpy())
        return np.concatenate(H), np.concatenate(G), np.concatenate(Y)
    Htr, Gtr, Ytr = collect(loader(tr_eps), NB_TR); Hte, Gte, Yte = collect(loader(ho_eps), NB_E); print("probe frames train/test", len(Htr), len(Hte), flush=True); out = {}
    for ki, k in enumerate((10, 30)):
        for ci, c in enumerate("xyz"):
            for nm, (Xa, Xb) in {"H": (Htr[:, ki], Hte[:, ki]), "G": (Gtr, Gte), "H+G": (np.c_[Htr[:, ki], Gtr * 10], np.c_[Hte[:, ki], Gte * 10])}.items():
                mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; m = RidgeCV(alphas=(0.1, 1, 10, 100, 1000)).fit((Xa - mu) / sd, Ytr[:, ki, ci]); p = m.predict((Xb - mu) / sd); y = Yte[:, ki, ci]; out[(k, c, nm)] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    pickle.dump({"probe": out, "n": (len(Htr), len(Hte))}, open(OUT / "g17_dataset_F.pkl", "wb")); print("F done")


if __name__ == "__main__":
    main()
