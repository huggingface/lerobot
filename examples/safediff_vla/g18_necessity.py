#!/usr/bin/env python
"""G18-A TARGET NECESSITY AUDIT (no training, no rollout). Frozen V2 decoder on the G11/G16/G17 episode-disjoint held-out set (10% of the 6-task episodes = 300). Visual memory, state, language, grounded feature are the dataset's own; ONLY target-related conditioning is changed:
  N0 normal (V2's own predicted target_xyz + grounded feature, exactly the training-time path) | N1 target_xyz branch OFF | N2 target_xyz of ANOTHER episode of the same task (shuffled within the batch) | N4 target_xyz AND grounded feature OFF (encoder(0) as in G10-Z3/Z4) | N1b grounded feature OFF only (extra).
  N3/N5 (same-scene alternative referent) are NOT evaluated: the counterfactual scenes (outputs/data/safediff_vla_g1_counterfactual) have no expert actions, so no BC loss exists for them.
Loss = per-horizon-bin MSE against the expert action in the policy's encoded space (position xyz normalised, rot sin/cos, grip), padded steps excluded, same masking as policy.forward. Strata from the G11 audit: EARLY t<20, MID t<tc-15, NEAR t<=tc, POST t>tc (tc = demonstrated close step). Sensitivity: N0 vs N2 chunks (frames whose task has >=2 samples in the batch).
Probe (diagnostic only): ridge from visual/state representation (before target conditioning) to the demonstration anchor xyz."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
import g17_decoder as D  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy, pad_or_crop_horizon, pad_or_crop_mask  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g18"); TASKS = ["select_poker", "select_mahjong", "select_fruit", "select_drink", "select_toy", "select_book"]; HB = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50)]; COND = ["N0", "N1", "N2", "N4", "N1b", "N0m"]; COMP = {"xy": slice(0, 2), "z": slice(2, 3), "pos": slice(0, 3), "rot": slice(3, 9), "grip": slice(9, 10)}; NB_E, NB_TR, BS = 90, 70, 64


def main():
    OUT.mkdir(parents=True, exist_ok=True); E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); E = E[E.task.isin(TASKS)]; eps = sorted(E.episode.astype(int).tolist()); perm = np.random.default_rng(0).permutation(eps); hold = set(perm[:int(0.1 * len(eps))].tolist()); tr_eps = [e for e in eps if e not in hold]; ho_eps = sorted(hold)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    FROM = np.array(LeRobotDatasetMetadata("lerobot/vlabench_unified").episodes["dataset_from_index"]); ep2task = {int(r.episode): TASKS.index(r.task) for r in E.itertuples()}; ep2tc = {int(r.episode): int(r.t_close) for r in E.itertuples()}; ep2anc = {int(r.episode): np.array(r.anchor, float) for r in E.itertuples()}; print("held-out episodes", len(ho_eps), "train", len(tr_eps), flush=True)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); dec = policy.decoder; D.capture_install(policy); mean, std = policy.action_pos_mean, policy.action_pos_std; s_mean, s_std = policy.state_pos_mean, policy.state_pos_std; H = policy.config.action_horizon
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    def loader(ep_list): cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", episodes=ep_list), policy=policy.config, rename_map=RENAME_MAP, batch_size=BS); return torch.utils.data.DataLoader(make_dataset(cfg), batch_size=BS, shuffle=True, num_workers=6, drop_last=True)
    def prep(b):
        for k in list(b):
            if k.startswith("observation.images") and b[k].dtype == torch.uint8: b[k] = b[k].float() / 255.0
        return pre(b)
    def stream(ld, nb):
        it = iter(ld); n = 0
        while n < nb:
            try: b = prep(next(it))
            except StopIteration: it = iter(ld); continue
            n += 1
            with torch.no_grad(): policy.forward(b)
            ctx = dec._ctx; epi = b["episode_index"].cpu().numpy().astype(int); fi = b["index"].cpu().numpy().astype(int) - FROM[epi]; yield b, ctx, epi, fi
    S = {c: {k: np.zeros((6, 4, 5)) for k in COMP} for c in COND}; CNT = {c: np.zeros((6, 4, 5)) for c in COND}; sens = []; rng = np.random.default_rng(1); maxdiff = 0.0
    for bi, (b, ctx, epi, fi) in enumerate(stream(loader(ho_eps), NB_E)):
        tk = np.array([ep2task[e] for e in epi]); tc = np.array([ep2tc[e] for e in epi]); st = np.where(fi < 20, 0, np.where(fi < tc - 15, 1, np.where(fi <= tc, 2, 3)))
        clean = policy._encode_action_target(pad_or_crop_horizon(b["action"], H)); valid = ~pad_or_crop_mask(b["action_is_pad"], H) if "action_is_pad" in b else torch.ones(clean.shape[:2], dtype=torch.bool, device=clean.device)
        T0 = ctx["target"]; idx = np.arange(len(epi)); sh = idx.copy(); ok = np.zeros(len(epi), bool)
        for t in np.unique(tk):
            g = idx[tk == t]
            if len(g) >= 2: sh[g] = rng.permutation(g) if True else g; ok[g] = True
        for t in np.unique(tk):  # make sure no frame keeps its own target (derangement by roll where permutation fixed a point or same episode)
            g = idx[tk == t]
            if len(g) >= 2:
                for i in g:
                    if sh[i] == i or epi[sh[i]] == epi[i]: j = [k for k in g if epi[k] != epi[i]]; sh[i] = rng.choice(j) if j else i; ok[i] = ok[i] and bool(j)
        ctx0 = {**ctx, "feat": torch.zeros_like(ctx["feat"])}; outs = {"N0": D.run(dec, ctx, T0)[0], "N1": D.run(dec, ctx, T0, in_scale=0.0)[0], "N2": D.run(dec, ctx, T0[torch.as_tensor(sh, device=T0.device)])[0], "N4": D.run(dec, ctx0, T0, in_scale=0.0)[0], "N1b": D.run(dec, ctx0, T0)[0]}; maxdiff = max(maxdiff, float((outs["N0"] - dec(ctx["tok"], ctx["pad"], ctx["state"], None, ctx["target"], None, None, ctx["feat"], ctx["phase"])).abs().max()))
        outs["N0m"] = outs["N0"]; sq = {c: (o - clean) ** 2 for c, o in outs.items()}; vm = valid.cpu().numpy(); okt = torch.as_tensor(ok, device=clean.device)
        for c in COND:
            e2 = sq[c].cpu().numpy(); use = ok if c in ("N2", "N0m") else np.ones(len(epi), bool)
            for i in np.where(use)[0]:
                for hi, (a, z) in enumerate(HB):
                    m = vm[i, a:z]
                    if not m.any(): continue
                    CNT[c][tk[i], st[i], hi] += m.sum()
                    for k, sl in COMP.items(): S[c][k][tk[i], st[i], hi] += (e2[i, a:z, sl].mean(-1) * m).sum()
        # sensitivity N0 vs N2
        xyz = lambda o: (o[..., :3] * std + mean).cpu().numpy(); x0, x2 = xyz(outs["N0"]), xyz(outs["N2"]); ee = (ctx["state"][:, :3] * s_std + s_mean).cpu().numpy(); Ta = (T0 * std + mean).cpu().numpy(); Tb = Ta[sh]
        def cos(x, T, i): v = x[i, :10, :2].mean(0) - ee[i, :2]; d = T[i, :2] - ee[i, :2]; return float(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))
        for i in np.where(ok)[0]: sens.append({"task": tk[i], "st": st[i], "anc_sep": np.linalg.norm(Ta[i, :2] - Tb[i, :2]), "sep10": np.linalg.norm((x0[i, :10, :2] - x2[i, :10, :2]).mean(0)), "sepEnd": np.linalg.norm(x0[i, 49, :2] - x2[i, 49, :2]), "cos0_true": cos(x0, Ta, i), "cos2_true": cos(x2, Ta, i), "cos2_used": cos(x2, Tb, i), "off0": np.linalg.norm(x0[i, 0] - ee[i]), "off2": np.linalg.norm(x2[i, 0] - ee[i])})
        if (bi + 1) % 20 == 0: print("batch", bi + 1, flush=True)
    print("runner vs dec.forward max diff", maxdiff, flush=True); pickle.dump({"S": S, "CNT": CNT, "sens": sens, "HB": HB}, open(OUT / "g18_loss.pkl", "wb"))
    # ---------------- probe ----------------
    @torch.no_grad()
    def collect(ld, nb):
        R = {k: [] for k in ("state", "vis", "hid", "v2", "y", "task", "ee")}
        for b, ctx, epi, fi in stream(ld, nb):
            tc = np.array([ep2tc[e] for e in epi]); pre_m = fi <= tc
            if not pre_m.any(): continue
            m = torch.as_tensor(pre_m, device=ctx["tok"].device); mem = dec.latent_projection(ctx["tok"]); pad = ctx["pad"].float().unsqueeze(-1); vis = (mem * pad).sum(1) / pad.sum(1).clamp(min=1); ctx0 = {**ctx, "feat": torch.zeros_like(ctx["feat"])}; _, rec = D.run(dec, ctx0, ctx["target"], in_scale=0.0, record=True)
            R["state"].append(ctx["state"][m, :7].cpu().numpy()); R["vis"].append(vis[m].cpu().numpy()); R["hid"].append(rec["x_final"].mean(1)[m].cpu().numpy()); R["v2"].append((ctx["target"] * std + mean)[m].cpu().numpy()); R["y"].append(np.stack([ep2anc[e] for e in epi[pre_m]])); R["task"].append(np.array([ep2task[e] for e in epi[pre_m]])); R["ee"].append((ctx["state"][:, :3] * s_std + s_mean)[m].cpu().numpy())
        return {k: np.concatenate(v) for k, v in R.items()}
    tr = collect(loader(tr_eps), NB_TR); te = collect(loader(ho_eps), NB_E); print("probe frames", len(tr["y"]), len(te["y"]), flush=True); P = {}
    def fit(Xa, Ya, Xb):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; m = RidgeCV(alphas=(0.1, 1, 10, 100, 1000)).fit((Xa - mu) / sd, Ya); return m.predict((Xb - mu) / sd)
    def r2(y, p): return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    feats = {"EE only": lambda d: d["ee"], "state": lambda d: d["state"], "vis (pooled memory)": lambda d: d["vis"], "vis+state": lambda d: np.c_[d["vis"], d["state"]], "hid (decoder, target+feature OFF)": lambda d: d["hid"], "hid+state": lambda d: np.c_[d["hid"], d["state"]]}
    for nm, f in feats.items():
        p = fit(f(tr), tr["y"], f(te)); P[(nm, "pooled")] = r2(te["y"], p)
        for t in (0, 2): mt, mte = tr["task"] == t, te["task"] == t; P[(nm, "within " + TASKS[t][7:])] = r2(te["y"][mte], p[mte]); P[(nm, "fit-in-task " + TASKS[t][7:])] = r2(te["y"][mte], fit(f(tr)[mt], tr["y"][mt], f(te)[mte]))
    P[("V2 target_point_head (uses grounded feature)", "pooled")] = r2(te["y"], te["v2"])
    for t in (0, 2): mte = te["task"] == t; P[("V2 target_point_head (uses grounded feature)", "within " + TASKS[t][7:])] = r2(te["y"][mte], te["v2"][mte])
    P[("anchor std (cm) x/y/z", "poker")] = te["y"][te["task"] == 0].std(0) * 100; P[("anchor std (cm) x/y/z", "fruit")] = te["y"][te["task"] == 2].std(0) * 100
    pickle.dump({"P": P, "n": (len(tr["y"]), len(te["y"]))}, open(OUT / "g18_probe.pkl", "wb")); print("done")


if __name__ == "__main__":
    main()
