#!/usr/bin/env python
"""G23 STEPS 1/3/4 on expert post-grasp states (held-out episodes, frozen V2 decoder, NO training). Frames: from the demonstrated grasp to the end of the episode (every 4th step), tasks poker / mahjong / toy (fruit secondary). Decoder conditioning exactly as in the closed-loop POST_GRASP pipeline: the FROZEN demonstration grasp anchor as target_xyz, grounded feature + phase from the frame itself (phase = 1 after the grasp).
Measured: decoder gripper chunk (decoded raw units; >0.5 = open command): any-open / first-open index / open fraction; versus the expert's own gripper chunk (does the expert release inside this 50-step window? where?). Interventions on the SAME frame: phase flag forced to 0 (counterfactual), image tokens masked (as G10-V1).
Release-driver regression (Step 4): per frame logistic regression of 'decoder opens within the chunk' on elapsed time since grasp, distance of the EE to the demonstration's own release point (goal proxy; the dataset stores no container pose) and the expert's remaining time to release."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import g17_decoder as D  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy, pad_or_crop_horizon  # noqa: E402
from lerobot.policies.safediff_vla.utils import IMAGE_MODALITY  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g23"); TASKS = ["select_poker", "select_mahjong", "select_toy", "select_fruit"]; BS = 64


def main():
    E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); E6 = E[E.task.isin(["select_poker", "select_mahjong", "select_fruit", "select_drink", "select_toy", "select_book"])]; eps = sorted(E6.episode.astype(int).tolist()); perm = np.random.default_rng(0).permutation(eps); ho = sorted(set(perm[:int(0.1 * len(eps))].tolist()))
    ho = [e for e in ho if E6[E6.episode == e].task.iloc[0] in TASKS]; ep2task = {int(r.episode): r.task for r in E6.itertuples()}; ep2tc = {int(r.episode): int(r.t_close) for r in E6.itertuples()}; ep2anc = {int(r.episode): np.array(r.anchor, float) for r in E6.itertuples()}
    RG = pd.read_pickle(OUT / "release_gt.pkl"); rel = {int(r.ep): (None if pd.isna(r.t_rel) else int(r.t_rel)) for r in RG.itertuples()}; eerel = {int(r.ep): r.ee_rel for r in RG.itertuples()}
    FROM = np.array(LeRobotDatasetMetadata("lerobot/vlabench_unified").episodes["dataset_from_index"]); policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); dec = policy.decoder; D.capture_install(policy); mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std; s_mean, s_std = policy.state_pos_mean, policy.state_pos_std
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", episodes=ho), policy=policy.config, rename_map=RENAME_MAP, batch_size=BS); ld = torch.utils.data.DataLoader(make_dataset(cfg), batch_size=BS, shuffle=True, num_workers=6, drop_last=True)
    def prep(b):
        for k in list(b):
            if k.startswith("observation.images") and b[k].dtype == torch.uint8: b[k] = b[k].float() / 255.0
        return pre(b)
    rows = []; it = iter(ld); nb = 0; H = policy.config.action_horizon; seen = {t: 0 for t in TASKS}
    while nb < 160:
        try: b = prep(next(it))
        except StopIteration: it = iter(ld); continue
        nb += 1
        with torch.no_grad(): policy.forward(b)
        ctx = dec._ctx; epi = b["episode_index"].cpu().numpy().astype(int); fi = b["index"].cpu().numpy().astype(int) - FROM[epi]; tc = np.array([ep2tc[e] for e in epi]); post = fi >= tc
        if not post.any(): continue
        ix = np.where(post)[0]; sel = torch.as_tensor(ix, device=ctx["tok"].device); c = {k: (v[sel] if torch.is_tensor(v) and k != "gt" else v) for k, v in ctx.items() if k != "gt"}
        T = ((torch.as_tensor(np.stack([ep2anc[e] for e in epi[ix]]), dtype=std.dtype, device=std.device) - mean) / std); exp_g = b["action"][sel][..., 6].cpu().numpy() if False else None
        clean = policy._encode_action_target(pad_or_crop_horizon(b["action"], H))[sel]; eg = (policy._decode_action_prediction(torch.cat([clean, clean[..., :0]], -1))[..., 6] * gs + gm).cpu().numpy() if False else None
        ee = (c["state"][:, :3] * s_std + s_mean).cpu().numpy(); variants = {}
        c0 = dict(c); variants["base"] = D.run(dec, c0, T)[0]; cp = dict(c); cp["phase"] = torch.zeros_like(c["phase"]); variants["phase0"] = D.run(dec, cp, T)[0]
        pm = c["pad"].clone(); pm[c["mod"] == IMAGE_MODALITY] = False; ci = dict(c); ci["pad"] = pm; variants["noimg"] = D.run(dec, ci, T)[0]
        graw = {k: (policy._decode_action_prediction(v)[..., 6] * gs + gm).cpu().numpy() for k, v in variants.items()}; raw_expert = b["action"][sel].cpu().numpy()
        for j, i in enumerate(ix):
            e = int(epi[i]); t = int(fi[i]); n_ep = int(FROM[e] * 0)  # placeholder
            tr = rel[e]; xp = raw_expert[j][:, 6] if raw_expert.ndim == 3 else None
            rows.append({"task": ep2task[e], "ep": e, "t": t, "tc": ep2tc[e], "t_rel": tr, "d_rel": (float(np.linalg.norm(ee[j] - eerel[e])) if eerel[e] is not None else np.nan), "ee_z": float(ee[j, 2]), **{f"g_{k}": graw[k][j] for k in graw}, "expert_g": (xp.copy() if xp is not None else None)})
    df = pd.DataFrame(rows); df.to_pickle(OUT / "g23_fixed.pkl"); print("frames", len(df), df.task.value_counts().to_dict())


if __name__ == "__main__":
    main()
