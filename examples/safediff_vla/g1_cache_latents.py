#!/usr/bin/env python
"""G1: cache the FROZEN v2 VLM prefix latents for every (scene, referent) of the counterfactual dataset (forward only).
Text enters the VLM prefix, so latents depend on the instruction -> cache per (scene, referent). Output per task:
    tokens [S,KMAX,NMAX,960] fp16, modality [S,KMAX,NMAX] int8 (3=pad), anchors [S,KMAX,3] m (robot frame), valid [S,KMAX],
    cellfrac [S,KMAX,192] (referent mask mass per image token, cam-major row-major; TRAINING-ONLY supervision),
    names/scene ids/split, and the policy's action_pos_mean/std.  No gradients, no weights changed."""
from __future__ import annotations

import copy, json, sys
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402

from lerobot.envs import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

D = Path("outputs/data/safediff_vla_g1_counterfactual"); KMAX = 4


@torch.no_grad()
def encode(policy, batch):
    tok, pad, mod = policy._encode_multimodal_latent(batch)
    mod = mod.clone(); mod[~pad.to(torch.bool)] = 3
    return tok[0].half().cpu(), mod[0].to(torch.int8).cpu()


def main():
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    consts = {"pos_mean": policy.action_pos_mean.cpu().numpy().tolist(), "pos_std": policy.action_pos_std.cpu().numpy().tolist()}
    (D / "policy_constants.json").write_text(json.dumps(consts))
    for task in ("select_poker", "select_fruit"):
        cfg = VLABenchEnv(task=task)
        pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        splits = json.load(open(D / task / "splits.json")); sids = sorted(splits)
        toks, mods, anc, val, cf, names, held, instr = [], [], [], [], [], [], [], []
        for n, sid in enumerate(sids):
            rec = json.load(open(D / task / f"{sid}.json")); z = np.load(D / task / f"{sid}.npz")
            raw = {"pixels": {k: z["rgb"][i][None] for i, k in enumerate(("image", "second_image", "wrist_image"))}, "agent_pos": z["state"][None].astype(np.float64)}
            T, M = [], []
            for c in rec["candidates"]:
                o = preprocess_observation(copy.deepcopy(raw)); o["task"] = [c["instruction"]]
                t, m = encode(policy, pre(env_pre(o))); T.append(t); M.append(m)
            toks.append(T); mods.append(M); anc.append([c["anchor_robot"] for c in rec["candidates"]]); names.append([c["name"] for c in rec["candidates"]])
            instr.append([c["instruction"] for c in rec["candidates"]]); cf.append(z["cellfrac"].reshape(len(T), -1))
            if (n + 1) % 50 == 0: print(task, n + 1, "/", len(sids), flush=True)
        nmax = max(t.shape[0] for T in toks for t in T); S = len(sids)
        tokens = torch.zeros(S, KMAX, nmax, 960, dtype=torch.float16); modality = torch.full((S, KMAX, nmax), 3, dtype=torch.int8)
        A = np.zeros((S, KMAX, 3), np.float32); V = np.zeros((S, KMAX), bool); C = np.zeros((S, KMAX, 192), np.float32)
        for s in range(S):
            for k, (t, m) in enumerate(zip(toks[s], mods[s])):
                tokens[s, k, :t.shape[0]] = t; modality[s, k, :m.shape[0]] = m; A[s, k] = anc[s][k]; V[s, k] = True; C[s, k] = cf[s][k]
        torch.save({"tokens": tokens, "modality": modality, "anchors": torch.from_numpy(A), "valid": torch.from_numpy(V), "cellfrac": torch.from_numpy(C),
                    "scene_ids": sids, "split": [splits[s] for s in sids], "names": names, "instructions": instr, "consts": consts}, D / task / "latent_cache.pt")
        print(task, "cached", tuple(tokens.shape), flush=True)


if __name__ == "__main__":
    main()
