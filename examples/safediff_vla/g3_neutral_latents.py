#!/usr/bin/env python
"""G3 representation diagnostic input: frozen-v2 VLM IMAGE-token latents of every poker scene encoded with a NEUTRAL instruction that names NO card
("primitive: Please pick the poker"). Any card-identity information in a card's region features is therefore visual (plus generic scene context), not
copied from the instruction text. Forward only."""
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

D = Path("outputs/data/safediff_vla_g1_counterfactual/select_poker"); OUT = Path("outputs/eval/safediff_vla_g3_offline"); NEUTRAL = "primitive: Please pick the poker"


def main():
    OUT.mkdir(parents=True, exist_ok=True); policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    cfg = VLABenchEnv(task="select_poker")
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
    splits = json.load(open(D / "splits.json")); sids = sorted(splits); X = torch.zeros(len(sids), 192, 960, dtype=torch.float16)
    for n, sid in enumerate(sids):
        z = np.load(D / f"{sid}.npz"); raw = {"pixels": {k: z["rgb"][i][None] for i, k in enumerate(("image", "second_image", "wrist_image"))}, "agent_pos": z["state"][None].astype(np.float64)}
        o = preprocess_observation(copy.deepcopy(raw)); o["task"] = [NEUTRAL]
        with torch.no_grad(): tok, pad, mod = policy._encode_multimodal_latent(pre(env_pre(o)))
        assert (mod[0, :192] == 0).all(); X[n] = tok[0, :192].half().cpu()
        if (n + 1) % 100 == 0: print(n + 1, flush=True)
    torch.save({"tokens": X, "scene_ids": sids, "split": [splits[s] for s in sids], "text": NEUTRAL}, OUT / "g3_neutral_image_latents.pt"); print("saved", tuple(X.shape))


if __name__ == "__main__":
    main()
