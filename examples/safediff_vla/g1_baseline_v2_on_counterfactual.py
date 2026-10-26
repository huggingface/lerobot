#!/usr/bin/env python
"""G1 reference point: the CURRENT GroundedGrasp-v2 20k target head scored on the G1 counterfactual scenes (forward only).
Observation = the stored RGB/state of each scene (exact policy inputs), text = the training-format instruction of each
candidate referent. Saves per-sample predictions (reused for figures) and the shared G1 metrics per split."""
from __future__ import annotations

import copy, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from g1_counterfactual_metrics import metrics  # noqa: E402
from phase_g06_counterfactual_instruction import forward_stages  # noqa: E402

from lerobot.envs import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

D = Path("outputs/data/safediff_vla_g1_counterfactual"); OUT = Path("outputs/eval/safediff_vla_g1_reference_v2_current"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); res = {}; dump = {}
    for task in ("select_poker", "select_fruit"):
        cfg = VLABenchEnv(task=task)
        pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        splits = json.load(open(D / task / "splits.json")); by = {}
        for sid, sp in splits.items():
            rec = json.load(open(D / task / f"{sid}.json")); z = np.load(D / task / f"{sid}.npz")
            raw = {"pixels": {k: z["rgb"][i][None] for i, k in enumerate(("image", "second_image", "wrist_image"))}, "agent_pos": z["state"][None].astype(np.float64)}
            anchors, preds, ids = {}, {}, {}
            for c in rec["candidates"]:
                o = preprocess_observation(copy.deepcopy(raw)); o["task"] = [c["instruction"]]
                preds[c["name"]] = forward_stages(policy, pre(env_pre(o)))["pred_robot"]; anchors[c["name"]] = np.array(c["anchor_robot"])
            by.setdefault(sp, []).append({"scene": sid, "anchors": anchors, "preds": preds})
            dump[sid] = {"split": sp, "preds": {k: v.tolist() for k, v in preds.items()}}
        res[task] = {sp: metrics(v) for sp, v in by.items()}; res[task]["ALL"] = metrics([s for v in by.values() for s in v])
        for sp, m in res[task].items():
            print(f"{task:12s} {sp:22s} scenes={m['n_scenes']:3d} R_follow={m['R_follow_xy_pooled']:+.3f} {m['R_follow_ci95_scene_bootstrap']} | pred_shift={m['pred_shift_xy_cm_mean']:.2f}cm vs gt_shift={m['gt_shift_xy_cm_mean']:.1f}cm | loc_err xy mean={m['loc_err_xy_cm_mean']:.1f} med={m['loc_err_xy_cm_median']:.1f}cm | sel_acc={m['selection_acc']:.2f} (chance {m['selection_chance']:.2f}) margin_med={m['selection_margin_cm_median']:.1f}cm", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1)); (OUT / "preds.json").write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
