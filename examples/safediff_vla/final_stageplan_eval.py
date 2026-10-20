#!/usr/bin/env python
"""FINAL SafeDiff-VLA StagePlan closed-loop evaluation: ONE task x ONE seed per OS process. The policy input is RGB + language +
proprio only; the privileged simulator reads in this file (snapshots, keypoints, goal poses) are LOGGING ONLY and never reach
the policy (they are read in env wrappers AFTER the action is produced). Writes episode record + per-step stage log + plan.

Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/final_stageplan_eval.py --task select_book --seed 3000 --head outputs/train/safediff_stageplan_v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import g25_common as C25  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan import SafeDiffVLAStagePlanPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

OUT_ROOT = Path("outputs/eval/safediff_stageplan_final")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--head", default="outputs/train/safediff_stageplan_v1"); ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args()
    out = Path(a.out) / a.task / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    policy = SafeDiffVLAStagePlanPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); policy.attach_stageplan(a.head, "cuda")
    prov = {"backbone_checkpoint": str(Path(CHECKPOINT).resolve()), "head_dir": str(Path(a.head).resolve()), "stageplan_config": json.load(open(Path(a.head) / "stageplan_config.json")),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    lang: list[dict] = []; all_snaps: list[list[dict]] = []; stage_log: list[list[dict]] = []; plans: list[dict] = []; geo: list[dict] = []
    orig_reset, orig_step = VLABenchEnvImpl.reset, VLABenchEnvImpl.step

    def _others(e) -> dict:
        t = e._env.task; d = {}
        for k, ent in t.entities.items():
            if k == t.target_entity: continue
            try: d[k] = np.asarray(ent.get_xpos(e._env.physics), float).tolist()
            except Exception:  # noqa: BLE001
                pass
        return d

    def reset(self, *args, **kw):
        r = orig_reset(self, *args, **kw)
        lang.append({"text": self.task_description, "source": self.task_description_source})
        t = self._env.task; ph = self._env.physics; base = robot_base(self)
        kps = np.asarray(t.entities[t.target_entity].get_grasped_keypoints(ph), float)
        geo.append({"target": t.target_entity, "kp_robot": (kps - base).tolist(), "base": base.tolist()})  # LOGGING ONLY (analysis of P1 error)
        all_snaps.append([{"others": _others(self), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in C25.snapshot(self._env).items()}}]); stage_log.append([])
        return r

    def step(self, action):
        r = orig_step(self, action)
        if self._env is None: return r
        all_snaps[-1].append({"others": _others(self), **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in C25.snapshot(self._env).items()}})
        return r

    VLABenchEnvImpl.reset, VLABenchEnvImpl.step = reset, step
    orig_sel = policy.select_action

    def sel(batch):
        act = orig_sel(batch)
        c = policy.stage_controller
        stage_log[-1].append({"phase": c.phase, "cmd": None if c.last_cmd is None else np.asarray(c.last_cmd).tolist(), "grip": float(c.last_grip)})
        if policy._sp_plan is not None and len(plans) < len(lang) and not any(p["ep"] == len(lang) - 1 for p in plans):
            plans.append({"ep": len(lang) - 1, "poses": np.asarray(policy._sp_plan["poses"]).tolist(), "release_prob": policy._sp_plan["release_prob"]})
        return act

    policy.select_action = sel
    set_seed(a.seed)
    try:
        info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre, postprocessor=post, n_episodes=1, max_episodes_rendered=1, videos_dir=out / "videos", start_seed=a.seed)
    finally:
        env.close(); VLABenchEnvImpl.reset, VLABenchEnvImpl.step = orig_reset, orig_step
    ep = info["per_episode"][0]
    n = min(len(all_snaps[0]), len(stage_log[0]) + 1)
    (out / "result.json").write_text(json.dumps({"seed": ep["seed"], "success": bool(ep["success"]), "len": len(stage_log[0]), "provenance": prov, "language": lang, "geo": geo, "plan": plans[:1]}, indent=1))
    (out / "stage_log.json").write_text(json.dumps(stage_log[0])); (out / "snapshots.json").write_text(json.dumps(all_snaps[0][:n]))
    print(f"[StagePlan] {a.task} seed {a.seed} success={ep['success']} len={len(stage_log[0])} language={lang[0]}")


if __name__ == "__main__":
    main()
