#!/usr/bin/env python
"""PHASE 1B: fixed-replanning-frequency ablation, isolating ONE variable --
`SafeDiffVLAConfig.execute_horizon` -- against the SAME frozen `GroundedGrasp-v2` checkpoint Phase
1A already profiled. NO training, NO architecture change, NO new module, NO smoothing, NO reactive-
close logic change: `execute_horizon` is the ONLY thing that differs between conditions.

Question: "committing to a 50-step open-loop chunk before re-observing/re-grounding causes the
chunk-boundary whiplash (10-30cm planner target jumps) Phase 1A found visually and quantitatively
-- does replanning (= re-running the FULL grounding pipeline + decoder) more often reduce it, and
does that actually translate into better closed-loop approach/grasp behavior?"

Mechanism (see `execution.py`'s `ActionExecutor`): `execute_horizon` is read ONLY once, inside
`ActionExecutor.__init__` (constructed fresh every `SafeDiffVLAPolicy.reset()`, i.e. once per
episode). This script loads the UNCHANGED checkpoint and mutates `policy.config.execute_horizon`
exactly once, right after loading and before any rollout -- every subsequent per-episode `reset()`
then builds its `ActionExecutor` with the new value. `action_horizon` (the decoder's own native
chunk width, 50) is NEVER changed: the full 50-step chunk is still predicted on every replan: with
`execute_horizon=10`, only the chunk's first 10 steps are ever pushed onto the queue
(`ActionExecutor.select_action`'s `chunk.transpose(0, 1)[: self._execute_horizon]`), discarding the
rest and triggering a fresh full re-observe/re-ground/re-plan 5x more often than the `execute_
horizon=50` baseline. No code path besides `ActionExecutor.__init__` reads `execute_horizon`
(verified by grep across `modeling_safediff_vla.py`/`execution.py`), so this is a pure
execution-cadence change.

Same monitor-only instrumentation as `eval_grounded_grasp_v2_monitor_baseline.py` (imported, not
duplicated) -- `monitor_predicted_target_xyz` is STILL recomputed every single env step regardless
of `execute_horizon`, so jitter/staleness are measured on a comparable basis across conditions;
`planner_target_robot_frame`/`planner_replanned_this_step` now flip far more often at smaller
`execute_horizon`, which is exactly the variable under test.

E50 is NOT re-run here -- it is byte-for-byte the same rollout Phase 1A already produced (same
checkpoint, same config, same seeds, deterministic MuJoCo physics, policy.eval()): reused directly
by `run_grounded_grasp_v2_execute_horizon_ablation_isolated.py` via `REUSE_E50_FROM`.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_execute_horizon_ablation.py \
        --execute-horizon 10 --tasks select_book
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import (  # noqa: E402
    CHECKPOINT,
    install_env_instrumentation,
    install_policy_instrumentation,
    uninstall_env_instrumentation,
    uninstall_policy_instrumentation,
)

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SEEDS = [1000, 1001, 1002]
OUT_DIR_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_execute_horizon_ablation")


def run_task(policy: SafeDiffVLAPolicy, device: str, task: str, seeds: list[int], out_dir: Path) -> None:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2"
    set_seed(seeds[0])

    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_policy_instrumentation(policy)
    env_state = install_env_instrumentation(policy_inst["log"])
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        policy_inst["log"].clear()
        logger.info("  [%s] episode %d len=%d n_steps=%d", task, episode_ix, ep_len, len(steps))

    try:
        info = eval_policy(
            env=env, policy=policy, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=len(seeds),
            max_episodes_rendered=len(seeds), videos_dir=videos_dir, start_seed=seeds[0], episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_policy_instrumentation(policy)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))
    logger.info(
        "[%s] wrote %s (successes: %s)", task, out_dir / "episodes_raw.json", [ep["success"] for ep in info["per_episode"]]
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute-horizon", type=int, required=True, choices=[1, 5, 10, 50])
    parser.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR_ROOT / f"E{args.execute_horizon}"

    logger.info("=== loading checkpoint: %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    assert policy.config.action_horizon == 50, "action_horizon must stay 50 -- only execute_horizon is ablated"
    policy.config.execute_horizon = args.execute_horizon
    logger.info("=== execute_horizon overridden to %d (action_horizon stays %d) ===", args.execute_horizon, policy.config.action_horizon)
    policy = policy.to(args.device)
    policy.eval()
    for task in args.tasks:
        logger.info("--- task=%s seeds=%s execute_horizon=%d ---", task, args.seeds, args.execute_horizon)
        run_task(policy, args.device, task, args.seeds, out_dir / task)


if __name__ == "__main__":
    main()
