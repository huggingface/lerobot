#!/usr/bin/env python
"""PHASE R0: rollout reproducibility / noise-floor audit. NO policy behavior change -- this script
only runs the UNCHANGED `GroundedGrasp-v2` 20k checkpoint (E50) repeatedly, under two
configurations ("current" = this codebase's actual defaults, "deterministic" = every determinism
knob PyTorch exposes turned on, best-effort), to separate genuine policy-intervention effects
(Phase 2C's A-vs-B) from run-to-run execution noise.

Note `lerobot.utils.random_utils.set_seed` (already called identically every run, every repeat, by
every eval script on this branch) already seeds Python `random`, NumPy, and `torch.manual_seed` /
`torch.cuda.manual_seed_all` -- so if repeats under "current" still diverge, RNG seeding is NOT the
cause; the remaining candidates are GPU kernel-level nondeterminism (cuDNN/cuBLAS algorithm
selection and reduction order), MuJoCo's own solver, and EGL rendering. "deterministic" mode adds,
on top of the same `set_seed` call:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn, never crash or change code
`CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set in the shell environment BEFORE this process starts
(CUDA context init reads it once) -- the driver script sets it for deterministic-mode subprocesses.
`warn_only=True` means any op lacking a deterministic implementation prints a UserWarning instead of
raising -- this script does NOT change code to route around such ops (explicitly instructed not
to); it only captures and reports which ops (if any) warned.

`--monitor {on,off}`: "on" reuses Phase 2C's `install_reanchor_policy(policy, enabled=False)` (the
monitor-only forward runs every step, applies no correction -- condition A from Phase 2C); "off"
leaves `SafeDiffVLAPolicy.select_action` completely unwrapped. Comparing on vs. off under
"deterministic" mode is how this script verifies the monitor forward is trajectory-level side-
effect free (Phase 2C only argued this logically; this audit checks it empirically).

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/phase_r0_reproducibility_audit.py \
        --task select_poker --seed 1001 --repeat-id 0 --monitor on [--deterministic]
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from eval_grounded_grasp_v2_phase_preserving_reanchor import (  # noqa: E402
    install_env_instrumentation,
    install_reanchor_policy,
    uninstall_env_instrumentation,
    uninstall_reanchor_policy,
)

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

OUT_DIR_ROOT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_r0_reproducibility")
captured_warnings: list[str] = []


def _warning_sink(message, category, filename, lineno, file=None, line=None):
    captured_warnings.append(f"{category.__name__}: {message}")


def configure_determinism(enabled: bool) -> dict[str, Any]:
    state = {
        "cudnn_deterministic_before": torch.backends.cudnn.deterministic,
        "cudnn_benchmark_before": torch.backends.cudnn.benchmark,
        "matmul_tf32_before": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32_before": torch.backends.cudnn.allow_tf32,
    }
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        warnings.showwarning = _warning_sink
        warnings.filterwarnings("always")
        torch.use_deterministic_algorithms(True, warn_only=True)
    state.update(
        {
            "cudnn_deterministic_after": torch.backends.cudnn.deterministic,
            "cudnn_benchmark_after": torch.backends.cudnn.benchmark,
            "matmul_tf32_after": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32_after": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms_requested": enabled,
        }
    )
    return state


def run_once(policy: SafeDiffVLAPolicy, device: str, task: str, seed: int, monitor_on: bool) -> dict:
    env_cfg = VLABenchEnv(task=task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_reanchor_policy(policy, enabled=False) if monitor_on else {"log": []}
    env_state = install_env_instrumentation(policy_inst["log"])
    episode_record: dict[str, Any] = {}

    def episode_callback(episode_ix, rollout_data, env_idx, done_index):
        episode_record["episode_len"] = done_index + 1
        episode_record["steps"] = list(env_state["current_episode_steps"])

    set_seed(seed)
    try:
        info = eval_policy(
            env=env, policy=policy, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=1,
            max_episodes_rendered=0, videos_dir=None, start_seed=seed, episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        if monitor_on:
            uninstall_reanchor_policy(policy)

    episode_record["success"] = bool(info["per_episode"][0]["success"])
    episode_record["seed"] = seed
    return episode_record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--monitor", choices=["on", "off"], default="on")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    det_state = configure_determinism(args.deterministic)
    logger.info("=== determinism config: %s ===", json.dumps(det_state, indent=2))

    mode = "deterministic" if args.deterministic else "current"
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR_ROOT / mode / f"monitor_{args.monitor}" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== loading checkpoint: %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    record = run_once(policy, args.device, args.task, args.seed, monitor_on=(args.monitor == "on"))
    record["determinism_state"] = det_state
    record["captured_warnings"] = sorted(set(captured_warnings))
    record["repeat_id"] = args.repeat_id

    out_path = out_dir / f"repeat_{args.repeat_id}.json"
    out_path.write_text(json.dumps(record, indent=2))
    logger.info(
        "[%s seed=%d mode=%s monitor=%s repeat=%d] len=%d success=%s n_warnings=%d -> %s",
        args.task, args.seed, mode, args.monitor, args.repeat_id,
        record["episode_len"], record["success"], len(record["captured_warnings"]), out_path,
    )


if __name__ == "__main__":
    main()
