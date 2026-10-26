#!/usr/bin/env python
"""LIBERO / LIBERO-Plus FINAL evaluation: native task success rate across all 40 tasks (the 4
standard suites: libero_spatial, libero_object, libero_goal, libero_10 -- LIBERO's own recommended
protocol, matching this HF dataset's 40-task coverage), same eval seeds/episode count applied
identically to SmolVLA and SafeDiff-VLA Final for a fair comparison.

For `--policy safediff_vla_final`, wires the same LIBERO camera mapping (LIBERO-Plus only) and
gripper-collapse observation step the smoke test verified, via `make_env_pre_post_processors` +
an appended `GripperCollapseObservationStep` -- `find_grasp_target`'s LIBERO patch is NOT needed
at eval time (it only affects the TRAINING-time auxiliary loss/label, not inference).

Usage:
    uv run python examples/safediff_vla/libero_final/eval_final.py \
        --dataset libero --policy smolvla \
        --checkpoint outputs/train/libero_final/libero_smolvla_final/checkpoints/050000/pretrained_model \
        --episodes-per-task 3 --output-dir outputs/libero_final/eval/libero_smolvla_50k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "libero_smoke_test"))

from adapter import GripperCollapseObservationStep  # noqa: E402

import torch  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.configs import LiberoPlusEnv as LiberoPlusEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy_all  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402

CONFIG_CLS = {"smolvla": SmolVLAConfig, "safediff_vla_final": SafeDiffVLAConfig}
STANDARD_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FPS = {"libero": 10, "libero_plus": 20}
# LIBERO-Plus's env emits image/image2 by default (see the smoke-test adapter's own docstring on
# why -- `LiberoPlusEnv` inherits `LiberoEnv`'s camera_name_mapping unchanged); the LIBERO-Plus
# dataset trains on its native front/wrist keys, so eval needs this override to match.
EVAL_CAMERA_NAME_MAPPING = {
    "libero": None,
    "libero_plus": {"agentview_image": "front", "robot0_eye_in_hand_image": "wrist"},
}


def build_env_cfg(dataset: str):
    common = dict(
        task=",".join(STANDARD_SUITES),
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        init_states=True,
        hard_reset=True,
        control_mode="relative",
        camera_name_mapping=EVAL_CAMERA_NAME_MAPPING[dataset],
    )
    if dataset == "libero":
        return LiberoEnvConfig(fps=10, **common)
    return LiberoPlusEnvConfig(fps=20, **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--policy", choices=("smolvla", "safediff_vla_final"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episodes-rendered", type=int, default=6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"

    env_cfg = build_env_cfg(args.dataset)
    envs = make_env(env_cfg, n_envs=args.batch_size)
    n_tasks = sum(len(g) for g in envs.values())
    print(f"[eval] dataset={args.dataset} policy={args.policy} suites={STANDARD_SUITES} "
          f"n_tasks={n_tasks} episodes_per_task={args.episodes_per_task} "
          f"fps={FPS[args.dataset]} action_horizon=50 -> physical_horizon={50 / FPS[args.dataset]:.1f}s")

    policy_cfg = CONFIG_CLS[args.policy].from_pretrained(args.checkpoint)
    device = get_safe_torch_device(policy_cfg.device)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    if args.policy == "safediff_vla_final":
        env_preprocessor.steps = list(env_preprocessor.steps) + [GripperCollapseObservationStep()]

    with torch.no_grad():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.episodes_per_task,
            max_episodes_rendered=args.max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=args.seed,
        )

    print("[eval] overall:", info.get("overall"))
    for group, group_info in info.items():
        if group == "overall":
            continue
        print(f"[eval] {group}:", group_info)

    (out_dir / "eval_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"[eval] wrote {out_dir / 'eval_info.json'} and videos to {videos_dir}")


if __name__ == "__main__":
    main()
