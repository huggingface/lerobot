#!/usr/bin/env python
"""Convert a legacy LeRobot checkpoint to the current processor-pipeline format.

Older hub checkpoints (e.g. ``lerobot/act_aloha_sim_insertion_human``) bake
normalization stats into the model weights and do not ship
``policy_preprocessor.json`` / ``policy_postprocessor.json``. Current ``main``
loads those processor configs from the checkpoint, so eval/rollout fail with
``FileNotFoundError: Could not find 'policy_preprocessor.json'``.

This script rebuilds the processors from the training dataset's stats and saves
a pipeline-format checkpoint locally that ``lerobot-eval`` can consume directly.

Usage:
    python examples/onnx/convert_legacy_checkpoint.py \
        --policy-path=lerobot/act_aloha_sim_insertion_human \
        --dataset-repo-id=lerobot/aloha_sim_insertion_human \
        --output-dir=outputs/converted/act_aloha_sim_insertion_human

Then:
    lerobot-eval \
        --policy.path=outputs/converted/act_aloha_sim_insertion_human \
        --env.type=aloha --env.task=AlohaInsertion-v0 \
        --eval.batch_size=10 --eval.n_episodes=50 \
        --eval.use_async_envs=false --policy.device=cuda
"""

import argparse
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Legacy checkpoint repo id or local dir")
    parser.add_argument(
        "--dataset-repo-id",
        required=True,
        help="Training dataset repo id, used only for normalization stats",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save the converted checkpoint")
    parser.add_argument("--device", default="cpu", help="Device for building the policy (cpu is fine)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading dataset stats from '{args.dataset_repo_id}' (metadata only)...")
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)

    print(f"[2/4] Loading policy weights from '{args.policy_path}'...")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device
    policy = make_policy(cfg, ds_meta=ds_meta)

    print("[3/4] Building processors from dataset stats...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=ds_meta.stats,
    )

    print(f"[4/4] Saving pipeline-format checkpoint to '{out}'...")
    policy.save_pretrained(out)
    preprocessor.save_pretrained(out, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(out, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")

    print(f"\nDone. Converted checkpoint at: {out}")
    print("Eval it with --policy.path=" + str(out))


if __name__ == "__main__":
    main()
