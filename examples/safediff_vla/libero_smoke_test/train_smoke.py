#!/usr/bin/env python
"""LIBERO / LIBERO-Plus smoke-test training driver (see `../libero_transfer_audit_report.md`).

Minimal, isolated smoke test per the transfer-experiment protocol: ONE task
("pick up the milk and place it in the basket" -- `adapter.SMOKE_TEST_TASK_TEXT`), a
20-50 episode subset (`adapter.py`'s saved episode-index list), 1k-2k steps, comparing SmolVLA
against SafeDiff-VLA's plain `architecture="temporal_decoder"` (= "TemporalDecoder-SinCos-PadFix"
-- sin/cos rotation encoding and the action-pad fix are both unconditional in that code path, see
`rotation_encoding.py`). LIBERO and LIBERO-Plus are trained as two fully independent runs (own
DatasetConfig, own output_dir) -- never concatenated.

No GroundedGrasp architecture, no `find_grasp_target` involvement -- those stay untouched until a
separate follow-up, after this smoke test's rollout video confirms action/gripper decoding.

No camera rename here: each dataset trains on its OWN native camera keys (LIBERO: `image`/
`image2`; LIBERO-Plus: `front`/`wrist`) -- `TrainPipelineConfig.rename_map` turned out to only
apply to a pretrained-checkpoint resume, not fresh init (see `adapter.py`'s
`EVAL_CAMERA_NAME_MAPPING` docstring), so the camera-key adapter lives at EVAL time instead
(`rollout_diagnostics.py` passes `LiberoPlusEnv(camera_name_mapping=...)`). No translation/
rotation scale or gripper-polarity correction is applied here either: training/eval both consume
the dataset's/env's native 7-D action as-is (MEAN_STD-normalize / unnormalize), which doesn't
need those semantics to be trained correctly -- they only matter for the rollout-video
diagnostics.

Usage:
    uv run python examples/safediff_vla/libero_smoke_test/train_smoke.py \
        --dataset libero --policy smolvla --steps 1500
    uv run python examples/safediff_vla/libero_smoke_test/train_smoke.py \
        --dataset libero_plus --policy safediff_vla --steps 1500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train

from adapter import SMOKE_TEST_TASK_TEXT, install_libero_gripper_collapse_training_patch

REPO_IDS = {"libero": "lerobot/libero", "libero_plus": "lerobot/libero_plus"}


def build_policy_config(policy: str) -> SmolVLAConfig | SafeDiffVLAConfig:
    if policy == "smolvla":
        return SmolVLAConfig(
            device="cuda",
            push_to_hub=False,
            load_vlm_weights=True,
            chunk_size=50,
            n_action_steps=50,
            optimizer_lr=1e-4,
        )
    if policy == "safediff_vla":
        return SafeDiffVLAConfig(
            device="cuda",
            push_to_hub=False,
            architecture="temporal_decoder",  # "TemporalDecoder-SinCos-PadFix"
            action_horizon=50,
            execute_horizon=50,
            use_temporal_ensembling=False,
            backbone_name=None,  # fresh SmolVLM2 base weights -- no VLABench-domain checkpoint
            vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            freeze_backbone=True,
            freeze_vision_encoder=True,
            use_lora=False,
            decoder_hidden_dim=512,
            decoder_num_layers=4,
            decoder_num_heads=8,
            decoder_ffn_dim=1024,
            decoder_dropout=0.1,
            use_backbone_domain_adapter=False,
            optimizer_lr=1e-4,
            optimizer_weight_decay=1e-6,
            scheduler_warmup_steps=100,
            scheduler_decay_steps=2000,
        )
    raise ValueError(f"Unknown policy {policy!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--policy", choices=("smolvla", "safediff_vla"), required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--episodes-file",
        default=str(
            Path("/tmp/claude-0/-home/b0991f05-9e54-4109-9e66-97772d456c71/scratchpad/libero_audit")
            / "smoke_test_episode_subset.json"
        ),
    )
    parser.add_argument("--output-root", default="outputs/train/libero_smoke_test")
    args = parser.parse_args()
    if args.steps > 2000:
        raise ValueError("Smoke test is capped at 2000 steps per the protocol -- do not extend here.")

    episodes = json.loads(Path(args.episodes_file).read_text())[args.dataset]
    if not (20 <= len(episodes) <= 50):
        raise ValueError(f"Episode subset must be 20-50 episodes, got {len(episodes)}")

    if args.policy == "safediff_vla":
        # LIBERO's real observation.state is 8-D (see adapter.py's docstring); SafeDiff-VLA's
        # temporal_decoder architectures hard-require 7-D. Collapse the two gripper-finger qpos
        # dims to their mean -- scoped to this process only, never touches repo files, never
        # applied to the SmolVLA branch above.
        install_libero_gripper_collapse_training_patch()
        if args.num_workers > 0:
            # DataLoader worker processes don't inherit this run's in-process monkeypatch (the
            # 8x8 vs 7 normalizer crash this guard prevents was hit and confirmed with
            # num_workers=4 on 2026-09-29) -- force single-process loading so every batch
            # actually goes through the patched `LeRobotDataset.__getitem__`.
            print("[smoke-test] forcing --num-workers=0 for safediff_vla: DataLoader workers "
                  "don't see the gripper-collapse monkeypatch, see adapter.py")
            args.num_workers = 0

    run_name = f"{args.dataset}_{args.policy}_smoke"
    output_dir = Path(args.output_root) / run_name

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_IDS[args.dataset], episodes=episodes),
        policy=build_policy_config(args.policy),
        output_dir=output_dir,
        job_name=run_name,
        resume=False,
        seed=1000,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=50,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=args.steps,
        use_policy_training_preset=True,
    )

    print(f"[smoke-test] task_text={SMOKE_TEST_TASK_TEXT!r} dataset={args.dataset} "
          f"policy={args.policy} n_episodes={len(episodes)} steps={args.steps}")
    train(cfg)


if __name__ == "__main__":
    main()
