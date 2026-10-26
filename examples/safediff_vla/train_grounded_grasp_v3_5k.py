#!/usr/bin/env python
"""Fresh (no resume) 5k-step training run for `architecture="temporal_decoder_grounded_grasp_v3"`
(see `grounded_grasp_v3.py` and `configuration_safediff_vla.py`'s architecture docstring): splits
v2's single "grounded feature -> direct xyz" `TargetPointHead` regression into an explicit
coarse-visual-selection stage (`SharpenedVisualGrounding`) followed by a coarse-then-bounded-residual
xyz head (`CoarseTargetHead` + `FineResidualHead`). v2's held-out eval (canonical baseline 2/10
success vs. v2 1/10; mean target-xyz error 0.205m vs. 0.145m) showed v2 improved localization but
never reached grasp-relevant precision and did not move task success -- this pilot changes ONLY the
grounding/localization head to test whether the coarse/fine split closes that gap.

Condition (identical to `train_grounded_grasp_v2_5k.py` except `architecture` and the new
`grounded_grasp_v3_*`/`lambda_target_coarse`/`lambda_target_refine` fields, all left at their
conservative config defaults -- no sweep in this pilot):
  - frozen backbone (`freeze_backbone=True`, `freeze_vision_encoder=True`).
  - action_horizon=execute_horizon=50.
  - use_temporal_ensembling=False (ensembling OFF).
  - lambda_smooth=0.0 (smoothness OFF).
  - subgoal OFF (architecture is not `temporal_decoder_subgoal` -- no subgoal predictor built).
  - lambda_target_coarse=lambda_target_refine=1.0 (both default, no sweep).
  - grounded_grasp_v3_temperature/top_k/max_refine_offset left at config defaults (untuned,
    conservative -- see `configuration_safediff_vla.py`'s field docstrings).
  - reactive-close threshold/phase logic: unchanged from v2 (`grounded_grasp_close_threshold_m`
    default) -- this pilot's only experimental variable is the grounding head.
  - sin/cos rotation encoding + padfix action-mask: both unconditional in the current
    `temporal_decoder` path (no toggle left to set).
  - same optimizer/scheduler/seed/batch_size/num_workers as every prior 5k run on this branch
    (lr=1e-4, wd=1e-6, warmup=1000, decay=30000, seed=1000, batch_size=4, num_workers=8).
  - genuine 10% held-out split: `DatasetConfig(eval_split=0.1)` -- same deterministic split
    (repo_id + eval_split are the only inputs) as every prior 5k run on this branch.

5k steps only -- do not extend to 20k.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v3_5k.py
"""

import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.scripts.lerobot_train import train

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}

EVAL_SPLIT = 0.1  # genuine held-out split; same repo_id+eval_split as every prior 5k run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_grounded_grasp_v3_5k")
    parser.add_argument("--job-name", default="safediff_vla_grounded_grasp_v3_5k")
    args = parser.parse_args()
    if args.steps > 5000:
        raise ValueError("This pilot is 5k steps only -- do not extend to 20k without a separate decision.")

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v3",
        action_horizon=50,
        execute_horizon=50,
        use_temporal_ensembling=False,
        backbone_name="lerobot/smolvla_vlabench",
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
        backbone_action_conversion_semantics="per_step",
        lambda_smooth=0.0,
        lambda_target_coarse=1.0,
        lambda_target_refine=1.0,
        target_head_hidden_dim=256,
        grounded_grasp_v3_temperature=0.5,
        grounded_grasp_v3_top_k=None,
        grounded_grasp_v3_max_refine_offset=0.5,
        grounded_grasp_gripper_open_threshold=0.5,
        grounded_grasp_close_threshold_m=0.03,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=EVAL_SPLIT),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=Path(args.output_dir),
        job_name=args.job_name,
        resume=False,
        seed=1000,
        num_workers=8,
        batch_size=4,
        steps=args.steps,
        log_freq=100,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=args.save_freq,
        use_policy_training_preset=True,
    )
    train(cfg)


if __name__ == "__main__":
    main()
