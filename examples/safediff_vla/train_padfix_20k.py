#!/usr/bin/env python
"""Fresh (no resume) 20k-step training run on *canonical* `libero-safety` code -- i.e. the sin/cos
rotation encoding (`a7ef15e4`) plus the padded-action-timestep loss masking fix
(`371c88c8`/upstream `585fc7db`, "Mask padded action timesteps in SafeDiffVLA loss"), with no
phase-conditioning or grasp-event-replan code at all (that work is confined to the
`experimental-phase-replan` branch, not this one).

Goal: reproducibility, not re-proving the padfix effect itself -- does the same canonical code,
run fresh on this machine/session, produce comparable training behavior to whatever run originally
validated the padfix (`A`)? Same optimizer/scheduler/loss hyperparameters as every other
`safediff_vla_temporal_decoder_*` run in this repo (`sincos_5k`/`sincos_20k`/`phase_5k`'s own saved
`train_config.json`) -- nothing here is picked to flatter this specific run.

Fixed condition: architecture=temporal_decoder (not temporal_decoder_subgoal -- no subgoal loss
at all), action_horizon=execute_horizon=50, freeze_backbone=True, use_temporal_ensembling=False,
lambda_smooth=0.0 (smoothness off, the architecture default). No phase_conditioning /
replan_on_gripper_close fields exist on this branch's `SafeDiffVLAConfig` at all -- there is
nothing to turn off, they're simply absent.

Usage:
    uv run python examples/safediff_vla/train_padfix_20k.py
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_temporal_decoder_padfix_20k")
    parser.add_argument("--job-name", default="safediff_vla_temporal_decoder_padfix_20k")
    args = parser.parse_args()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder",
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
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=0.0),
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
