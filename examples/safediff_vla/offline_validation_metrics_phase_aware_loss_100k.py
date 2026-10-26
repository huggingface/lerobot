#!/usr/bin/env python
"""Offline/validation-set metric comparison across the 100k-step phase-aware-loss long run's
checkpoints (`train_grounded_grasp_v2_5_phase_aware_loss_100k.py`), on a FIXED set of held-out
batches so every checkpoint is scored on identical data -- no rollout, no GPU-heavy simulation,
just repeated `policy.forward(batch)` calls.

Held-out set: the SAME deterministic eval split training itself used (`make_train_eval_datasets`
with the same `episodes`/`eval_split` -- entirely a function of those two inputs, no randomness),
then a fixed-seed `RandomSampler` subset of that eval set (same subset every run of this script,
across every checkpoint).

"step 0" reference: NOT the Stage 2 checkpoint loaded as-is (its own saved config has
`local_grasp_use_phase_aware_loss=False`, so the new loss terms would read as an uninformative flat
zero) -- instead, Stage 2's checkpoint weights loaded WITH `local_grasp_use_phase_aware_loss=True`
turned on for this evaluation only, i.e. exactly what the phase-aware run's own LocalGraspRefiner
looked like the instant before step 1 of that run.

Usage:
    uv run python examples/safediff_vla/offline_validation_metrics_phase_aware_loss_100k.py
    uv run python examples/safediff_vla/offline_validation_metrics_phase_aware_loss_100k.py --checkpoints 0 10000 25000
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, REPO_ID, STAGE2_TASK_NAMES, build_episode_task_map  # noqa: E402

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_train_eval_datasets  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.processor.rename_processor import rename_batch_keys  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

EVAL_SPLIT = 0.1
STAGE2_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/checkpoints/005000/pretrained_model"
RUN_DIR = Path("outputs/train/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k")
OUT_PATH = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_100k/offline_validation_metrics.json")

N_HELDOUT_BATCHES = 32
BATCH_SIZE = 8
HELDOUT_SEED = 42

METRIC_KEYS = (
    "loss_local_pos", "loss_local_rot", "loss_local_grip",
    "loss_local_terminal_xyz", "loss_local_approach_xyz", "loss_local_grip_weighted",
    "terminal_grasp_distance_m", "predicted_local_residual_magnitude_m",
    "local_eligible_frac",
)

# Shared policy_cfg fields (identical to the 100k training script -- see that file, not re-derived
# here) needed to build a `SafeDiffVLAConfig` for the "step 0" (Stage 2 init) reading.
_BASE_POLICY_KWARGS = {
    "device": "cuda",
    "push_to_hub": False,
    "architecture": "temporal_decoder_grounded_grasp_v2_5",
    "action_horizon": 50,
    "execute_horizon": 50,
    "use_temporal_ensembling": False,
    "backbone_name": "lerobot/smolvla_vlabench",
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "freeze_backbone": True,
    "freeze_vision_encoder": True,
    "use_lora": False,
    "decoder_hidden_dim": 512,
    "decoder_num_layers": 4,
    "decoder_num_heads": 8,
    "decoder_ffn_dim": 1024,
    "decoder_dropout": 0.1,
    "use_backbone_domain_adapter": False,
    "backbone_action_conversion_semantics": "per_step",
    "lambda_smooth": 0.0,
    "lambda_target": 1.0,
    "target_head_hidden_dim": 256,
    "grounded_grasp_gripper_open_threshold": 0.5,
    "grounded_grasp_close_threshold_m": 0.03,
    "local_grasp_radius_m": 0.10,
    "local_grasp_max_delta_xyz_m": 0.03,
    "local_grasp_max_delta_rot_rad": 0.05,
    "local_grasp_action_horizon": 5,
    "local_grasp_execute_horizon": 1,
    "local_grasp_window_k": 10,
    "lambda_local_grasp": 1.0,
    "local_grasp_hidden_dim": 256,
    "local_grasp_freeze_global_modules": True,
    "local_grasp_use_correction_head": False,
    "local_grasp_use_phase_aware_loss": True,
    "lambda_local_terminal_xyz": 2.0,
    "lambda_local_approach_xyz": 1.0,
    "lambda_local_grip_weighted": 2.0,
    "local_grasp_grip_close_weight": 3.0,
    "local_grasp_approach_weight_min": 0.1,
    "local_grasp_approach_weight_max": 1.0,
    "local_grasp_huber_delta": 1.0,
}


def checkpoint_path_for(step: int) -> str:
    return str(RUN_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model")


def load_policy(step: int, device: str) -> SafeDiffVLAPolicy:
    if step == 0:
        # Stage 2's OWN checkpoint (`local_grasp_use_phase_aware_loss=False` in its saved config)
        # loaded normally, then the flag flipped on in-place for this evaluation only -- gives the
        # new loss terms' reading on the EXACT weights the 100k run started from, without
        # reconstructing input/output features by hand (those come from the checkpoint's own saved
        # config, not from `_BASE_POLICY_KWARGS` alone).
        policy = SafeDiffVLAPolicy.from_pretrained(STAGE2_CHECKPOINT)
        policy.config.local_grasp_use_phase_aware_loss = True
        for k in (
            "lambda_local_terminal_xyz", "lambda_local_approach_xyz", "lambda_local_grip_weighted",
            "local_grasp_grip_close_weight", "local_grasp_approach_weight_min",
            "local_grasp_approach_weight_max", "local_grasp_huber_delta",
        ):
            setattr(policy.config, k, _BASE_POLICY_KWARGS[k])
    else:
        ckpt = checkpoint_path_for(step)
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint for step {step} not found at {ckpt}")
        policy = SafeDiffVLAPolicy.from_pretrained(ckpt)
    return policy.to(device)


def build_heldout_batches(reference_policy_cfg: SafeDiffVLAConfig, device: str) -> list[dict]:
    episode_task = build_episode_task_map()
    union_episodes = sorted(ep for ep, t in episode_task.items() if t in STAGE2_TASK_NAMES)

    # `reference_policy_cfg` comes from an already-`from_pretrained`-loaded policy (see `main()`),
    # so `input_features`/`output_features` are already populated from that checkpoint's own saved
    # config -- a bare `SafeDiffVLAConfig(**kwargs)` here lacks them, which both breaks
    # `SafeDiffVLAPolicy.__init__`'s own `validate_features()` AND silently skips a dtype-cast step
    # in the real image-preprocessing pipeline (uint8 frames never get converted to float).
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=REPO_ID, episodes=union_episodes, eval_split=EVAL_SPLIT),
        policy=reference_policy_cfg,
        rename_map=RENAME_MAP,
    )
    _, eval_dataset = make_train_eval_datasets(cfg)
    logger.info("held-out eval dataset: %d episodes, %d frames", eval_dataset.num_episodes, eval_dataset.num_frames)

    generator = torch.Generator().manual_seed(HELDOUT_SEED)
    sampler = torch.utils.data.RandomSampler(
        eval_dataset, replacement=False, num_samples=N_HELDOUT_BATCHES * BATCH_SIZE, generator=generator
    )
    collate_fn = lerobot_collate_fn if eval_dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=4,
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=reference_policy_cfg,
        pretrained_path=STAGE2_CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    # Mirrors `lerobot_train.py`'s own `_preprocess_dataset_batch` (a raw dataset batch's image
    # tensors are uint8 -- `return_uint8=True` above, same as real training -- and need the SAME
    # uint8->float32/255.0 cast + rename BEFORE `preprocessor(batch)`; not reused directly since
    # that's a module-private helper).
    camera_keys = eval_dataset.meta.camera_keys
    batches = []
    for batch in dataloader:
        for cam_key in camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = rename_batch_keys(batch, RENAME_MAP)
        batches.append(preprocessor(batch))
    logger.info("built %d fixed held-out batches (batch_size=%d, seed=%d)", len(batches), BATCH_SIZE, HELDOUT_SEED)
    return batches


def evaluate_checkpoint(policy: SafeDiffVLAPolicy, batches: list[dict]) -> dict:
    sums = dict.fromkeys(METRIC_KEYS, 0.0)
    n = 0
    policy.eval()
    with torch.no_grad():
        for batch in batches:
            _, metrics = policy(batch)
            for k in METRIC_KEYS:
                sums[k] += metrics[k]
            n += 1
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 10_000, 25_000, 50_000, 75_000, 100_000])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(OUT_PATH))
    args = parser.parse_args()

    reference_policy_cfg = SafeDiffVLAPolicy.from_pretrained(STAGE2_CHECKPOINT).config
    batches = build_heldout_batches(reference_policy_cfg, args.device)

    results: dict[str, dict] = {}
    for step in args.checkpoints:
        label = "step0_stage2_init" if step == 0 else f"step{step}"
        try:
            policy = load_policy(step, args.device)
        except FileNotFoundError as e:
            logger.warning("skipping %s: %s", label, e)
            continue
        metrics = evaluate_checkpoint(policy, batches)
        results[label] = metrics
        logger.info("[%s] %s", label, json.dumps({k: round(v, 5) for k, v in metrics.items()}))
        del policy
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
