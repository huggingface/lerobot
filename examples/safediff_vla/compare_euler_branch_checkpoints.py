#!/usr/bin/env python
"""Compare the old raw-Euler checkpoint (B) against the new sin/cos-representation checkpoint on
the *same* fixed sample sets already saved by `euler_branch_audit.py` (both the general sample
set and the branch-cut-near sample set).

B's checkpoint weights are for a 7-D decoder (`decoder.action_head.weight: [7, 512]`); the
*current* code always builds a 10-D decoder now (the whole point of this change), so
`SafeDiffVLAPolicy.from_pretrained(B)` genuinely cannot load anymore -- exactly the incompatibility
flagged up front ("기존 20k checkpoint는 이 새 representation과 호환되지 않습니다"). So B's numbers
here are *not* re-run: the general-sample-set summary is read straight from
`euler_branch_audit_B.json` (already computed, before this change), and the branch-cut-sample-set
summary is *recomputed* (no model inference) from `euler_branch_samples_B.json`'s already-saved
per-sample `gt_raw_euler_first10`/`pred_raw_euler_first10` arrays. Only the new sincos checkpoint
actually runs inference here, via `euler_branch_audit.run_model_comparison` unchanged.

Usage:
    uv run python examples/safediff_vla/compare_euler_branch_checkpoints.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import euler_branch_audit as eba  # noqa: E402  (reuse run_model_comparison, don't reimplement)
from eval_baseline_rollout import geodesic_angle  # noqa: E402

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

OLD_CHECKPOINT = eba.CHECKPOINT  # raw-Euler "B" checkpoint (v2 baseline, 20k steps)
NEW_CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_sincos_5k/checkpoints/005000/pretrained_model"


def summarize_from_saved_records(records: list[dict]) -> dict:
    """Recompute the same summary `run_model_comparison` would, from already-saved per-sample
    `gt_raw_euler_first10`/`pred_raw_euler_first10` -- no model inference, no dataset access."""
    raw_mse5, raw_mse10, unwrap_mse5, unwrap_mse10, geo5, geo10 = [], [], [], [], [], []
    for r in records:
        gt_rot = np.array(r["gt_raw_euler_first10"])
        pred_rot = np.array(r["pred_raw_euler_first10"])
        gt_unwrapped = np.unwrap(gt_rot, axis=0)
        pred_unwrapped = np.unwrap(pred_rot, axis=0)

        def mse(a, b, n):
            return float(np.mean((a[:n] - b[:n]) ** 2))

        raw_mse5.append(mse(pred_rot, gt_rot, 5))
        raw_mse10.append(mse(pred_rot, gt_rot, 10))
        unwrap_mse5.append(mse(pred_unwrapped, gt_unwrapped, 5))
        unwrap_mse10.append(mse(pred_unwrapped, gt_unwrapped, 10))
        geo = geodesic_angle(torch.from_numpy(pred_rot).unsqueeze(0), torch.from_numpy(gt_rot).unsqueeze(0))[0]
        geo5.append(float(geo[:5].mean()))
        geo10.append(float(geo[:10].mean()))
    return {
        "n_samples": len(records),
        "raw_euler_mse_first5": float(np.mean(raw_mse5)),
        "raw_euler_mse_first10": float(np.mean(raw_mse10)),
        "unwrapped_euler_mse_first5": float(np.mean(unwrap_mse5)),
        "unwrapped_euler_mse_first10": float(np.mean(unwrap_mse10)),
        "geodesic_orientation_error_first5_rad": float(np.mean(geo5)),
        "geodesic_orientation_error_first10_rad": float(np.mean(geo10)),
    }


def load_checkpoint(checkpoint: str, device: str):
    eba.CHECKPOINT = checkpoint
    logger.info("=== loading policy from %s ===", checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint)
    policy = policy.to(device)
    policy.eval()

    eba.PREPROCESSOR, eba.POSTPROCESSOR = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=eba.DATASET_REPO_ID),
        policy=policy.config,
        rename_map=eba.RENAME_MAP,
        batch_size=1,
    )
    dataset = make_dataset(train_cfg)
    return policy, dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-file", default="outputs/eval/euler_branch_audit/euler_branch_audit_B.json")
    parser.add_argument("--samples-file", default="outputs/eval/euler_branch_audit/euler_branch_samples_B.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/euler_branch_audit/checkpoint_comparison_B_vs_sincos.json")
    args = parser.parse_args()

    audit = json.loads(Path(args.audit_file).read_text())
    fixed = audit["fixed_sample_set"]
    general_samples = [{"sample_index": i} for i in fixed["general_sample_indices"]]
    branch_cut_samples = [{"sample_index": i} for i in fixed["branch_cut_sample_indices"]]
    logger.info(
        "reusing fixed sample sets from %s: %d general, %d branch-cut (seed=%d)",
        args.audit_file,
        len(general_samples),
        len(branch_cut_samples),
        fixed["seed"],
    )

    results = {}

    logger.info("=== B (raw-Euler): reusing already-saved results, not re-running inference ===")
    logger.info(
        "B's decoder.action_head is [7, 512]; current code always builds a 10-D decoder now, so "
        "SafeDiffVLAPolicy.from_pretrained(B) cannot load anymore (exactly the incompatibility "
        "flagged up front). Recomputing summaries from saved data instead of re-running the model."
    )
    branch_cut_records_B = json.loads(Path(args.samples_file).read_text())["samples"]
    results["B_raw_euler"] = {
        "checkpoint": OLD_CHECKPOINT,
        "note": "recomputed from already-saved audit output, not re-run (incompatible with current code)",
        "general_sample_set": audit["model_vs_gt_general_sample_set"],
        "branch_cut_sample_set": summarize_from_saved_records(branch_cut_records_B),
    }
    logger.info("[B_raw_euler] branch_cut_summary: %s", results["B_raw_euler"]["branch_cut_sample_set"])

    logger.info("=== sincos: %s ===", NEW_CHECKPOINT)
    policy, dataset = load_checkpoint(NEW_CHECKPOINT, args.device)
    general_summary, _ = eba.run_model_comparison(policy, args.device, dataset, general_samples)
    logger.info("[sincos] general_summary: %s", general_summary)
    branch_cut_summary, _ = eba.run_model_comparison(policy, args.device, dataset, branch_cut_samples)
    logger.info("[sincos] branch_cut_summary: %s", branch_cut_summary)
    results["sincos"] = {
        "checkpoint": NEW_CHECKPOINT,
        "general_sample_set": general_summary,
        "branch_cut_sample_set": branch_cut_summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fixed_sample_set_source": args.audit_file, "results": results}, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
