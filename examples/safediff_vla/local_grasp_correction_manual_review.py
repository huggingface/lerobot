#!/usr/bin/env python
"""Manual (human/Claude-in-the-loop) visual classification of the 18 rollout episodes already on
disk from the v2.5 multitask Stage 2 checkpoint (`outputs/train/
safediff_vla_grounded_grasp_v2_5_multitask_stage2_5k/checkpoints/005000/pretrained_model`, the same
one `grasp_point_bias_diagnostic.py`/`trajectory_shape_diagnostic.py` already analyzed) -- done
because numeric bias/failure-mode metrics ALONE were judged insufficient to conclude what's
actually wrong (see the design note this feeds): each episode was inspected via its rendered
`image_overlay.png` + `trajectory_3d.png` (both already-generated real-camera-frame renders, not
abstract plots) and, for two episodes where those artifacts either didn't exist or the numeric
"close point" heuristic looked contradicted by the final outcome, a raw rollout `.mp4` frame was
extracted directly with ffmpeg and the full `episodes_raw.json` step trace was read instead of
trusting either diagnostic script's own "close point" pick in isolation. NOT a re-derivable
computation -- this file's `EPISODES` list IS the record of that manual judgment; re-running this
script only recomputes the residual_A statistics FROM it, it does not redo the classification.

Two corrections this manual pass made to what a purely numeric/single-script read would have
concluded (both explained in each episode's own `note` field below):
  - select_book/seed=1000: `trajectory_shape_diagnostic.py` labeled this "never entered
    LOCAL_GRASP" (its `stage==1` filter found none), but `grasp_point_bias_diagnostic.py`'s
    latch-search found a real `just_latched_gripper` step -- the raw `episodes_raw.json` trace
    shows step 30 truly WAS computed under LOCAL_GRASP (`local_delta_xyz_m` populated) but got
    logged with `stage=2` because `_v25_stage` is updated (LOCAL_GRASP->POST_GRASP) before that
    same step's telemetry is captured -- an eval-harness logging-order artifact, not a modeling
    bug, and orthogonal to anything this design touches (documented here rather than silently
    picking one script's label).
  - select_poker/seed=1000: the short LOCAL_GRASP-window plot shows the FIRST `just_latched_gripper`
    event (step 64, `ee_to_target_dist`=13.6cm, looks like a close on the wrong card) but the
    episode continues for 328 more steps and the REAL, simulator-confirmed grasp (`is_grasped`,
    `success=True`) only happens at step 392 -- the short-window plot simply doesn't cover the part
    of the episode where the outcome was decided. Classified D_OK_OR_OTHER (succeeded), NOT
    B_WRONG_OBJECT as an earlier, single-image read of this same episode had concluded.

Classes (see the design note for full definitions):
    A_FINE_GRASP_ERROR   -- identity correct, predicted target/close point off the graspable region
    B_WRONG_OBJECT       -- a different object/instance was targeted
    C_TIMING_OR_POST_GRASP -- approach/target correct, failure is timing/overshoot/post-contact
    D_OK_OR_OTHER        -- succeeded, or genuinely doesn't fit the above (incl. "never entered
                             LOCAL_GRASP at all", a PRE_GRASP-level failure out of this design's scope)

Outputs:
    outputs/eval/safediff_vla_local_grasp_correction_manual_review/manual_review.json

Usage:
    uv run python examples/safediff_vla/local_grasp_correction_manual_review.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/eval/safediff_vla_local_grasp_correction_manual_review")

# predicted_target_to_gt_cm / actual_tcp_to_gt_cm: from `grasp_point_bias_diagnostic.py`'s own
# per-episode residual vectors (||predicted_target - GT||, ||actual_TCP - GT||), re-derived here
# only as norms for this table -- not recomputed independently.
EPISODES = [
    {"task": "select_fruit", "seed": 1000, "cls": "C_TIMING_OR_POST_GRASP", "pred_to_gt_cm": 8.0, "tcp_to_gt_cm": 5.2,
         "grasp_success": True, "task_success": False,
         "evidence": "Precise reach to the orange next to the plate (close point on GT star); grasped at "
                  "some point (ever_grasped=True) but task ultimately failed -- post-contact/placement "
                  "issue, not a targeting error (triggered_modes: post_contact_instability)."},
    {"task": "select_fruit", "seed": 1001, "cls": "C_TIMING_OR_POST_GRASP", "pred_to_gt_cm": 3.4, "tcp_to_gt_cm": 8.8,
         "grasp_success": False, "task_success": False,
         "evidence": "predicted_target path lands right on the GT star (image_overlay.png) -- accurate "
                  "grounding -- but no commanded_EE/actual_TCP path is drawn at all near the table: the "
                  "gripper closed while still up near the arm joint, before actually descending. "
                  "insufficient_approach flag fired; min_dist==close_dist (stable, not an overshoot)."},
    {"task": "select_fruit", "seed": 1002, "cls": "C_TIMING_OR_POST_GRASP", "pred_to_gt_cm": 46.8, "tcp_to_gt_cm": 42.4,
         "grasp_success": False, "task_success": False,
         "evidence": "predicted_target path is a large chaotic scribble near the gripper's own hand, far "
                  "from the GT star; BUT min_dist_m=7.5cm (got genuinely close at some point) while "
                  "close_dist_m=42.4cm -- overshoot+premature_close+delayed_close all fired, i.e. an "
                  "approach that found the right area then badly overshot/mistimed the close, not a "
                  "stable wrong-region lock (reclassified from an initial 'looks like B' read of the "
                  "image alone once min_dist was cross-checked)."},
    {"task": "select_toy", "seed": 1000, "cls": "D_OK_OR_OTHER", "pred_to_gt_cm": 17.9, "tcp_to_gt_cm": 10.0,
         "grasp_success": True, "task_success": True,
         "evidence": "success=True. predicted_target/close point both near the correct small figure; "
                  "first_grasped_ix==latch_ix (no long recovery needed)."},
    {"task": "select_toy", "seed": 1001, "cls": "A_FINE_GRASP_ERROR", "pred_to_gt_cm": 16.4, "tcp_to_gt_cm": 17.6,
         "grasp_success": False, "task_success": False,
         "evidence": "Correct figure region (distinct from the other two figures on the table) but "
                  "consistently ~16-17cm off (min_dist==close_dist, stable -- no overshoot/premature/"
                  "delayed_close flags, just insufficient_approach): a stable, moderate localization "
                  "error on the right object, not a timing spike or wrong instance."},
    {"task": "select_toy", "seed": 1002, "cls": "A_FINE_GRASP_ERROR", "pred_to_gt_cm": 8.4, "tcp_to_gt_cm": 13.4,
         "grasp_success": False, "task_success": False,
         "evidence": "Correct small-figure cluster identified (distinguished from the clearly-different "
                  "Superman/Batman figures nearby); stable 8-13cm miss, no overshoot signature."},
    {"task": "select_drink", "seed": 1000, "cls": "C_TIMING_OR_POST_GRASP", "pred_to_gt_cm": 21.3, "tcp_to_gt_cm": 5.6,
         "grasp_success": True, "task_success": False,
         "evidence": "Wrist-camera view: predicted_target path is detached onto an irrelevant fridge-door "
                  "control-panel region (21.3cm off -- a real, separately-notable grounding-signal "
                  "defect, flagged but not used to override this outcome-based label), yet "
                  "commanded_EE/actual_TCP still converge tightly to the true bottle (5.6cm, stable); "
                  "grasped but task ultimately failed afterward -- post-contact, not targeting."},
    {"task": "select_drink", "seed": 1001, "cls": "D_OK_OR_OTHER", "pred_to_gt_cm": 7.1, "tcp_to_gt_cm": 1.2,
         "grasp_success": True, "task_success": True,
         "evidence": "success=True, 1.2cm final precision. Same predicted_target/control-panel detachment "
                  "as seed=1000 noted for completeness, again not overriding the (successful) outcome."},
    {"task": "select_drink", "seed": 1002, "cls": "D_OK_OR_OTHER", "pred_to_gt_cm": None, "tcp_to_gt_cm": None,
         "grasp_success": False, "task_success": False,
         "evidence": "Never entered LOCAL_GRASP at all (stage stays 0 the whole episode, no latch, no "
                  "local_delta_xyz_m steps, min ee_to_target_dist=9.1cm briefly near the 10cm entry "
                  "radius but never sustained) -- a PRE_GRASP-level approach failure, out of scope for "
                  "a LOCAL_GRASP-only correction head by construction."},
    {"task": "select_book", "seed": 1000, "cls": "A_FINE_GRASP_ERROR", "pred_to_gt_cm": 12.7, "tcp_to_gt_cm": 17.3,
         "grasp_success": False, "task_success": False,
         "evidence": "No image_overlay.png exists for this episode (trajectory_shape_diagnostic's "
                  "stage==1 filter found zero steps -- see the eval-harness logging-order artifact "
                  "explained in the module docstring). Resolved by reading episodes_raw.json directly "
                  "(step 30: local_delta_xyz_m populated, just_latched_gripper=True, stage logged as 2) "
                  "plus two extracted rollout-video frames (steps ~23 and ~49): target_entity= "
                  "'biopolymer_composites_in_electronics' (the green book) and the gripper is genuinely "
                  "hovering over that same green book in both frames -- correct identity, but it closed "
                  "17.3cm away and never actually grasped (ends the episode 20.9cm from GT, "
                  "predicted_target_dist_to_gt_target=25.9cm at the final step)."},
    {"task": "select_book", "seed": 1001, "cls": "A_FINE_GRASP_ERROR", "pred_to_gt_cm": 13.6, "tcp_to_gt_cm": 12.9,
         "grasp_success": False, "task_success": False,
         "evidence": "Close point (red X) sits at the edge of the correct 'Contract Law in Japan' book, "
                  "near (not on) the GT star; predicted_target cluster offset further left, but the "
                  "final close position is plausibly the same book. min_dist==close_dist (stable "
                  "12.1-12.9cm), no overshoot/timing flags -- fine-localization error, not identity."},
    {"task": "select_book", "seed": 1002, "cls": "B_WRONG_OBJECT", "pred_to_gt_cm": 21.7, "tcp_to_gt_cm": 12.3,
         "grasp_success": False, "task_success": False,
         "evidence": "target_entity/GT star is at 'The Story of American Freedom' (Eric Foner, left edge "
                  "of frame); predicted_target cluster sits on an unrelated lower-shelf item, and the "
                  "close point (red X) lands on a COMPLETELY DIFFERENT book ('Data Visualization', "
                  "clearly separated in the image) -- min_dist==close_dist (12.1-12.3cm, stable: it "
                  "consistently locked onto the wrong book, this was not noisy)."},
    {"task": "select_mahjong", "seed": 1000, "cls": "C_TIMING_OR_POST_GRASP", "pred_to_gt_cm": 15.5, "tcp_to_gt_cm": 21.8,
         "grasp_success": False, "task_success": False,
         "evidence": "LOW CONFIDENCE / visually ambiguous: the GT star projects onto a position "
                  "overlapping the arm's own wrist hardware at this camera angle (likely occluding the "
                  "true tile), so object identity could not be confirmed from the image alone. "
                  "min_dist=16.8cm vs close_dist=21.8cm (overshoot+delayed_close both fired) -- given "
                  "the divergence pattern matches select_fruit/1002's overshoot signature more than a "
                  "stable wrong-object lock, tentatively grouped as C, flagged low-confidence rather "
                  "than asserted."},
    {"task": "select_mahjong", "seed": 1001, "cls": "D_OK_OR_OTHER", "pred_to_gt_cm": 7.8, "tcp_to_gt_cm": 2.9,
         "grasp_success": True, "task_success": True,
         "evidence": "success=True; predicted_target and close point both essentially on the GT star "
                  "(one of 7 tiles laid out) -- precise and correct."},
    {"task": "select_mahjong", "seed": 1002, "cls": "B_WRONG_OBJECT", "pred_to_gt_cm": 3.7, "tcp_to_gt_cm": 8.4,
         "grasp_success": False, "task_success": False,
         "evidence": "5 near-identical green tiles in a row; GT star is the RIGHTMOST tile, but "
                  "predicted_target and close point both land on the tile immediately to its LEFT (one "
                  "position off). NOTE: pred_to_gt is only 3.7cm here -- a SMALL coarse-target error --"
                  " which was still enough to land on the wrong tile because tile pitch in this scene "
                  "is smaller than that error. Important caveat for the outlier-clip mitigation below: "
                  "B_WRONG_OBJECT is not always large-magnitude in cluttered scenes."},
    {"task": "select_poker", "seed": 1000, "cls": "D_OK_OR_OTHER", "pred_to_gt_cm": 17.5, "tcp_to_gt_cm": 13.6,
         "grasp_success": True, "task_success": True,
         "evidence": "success=True (confirmed in episodes_raw.json). The short LOCAL_GRASP-window image "
                  "only shows the FIRST just_latched_gripper event (step 64, 13.6cm from GT, looks like "
                  "the wrong card in isolation); the real outcome (is_grasped=True, success=True) only "
                  "resolves 328 steps later at step 392 -- outside that plot's window. RETRACTS an "
                  "earlier same-episode read (from a prior turn) that called this B_WRONG_OBJECT off "
                  "the short-window image alone, without checking the full episode."},
    {"task": "select_poker", "seed": 1001, "cls": "C_TIMING_OR_POST_GRASP", "pred_to_gt_cm": 18.0, "tcp_to_gt_cm": 4.4,
         "grasp_success": True, "task_success": False,
         "evidence": "Close point/actual_TCP land tightly on the correct (middle, '2_of_diamonds') deck "
                  "(4.4cm); predicted_target itself is detached/wrong (18.0cm, same pattern as "
                  "select_drink) but not used to override this outcome-based label. Grasped but task "
                  "ultimately failed -- overshoot+premature_close+delayed_close all fired."},
    {"task": "select_poker", "seed": 1002, "cls": "B_WRONG_OBJECT", "pred_to_gt_cm": 22.5, "tcp_to_gt_cm": 14.6,
         "grasp_success": False, "task_success": False,
         "evidence": "3 decks visible; GT star on the LEFT deck, predicted_target and close point both "
                  "on the MIDDLE deck (one position over) -- min_dist==close_dist (13.3-14.6cm, "
                  "stable): a consistent wrong-instance lock, not noise."},
]


def main() -> None:
    by_class: dict[str, list[dict]] = {}
    for e in EPISODES:
        by_class.setdefault(e["cls"], []).append(e)

    logger.info("=== class counts (n=%d episodes) ===", len(EPISODES))
    for cls, eps in sorted(by_class.items()):
        logger.info("%-24s %d  (%s)", cls, len(eps), ", ".join(f"{e['task']}/{e['seed']}" for e in eps))

    # residual_A = demo_grasp_TCP_xyz - coarse_predicted_target_xyz. For these ROLLOUT episodes the
    # only available "demo grasp TCP" proxy is the privileged sim GT_object_center_xyz that
    # `grasp_point_bias_diagnostic.py` already logs (Entity.get_xpos) -- NOT the same thing as the
    # dataset's `find_grasp_target`-derived label the actual training loss uses (see
    # `grasp_correction_head.py`'s docstring); this magnitude-only substitution is only for choosing
    # `local_grasp_max_grasp_correction_m` from real rollout evidence, spelled out here rather than
    # silently conflated with the training-time label.
    a_episodes = by_class.get("A_FINE_GRASP_ERROR", [])
    per_task_a: dict[str, list[float]] = {}
    for e in a_episodes:
        per_task_a.setdefault(e["task"], []).append(e["pred_to_gt_cm"])

    def stats(vals: list[float]) -> dict:
        arr = np.asarray(vals, dtype=float)
        return {
            "n": len(arr),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "max": round(float(arr.max()), 2),
        }

    residual_a_report = {task: stats(vals) for task, vals in per_task_a.items()}
    all_a_vals = [v for vals in per_task_a.values() for v in vals]
    residual_a_report["overall"] = stats(all_a_vals)

    logger.info("=== residual_A (||GT_object_center - predicted_target||, cm) -- A_FINE_GRASP_ERROR only ===")
    logger.info("CAVEAT: n=%d total across %d tasks -- FAR too small for a reliable p90; treat as a "
                "directional first cut, not a settled distribution.", len(all_a_vals), len(per_task_a))
    for task, s in residual_a_report.items():
        logger.info("%-14s %s", task, s)

    suggested_max_grasp_correction_m = round(residual_a_report["overall"]["p90"] / 100 + 0.005, 3)
    logger.info(
        "=== suggested local_grasp_max_grasp_correction_m ~= %.3fm (A-class p90=%.1fcm, small margin) ===",
        suggested_max_grasp_correction_m, residual_a_report["overall"]["p90"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "episodes": EPISODES,
        "class_counts": {cls: len(eps) for cls, eps in by_class.items()},
        "residual_A_cm": residual_a_report,
        "suggested_local_grasp_max_grasp_correction_m": suggested_max_grasp_correction_m,
    }
    (OUT_DIR / "manual_review.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", OUT_DIR / "manual_review.json")


if __name__ == "__main__":
    main()
