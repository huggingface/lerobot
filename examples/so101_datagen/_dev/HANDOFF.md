# Handoff: so101_datagen v2 (2026-10-30)

This file and the rest of `_dev/` are a temporary handoff. Remove `_dev/` before the PR is reviewed.

- **The plan:** `PLAN.md`, approved by the user. It has six checkpoints (0–5), each signed off visually by the user before the next.
- **Paper notes:** `SOTA_ANALYSIS.md`.
- **Scripts:** run from the worktree root with `PYTHONPATH=.`:
  - `carry_stats.py`
  - `review_clips.py`
  - `sag_check.py` (needs `matplotlib`)

## Status: checkpoint 0 (carry height) in progress, untested last step

**Why checkpoint 0 exists.** On the real SO-101, `g05_so101_sim4k` picked objects but carried them too low toward the box.

### Measurements done

**Gravity sag (`sag_check.py`)** fits the action − state gap against gravity torque, on still frames with each episode's mean removed.

| Data                                             | shoulder_lift slope | Over the typical torque range            |
| ------------------------------------------------ | ------------------- | ---------------------------------------- |
| Real (hungho77, IlyaGusev, svla_so101_pickplace) | about +1.0 deg/Nm   | about 0.8°                               |
| Our sim                                          | −2.1 deg/Nm         | wrong sign; probably a backlash artifact |

- The elbow results are noisy and depend on joint conventions.
- **Conclusion:** sag is a minor cause, a few millimetres. The optional `GravitySag` actuation step is low priority.

**Undertraining.** The checkpoint saw 4000 steps × batch 24 = 96k samples, about 9% of one epoch of 1.02M frames. It is very likely undertrained. Resuming toward 20k steps (pushed to a public repo) is the cheapest test.

**Carry stats (`carry_stats.py`, 400 seeds, place_in + place_near only).** Heights are in cm.

| Variant                                       | Successes | Planning failures | place_in clearance over path obstacles, p10 / p50 |
| --------------------------------------------- | --------- | ----------------- | ------------------------------------------------- |
| Baseline (HEAD `48384bbe1`)                   | 256       | 53                | 2.2 / 5.0                                         |
| Hard hover floor (`hover_z` = clear + 4–8 cm) | 177       | 156               | 4.4 / 6.5                                         |
| + vertical hover fallback                     | 209       | 120               | 4.4 / 6.5                                         |
| + soft target with a 2 cm hard minimum        | 257       | 56                | 1.8 / 5.0                                         |

- **Hard hover floor:** clearance improves, but the hover is unreachable near the base.
- **Soft target:** success is back to baseline, but clearance is back to baseline too.
- **Lift height** didn't rise despite `lift_height` (0.08, 0.15): the lift keeps the grasp orientation, and the SO-101 can't go higher pointing down.

**Root cause.** The arm can't be high while its gripper points down near the base. `_via` silently skipped the raise when IK failed.

### Code changes in this commit

- `config.py`:
  - `lift_height` is now (0.08, 0.15);
  - new `carry_margin` (0.04, 0.08);
  - new `min_carry_margin` 0.02.
- `tasks/__init__.py`: `obstacle_top(scene, start, end, exclude)`. `place_in` and `place_near` pass `hover_z` and `min_hover_z` to `planner.place`.
- `motion.py`:
  - `place(pos, tilt, hover_z, min_hover_z)` hovers back along the approach, or straight above the release when that's out of reach;
  - `_highest(..., away=, floor=)` replaces the old `grasp=` argument;
  - `transit(..., min_z=)`.
  - **Last change, not yet measured:** `_via(middle, min_z)` falls back to position-only IK, so the wrist may tilt to raise the carry arc.
- `so101.py`: `Arm.ik` accepts `approach=None` (position only). The convergence check is the same as before, per block.

### Next steps

1. Measure the last change:
   - run `carry_stats.py` after the change;
   - run it on the baseline too. For the baseline, `git archive 48384bbe1 examples/so101_datagen | tar -x -C <tmp>`, add the `obstacle_top` function into `<tmp>`'s `tasks/__init__.py`, then `PYTHONPATH=<tmp>`.
   - The "lowest over the box walls" metric is the one that matters.
   - Target: place_in p10 clearance of at least 4 cm with success at least as good as baseline.
2. Render review clips. Before (seeds 5 8 11 14 20 27, `--tasks place_in place_near`) is the baseline code; after is the new code. Write them to `review/0_carry/{before,after}`.
   - Clips put the front view next to a side witness camera, with the object-bottom and rim height burned in.
   - Plot the carry curves (`*.npy` from `carry_stats --out`) with the rim.
   - Show them to the user and wait for an OK.
3. Then checkpoints 1–5 from `PLAN.md`:
   1. Side grasps: wrist roll 90° so the jaws close tangentially, tilt 60–90°. Steep top grasps get tangential jaws too.
   2. In-episode retries via the `Rollout` refactor.
   3. A 3 Hz front-box stream plus a grasp-point keypoint.
   4. A `wide` camera mode plus an occlusion check.
   5. `MotionNoise` (Jacobian).

## Rules

- **Never 14 workers locally** (OOM); use 6–8.
- **Tests:** scoped only, `pytest tests/test_so101_datagen.py`, never the full suite.
- **Lint:** `uvx ruff@0.15.0 format/check examples/so101_datagen`.
- **Git:**
  - commit only when asked, one-line messages, no trailers;
  - push with an explicit refspec, `git push origin feat/so101-datagen`;
  - don't touch `uv.lock`.
- **Hub:** push models and datasets public; private storage is full.
