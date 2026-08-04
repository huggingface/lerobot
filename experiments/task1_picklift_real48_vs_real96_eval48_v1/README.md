# Task1 Real48 versus Real96 paired Eval48

This directory owns the frozen software contract for
`task1_picklift_real48_vs_real96_eval48_v1`.

- Research source: commit `73908355df1add52cd04753216c13f8b1c0b400a`.
- Plan: 48 frozen poses, two fixed ACT checkpoints per pose, 96 scored trials.
- Pairing: both models run consecutively at the same nominal manual placement;
  first-model order is copied directly from the research pose manifest.
- Runtime: canonical front 640x480 at 20 Hz for a complete 30-second wall
  window; success never shortens the policy window.
- Action path: official `SO101Follower.send_action`,
  `max_relative_target=None`, no runner absolute clamp, no custom 5-degree
  limiter, and no catch-up burst.
- Ready/return: the same frozen Task1 ready pose before and after every trial;
  reset the ACT action queue only after ready is observed.
- Manual placements are nominal protocol coordinates, not millimetre-accurate
  measured ground truth.

The software gate is frozen in `software_gate_result.json`. It contains no
real-camera, serial, robot, torque, 12 V, policy-rollout, or success-rate
evidence. `--execute-hardware` is reserved for a later explicit research-control
hardware GO.

After reviewed real evidence exists, `summarize_reviewed_results.py` freezes the
overall, coverage-tier, cell, yaw, failure-category, and paired Real96-minus-
Real48 summaries while preserving operator, video-review, and adjudication
labels.
