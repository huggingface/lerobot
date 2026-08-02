# Task1 Real24 budget-extension ACT v1

Pure-offline, from-scratch ACT training for the frozen deterministic Real24 subset derived from the new Real48 dataset. The recipe is matched to the frozen Real48/Real96 ACT runs; only dataset membership and its derived state/action statistics differ.

- Research commit: `428bdcd`
- Plan: `task1_picklift_real24_act_budget_extension_v1`
- Recipe: `task1_act100k_front20hz_chunk67_v1`
- Dataset: 24 episodes / 4263 frames, front-only 640x480 at 20 Hz
- Smoke: independent 500 steps from scratch
- Full: independent 100000 steps from scratch, checkpoints every 20000, fixed selection at 100000
- Hardware and all rollout paths remain forbidden.

Large datasets, logs, checkpoints, and evidence remain under `/home/ubuntu24/Teleop/artifacts` and are not committed.
