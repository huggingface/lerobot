# Task1 Real24 + Quest-Sim48 v7 ACT v1

This experiment builds a new immutable LeRobot Dataset v3 training view from
the frozen Real24 accepted subset and the frozen Quest-Sim48 v7 accepted
subset. Every accepted source episode appears exactly once. High-resolution
simulation sidecars are excluded, and both raw source trees remain unchanged.

Training uses the frozen Real24-only ACT recipe and the deterministic sampler
already validated for controlled Mixed v2: every batch of eight contains four
Real frames and four Sim frames. Visual preprocessing uses ImageNet statistics.
The policy is trained from scratch with seed 1000; a 500-step CUDA smoke must
pass before the fixed 100,000-step run starts. Only checkpoint 100,000 is
selected.

This is an offline engineering baseline. It does not run a simulator or real
robot, and it does not establish a paper effect.

Key commands, from the owning repository with its Python 3.12 environment:

```bash
python experiments/task1_picklift_real24_questsim48_v7_act_v1/prepare_combined_dataset.py
python experiments/task1_picklift_real24_questsim48_v7_act_v1/verify_combined_dataset.py
python experiments/task1_picklift_real24_questsim48_v7_act_v1/verify_training_contract.py \
  --output /home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_questsim48_v7_act_v1/combined72_freeze_v1/training_contract_verification.json
```

The smoke and full runs use `train_config_smoke.json` and
`train_config_full.json`. Result artifacts remain under `/home/ubuntu24/Teleop/artifacts`;
only compact manifests, scripts, and result indices are committed.
