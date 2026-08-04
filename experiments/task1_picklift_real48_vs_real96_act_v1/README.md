# Task1 Real48 versus Real96 matched ACT training v1

This experiment trains exactly two pure-Real ACT policies under research
contract `73908355df1add52cd04753216c13f8b1c0b400a`:

- `ACT_Real48_seed1000_step100000`
- `ACT_Real96_seed1000_step100000`

The conditions differ only in the frozen dataset identity and the state/action
MEAN_STD statistics necessarily derived from that dataset. Model, optimizer,
runtime, seed, sampling, step budget, checkpoint schedule, and ImageNet visual
normalization are matched. Both policies start from scratch. The smoke
checkpoints are never resumed by the formal runs.

The accepted Real96 view has 96 episodes and 17,439 frames. The two retained
discard attempts account for another 242 raw frames and are not in either
training condition. No simulation data or prior checkpoint is used.

## Offline sequence

```bash
cd /home/ubuntu24/Teleop/lerobot

.venv/bin/python \
  experiments/task1_picklift_real48_vs_real96_act_v1/audit_datasets.py
.venv/bin/python \
  experiments/task1_picklift_real48_vs_real96_act_v1/verify_training_contract.py

.venv/bin/lerobot-train --config_path \
  experiments/task1_picklift_real48_vs_real96_act_v1/real48_train_config_smoke.json
.venv/bin/lerobot-train --config_path \
  experiments/task1_picklift_real48_vs_real96_act_v1/real96_train_config_smoke.json

# Run only after both independent smokes and their offline checkpoint reloads pass.
.venv/bin/lerobot-train --config_path \
  experiments/task1_picklift_real48_vs_real96_act_v1/real48_train_config_full.json
.venv/bin/lerobot-train --config_path \
  experiments/task1_picklift_real48_vs_real96_act_v1/real96_train_config_full.json
```

Only the fixed step-100000 checkpoints are selected. The 20k, 40k, 60k, and
80k checkpoints are retained only for provenance.

This directory does not authorize or implement serial, camera, robot, torque,
12 V, Quest, Remote, MuJoCo, or real-robot execution. Training loss and offline
inference are engineering validation, not a success-rate or paper-effect
conclusion.
