# Task1 response-v3 C-aligned-v2 ACT200k

Status: `implementation_ready_training_not_started_hardware_not_authorized`.

This experiment trains exactly one ACT model from frozen Real24 plus the newly
collected human Quest response-v3 Sim-gap24 trajectories. The Sim rows are a
genuinely new trajectory Dataset (24 episodes / 3023 frames), so this binder
does **not** compare them to the historical LocalSim-gap24 state/action rows.
It instead validates the frozen finalization identity, the new human-source to
real-appearance-derived row mapping, the official loader contract, and the
fixed 4+4 sampler. Only the Real24 sampler stream must byte-match old C.

The formal recipe is cloned from old C and may differ only in Dataset
root/repo-id, output directory and job name. It keeps frozen Real24
normalization, seed 1000, ACT chunk/action steps 67, and fixed step 200000.

After training, software preparation is limited to the frozen Sim-seen6 paired
gate: six poses x C-source/C-aligned-v2 = 12 trials. There is no automatic
24-pose/48-trial fallback. Hardware remains unauthorized until a separate GO.

## Main-lane commands

First copy the frozen research finalization and Sim-seen6 contracts to a stable
Ubuntu read-only evidence path. Then run:

```bash
cd /home/ubuntu24/Teleop/lerobot
.venv/bin/python experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1/bind_and_prepare.py \
  --finalization-manifest /home/ubuntu24/Teleop/artifacts/research-control/task1-picklift-localsim24-gap-response-v3-real-appearance-finalization-v1.json \
  --finalization-manifest-sha256 a271bcd613bb8cc19e8dd506380a722a0020f1bbd0b1d0cf2a0b8352e4e65b25 \
  --materialize
```

The binder prints the two generated config paths. Run the 500-step smoke first;
after it completes naturally, run the formal config in a fresh output path:

```bash
mkdir -p /home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_act_caligned_v2_response_v3_200k_v1/training_v1
.venv/bin/python -m lerobot.scripts.lerobot_train \
  --config_path experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1/bound_configs/c_aligned_v2_smoke_500.json \
  > /home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_act_caligned_v2_response_v3_200k_v1/training_v1/smoke.log 2>&1
.venv/bin/python -m lerobot.scripts.lerobot_train \
  --config_path experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1/bound_configs/c_aligned_v2_full_200k.json \
  > /home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_act_caligned_v2_response_v3_200k_v1/training_v1/formal.log 2>&1
.venv/bin/python experiments/task1_picklift_act_caligned_v2_response_v3_200k_v1/validate_and_freeze.py
```

Do not resume formal training from smoke and do not select any checkpoint other
than step 200000.
