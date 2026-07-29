# Task1 controlled Mixed v2

This is one pre-fixed, seed-1000 ACT training run from scratch. It bundles three
engineering-contract checks/corrections and is not an ablation:

- Visual preprocessing matches the verified Real24-only run:
  `use_imagenet_stats=true`, ResNet18 ImageNet initialization, no image
  augmentation, front-only 640x480 at 20 Hz.
- The frozen collectors are audited before derivation. Both domains already
  encode gripper position as `MotorNormMode.RANGE_0_100`, with `0=closed` and
  `100=open`. The versioned canonicalization is therefore an exact identity
  transform in both domains. The narrower Real demonstration range is retained;
  no binarization, quantile mapping, scaling, or clipping is applied.
- Every batch is deterministic and domain-balanced: batch size 8 contains
  exactly 4 Real and 4 Quest-Sim frames. Each source frame is used at most once
  per sampler epoch; the smaller domain defines the number of complete batches,
  so no domain is oversampled.

The derived Dataset v3 view contains all 24 accepted Real and 24 accepted
Quest-Sim episodes exactly once (7,966 frames total). Source videos remain
separate and are not re-encoded; high-resolution sidecars are excluded. Source
trees stay immutable and all source episode identities remain in the external
provenance map.

All other ACT settings match the frozen Real24-only baseline: state plus front
RGB to action, 67-step chunks/actions, MEAN_STD, model width 512, 8 heads,
feed-forward 3200, encoder 4, decoder 1, VAE latent 32, KL 10, learning rate
1e-5, weight decay 1e-4, batch 8, 500-step CUDA smoke, then 100,000 steps with
20,000-step checkpoints. Only checkpoint 100,000 is selected.

Artifacts live under `/home/ubuntu24/Teleop/artifacts`. This task performs no
hardware access and no simulation or real rollout.

## Completed offline result

- Derived Dataset v3 tree SHA-256:
  `b137d32531e9455afdaea385ad93fea7584f4b790146a861cba217dec7a79b7c`.
- Smoke: 500 steps, final loss `2.669`; actual samples were exactly 2,000
  Real and 2,000 Sim.
- Full run: 100,000 steps in 5,016 seconds; final loss `0.051`, L1 `0.050`,
  KLD `0.000`; actual samples were exactly 400,000 Real and 400,000 Sim.
- Fixed step-100000 model SHA-256:
  `b7faae880393bdbf5e44ebeaab1f399f732d6ee325be698f999c90eb865cee68`.
- CUDA reload and offline inference passed for one Real and one Sim sample;
  both returned finite `[1,6]` actions. The saved processor contains the
  ImageNet mean/std values.
- Frozen training evidence:
  `/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_questsim24_act_v2/training_result_v1`.

These are offline engineering facts only; Mixed v2 has not been evaluated in
MuJoCo or on the real robot.
