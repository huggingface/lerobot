# Task1 Real24 + LocalSim48 contract (software-ready)

This is preparation only. No LocalSim48 accepted manifest is currently available, so materialization, training, checkpoint selection, rollout, and hardware access are prohibited.

The future materializer must verify fixed Real24 accepted identity, the selected sim subset and state-bank hashes, success-only training view, raw failed attempts excluded from training, front 640x480 at 20 FPS, `float32[6]` state and actual-applied action, joint/units/camera/alignment/runtime identities, and no held-out Real48 state provenance.

Conditions are overlap, gap (primary), and full Sim48. All new ACT runs are seed 1000, 100k steps, fixed 100k checkpoint, and deterministic 4 Real + 4 Sim per batch. The existing Real24-only 23/48 is reference-only.

The only permitted future start is after the validator reports `dataset_ready=true` for a concrete accepted manifest; otherwise status remains `software_ready_waiting_for_local_dataset`.
