# Task1 Real24 budget extension — same Eval48 software gate

This experiment evaluates the fixed Real24 budget-extension ACT 100k checkpoint on the unchanged 48-pose Eval48 bank. The software gate is frozen with `hardware_authorized=false`; no hardware or rollout was accessed.

- Plan: `evaluation_plan.json` (`ada1a17eecc972a999fe8e8540015b42ebc3115577bd34a73985a9f97eb29abf`)
- Trials: 48, one Real24 trial per original pose, original pose order unchanged.
- Real24 membership tiers: 12 seen by Real24, 12 added by Real48, 18 added by Real96, 6 unseen by both.
- Runtime contract: official send, frozen ready before/after, policy reset after ready, 20 Hz, full 30 s, no catch-up, `max_relative_target=None`, no runner clamp or step limiter.
- Software evidence: `/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_budget_extension_same_eval48_v1/software_gate_v1`.

Do not execute hardware without a later explicit research-control GO.
