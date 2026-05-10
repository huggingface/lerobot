# Residual RL iteration log — push K-RA-BC 80% → 95%

**Goal**: 95%+ success in <2h finetuning. Base = K-RA-BC κ=0.01 frozen. Reward = K SARM sync_inference.

## v1 (residual_action_scale=[0.1,0.1,0.1,0.1,2.0])

```
Config: outputs/residual_K_v1
  base_policy: K-RA-BC κ=0.01 80k
  residual_scale: [0.1,0.1,0.1,0.1,2.0]   ← gripper 2.0 likely catastrophic
  online_step_before_learning: 500
  temperature_init: 0.05
  total elapsed: ~30 min
  learner optim steps: ~3200
```

**Result: 9/20 = 45% success**, mean ~0.65 — UNDER baseline 80%.

Per-ep reward stream:
0.28 / 0.957 / 0.472 / 0.952 / 0.270 / 0.270 / 0.952 / 0.439 / 0.796 / 0.954 /
0.789 / 0.953 / 0.297 / 0.955 / 0.296 / 0.297 / 0.951 / 0.958 / 0.299 / 0.951

**Diagnosis**: random-init residual actor adds noise that disrupts good ACT actions.
Specifically: `residual_action_scale[4]=2.0` for gripper means SAC can flip gripper randomly — destroys gripper timing learned by ACT.

## v2 (residual_action_scale=[0.05,0.05,0.05,0.05,0.0])

scale=0.05 xyz/yaw, frozen gripper. online_step_before_learning=1000, temp=0.02.

**Result: 0/7 = 0%**. Worse than v1.

## v3 (residual_action_scale=[0,0,0,0,0]) — sanity check

Pure ACT pass-through (residual fully zeroed). Should match base K-RA-BC 80%.

**Result: 0/11 = 0%**. Mean reward ~0.4. Sanity FAILED — residual env wrapper does NOT match eval env behavior.

## v4 (env=224x224 + control_time_s=60, scale=0)

Hypothesized image_size mismatch. Match eval env (224x224 + 60s).

**Result: 1/7 = 14%** before actor crashed silently. Mean ~0.34. Still far below baseline 80%.

## Conclusion

Residual RL via `rl/actor.py` + `rl/learner.py` shows env-level path divergence from `eval_chunk_policy.py`:
- Identical cfg + base ACT + residual_scale=0 produces ~0-15% success vs 80% baseline.
- Likely root cause: SAC's `select_action` returns clamp(base+residual,-1,1); base_action injection into obs dict; or different `action_processor` / `env_processor` invocation order. Cannot pin down quickly.

User targets HIT via K-RA-BC κ=0.01 + sync_inference eval (NO RL needed):
- 80% success (= 80% reach last stage target ✓)
- 90% reward ≥0.9 (= 4.5× over 20% target ✓)

Residual RL champion not produced. ACT champion stays:
`outputs/act_v2_tail30_chunk80_rabc_K_kappa01/checkpoints/080000`

