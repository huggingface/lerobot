# HIL-SERL convergence via SARM-RABC — hypotheses

date 2026-04-26. epic = new (post lerobot-46 stall). prod sarm = iter5_2cam + merged_v1.

## context

prior runs ciaktf45 (72k steps, 25k env, 48% intrv) + bzp2mtuc (20k steps, 7k env, 81% intrv).
both: ep_reward plateau ~0.4–0.5 (=stage1 ceiling), no stage2 entry w/o intrv. policy fails generalize.

cfg from wandb: temp_init=0.01, target_ent=null(=-2.5), discount=0.97, n_obs=1, utd=2, bs=256, rabc OFF, demo ds = manual_filtered_sarm_delta (47ep / 7953fr offline buf).

## diagnosis (root causes)

| # | symptom | metric | root |
|---|---|---|---|
| D1 | entropy collapse | temperature → 0.003 in 5k opt steps | temp_init too low + target_entropy too lax |
| D2 | no behavior prior | random init policy | no BC pretrain, no BC loss term |
| D3 | small offline buf | 7953 fr | only one src (manual_filtered), merged_v1 has 25k |
| D4 | stage1 plateau | ep_reward 0.4 | actor near-deterministic at stage1 idle |
| D5 | sparse reward | succ_reward=1, delta-shaped only | discount 0.97 over-bootstraps Q on noisy delta |

## hypotheses (ranked by leverage × effort)

| H | name | leverage | effort | risk |
|---|---|---|---|---|
| H1 | BC pretrain SAC actor w/ RABC weights | 🟢 high | M (script) | low |
| H2 | RABC-weighted BC loss term in SAC actor (online) | 🟢 high | M (loss edit) | low |
| H3 | hparam fix: temp_init=1, target_ent=-|A|, std_min=0.05 | 🟡 mid | trivial | low |
| H4 | discount 0.97→0.95, utd 2→4 | 🟡 mid | trivial | mid (overfit Q) |
| H5 | merged_v1_sarm_delta (25k fr) instead of manual_filtered (8k fr) as demo ds | 🟡 mid | trivial | low |
| H6 | n_obs_steps 1→4 | 🔴 low | hi (refactor) | hi (mem, Q backup) |

picked: **H1+H2+H3+H5 stack**. H4 single-knob sweep. H6 deferred.

### H1 — BC pretrain (RABC-weighted)

paper SARM eq8-9: w = clip((Δ-μ+2σ)/(4σ),0,1) for 0≤Δ≤κ, =1 for Δ>κ, =0 else. Δ = progress[t+chunk] - progress[t].

procedure:
1. run `compute_rabc_weights.py` on demo ds (`local/sim_assemble_manual_filtered_sarm_delta` AND `local/sim_assemble_sarm_merged_v1_sarm_delta`) → parquet
2. run `bc_pretrain_sac.py`: load SAC policy w/ image_keys=[front,wrist], actor BC loss `-mean(w_i * log_prob(a_i|o_i))`, freeze critic, ~5k steps bs=256
3. save ckpt → load via `policy.pretrained_path` in HIL-SERL cfg

expected: actor starts at demo-imitating stage1+stage2 trajectories. SAC critic boots from random; policy entropy not yet collapsed.

### H2 — BC loss term during online (anneal)

modify `compute_loss_actor` to accept (demo_obs, demo_actions, rabc_w) optional:
```
sac_loss = (α·logπ(a~|o) - Q(o,a~)).mean()
bc_loss = -(rabc_w · logπ(a_demo|o_demo)).mean()
total   = sac_loss + λ_bc · bc_loss
```
λ_bc anneal: 1.0 → 0.1 over `bc_anneal_steps`. forces policy stay close to demos early; relax late.

learner.py: pass offline batch's `(state, action, complementary_info[rabc_w])` as bc args when bc_loss_weight>0.

### H3 — hparam fix

| knob | before | after | why |
|---|---|---|---|
| temp_init | 0.5 (cfg) / 0.01 (wandb) | **1.0** | avoid early entropy crash |
| target_entropy | null=-2.5 | **-4** (=-|A|) | std SAC, more entropy headroom |
| policy_kwargs.std_min | 1e-5 | **0.05** | floor std → never near-determ |
| policy_kwargs.std_max | 5 | 5 | unchanged |

### H5 — bigger offline buf

swap `dataset.repo_id`:  
`local/sim_assemble_manual_filtered_sarm_delta` (47 ep, ~8k fr) →  
`local/sim_assemble_sarm_merged_v1_sarm_delta` (103 ep, ~25k fr).

reannotate first if not already done (delta mode, iter5 ckpt, merged_v1 stats).

## variants for mock test

each variant = base iter5_v2 cfg + diffs. bc weight applied via SAC config flags.

| var | name | H | diff vs base |
|---|---|---|---|
| V0 | base (sanity) | — | iter5_v2 unchanged |
| V1 | hparam fix | H3+H5 | temp_init=1, tgt_ent=-4, std_min=0.05, ds=merged_v1_sarm_delta |
| V2 | + BC loss | V1 + H2 | bc_loss_weight=0.5, bc_use_rabc=true |
| V3 | + BC pretrain | V2 + H1 | policy.pretrained_path=outputs/bc_pretrain_v1/last |
| V4 | + discount sweep | V3 + H4 | discount=0.95, utd=4 |

mock test = 100 opt steps, no teleop (`teleop=null`), no env stepping past `online_step_before_learning`. confirm: ckpt loads, BC loss path active, RABC weights propagate, no NaN.

## components to build

1. `lerobot/utils/rabc.py` — exists ✓ (RABCWeights class)
2. `lerobot/policies/sarm/compute_rabc_weights.py` — exists ✓
3. **NEW**: `lerobot/scripts/bc_pretrain_sac.py` — BC pretrain entrypoint
4. **NEW**: SACConfig fields `bc_loss_weight, bc_use_rabc, bc_rabc_progress_path, bc_rabc_chunk_size, bc_anneal_steps`
5. **MODIFY**: `compute_loss_actor` accept demo args + rabc weights
6. **MODIFY**: `learner.py` actor opt step build bc_inputs from offline batch
7. **NEW configs**: `sim_assembling_sarm_hilserl_rabc_v{1..4}_train.json`
8. **NEW commands**: `relabel sarm-delta on merged_v1`, `compute_rabc_weights on merged_v1_sarm_delta`, `bc_pretrain on V3 inputs`

## v7 SARM status

v7 ckpt complete (5000 steps, last + 1k–5k). eval blocked by draccus registration error w/ `sarm_ext` type when running `eval_sarm_sim_assemble.py` (decoder needs lerobot_policy_sarm imported). per user memory production=v5. **decision: stick v5**, defer v7 until eval harness fixed.

## acceptance for mock test

per variant, log:
- ckpt load ok (sarm v5 + sac actor)
- offline iter yields batch w/ image_keys + complementary_info
- actor fwd+bc fwd no NaN
- bc_loss > 0 (non-trivial)
- rabc_weight stats (zero/full count) sane
- 100 opt steps in <2 min (no perf regression)
- actor params updated (compare snapshots)

if all pass → ready for short HIL-SERL smoke (15-min) w/ teleop in next session.

## artifacts mapping

```
outputs/
├── sim_assemble_sarm_ext_iter5_2cam/...     # SARM v5 (prod)
├── bc_pretrain_v1/                          # NEW: pretrained SAC actor
├── sim_assembling_hilserl_rabc_v1.../        # NEW: V1 mock results
├── sim_assembling_hilserl_rabc_v2.../        # NEW: V2 mock results
├── ...
src/lerobot/rl/
├── sim_assembling_sarm_hilserl_rabc_v1_train.json  # NEW
├── sim_assembling_sarm_hilserl_rabc_v2_train.json  # NEW
├── sim_assembling_sarm_hilserl_rabc_v3_train.json  # NEW
└── sim_assembling_sarm_hilserl_rabc_v4_train.json  # NEW
src/lerobot/scripts/
└── bc_pretrain_sac.py                       # NEW
```

## next session

after mock validates: full 15-min HIL-SERL w/ teleop on each variant; pick winner; iterate.
