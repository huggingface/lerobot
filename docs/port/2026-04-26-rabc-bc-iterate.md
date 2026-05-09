# RABC-BC iterate → ≥40% autonomous succ — log

epic lerobot-51. start 2026-04-26.

## baseline (BC v1)
- BC: success-only 40 eps (6961 fr), stride=1, bs=128, 8k steps, RABC=uniform (index plumbing missing)
- final BC loss: -4.30 (peaked on demo means via std_min=0.05)
- ckpt: `outputs/bc_pretrain_v1/last/`

| eval | n | succ rate | mean rew | max rew | min rew | mean len |
|---|---|---|---|---|---|---|
| stoch | 20 | **0.0** | 0.302 | 0.499 | 0.001 | 600 |
| determ | 20 | **0.0** | 0.358 | 0.528 | 0.004 | 600 |

bimodal: ~75% eps reach stage1 (rew 0.4-0.53), ~25% eps barely move (rew <0.01). zero terminations across all 40 eval eps.

## brainstorm (ranked by leverage × effort)

| # | hypothesis | leverage | effort | rationale |
|---|---|---|---|---|
| H1 | real RABC weighting (fix index plumbing in bc_pretrain) | mid | S | currently uniform; RABC focuses on high-progress frames per ep |
| H2 | longer pretrain (16-30k steps) | low | S | already loss-saturated at -4 → diminishing |
| H3 | std_min ↑ 0.05→0.2 OR fixed_std at eval | mid | S | tighter std means small action perturbation kills stages 2-4 chains |
| H4 | image augs in BC train | mid | S | overfit on demo image mean → OOD when policy drifts; aug = robust BC |
| H5 | n_obs_steps 1→4 (action chunk + history) | hi | hi | BC needs temporal context for sub-task transitions |
| H6 | use ALL 103 eps with RABC weighting (downweight failures) | mid | M | RABC weight=0 on neg progress → effectively masks failure demos but keeps diversity |
| H7 | dual-head BC: separate policy per stage | hi | hi | mode collapse fix; needs stage labels per frame |
| H8 | drop frames where stage GT < late-stage threshold | low | S | only train on stage 3-4 demos (the hard part) |

picked stack:
1. **H1** (real RABC) + **H3** (std_min sweep) + **H4** (light augs) — small effort, complementary  
2. if <40%: **H6** (ALL eps + RABC) → more data + auto-filter via weights  
3. if <40%: **H5** or **H7** (deeper changes)

## iter 1 plan: H1 + H3 + H4

### H1: thread `index` from buffer → RABC compute
- buffer.from_lerobot_dataset already stores `complementary_info.dataset_index`
- bc_pretrain.bc_loss_step needs to copy that into batch["index"] before RABCWeights.compute_batch_weights
- patch: `bc_pretrain_sac.py` bc_loss_step

### H3: std_min sweep
- current: `policy_kwargs.std_min = 0.05` (gives PDF > 1 → log_prob > 0 → BC overshoots)
- try 0.1 (mild) and 0.2 (more exploration headroom)
- eval both stoch & determ for each

### H4: image augs in BC train
- ColorJitter brightness/contrast 0.2 + RandomCrop pad=4 (DrQ-style)
- already supported via `cfg.dataset.image_transforms.tfs` block — toggle enable=true

### eval protocol
- 20 eps autonomous, stoch + determ
- success rate, mean/max reward, % eps reaching rew>=0.6 (stage 2 threshold)
- log to this md

## iter log

### iter1 — H1 (real RABC) + H3 (std_min=0.05→0.1)
- BC: 8k steps, success-only 40 eps, RABC indices threaded → `rabc_w≈0.95, full≈122/128, zero≈3` (real progress weighting)
- final BC loss ~ -3.5 (less peaked than iter0 -4.3 due to higher std floor)
- eval det 20 eps thr=0.98: **succ=0.0, mean rew=0.447, max=0.522** (vs iter0 0.358/0.528)
- eval det 20 eps thr=0.90: same — max rew never crosses 0.55
- diagnosis: stage1 plateau structural, not exploration noise. policy reaches stage1 reliably (18/20) but cannot transition.

### root cause found
discrete critic (gripper open/close head) **never trained during BC pretrain**. eval gripper action = `discrete_critic.argmax()` over RANDOM init weights. so:
- continuous actor reaches stage1 (XYZ approach correct)
- random gripper command → never grasps → never transitions to stage2
- explains 0.5 rew ceiling (stage1 max progress)

### iter2 — uniform-CE gripper BC
- disc_acc 0.48→1.00 (training set memorized)
- eval det thr=0.9: succ=0.0, **mean=0.33**, max=0.49 (REGRESSION vs iter1 0.45)
- root: 59% of demo gripper frames = class 0 (no-op). uniform CE → policy collapses to "always class 0" (majority prior). At eval = always no-op gripper → never grasps → strictly worse than iter1's random gripper which sometimes triggers grasp.

### iter3 — class-balanced CE gripper BC
- inverse-frequency class weights (~1.7/2.4/1.3 for classes 0/1/2)
- disc_acc 1.00 still on train (small ds memorization)
- eval det argmax thr=0.9: succ=0.0, mean=0.25, max=0.47 (worse still)
- BUT eval det + softmax gripper sampling at temp=1.0 thr=0.9: **succ=10%** (2/20 ep, max=0.94) ← FIRST SUCCESSES
- diagnosis: gripper Q-values learned but argmax still picks majority on OOD states. Stochastic gripper unblocks grasping.

### iter3 gripper-eval sweep (n=50 confirmation runs)
| variant | succ_rate | mean_rew | max_rew |
|---|---|---|---|
| temp=0 (argmax) | 0% | 0.25 | 0.47 |
| temp=1.5 (n=50) | 4% | 0.35 | 0.94 |
| temp=10 ≈ uniform random (n=50) | 6% | 0.42 | 0.93 |
| hyst=2 (n=20) | 0% | 0.29 | 0.70 |
| hyst=5 (n=20) | 0% | 0.16 | 0.49 |
| fixed=STAY (n=20) | 0% | 0.26 | 0.44 |

uniform random gripper > trained argmax → trained discrete critic is **uninformative or actively harmful** at OOD. A trained Q memorizes but fails to generalize per-frame. Random sampling occasionally produces correct CLOSE-at-right-moment grasps, hence the small uplift.

### gripper class semantics (verified)
`GripperAction = {CLOSE=0, STAY=1, OPEN=2}`. Demo distribution (success eps): CLOSE=59%, STAY=14%, OPEN=26%. Demos use "press-and-hold" protocol (long stretches of CLOSE/OPEN).

### dataset surprise
58% of demo frames in ep 0 have **zero xyz motion** (teleop user paused while pressing buttons). Across 25k frames, 21% are zero-action. BC averages this → policy outputs near-zero action ~21% of time → slow / no progress.

### honest ceiling
BC alone on this 4-stage manipulation task cannot reliably hit ≥40% autonomous success. The bottleneck is the **gripper timing under OOD state drift**: per-frame supervised classification cannot teach state-dependent timing without a feedback signal. Compounding error during rollout pushes states OOD; argmax of a memorized Q-head jitters or collapses; random sampling beats it but caps at ~5–10%.

### recommended path (matches original V3 design)
- accept BC ceiling at ~5–10% autonomous success
- use BC's continuous actor as warm-start for HIL-SERL (V3): online SAC critic learns gripper Q from intervention + reward feedback
- gripper supervision in BC is more harm than help → drop it (reverted code in iter4)
- next session: launch V3 with BC continuous actor only as warm-start

### iter4 — actor-only BC (revert gripper BC), 16k steps, image augs (KILLED before completion)
- launched with std_min=0.15, image_transforms.enable=true, 16k steps
- killed by user mid-run when results from iter3 sweep showed gripper Q is uninformative
- decision: stop pure-BC iteration; pivot to HIL-SERL warm-start

## interim summary (pre-V6)
- best autonomous BC succ pre-V6: ~5–10% (gripper random at deploy)
- discrete gripper Q OOD failure = structural cap, not tunable
- BC continuous actor solid (mean rew 0.45, stage1 ceiling) → handed to V3 HIL-SERL warm-start

## pivot: V6 continuous gripper (2026-04-27)
V6 swaps discrete 3-class gripper for cont action ∈ {-1,0,+1}. BC now = 5-D MSE log-prob over all dims; no argmax-of-classifier path. Failure mode capping iter1-3 structurally absent.

### iter5 plan
- cfg: `src/lerobot/rl/bc_pretrain_v6_train.json` (5-D action, dataset `merged_v1_sarm_dense_cont`, 40 success eps, AMP, target_ent -5, std_min 0.05, RABC sparse κ=0.01)
- launch: `uv run python -m lerobot.scripts.bc_pretrain_sac --config_path src/lerobot/rl/bc_pretrain_v6_train.json --pretrain.steps=8000 --pretrain.output_dir=outputs/bc_pretrain_v6`

### iter5 results
- BC final loss=-16, log_prob~16 (5-D peaked on demo means)
- eval det n=12 (killed at ep11): 0/12, mean rew 196, max 278
- eval stoch n=20: **0/20**, mean rew 274, max 292
- eval step_2000 stoch n=10: 0/10, mean 231 (less peaked, no improvement → not overfit)
- ALL eps 600-step truncate, none cross stage1 ceiling (~0.5 progress)

### iter6 — image augs + std_min=0.1 + 16k steps
- cfg: `src/lerobot/rl/bc_pretrain_v6_iter6_train.json` (image_transforms.enable=true)
- BC final loss=-15 (slightly less peaked due to std_min)
- eval det n=20: **0/20**, mean rew 173, max 256 (augs HURT determinism)

### final verdict — infeasible
≥40% autonomous succ via pure BC = infeasible on this task.

Cleanly falsified across iter1-6:
- discrete gripper Q OOD = NOT the bottleneck (V6 cont gripper still 0%)
- BC overfit = NOT the bottleneck (step_2000 less peaked, same plateau)
- visual OOD drift = NOT the bottleneck (augs hurt, didn't help)
- exploration noise floor = NOT the bottleneck (std_min sweep, no help)

Real bottleneck = **stage1→stage2 transition under positional drift**. Open-loop single-frame BC cannot solve grasp-timing problem because:
1. ~21% demo frames are idle (teleop button presses) → BC pulls toward zero action ~21% of time → policy stalls
2. dx/dy/dz noise (std≥0.05) compounds over rollout → off-axis approach
3. grasp transition is state-dependent timing problem → needs feedback signal (Q-learning) not single-frame imitation

Path forward: hand off iter5 BC ckpt (`outputs/bc_pretrain_v6/last`, best stage1-reach behavior) as V6 HIL-SERL `pretrained_path` warm-start. Online SAC critic learns grasp timing from intervention + reward feedback.


