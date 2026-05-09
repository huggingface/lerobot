# Project state handoff — SARM/ACT 6-stage v2

Beads: epic `lerobot-123`. mds: `outputs/sarm_iterations.md`, `outputs/act_iterations.md`.

## Goal

IL-only (no HIL-SERL/residual): SARM passes 10 gates, ACT 95% threshold + ≥10% succ@CNN-cls / ≥80% reach last stage.

## Production winners (local paths, also on remote DL_A6000)

### SARM
- **Best (5/10 gates)**: `outputs/sim_3stage_sarm_v2_full_v2_paperfull_lowlr/checkpoints/024000/pretrained_model`
- recipe: paper arch (`epstart_anchor=true, use_causal_mask=false, mlp_heads=true`) + `peak_lr=2e-5` + sw=10 + 2cam + 24k steps + dataset `local/sim_3stage_v2_full_v2_nostale` (307 eps mixed, 205 succ + 102 partial)
- gates pass: lin_mad=0.24 ✓, mean_mid=0.45 ✓, fail_term=0 ✓, zero_max=0 ✓✓, stage_ne=0.97 ✓
- gates fail: succ_term=0.44, mono=0.73, last_stage=0.87, plateau=0.29, stage_nb=0.39
- alt: `paperfull_24k` (4/10, succ_term=0.85 better but zero_max=0.11)

### ACT (best-by-mean_reward, all 0/20 CNN succ)
- **Best**: `outputs/act_v2_full_v2_tail30_chunk80_lowlr_rabc/checkpoints/080000/pretrained_model` (mean=0.242, max=0.78, cnn_max=0.31)
- recipe: chunk=80, n_obs=1, ResNet18, kl=10, RABC kappa=0.01, source = paperfull_lowlr-relabeled tail-30-destale dataset
- chunk=80 = sweet spot (cnn_max=0.37 BC, 0.31 RABC). Larger chunks ↑ mean_rew, ↓ cnn signal.

### CNN binary success classifier
- `outputs/cnn_v2_front_v1/best.pt` (ResNet18 + `net.` prefix)
- trained on `local/sim_3stage_v2_full_v2_nostale` front cam
- v2 holdout: 100% recall last5_max>0.5, 0% FP first5_max>0.5
- recommended threshold: P(succ) ≥ 0.5

## Datasets

| repo_id | eps | use |
|---|---|---|
| `domrachev03/sim_3stage_v2_train_fs` | 158 | base recordings (raw) |
| `domrachev03/sim_3stage_v2_extra` | 122 (drop 4,84) | newer recordings, 224x224 |
| `domrachev03/sim_3stage_v2_extra_partial` | 29 | partial-fail demos |
| `local/sim_3stage_v2_full_v2_nostale` | 307 | SARM train (mixed, ds 128x128) |
| `local/sim_3stage_v2_full_v2_succonly_nostale` | 205 | success-only filtered |
| `local/sim_3stage_v2_full_v2_succonly_destale_tail30` | 205 | tail-30 destale ACT base |
| `local/sim_3stage_v2_full_v2_succonly_paperfull_lowlr_delta` | 205 | RA-BC reward-relabeled |
| `domrachev03/sim_3stage_v2_val_fs` | 158 (100 full + bucket 0/6..5/6) | SARM 10-gate eval |

## Eval pipeline

### SARM 10-gate
```
uv run python -m lerobot_policy_sarm.eval_sarm_sim_assemble \
  --dataset domrachev03/sim_3stage_v2_val_fs \
  --pretrained <ckpt> \
  --task "Three-stage assembly" \
  --stats <ckpt-train-stats> \
  --image-key observation.images.wrist \
  --type sarm_ext --head-mode sparse \
  --out outputs/sarm_gate_eval_<name> --label <name>
```

Gates (fully implemented, including `succ_term_max5_rate` from last5 frames):
- succ_term_rate ≥ 0.95, succ_term_max5_rate ≥ 0.95
- lin_mad ≤ 0.25
- mean_mid ≥ 0.25
- monotonicity ≥ 0.85
- last_stage_max_prog_rate ≥ 1.0
- fail_term_rate = 0
- zero_max_ge_0.5 = 0 (priority for non-progress eps)
- plateau_ok_rate ≥ 0.8
- stage_not_exceed_rate ≥ 0.9 (#1 priority)
- stage_not_below_rate ≥ 0.7 (#3 priority)

### ACT eval (gym_manipulator + SARM-reward + CNN cls)
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python -m lerobot.scripts.eval_chunk_policy \
  --config_path=src/lerobot/rl/sim_3stage_act_succonly_sarmrew_eval_env.json \
  --pretrained=<act-ckpt> \
  --n-episodes=20 --policy-type=act --task="Three-stage assembly" \
  --video-dir=outputs/<rollouts> \
  --cnn-ckpt=outputs/cnn_v2_front_v1/best.pt --cnn-thr=0.5
```
- eval env JSON points to **paperfull_24k** SARM as reward (see `src/lerobot/rl/sim_3stage_act_succonly_sarmrew_eval_env.json`)
- patched eval_chunk_policy.py adds `max_cum`, `cnn_succ`, `cnn_max_prob` per ep
- success_thr at sim cfg = 0.5 (NOT used for cls — uses CNN as honest oracle)

## Teleop env (current target)

`src/lerobot/rl/sim_3stage_sarm_teleop_env.json` — wired to **succonly_paperfull** (last user toggled). Run:
```
DISPLAY=:0 uv run python -m lerobot.rl.gym_manipulator \
  --config_path=src/lerobot/rl/sim_3stage_sarm_teleop_env.json
```

## Key code mods

- `lerobot_policy_sarm/.../configuration_sarm.py`: `epstart_anchor`, `mlp_heads`, `use_causal_mask`, `progress_loss_weight`, `stage_class_weights`, `peak_lr`
- `lerobot_policy_sarm/.../eval_sarm_sim_assemble.py`: added `terminal_max5` metric + `succ_term_max5_rate` gate
- `lerobot/.../policies/act/modeling_act.py`: `forward(batch, reduction="none")` for per-sample loss → RA-BC weighting
- `lerobot/.../scripts/lerobot_train.py`: wires `rabc_weights_provider` → per-sample-weighted loss
- `lerobot/.../scripts/eval_chunk_policy.py`: `--cnn-ckpt`, `--cnn-thr`, per-ep `cnn_max_prob`, `cnn_success`
- `simulator_for_IL_RL/simulator_for_il_rl/env.py`: `Renderer(model, height=128, width=128)` (was 224, broke v2 visual match)
- `scripts_local/`: `combine_datasets.py`, `resize_videos.py`, `filter_stale_state_frames.py`, `discretize_gripper.py`, `train_cnn_success_classifier.py`, `cnn_eval_v2.py`, `subset_episodes.py`
- `src/lerobot/scripts/destale_actions.py`: `--last-frac 0.3` for tail-30 destale

## SARM iteration history (24+ variants on v2_full_v2_nostale)

Best by zero_max (priority gate):
- paperfull (M2+M3+M4 sw=10, 14k): zero_max=0.000 ✓ — first to hit
- paperfull_lowlr (peak_lr=2e-5, 24k): zero_max=0.000, **5/10 gates** ← prod
- paperfull seed=42: high seed variance (3/10 vs 4/10 seed=1000)

Levers tested (didn't beat paperfull_lowlr):
- sw ∈ {3,5,10,20}: sw=10 optimal (+ paperfull arch)
- plw ∈ {0.3,1,1.5,2,3,5}: non-monotonic, plw=1 default best with paperfull
- frame_gap ∈ {3,5,10}: 5 optimal
- n_obs ∈ {8,12,16}: 8 optimal
- max_rewind ∈ {3,5}: 3 optimal
- batch ∈ {16,32}: 32 optimal w/ paperfull
- steps: 14k/24k/40k → 24k optimal for paperfull, lowlr stable longer
- lr: 5e-5 default → 2e-5 (lowlr) better
- iter4 recipe (5k steps, sw=3, wrist-only, batch=16, no paper-arch): 2/10 → undertrained for 6-stage
- succplusfew (succ + few partials): 3/10 (zero_max=0.78)
- succonly: 2/10 (zero_max=1.0 — never sees no-progress)
- raw (no nostale): 3/10
- inverse_freq stage weights: no help
- bigger CLIP backbone B/16, L/14: worse on v3 era
- drop_n_last_frames=0: no help

## ACT iteration history

Phase 1 baseline = chunk20 RABC w/ succonly_sw3 SARM. SARM hallucinated → mean_reward=0.62 misleading.

After CNN added as honest oracle: all variants 0/20 CNN succ.

Best by mean_reward + cnn_max_prob:
- chunk=80 BC tail30: cnn_max=0.368 (peak)
- chunk=80 + lowlr RABC: mean_rew=0.242 (peak)
- chunk=160 BC tail30: mean_rew=0.233

Levers tested:
- chunk ∈ {10,20,40,80,120,160}: chunk=80 optimal for CNN
- ResNet18 vs ResNet50: RN50 doesn't help when combined w/ chunk=40 (combo dropped CNN signal)
- RA-BC kappa ∈ {0.005,0.01,0.05}: 0.01 default best
- BC vs RA-BC: similar, RA-BC slightly better w/ best SARM
- RABC source SARM: paperfull, paperfull_24k, paperfull_lowlr — no major diff
- tail-30 destale: helps mean_reward 5x but doesn't break CNN ceiling

## Lessons

- **paperfull (M2+M3+M4 paper arch) breakthrough**: zero_max=0 first time. Critical lever.
- **lowlr**: paperfull's seed-variance problem solved by peak_lr=2e-5. 4/10→5/10.
- **paperfull SARM hallucinates on ACT rollouts**: passes recorded-demo gates but rates ACT visual gripper-open as completion. CNN is orthogonal honest oracle.
- **chunk size matters more than RABC weighting**: chunk=80 BC alone beats chunk=20 RABC w/ paperfull SARM.
- **2-stage 40% champion (epic-52)**: used 15-D state w/ EE pose + tail-30 destale + chunk=10. Untested on 6-stage: state augmentation w/ FK is the only big remaining lever.
- **ACT plateau hypothesis**: 6-stage genuinely harder + state lacks EE pose precision needed for cover placement. CNN ceiling ~0.37.

## Open paths

1. **15-D state w/ FK-augmented EE pose** (untested, biggest pending lever)
2. record more demos (current ≤ 200 succ may be insufficient)
3. CNN-as-reward for RABC (user dropped this option)
4. stage-conditioned ACT (user dropped)

## Resource notes

- `DL_A6000` ssh alias = irislab.asuscomm.com:8003 (DNS flaky); fallback IP 143.248.121.169:8003 (need `-o StrictHostKeyChecking=no`)
- Disk on remote: 1.8TB used, ~30G free typical, fills mid-train. Cleanup ckpts often.
- Recording requires `DISPLAY=:1` (X session); EGL doesn't work for teleop

## Recent active runs

paperfull_lowlr_40k done (4/10 best ckpt — no improvement vs 24k). Production stays paperfull_lowlr 24k.

succonly_paperfull retrain killed (per user request). Local `outputs/sim_3stage_sarm_v2_full_v2_succonly_paperfull/checkpoints/014000` exists from earlier — wired to teleop now.
