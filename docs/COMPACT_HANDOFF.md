# Project handoff (compact-ready) — SARM 6-stage v2

**Read first.** Authoritative state.

## Champion artifacts (2026-05-11)

```
SARM:        outputs/sarm_shorthor_k_n4g2_sw20/checkpoints/002500/pretrained_model
             10/11 gates  mean_max=0.961  succ=0.83  lag=0.30s

ACT RA-BC:   outputs/act_v2_tail30_chunk80_rabc_K_kappa01/checkpoints/080000/pretrained_model
             80% success  mean_rew=0.82  max=0.96  mean_len=270  (K SARM sync_inference eval)

Eval cfg:    src/lerobot/rl/sim_3stage_act_eval_env_K_sync.json   (sync_inference=true)

Progress:    ~/.cache/huggingface/lerobot/local/sim_3stage_v2_full_v2_succonly_destale_tail30/sarm_progress.parquet
             (K SARM, 205 eps × 42094 frames, stride=5)
```

## How to launch (recipes)

### 1. Train SARM (shorthor variant)

```bash
ssh -p 8003 dom_iva@143.248.121.169
cd ~/github.com/orel/lerobot/lerobot
df -h ~ | tail -1   # CHECK DISK FIRST. CLIP-FT 2cam ~9GB/run. Need >= 9N+5GB.
ps -eo pid,etime,pcpu,cmd --sort=-pcpu | grep -aE "python|uv " | head   # CHECK STALE PROCS
CUDA_VISIBLE_DEVICES=N nohup ~/.local/bin/uv run python -m \
  lerobot.scripts.lerobot_train \
  --config_path=src/lerobot/rl/sarm_shorthor_k_n4g2_sw20_train.json \
  > /tmp/<name>.log 2>&1 &
```

K's recipe (n_obs=4, gap=2, sw=20, freeze_clip=false, clip_lr=5e-7, paperfull base, 2.5k steps).

### 2. Generate RA-BC progress parquet

```bash
# /tmp/compute_rabc_weights_ext.py is the 2cam ext-aware patched script.
ssh -p 8003 dom_iva@143.248.121.169 \
  'cd ~/github.com/orel/lerobot/lerobot && \
   CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python /tmp/compute_rabc_weights_ext.py \
     --dataset-repo-id local/sim_3stage_v2_full_v2_succonly_destale_tail30 \
     --reward-model-path outputs/sarm_shorthor_k_n4g2_sw20/checkpoints/002500/pretrained_model \
     --stride 5'
# Writes ~/.cache/.../sarm_progress.parquet (~500KB, ~10min @ stride=5).
# Crash on HF push at end is harmless — parquet already saved.
```

### 3. Train ACT RA-BC

```bash
CUDA_VISIBLE_DEVICES=N nohup ~/.local/bin/uv run python -m \
  lerobot.scripts.lerobot_train \
  --config_path=src/lerobot/rl/act_v2_tail30_chunk80_rabc_K_kappa01_train.json \
  > /tmp/act_rabc_K_kappa01.log 2>&1 &
# 80k steps ~1h on A6000. Saves at 20k/40k/60k/80k.
```

### 4. Evaluate ACT (10 eps)

```bash
DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 uv run python -m lerobot.scripts.eval_chunk_policy \
  --config_path=src/lerobot/rl/sim_3stage_act_eval_env_K_sync.json \
  --pretrained=outputs/act_v2_tail30_chunk80_rabc_K_kappa01/checkpoints/080000/pretrained_model \
  --task "Three-stage assembly" \
  --policy-type=act \
  --n-episodes=10
```

**Use sync env config** (`_K_sync.json` not `_K_async.json`) — sync_inference=true gives 2× ACT success rate (in-distribution SARM input).

### 5. Eval SARM 11-gate

```bash
ssh -p 8003 dom_iva@143.248.121.169 \
  'cd ~/github.com/orel/lerobot/lerobot && \
   CUDA_VISIBLE_DEVICES=N ~/.local/bin/uv run python -m \
     lerobot_policy_sarm.eval_sarm_sim_assemble \
     --dataset domrachev03/sim_3stage_v2_val_fs \
     --pretrained outputs/sarm_shorthor_k_n4g2_sw20/checkpoints/002500/pretrained_model \
     --task "Three-stage assembly" \
     --stats local/sim_3stage_v2_full_v2_nostale \
     --image-key observation.images.wrist \
     --type sarm_ext --head-mode sparse \
     --out outputs/sarm_gate_eval_<label> --label <label>'
```

## SARM iteration leaderboard (24 ckpts evaluated)

```
ckpt              gates  mean_max  succ   lag    recipe
K_2.5k            10/11  0.961    0.83   0.30s  n=4 g=2 sw=20  ← CHAMPION
E_2.5k             9/11  0.958    0.84   0.30s  n=4 g=2 sw=15
N_2.5k             9/11  0.931    0.00   0.20s  n=3 g=2 sw=20  (lin best)
L_2k               9/11  0.936    0.00   0.40s  n=5 g=2 sw=15
A_2.5k             9/11  0.935    0.00   0.30s  n=4 g=2 sw=10
PROD paperfull/24k 5/11  0.953    0.44   1.75s  n=8 g=5 sw=10
```

Sw scaling: 10→15 cured succ_term gates (0→0.84). 15→20 cured zero_max (10→11/11).
Lag floor 0.30s. Tighter (gap=1 or n_obs=3) loses terminal grasp.

## ACT eval leaderboard (10 eps each, K SARM sync mode)

```
ckpt              mode    succ%  mean_rew  max    len    notes
K RA-BC κ=0.01    sync    80%    0.82     0.96   270    🏆 CHAMPION
K RA-BC κ=0.05    async   60%    0.68     0.96   519
K RA-BC κ=0.01    async   50%    0.62     0.96   681
PROD BC chunk80   sync    40%    0.58     0.95   751
BC chunk40        sync     0%    0.45     0.80   1200   chunk40 worse
PROD BC chunk80   async   10%    0.47     0.95   1098
PROD RA-BC (old)  async    0%    0.27     0.29   1200   PROD SARM was bad
K RA-BC κ=0.05    sync     0%    0.28     0.31   1200   κ=0.05+sync unstable
```

## Infra reference

### Remote DL_A6000

```
host: irislab.asuscomm.com:8003 (DNS flaky)
fallback: ssh -p 8003 -o StrictHostKeyChecking=no dom_iva@143.248.121.169
ssh alias: DL_A6000
4× A6000 (G0-G3), uv at ~/.local/bin/uv
disk: 1.8TB, ~58G free post-cleanup; PRUNE OFTEN
code sync: rsync (no git remote on remote)

# Quick state check
ssh -p 8003 dom_iva@143.248.121.169 'df -h ~ | tail -1; \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; \
  pgrep -af "python|uv " | grep -v grep | head'
```

### Local

```
RTX 4070 Ti Super (G0, 16GB)
DISPLAY=:1 (mujoco rendering works; remote DOES NOT have display)
ACT eval MUST run local (sim_3stage_act_eval_env_K_sync.json + DISPLAY)
```

## Datasets (HF cache)

```
local/sim_3stage_v2_full_v2_nostale          307 eps  (SARM K train)
domrachev03/sim_3stage_v2_val_fs              158 eps  (SARM eval)
local/sim_3stage_v2_full_v2_succonly_destale_tail30  205 eps  (RA-BC train, ACT)
local/sim_3stage_v2_honest_val                30 full eps
```

## Beads epic state

```
lerobot-146 SARM 9/10 sync gates @ lag <0.4s        DONE (10/11 hit + targets exceeded)
lerobot-147 train 4 short-horizon variants          CLOSED
lerobot-148 sync val_fs eval all ckpts              CLOSED
lerobot-149 pick winner + wire teleop               in_progress
lerobot-XXX K-RA-BC kappa=0.01 sync hits 80%        OPEN (logged)
```

## Memory pointers

- `reference_local_weights_inventory.md` — every kept ckpt path
- `reference_dl_a6000_server.md` — DNS, ssh, disk
- `feedback_check_stale_procs_first.md` — `ps -eo etime,pcpu` first when slow
- `feedback_no_eval_during_training.md` — eval+train co-run = 6× slowdown
- `feedback_check_disk_before_train.md` — `df -h ~` first; 9GB/run; corrupt safetensors on disk-full
- `feedback_act_rabc_sync_inference.md` — sync_inference=true → 2× ACT success
- `feedback_keep_iterating.md` — don't stop on plateau
- `feedback_no_kill_in_progress.md` — never pkill without explicit ask

## Caveman ultra mode

Active. Drop articles/filler/hedging; abbreviate aggressively. Code/commits/security: write normal.
