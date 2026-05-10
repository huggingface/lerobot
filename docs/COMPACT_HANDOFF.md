# Project handoff (compact-ready) — SARM 6-stage v2

**Read first.** Authoritative state.

## Current goal (epic lerobot-146)

**9/10 sync gates @ lag ≤0.4s (max ≤0.2s)**.

Sync gates evaluate SARM offline using REAL future frames in the obs window
(`observation_delta_indices = [-1M, 0, gap, 2*gap, ..., (n_obs-1)*gap]`).
Teleop online with `sync_inference=true` flag = real past frames as proxy →
output describes frame `t - max_future_delta`. Lag = max_future_delta * dt.

```
PROD paperfull:  n_obs=8, frame_gap=5  →  max_future = 35 frames @ 20fps = 1.75s lag
Target:                                                       ≤  8 frames =  0.40s
Max plan:                                                     ≤  4 frames =  0.20s
```

## Active iteration (T1: lerobot-147 in_progress)

4 short-horizon variants on remote DL_A6000, 12k steps each, 4 ckpts each
(save_freq=3000), ETA ~1h parallel. paperfull recipe otherwise (sw=10,
epstart_anchor=true, mlp_heads=true, use_causal_mask=false, peak_lr=2e-5,
batch=32, 2cam, frozen CLIP).

| variant | n_obs | gap | max_delta | lag | output_dir |
|---|---|---|---|---|---|
| A | 4 | 2 |  6 | 0.30s | `outputs/sarm_shorthor_a_n4g2` |
| B | 4 | 1 |  3 | 0.15s | `outputs/sarm_shorthor_b_n4g1` |
| C | 6 | 2 | 10 | 0.50s | `outputs/sarm_shorthor_c_n6g2` |
| D | 8 | 1 |  7 | 0.35s | `outputs/sarm_shorthor_d_n8g1` |

PIDs `1131807,1131809,1131811,1131813` on G0/G1/G2/G3. Logs `/tmp/shorthor_*.log`.

## Recent findings

### Sync vs async inference (the key insight)

- Paper-arch SARM trained with **bidirectional** windows (real future frames at
  offsets +5..+35 from anchor frame).
- Old async inference (default before May 10): replicates current frame for
  all positive deltas → window OOD vs training → noisy stage transitions.
- New `sync_inference: true` (added in `lerobot/processor/reward_model/sarm.py`):
  shifts deltas to non-positive (use real past frames). Window matches
  training distribution. Trade-off: 1.75s lag.
- Verified on user teleop run: per-stage SARM-vs-GT lag 28-40 steps ≈ design
  35 = 1.75s. All 6 stages tracked correctly. Mono.

### CLIP fine-tune outcomes

`clipft_a/b/e 2k = 9/10 sync gates` (vs PROD 5/10). All overfit past 2k. But
in async/teleop tests, clipft models DEGRADED — overfit to future-frame
context that doesn't exist online. PROD paperfull_lowlr/24k stays the
production teleop ckpt.

### Buffer prewarm (in `processor/reward_model/sarm.py`)

When sync_inference active and buffer empty (episode start), replicate
first observation to fill ring buffer maxlen → no "all-zero progress for
first 35 steps" artifact.

### Logging + analysis

- `SARMRewardConfig.log_jsonl_path` → per-step JSONL (progress, stage_idx,
  stage_conf, stage_probs, delta_indices, buffer_len, gt_stage_idx,
  gt_stage_started_this_frame, ts).
- `scripts_local/analyze_teleop_log.py` → per-episode plot, `detect_sync_lag`
  reads anchor-slot delta (slot 1 for epstart_anchor) for shift, plots raw
  + lag-shifted views.

## Infra reference

### Remote DL_A6000

```
host: irislab.asuscomm.com:8003 (DNS flaky)
fallback: ssh -p 8003 -o StrictHostKeyChecking=no dom_iva@143.248.121.169
ssh alias: DL_A6000
4× A6000 (G0-G3), uv-managed venv at ~/.local/bin/uv
disk: 1.8TB, ~58G free post-cleanup; PRUNE OFTEN
code sync: rsync (no git remote on remote machine)
```

### Train: launch + monitor

```bash
# Launch (template)
ssh DL_A6000 'cd ~/github.com/orel/lerobot/lerobot && \
  CUDA_VISIBLE_DEVICES=N nohup ~/.local/bin/uv run python -m \
    lerobot.scripts.lerobot_train \
    --config_path=src/lerobot/rl/<cfg>.json > /tmp/<name>.log 2>&1 & echo $!'

# Track progress (poll loop)
ssh DL_A6000 'tail -1 /tmp/<name>.log | tr "\r" "\n" | tail -2'

# Ckpt list
ssh DL_A6000 'ls ~/github.com/orel/lerobot/lerobot/outputs/<run>/checkpoints/'
```

Use `Monitor` tool with `until <ckpt-saved>; do sleep 60; done` for
event-driven notifications instead of polling.

### Eval: sync (default) val_fs

```bash
ssh DL_A6000 'cd ~/github.com/orel/lerobot/lerobot && \
  CUDA_VISIBLE_DEVICES=N ~/.local/bin/uv run python -m \
    lerobot_policy_sarm.eval_sarm_sim_assemble \
    --dataset domrachev03/sim_3stage_v2_val_fs \
    --pretrained outputs/<run>/checkpoints/<step>/pretrained_model \
    --task "Three-stage assembly" \
    --stats local/sim_3stage_v2_full_v2_nostale \
    --image-key observation.images.wrist \
    --type sarm_ext --head-mode sparse \
    --out outputs/sarm_gate_eval_<label> --label <label>'

# Gates: 11 total (sT, sT5, lin_mad, mid, mono, last, ft, zero_max,
# plateau, stNE, stNB). PROD = 5/10, clipft 2k = 9/10.

# Async eval (matches teleop if sync_inference=false): add --mode async
```

Leaderboard dump (run on remote):

```bash
ssh DL_A6000 'python3 /tmp/dump_all.py' | head -25
```

(`/tmp/dump_all.py` reads every `outputs/sarm_gate_eval_*/metrics.json`;
sorts by `gates_passed`.)

### Teleop test

```bash
DISPLAY=:0 uv run python -m lerobot.rl.gym_manipulator \
  --config_path=src/lerobot/rl/sim_3stage_sarm_teleop_env.json
```

Currently wired: `paperfull_lowlr/024000` + `sync_inference: true` +
`log_jsonl_path: outputs/teleop_logs/sarm_teleop.jsonl`.

User presses gamepad stage-advance to log GT. After episode:

```bash
uv run python scripts_local/analyze_teleop_log.py \
  outputs/teleop_logs/sarm_teleop.jsonl
```

Auto-detects sync lag, plots raw + shifted views.

## Storage paths

### Local (`/home/dom-iva/github.com/orel/lerobot/lerobot/outputs/`)

See `MEMORY.md → reference_local_weights_inventory.md` for full inventory.
Highlights:

```
sim_3stage_sarm_v2_full_v2_paperfull_lowlr/checkpoints/024000/  # PROD 5/10
sarm_clipft_a_baseline/checkpoints/002000/                       # 9/10 sync
act_v2_full_v2_tail30_chunk80/checkpoints/080000/                # ACT prod
cnn_v2_front_v1/best.pt                                          # success cls
```

### Remote (~58G free, kept minimal)

- `outputs/sim_3stage_sarm_v2_full_v2_paperfull_lowlr` (PROD baseline for retrain)
- `outputs/act_v2_full_v2_tail30_chunk80*` (ACT prod ×2)
- `outputs/sarm_gate_eval_*/` (~50 metric dirs, ~2GB)
- `outputs/sarm_shorthor_{a,b,c,d}*` (active iteration)

### Datasets (HF cache)

```
~/.cache/huggingface/lerobot/local/sim_3stage_v2_full_v2_nostale          # train, 307 eps
~/.cache/huggingface/lerobot/domrachev03/sim_3stage_v2_val_fs              # val, 158 eps
~/.cache/huggingface/lerobot/local/sim_3stage_v2_full_v2_succonly_nostale  # 205 succ
~/.cache/huggingface/lerobot/local/sim_3stage_v2_honest_val                # 30 full eps
```

## Beads epic open

```
lerobot-146 EPIC: SARM 9/10 sync gates @ lag <0.4s
  lerobot-147 T1: train 4 short-horizon variants    (in_progress)
  lerobot-148 T2: sync val_fs eval all ckpts        (open)
  lerobot-149 T3: pick winner + wire teleop         (open)
  lerobot-150 T4: brainstorm next iter              (open)
```

## Iteration cycle (autonomous)

1. **Wait** for ckpts to save (Monitor on output_dir/checkpoints/).
2. **Eval** each ckpt sync mode on val_fs.
3. **Dump leaderboard**, identify gate_pass max.
4. If best <9/10: brainstorm + deploy next variants (e.g. longer steps,
   different sw/plw, dataset variant).
5. If best ≥9/10: wire winner into teleop, verify lag matches design,
   close epic.
6. **Don't stop** until target met OR user explicitly halts.

## Caveman ultra mode

Active. Drop articles/filler/hedging; fragments OK; abbreviate aggressively.
Code/commits/security: write normal. User says `normal` to revert.

## Memory pointers

- `feedback_keep_iterating.md` — don't stop on plateau; brainstorm + deploy
- `feedback_dump_state_at_95pct.md` — write COMPACT_HANDOFF.md when ctx near full
- `reference_local_weights_inventory.md` — every kept ckpt path
- `reference_dl_a6000_server.md` — DNS, ssh, disk
- `feedback_no_kill_in_progress.md` — never pkill without explicit ask
- `feedback_eval_on_same_gpu.md` — pin CUDA_VISIBLE_DEVICES on freed GPU
- `feedback_check_config_alignment_before_model_fixes.md` — diff env/scene before model fixes

## Compact prompt

After compaction, run `bd ready` to refresh open task list. Re-read
`docs/COMPACT_HANDOFF.md` for state. Caveman ultra is on (`/caveman ultra`
re-arm if needed). User reserves local G0; remote is iteration target.
