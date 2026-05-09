# Reboot handoff (2026-05-09 21:20)

## Remote DL_A6000 — runs persist via nohup

```
G0 PID 93080  sarm_clipft_a_baseline   bs=8  6k steps  CLIP-FT freeze=False  ~70min ETA
G1 PID 62461  sarm_frozen_balanced_v3  bs=32 36k steps stage_balanced=True   ~2.5h ETA
G2 PID 62462  sarm_frozen_sw20         bs=32 24k steps sw=20                 ~1.9h ETA
G3 PID 62463  sarm_frozen_balanced_sw15 bs=32 24k stage_balanced + sw=15     ~1.8h ETA
```

Logs: `/tmp/clipft_a.log /tmp/frozen_balanced_v3.log /tmp/frozen_sw20.log /tmp/frozen_balanced_sw15.log`

## In-flight evals on G2/G3 (PID 112601, 112602)

`sarm_frozen_sw20/006000` + `sarm_frozen_balanced_sw15/006000` on val_fs. ~90% done at ep 141/158. ETA <5min.

Output dirs:
```
outputs/sarm_gate_eval_frozen_sw20_006000/metrics.json
outputs/sarm_gate_eval_frozen_balanced_sw15_006000/metrics.json
```

## Resume commands after reboot

### Check all 4 trainings + evals
```
ssh DL_A6000 'pgrep -af lerobot.scripts.lerobot_train; pgrep -af eval_sarm_sim_assemble; echo ---; for d in sarm_clipft_a_baseline sarm_frozen_balanced_v3 sarm_frozen_sw20 sarm_frozen_balanced_sw15; do echo "$d: $(ls ~/github.com/orel/lerobot/lerobot/outputs/$d/checkpoints/ 2>/dev/null | grep -v last | tr "\n" ",")"; done'
```

### Dump leaderboard
```
ssh DL_A6000 "python3 << 'PYEOF'
import json,os
b='/home/dom_iva/github.com/orel/lerobot/lerobot/outputs'
def fmt(s,n):
    g=s['gates']; p=sum(1 for k in g if g[k].get('pass'))
    return '%-32s %d/10 sT=%.2f sT5=%.2f mid=%.2f mono=%.3f last=%.2f stNE=%.3f stNB=%.3f' % (n,p,g['succ_term_rate']['value'],g['succ_term_max5_rate']['value'],g['mean_mid']['value'],g['monotonicity']['value'],g['last_stage_max_prog_rate']['value'],g['stage_not_exceed_rate']['value'],g['stage_not_below_rate']['value'])
print(fmt(json.load(open(b+'/sarm_gate_eval_paperfull_lowlr/metrics.json'))['summary'],'PROD paperfull_lowlr_24k'))
for d in sorted(os.listdir(b)):
    if d.startswith('sarm_gate_eval_') and ('clipft' in d or 'frozen_' in d):
        p=os.path.join(b,d,'metrics.json')
        if os.path.exists(p):
            print(fmt(json.load(open(p))['summary'],d.replace('sarm_gate_eval_','')))
PYEOF"
```

### Eval next ckpts (run for each new save)
Pattern (replace NAME + STEP):
```
ssh DL_A6000 'cd ~/github.com/orel/lerobot/lerobot && CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python -m lerobot_policy_sarm.eval_sarm_sim_assemble \
  --dataset domrachev03/sim_3stage_v2_val_fs \
  --pretrained outputs/NAME/checkpoints/STEP/pretrained_model \
  --task "Three-stage assembly" \
  --stats local/sim_3stage_v2_full_v2_nostale \
  --image-key observation.images.wrist \
  --type sarm_ext --head-mode sparse \
  --out outputs/sarm_gate_eval_NAME_STEP --label NAME_STEP'
```

### Attention sanity
```
CUDA_VISIBLE_DEVICES=0 uv run python scripts_local/sarm_attention_metric.py \
  --ckpt outputs/NAME/checkpoints/STEP/pretrained_model \
  --dataset-root ~/.cache/huggingface/lerobot/local/sim_3stage_v2_full_v2_nostale \
  --n-frames 32 --n-eps 4 --task "Three-stage assembly"
```

## Disk

Remote ~40G free. Each ckpt ~2.5GB (frozen) or ~5GB (CLIP-FT). 4 runs may approach disk limit by completion. Prune after eval if needed.

## Beads

Epic `lerobot-133` open. T1/T2/T3/T4 closed. T5 in_progress.

## Targets

- Surpass PROD `paperfull_lowlr/024000` = 5/10 gates, sanity_score=0.562
- Match/exceed legacy 2-stage SARM (4 gates pass: lin/mid/mono/stNB, but model collapsed succ_term=0)

## Production wired teleop

`src/lerobot/rl/sim_3stage_sarm_teleop_env.json` → `paperfull_lowlr_40k/032000` (4/10, sw=20 mode). Currently NOT prod 24k. Re-wire to 24k if needed for baseline test:
```
"pretrained_path": "outputs/sim_3stage_sarm_v2_full_v2_paperfull_lowlr/checkpoints/024000/pretrained_model",
"stats_dataset_repo_id": "local/sim_3stage_v2_full_v2_nostale"
```
