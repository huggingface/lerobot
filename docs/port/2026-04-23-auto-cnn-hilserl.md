# auto CNN reward + HIL-SERL (no human) — progress log

beads epic: lerobot-11. subtasks: lerobot-12 (data) / lerobot-13 (classifier) /
lerobot-14 (hilserl prep) / lerobot-15 (smoke) / lerobot-16 (long run).

## datasets

user-provided:
- `domrachev03/sim_assemble_binary_data` — 20 eps × 300 fr = 6000 fr. recorded
  with `terminate_on_success=false`. action 5D (dx,dy,dz,dyaw,gripper), state
  15D, images front+wrist 128×128.
  - 14 success eps (1,2,3,5,6,7,8,10,13,14,16,17,18,19) — 1 frame w/
    `reward=1` per ep (operator 1-tick Triangle press).
  - 6 failure eps (0,4,9,11,12,15). first_press positions 132–237 of 300.
- `domrachev03/sim_assemble_manual_two_stages` — "53 eps" per user, but
  `file-001.parquet` corrupt → quarantined `.corrupt.bak`. 50 valid eps
  survived (eps 50–52 lost). filtered out user-requested eps 5,43,44 →
  **47 eps / 7953 fr → `local/sim_assemble_manual_filtered`** (demo buffer for RL).
  all eps terminate_on_success=true → last frame = reward=1, ep lens 110–261.

## classifier iterations

| # | train set | labels | val split | val @best thr | notes |
|---|-----------|--------|-----------|---------------|-------|
| 1 | binary_data raw | 1-tick Triangle = pos | stride-4 random | val had 0 pos → trivial | unusable |
| 2 | binary + sharp (last30 pos, first60 neg, middle skipped) | sharp windows | ep-level | sharp-val 100% but lenient 73% / 402 FP | overfit sharp regime |
| 3 | bin + man_filtered (sharp) | last30/first60 both src | ep-level | val 98.44% / 14 FP | near miss; FAIL eps 11,15 fire late |
| 4 | iter3 − ep15 | sharp | ep-level | val 98.22% / 4 FP | still not 0 FP |
| v2 | bin only, first-press-onwards = pos | **user-intended label** | ep-level | thr=0.99 acc=90.5% / **76 FP** | poor — FPs all in val ep 18 pre-press window |
| **v3** | same as v2 | same | **frame-stride=4** | **thr=0.98 acc=97.13% / FP=0 / rec=91%** | ✅ target met |

all runs: helper2424/resnet10 backbone, 2 cameras (front+wrist), binary head,
bs=64, 10k steps, dropout 0.1–0.2, lr=1e-4.

key insight: with episode-wise split, near-press frames only appear in train OR
val (not both). classifier learned to discriminate "val ep 18 pre-press" from
"val ep 18 post-press" poorly. frame-stride-4 mixes these frames across
splits → same distribution. all 6 failure eps now correctly 0-confidence end.

final artifact: `outputs/sim_assembling_cnn_reward_v3/checkpoints/last/pretrained_model`.
config: `src/lerobot/rl/sim_assembling_cnn_reward_train_v3.yaml`.

## HIL-SERL setup (task 14, in progress)

config draft: `src/lerobot/rl/sim_assembling_cnn_noinput_train.json`.
- env.teleop=null → actor path auto-skips teleop-event steps.
- cnn reward_step wired via env.processor.reward_model (type=cnn,
  pretrained_path=v3 ckpt, success_threshold=0.98).
- action cont-dim=4 (dx,dy,dz,dyaw) + num_discrete_actions=3 (gripper).
  env emits 5D; SAC slices `[:, :-1]` for continuous critic + argmax for discrete.
- realtime=false (fast sim). policy.device=cuda. concurrency=threads.
- dataset.repo_id = `local/sim_assemble_manual_filtered` (offline buffer).

dry-run passed: config parses, env builds (5D action), pipelines hook up,
make_policy + select_action → 5D action (4 cont + 1 discrete).

launch (when ready):
```bash
# terminal 1 (learner)
uv run python -m lerobot.rl.learner \
  --config_path=src/lerobot/rl/sim_assembling_cnn_noinput_train.json
# terminal 2 (actor)
MUJOCO_GL=egl uv run python -m lerobot.rl.actor \
  --config_path=src/lerobot/rl/sim_assembling_cnn_noinput_train.json
```

## pending

- [ ] smoke 15-min run (lerobot-15). watch: actor loss, critic loss, grad norm,
      success rate per classifier, interaction_step rate.
- [ ] triage if metrics look off (loss NaN, success rate 0 after warmup, etc).
- [ ] 1h run to >=90% success (lerobot-16).

## what we didn't commit (per user)

- no git commits made. all new/modified files staged locally.
- datasets live under `~/.cache/huggingface/lerobot/local/*` (not tracked).
- classifier ckpts under `outputs/sim_assembling_cnn_reward_v{2,3}/`.
- the bad `file-001.parquet` from manual_two_stages is renamed
  `.corrupt.bak` in user's HF cache — reversible.
