# LingBot-VLA v2 on RoboTwin: Developer Guide

> **Audience:** a lerobot developer who wants to fine-tune LingBot-VLA v2 on RoboTwin
> simulation data and evaluate it on the RoboTwin benchmark — using the lerobot framework
> end-to-end, without adopting the upstream LingBot training stack.
>
> **Scope:** data prep → norm stats → checkpoint conversion → fine-tune → sim evaluation.

The lerobot PR gives you the policy, the checkpoint converter, and the train loop. RoboTwin
gives you the simulator and the evaluation client. Two small bridges connect them, and both
ship with this policy:

```
 RoboTwin HDF5 ──robotwin_to_lerobot.py──▶ lerobot dataset ──lerobot-train──▶ lerobot ckpt
                                                                                  │
 RoboTwin sim ◀──eval client (official)──websocket── lingbot_vla_v2_policy_lerobot.py ◀┘
```

| Piece | Who provides it | Where |
|---|---|---|
| Policy / pre/post-processors | **this PR** | `lerobot.policies.lingbot_vla_v2` |
| Checkpoint converter + `--profile robotwin` (L1_fm + bounds_99_woclip) | **this PR** | `...lingbot_vla_v2/scripts/convert_upstream_checkpoint.py` |
| Data converter (RoboTwin HDF5 → lerobot) | **this PR** | repo-root `scripts/robotwin_to_lerobot.py` |
| lerobot-ckpt policy server | **this PR** | repo-root `scripts/lingbot_vla_v2_policy_lerobot.py` (copy into upstream `deploy/` to run) |
| Simulator + eval client + launcher | **RoboTwin / upstream** | `RoboTwin` repo, `experiment/robotwin/` |
| Raw datasets, base ckpt, Qwen3-VL | **download** | HuggingFace (see below) |

---

## 0. Downloads and environments

What you must fetch yourself (nothing else is redistributed):

| Asset | Size | Get it from | Needed for |
|---|---|---|---|
| RoboTwin repo | ~GB | `github.com/RoboTwin-Platform/RoboTwin` | sim + eval |
| RoboTwin raw data (HDF5) | per-task | `huggingface.co/datasets/TianxingChen/RoboTwin2.0` | training data (or collect your own) |
| Base ckpt `robbyant/lingbot-vla-v2-6b` | ~26 GB | HF (gated — request access) | fine-tune start |
| Qwen3-VL backbone | ~13 GB | HF | policy VL encoder, also at eval |

Two conda/venv environments, kept separate (the upstream launcher assumes this split):

- **inference / training env** — `pip install lerobot[lingbot_vla2]` plus `h5py`, and
  `pip install -U "av>=15,<16"` if you want video-mode datasets (PyAV ≥ 15; some mirrors 403
  it — use `--index-url https://pypi.org/simple` with proxies unset).
- **sim env** — the RoboTwin env (sapien / mplib / curobo / open3d, numpy 1.26.x). Only
  needed for the evaluation step, not for data conversion or training.

> You do **not** need the RoboTwin repo to prepare data or train — only to run the sim
> benchmark at the end. The data converter depends only on `h5py + lerobot`.

---

## 1. RoboTwin HDF5 → lerobot dataset

RoboTwin episodes ship as HDF5 (`state/{left,right}_{arm,ee}_joint_states`,
`vision/cam_*/colors` JPEG bytes, `instructions`). Convert to a lerobot v3 dataset:

```bash
python scripts/robotwin_to_lerobot.py \
    --input-dir /path/to/robotwin_task_episodes \
    --repo-id   my_robotwin_task \
    --fps 15 --mode video
```

Output schema (matches what `robotwin.yaml` expects):

- `observation.state`, `action` — `(14,)`, left arm 6 + left gripper + right arm 6 + right gripper; `action[t] = state[t+1]` (teacher shift).
- `observation.images.{cam_high, cam_left_wrist, cam_right_wrist}` — `(3, H, W)` video.
- `task` — from the HDF5 `instructions`.
- `meta/stats.json` carries real `q01/q99` — step 2 consumes them directly.

Alternatives / gotchas:

- The RoboTwin repo ships its own `policy/pi0/process_data_pi0.sh` that produces the **same**
  schema — you can use it instead. `robotwin_to_lerobot.py` exists so you don't need the full
  RoboTwin checkout just to convert data, and it handles the XPolicyLab HDF5 variant.
- Images are passed HWC `uint8` to `add_frame` (lerobot converts internally); don't pre-permute.
- The default PyAV codec `libsvtav1` is unavailable on some setups — the script pins `h264`.

---

## 2. Norm stats (robotwin profile)

```bash
python gen_rebot_norm_stats.py \
    --dataset-root /path/to/lerobot_dataset \
    --quantiles --out norm_stats.robotwin.json
```

`--quantiles` slices `q01/q99` out of the dataset's `meta/stats.json` — required for the
`bounds_99_woclip` normalization the robotwin profile uses.

## 3. Convert the base checkpoint

```bash
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
    --input  robbyant/lingbot-vla-v2-6b \
    --output ./lingbot-robotwin-6b \
    --robot-config-path robotwin.yaml \
    --norm-stats-path   norm_stats.robotwin.json \
    --profile robotwin
```

`--profile robotwin` bakes `loss_type=L1_fm` and `canonical_norm_type[arm/end/effector]
=bounds_99_woclip` into the ckpt config; training and inference then follow automatically.
(Use `--profile real` for a real-robot target: `fm` + `meanstd`.)

## 4. Fine-tune

```bash
lerobot-train \
    --dataset.repo_id=my_robotwin_task --dataset.root=/path/to/lerobot_dataset \
    --policy.path=./lingbot-robotwin-6b \
    --policy.device=cuda --policy.dtype=bfloat16 \
    --batch_size=... --steps=...
```

## 5. Evaluate on the RoboTwin benchmark

Evaluation is client-server over websocket. The **eval client and launcher are official and
unchanged** — you only swap the policy server for the lerobot-ckpt one via the launcher's
`--inference_script` flag:

```bash
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path       ./lingbot-robotwin-6b \
    --eval_workdir     /path/to/RoboTwin \
    --inference_script deploy/lingbot_vla_v2_policy_lerobot.py \
    --conda_sh         /path/to/miniconda3/etc/profile.d/conda.sh \
    --inference_env    lerobot --sim_env RoboTwin \
    --num_tasks 1 --num_gpus 1 --num_per_gpu 1     # smoke: 1 task, 1 GPU
```

`--inference_script deploy/lingbot_vla_v2_policy_lerobot.py` is the whole trick: the default
server loads the upstream ckpt format; this one loads a lerobot ckpt
(`LingbotVLAV2Policy.from_pretrained` + the PR's pre/post-processors). The launcher needs no
other change. Bump `--num_tasks 50` for the full benchmark once the smoke passes.

> The server sits next to the upstream eval code: it imports `deploy/websocket_policy_server.py`
> (msgpack transport) from the upstream checkout and is resolved by the launcher relative to
> `--inference_workdir`. **Copy this PR's `scripts/lingbot_vla_v2_policy_lerobot.py` into the
> upstream checkout's `deploy/`** (the RoboTwin experiments tree carries that `deploy/`), then
> pass `--inference_script deploy/lingbot_vla_v2_policy_lerobot.py`:

### What the server does (the contract)

- Receives upstream-keyed obs (`cam_high/cam_left_wrist/cam_right_wrist` HWC uint8,
  `observation.state`, `task`) from the official client; maps them to lerobot canonical keys.
- `preprocessor(raw_frame) → policy.select_action(batch) → postprocessor(action)` — the
  postprocessor unnormalizes, re-adds state (robotwin profile: `subtract_state=False`,
  absolute joints), and maps canonical 55-dim → the robot's raw 14-dim.
- Returns `{"action": np.ndarray}`; the client drives the sim.

---

## Checklist

- [ ] Download RoboTwin repo + raw data (or collect), base ckpt, Qwen3-VL
- [ ] Convert data: `scripts/robotwin_to_lerobot.py` (or RoboTwin's pi0 converter)
- [ ] Norm stats: `gen_rebot_norm_stats.py --quantiles`
- [ ] Convert ckpt: `--profile robotwin`
- [ ] Fine-tune: `lerobot-train`
- [ ] Eval: copy the server into upstream `deploy/`, then official launcher + `--inference_script deploy/lingbot_vla_v2_policy_lerobot.py`

**Status:** the data converter and the lerobot policy server are written and verified
(unit + synthetic-episode smoke: 14-dim dual-arm schema, 3 cameras, q01/q99 present). An
end-to-end sim run additionally needs a converted robotwin-profile ckpt and the RoboTwin sim
env on the machine.
