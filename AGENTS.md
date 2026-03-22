# agents.md — Research Assistant Context for LeRobot

This file provides Claude with the context needed to act as an effective research assistant
for robot learning experiments using the LeRobot codebase.

---

## Project Overview

**LeRobot** (v0.4.4) is a HuggingFace library for real-world robot learning in PyTorch.
This fork is used for behaviour cloning research on robot manipulation tasks.

**Research focus:** Behaviour cloning (BC) policies for robot manipulation, with custom extensions:
- `awm` — Action-World-Model policy for PushT (action-chunking transformer with world-model head, discrete action tokenization, EMA targets, cosine WM loss)
- `act_awm` — ACT variant with an Action-World-Model using a pretrained SigLIP2 encoder for future state alignment
- `pi0_fast_awm_head` — Pi0-FAST with AWM decoder head for LIBERO (frozen VLA context, trains WM decoder only)
- `siglip_decoder` — SigLIP feature decoder for LIBERO (used to decode AWM latent predictions to images for visualization)
- `act_simple_with_awm_head` — ACT Simple + World Model head: combines act_simple's non-autoregressive encoder-decoder (continuous actions, L1 loss) with an AWM-style WM decoder that uses continuous action embeddings (no tokenizer) and cosine similarity loss on future encoder representations. Supports EMA targets, WM loss warmup, image reconstruction decoder for debugging.
- `act_simple` branch — simplified ACT implementation
- RA-BC (Reward-Aligned Behaviour Cloning) — reward-weighted training via `use_rabc` flag in `TrainPipelineConfig`
- Gradient-Based Planning (GBP) — online planning via `gradient_planner.py` that optimizes action embeddings toward oracle goal observations using the AWM world-model

**Key source layout:**
```
src/lerobot/
  policies/          # Policy implementations (act, awm, act_awm, diffusion, sarm, smolvla, ...)
    act/             # Baseline ACT (Action Chunking Transformer)
    awm/             # Custom: Action-World-Model policy (PushT research)
      modeling_awm.py         # AWMPolicy with AR decoder + WM decoder head
      configuration_awm.py    # AWMConfig (wm_loss_weight, use_ema_target, etc.)
      tokenizer_awm.py        # Uniform action tokenizer (continuous ↔ discrete)
      gradient_planner.py     # GBP: oracle-guided gradient-based planning eval script
    act_awm/         # Custom: ACT + Action-World-Model (research branch)
    act_simple_with_awm_head/  # ACT Simple + WM decoder head (continuous actions, no tokenizer)
    pi0_fast_awm_head/  # Pi0-FAST with AWM decoder head (LIBERO)
    siglip_decoder/  # SigLIP feature decoder for AWM visualization
    diffusion/       # Diffusion policy
    sarm/            # SARM policy
  configs/
    train.py         # TrainPipelineConfig (batch_size, steps, RA-BC params, etc.)
    eval.py          # EvalPipelineConfig
    default.py       # DatasetConfig, EvalConfig, WandBConfig
  scripts/           # Entry-point scripts (train, eval, record, replay, ...)
  datasets/          # LeRobotDataset (Parquet + MP4 format)
  envs/              # Simulation environments (LIBERO, PushT, MetaWorld, ...)
```

Each policy follows the pattern:
- `configuration_<policy>.py` — dataclass config
- `modeling_<policy>.py` — `nn.Module` + `PreTrainedPolicy` subclass
- `processor_<policy>.py` — observation/action preprocessing

---

## Running Training Experiments

The main training entry point is `lerobot-train` (maps to `src/lerobot/scripts/lerobot_train.py`).

**Basic BC training:**
```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --output_dir=outputs/train/my_run
```

**ACT-AWM (custom research policy):**
```bash
lerobot-train \
  --policy=act_awm \
  --dataset.repo_id=<your_dataset> \
  --output_dir=outputs/train/act_awm_run
```

**RA-BC (reward-aligned training):**
```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=<your_dataset> \
  --use_rabc=true \
  --rabc_progress_path=<path_to_sarm_progress.parquet> \
  --rabc_kappa=0.01 \
  --rabc_head_mode=sparse \
  --output_dir=outputs/train/rabc_run
```

**Key `TrainPipelineConfig` fields** (`src/lerobot/configs/train.py`):
| Field            | Default  | Description                          |
| ---------------- | -------- | ------------------------------------ |
| `batch_size`     | 8        | Training batch size                  |
| `steps`          | 100,000  | Total training steps                 |
| `eval_freq`      | 20,000   | Steps between evaluations            |
| `save_freq`      | 20,000   | Checkpoint interval                  |
| `log_freq`       | 200      | WandB log interval                   |
| `seed`           | 1000     | Global random seed                   |
| `num_workers`    | 4        | DataLoader workers                   |
| `use_rabc`       | false    | Enable reward-weighted BC            |
| `rabc_kappa`     | 0.01     | Hard threshold for sample quality    |
| `rabc_head_mode` | "sparse" | Dual-head mode: "sparse" or "dense"  |
| `rename_map`     | {}       | Remap observation keys (image/state) |

**Resuming a run:**
```bash
lerobot-train --config=outputs/train/my_run/train_config.json --resume=true
```

**WandB logging** is on by default. Disable with `--wandb.enable=false`.

Experiuments are logged in `outputs/train` and are ordered by launch times
```
outputs/
  - 2026-02-23/
    - ...
    - ...
  - 2026-03-01/
    - checkpoints/
      - 010000/
      - 020000/
        - pretrained_model/
        - training_state/
      - ...
    - eval/
    - wandb/
      - latest_run/
```

You can access the checkpoints for loading in the `checkpoints` folder. You can also read the wandb training logs directly in the `wandb` folder. Use the wandb API to read logs whenever necessary.

---

## Running Evaluation

The main evaluation entry point is `lerobot-eval` (maps to `src/lerobot/scripts/lerobot_eval.py`).

**Evaluate a checkpoint on LIBERO:**
```bash
lerobot-eval \
  --policy.path=outputs/train/my_run/checkpoints/last \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=50 \
  --output_dir=outputs/eval/my_run
```

**Evaluate on a HF Hub model:**
```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

**`EvalPipelineConfig` key fields** (`src/lerobot/configs/eval.py`):
| Field             | Description                                        |
| ----------------- | -------------------------------------------------- |
| `policy.path`     | Local checkpoint dir or HF Hub repo ID             |
| `env.type`        | Environment type (libero, aloha, pusht, metaworld) |
| `env.task`        | Task within the environment                        |
| `eval.n_episodes` | Number of evaluation rollouts                      |
| `seed`            | Random seed for reproducibility                    |
| `rename_map`      | Remap observation keys                             |

---

## Code Style Guidelines

The project uses **Ruff** for linting/formatting (line length 110, Python 3.10 target).

**Rules enforced:** pycodestyle (E/W), PyFlakes (F), isort (I), pyupgrade (UP),
flake8-bugbear (B), flake8-comprehensions (C4), flake8-print (T20), pep8-naming (N), flake8-simplify (SIM).

**Ignored:** E501 (line length — handled by formatter), T201/T203 (print statements allowed).

**Docstring style:** Google convention.

**Quote style:** Double quotes, space indentation.

**Run checks manually:**
```bash
pre-commit run --all-files   # lint + format + typo checks
ruff check src/              # linting only
ruff format src/             # formatting only
```

**Key conventions:**
- Policy files follow the `configuration_`, `modeling_`, `processor_` naming pattern.
- Use `PreTrainedPolicy` as the base class for all policies.
- Observation constants (`OBS_IMAGES`, `OBS_STATE`, `OBS_ENV_STATE`, `ACTION`) live in `lerobot.utils.constants`.
- Config dataclasses use `draccus` for CLI parsing; fields map directly to CLI args (e.g., `--batch_size=32`).
- `__init__.py` files are exempt from F401/F403 (re-export allowed).
- `mypy` is partially enabled — strict checking applies to `configs.*`, `optim.*`, `model.*`, `cameras.*`, `transport.*`.

---

## Running Tests

```bash
# Full test suite
pytest -sv ./tests

# Single file
pytest -sv tests/test_specific_feature.py

# With coverage
pytest --cov=src/lerobot ./tests
```

Tests require git-lfs artifacts (`git lfs pull`).

---

## Making Pull Requests

**Workflow:**
1. Fork upstream and add remote: `git remote add upstream https://github.com/huggingface/lerobot.git`
2. Work on a descriptive branch (never `main`).
3. Rebase on `upstream/main` before opening a PR: `git rebase upstream/main`
4. Run `pre-commit run --all-files` and `pytest` locally — both must pass.
5. Open PR using the `.github/PULL_REQUEST_TEMPLATE.md` template.
6. A LeRobot team member will review.

**For research branches (this repo):**
- Branch off `main` for experiments; use `simple-act` or similar descriptive names.
- Keep research-specific changes (e.g., `act_awm`, RA-BC params) scoped to their own modules.
- Do not upstream experimental code unless it is clean, tested, and general-purpose.

**Issue reporting:** Use `.github/ISSUE_TEMPLATE/bug-report.yml`.

---

## Running SLURM Jobs

All GPU experiments are submitted via SLURM using wrapper scripts in the repo root.
The user will specify which compute format to use (GPU type, DDP setup). See
`experiments.sh` for examples.

**Available compute scripts:**
```
compute_rtx6000.sh       # 1x RTX 6000
compute_h200.sh          # 1x H200
compute_ddp_2_h200.sh    # 2x H200 (DDP)
compute_ddp_4_h200.sh    # 4x H200 (DDP)
compute_ddp_4_h100.sh    # 4x H100 (DDP)
compute_inference.sh     # 1x L40s (inference / eval)
```

**Submitting a job:**
```bash
sbatch compute_rtx6000.sh <command and args>
# Example:
sbatch compute_rtx6000.sh lerobot-train --policy.type=awm --dataset.repo_id=lerobot/pusht ...
sbatch compute_inference.sh lerobot-eval --policy.path=lerobot/diffusion_pusht ...
```

**Job monitoring workflow (CRITICAL — minimize token usage):**
1. `sbatch` prints the job ID (e.g. `Submitted batch job 4930059`)
2. To check job status, **ONLY use `squeue`** — never read log files to determine
   whether a job is still running. A single `squeue -j <jobid>` call uses minimal
   tokens and gives you everything you need: `PD` (pending), `R` (running), or
   absent (completed/failed).
3. **Do NOT read log files while the job is queued or running.** Log files are
   buffered and incomplete — reading them wastes tokens and provides no useful
   signal. Wait until the job disappears from `squeue`.
4. Once the job is no longer in `squeue` output, **then** read the log:
   `slurm_out/Report-<jobid>.out`
5. If error → fix code → resubmit. If success → done.

**Job states in squeue:**
- `PD` — pending (waiting for resources)
- `R`  — running
- Job disappears from squeue → completed (or failed)

**Debug runs:**
- Use `--output_dir=outputs/debug/<descriptive_name>` for debugging.
- Output directories must be unique. Before rerunning a debug job, **ask the user
  for permission** to delete the previous output directory, then `rm -rf` it.
- `--job_name` is only needed for formal sweeps — skip it for debug runs.

**Known harmless warning to ignore in logs:**
```
CommandNotFoundError: Your shell has not been properly configured to use 'conda deactivate'.
```
This appears at the top of every job log and does not affect execution.

---

## Useful CLI Commands

```bash
lerobot-info                   # Show install info
lerobot-dataset-viz            # Visualise a LeRobotDataset
lerobot-imgtransform-viz       # Visualise image transforms
lerobot-edit-dataset           # Edit dataset episodes/features
lerobot-record                 # Record demonstrations
lerobot-replay                 # Replay recorded episodes
lerobot-teleoperate            # Teleoperate a robot
```

---

## AWM / Gradient-Based Planning

The AWM (Action-World-Model) policy on PushT uses discrete action tokenization with a
transformer-based world-model head that predicts future latent states from action token
embeddings.

**Training AWM on PushT** (see `experiments.sh` for full sweep configs):
```bash
sbatch compute_rtx6000.sh lerobot-train \
    --policy.type=awm --dataset.repo_id=lerobot/pusht \
    --policy.wm_loss_weight=0.1 --steps=200000 ...
```

**Gradient-Based Planning (GBP)** runs as a standalone script:
```bash
python -m lerobot.policies.awm.gradient_planner \
    --awm_policy_path outputs/awm_pusht_.../checkpoints/last/pretrained_model \
    --oracle_policy_path lerobot/diffusion_pusht \
    --n_episodes 1 --gbp_n_iters 10 --gbp_lr 0.01 \
    --output_dir outputs/debug/gbp_test --device cuda
```

GBP collects oracle trajectories from a diffusion policy, then optimizes AWM action
embeddings toward oracle goal observations. It retries seeds automatically until the
oracle succeeds. Outputs include:
- `oracle_goals/` — saved oracle goal observation images
- `action_plots/` — env frames with BC (red) and GBP iteration (green gradient) trajectories overlaid
- `episode_plots/` — per-episode cosine similarity improvement plots
- `gbp_summary.json` — aggregate planning metrics

**Older HF Hub models** (e.g. `lerobot/diffusion_pusht`) lack `policy_preprocessor.json`.
A fallback in `src/lerobot/policies/factory.py` automatically extracts normalization
stats from the model's old embedded buffers when processor configs are missing.

**AWM checkpoints** live under `outputs/awm_pusht_*`. The most-trained baseline is
`outputs/awm_pusht_constlr_wm0.1_200k/checkpoints/last/pretrained_model`.

---

## Job Babysitting (Batch Run Monitoring & Auto-Resume)

When the user launches a batch of SLURM training jobs, Claude should monitor them and
automatically resume any that fail unexpectedly. This is the "babysit" workflow.

### 1. Launching Jobs

Submit jobs via `sbatch` and **record every (job_id, output_dir, wandb_run_id) tuple**:

```bash
# Example: submit a training run
sbatch compute_rtx6000.sh lerobot-train \
    --policy.type=... \
    --dataset.repo_id=... \
    --output_dir=outputs/train/my_experiment \
    --wandb.run_id=my_wandb_run_id \
    ...
```

After submission, `sbatch` prints `Submitted batch job <JOBID>`. Store the mapping.
Note: when a job is resumed, its **new job ID** and **new wandb.run_id** replace the old
entry for that output_dir (the output_dir stays the same across resumes):

| Job ID  | Output Dir                        | WandB Run ID                          |
| ------- | --------------------------------- | ------------------------------------- |
| 4930059 | `outputs/train/my_experiment`     | `20260318_100000-my_experiment`       |
| 4930060 | `outputs/train/my_experiment_2`   | `20260318_100001-my_experiment_2`     |

### 2. Heartbeat Checking (Token-Efficient Monitoring)

**Only use `squeue` to check job status — never read log files while jobs are running.**

```bash
squeue -j <jobid1>,<jobid2>,<jobid3> --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

- `R`  → running (healthy, do nothing)
- `PD` → pending (waiting for resources, do nothing)
- Job absent from output → completed or failed (proceed to step 3)

### 3. Detecting Completion vs Failure

Once a job disappears from `squeue`, **then** read the tail of its log to determine outcome:

```bash
tail -n 50 slurm_out/Report-<jobid>.out
```

- **Success:** Log contains `Training completed` or similar end-of-training message → mark as done.
- **Failure:** Log shows an error, OOM, or unexpected termination → resume the job (step 4).

### 4. Auto-Resuming a Failed Job

When a job fails, resume from its **last checkpoint** using the saved `train_config.json`.
**Each resume segment MUST get a new, unique `wandb.run_id`** to avoid W&B overlapping-step
merge conflicts (W&B cannot merge runs that log the same step twice).

**Generating a unique run ID:**
Use the pattern `<YYYYMMDD_HHMMSS>-<job_name>` (timestamp + job name) for each new segment.
For example: `20260318_143022-my_experiment`.

```bash
sbatch <compute_script>.sh lerobot-train \
    --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --wandb.run_id=<NEW_unique_run_id>
```

**Critical rules for resuming:**
- Use the **same output_dir** as the original job (checkpoints and training state are shared).
- Use a **NEW unique `wandb.run_id`** for each resume segment. Do NOT reuse the original
  run ID — W&B offline runs with overlapping steps cause sync conflicts. Each segment
  becomes a separate W&B run that can be grouped by tags or viewed together in the project.
- The `checkpoints/last/` directory is auto-updated during training, so `--config_path`
  pointing to `last/pretrained_model/train_config.json` always resumes from the most
  recent checkpoint.
- After resubmission, record the **new job ID** and **new wandb.run_id** and continue monitoring.
- Only resume the specific job that failed — do not restart jobs that are still running.

### 5. Rules During Monitoring

- **NEVER cancel running jobs.** Even if a job appears to be in a bad state (NaN, high
  loss, crash loop during eval), do not `scancel` it. Let the user decide.
- **Only check `squeue` status during babysitting.** Do not read internal log files
  while jobs are running. Only read logs after a job falls out of the queue.

### 6. Termination Condition

Do **not** stop monitoring until **all** jobs meet one of:
- Log confirms `Training completed` (success).
- User explicitly says to stop.

### 7. How to Activate Job Babysitting

Use the `/loop` slash command to set up recurring heartbeat checks:

```
/loop 5m Check all tracked SLURM job IDs with squeue. For any job that disappeared,
read the tail of its log. If it failed, generate a new unique wandb.run_id
(YYYYMMDD_HHMMSS-job_name) and resume it with the auto-resume command.
Report status of all jobs.
```

This runs the check every 5 minutes (adjustable). The loop keeps running until all
jobs are confirmed complete or the user cancels.

**Alternatively**, the user can simply ask:
> "Babysit these jobs: <job_id_1>, <job_id_2>, ... with output dirs ... and wandb run IDs ..."

And Claude will set up the monitoring loop automatically.

---

## Key References

- Upstream repo: https://github.com/huggingface/lerobot
- HF Hub datasets: https://huggingface.co/lerobot
- Documentation: https://huggingface.co/docs/lerobot/index
- ACT paper: https://huggingface.co/papers/2304.13705
