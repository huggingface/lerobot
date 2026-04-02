# run-and-eval

You are an autonomous execution agent. Your job is to **configure** hyperparameters, **commit and tag** each experiment, **run** the self-improvement script on SLURM, **babysit** the job, **log results** to a TSV file, and **report** the final evaluation score.

You do NOT design experiments or write new logic. The user has already set up `src/lerobot/scripts/self_improvement_playground.py` — you configure it, commit it, execute it, and collect results.

## Orchestrating multiple experiments

When the user requests multiple experiments, you must run them through a **serial submit pipeline** with **parallel babysitting**. This is critical because SLURM reads the script from disk when the job *starts* (not when `sbatch` is called) — if you edit the config while a prior job is still in `PD` (pending) state, that job will read the wrong config.

### The pattern

**You (the parent agent)** own the critical section: edit config → commit → submit → wait for `R` state. This is strictly serial — one experiment at a time through this pipeline.

**Background subagents** handle babysitting. Once a job reaches `R` state, you spawn a background subagent with the job ID and commit hash, then immediately move on to configuring the next experiment.

### Concrete workflow for N experiments

```
for each experiment:
  1. YOU: edit config, commit, tag, submit via sbatch        ← serial, you do this
  2. YOU: poll squeue until job transitions to R state        ← serial, you wait
  3. YOU: spawn background babysit subagent with:             ← non-blocking
         - SLURM job ID
         - commit hash
         - experiment description
         - instructions: monitor squeue, read logs when done,
           extract metrics, append to TSV, report back
  4. YOU: switch back and proceed to next experiment          ← immediately
```

### What the babysit subagent does

Each babysit subagent receives a job ID and commit hash. It:
1. Polls `squeue -u $USER -j <jobid>` every 30-60 seconds until the job disappears.
2. Reads the SLURM log: `grep 'EVAL_RESULTS\|EVAL_DIR' slurm_out/Report-<jobid>.out`
3. Extracts full metrics from `<EVAL_DIR>/eval_info.json`.
4. Appends a row to `results_self_improvement_deterministic.tsv`.
5. Reports results back.

**Babysit subagents must NEVER edit source files.** They only read logs and write to the TSV.

### Safety rules

- **Never** edit the playground script while any of your submitted jobs is in `PD` state.
- **Never** spawn a subagent that edits source files — only you (the parent) do that.
- If you need to submit 5 experiments, you will edit→commit→submit→wait-for-R five times sequentially, but all 5 babysit subagents run concurrently in the background.
- When all babysit subagents have reported back, compile a summary table for the user.

## Understanding the codebase

Before making any changes, **read both files** to understand the current config block and available knobs:

1. `src/lerobot/scripts/self_improvement_playground.py` — the top-level script you will configure and run.
2. `src/lerobot/scripts/self_improvement_utils.py` — the library of functions it calls (do NOT modify this file).

The playground has two configurable sections near the top:

- **`POLICY`** — path to the base pretrained model checkpoint.
- **Config block** — all experiment hyperparameters between the `Config — edit these` markers.

The **only** lines you may edit are `POLICY`, `OUTPUT_DIR`, and the config block. Do not touch any other code.

### Deep-dive: policy architecture and planning

When the user asks you to **suggest how to improve performance**, or asks "what should I try next?", you need to understand the model architecture and planning algorithms. Read these files **on demand** (not upfront — they are large and not needed for routine runs):

- `src/lerobot/policies/act_simple_with_awm_head/configuration_act_simple_with_awm_head.py` — all model hyperparameters (WM loss weight, encoder dims, EMA, normalization, etc.)
- `src/lerobot/policies/act_simple_with_awm_head/modeling_act_simple_with_awm_head.py` — the model: encoder, BC head, world-model decoder, forward pass, loss computation.
- `src/lerobot/policies/act_simple_with_awm_head/planning.py` — `PlanningConfig`, `GBPPlanner`, `MPPIPlanner`: the planning algorithms and all their hyperparameters (lr, n_iters, n_samples, noise_std, etc.)

This context lets you make informed suggestions like "try increasing planning n_iters from 10 to 30" or "the WM cosine similarity is low — try more finetune steps" rather than generic advice. When reporting results, use this understanding to populate the `suggestions` column in the TSV with actionable next steps.

## Config block reference

```python
POLICY = "..."                      # base model checkpoint path
OUTPUT_DIR = str(Path(POLICY).parent / "self_improvement")

# ── Self-improvement loop ──────────────────────────────────────
N_ITERS = 1                 # 0 = eval-only (skip loop), 1+ = collect→finetune cycles
N_COLLECT_EPISODES = 50     # episodes per eval_and_collect
FINETUNE_STEPS = 100        # 0 = skip end-to-end finetune
FINETUNE_LR = 5e-6          # LR for end-to-end finetune
FINETUNE_WM_STEPS = 0       # 0 = skip WM-only finetune
FINETUNE_WM_LR = 1e-6       # LR for WM-only finetune
BATCH_SIZE = 8
MIXING = "naive"            # "naive" = concat online+pretrain, uniform sample
                            # "ratio" = fixed pretrain/online split per batch
PRETRAIN_RATIO_FT = 0.5     # pretrain fraction for finetune    (only if MIXING="ratio")
PRETRAIN_RATIO_WM = 0.6     # pretrain fraction for finetune_wm (only if MIXING="ratio")
LOAD_OPTIMIZER = True       # load Adam state from checkpoint (False = fresh optimizer)
FINETUNE_ONLINE_MODE = "success_only"  # "success_only", "all", or "failure_only" for e2e finetune
BC_MASK_MODE = "none"       # "none"    = full BC+WM loss on all episodes (default)
                            # "failure" = zero BC loss on failure episodes, full on successes
                            # "all"     = zero BC loss on ALL episodes (WM-only loss, all params trainable)
WM_ONLINE_MODE = "all"      # "all", "success_only", or "failure_only"

# ── Final eval ─────────────────────────────────────────────────
EVAL_N_EPISODES = 250       # episodes for final evaluation
EVAL_USE_PLANNING = False   # True to enable GBP/MPPI planning at eval time
EVAL_PLANNING_ALGORITHM = "gcp"  # "gcp" or "mppi"
EVAL_PLANNING_OVERRIDES = {}     # e.g. {"lr": 0.3, "lr_decay": 1.0, "n_iters": 20}
```

### Common experiment patterns

| Experiment type | Key settings |
|---|---|
| **BC eval only** | `N_ITERS=0`, `EVAL_USE_PLANNING=False` |
| **BC + GBP eval** | `N_ITERS=0`, `EVAL_USE_PLANNING=True`, `EVAL_PLANNING_ALGORITHM="gcp"`, `EVAL_PLANNING_OVERRIDES={"lr": 0.3, ...}` |
| **Finetune + eval** | `N_ITERS=1`, `FINETUNE_STEPS=1000`, `EVAL_USE_PLANNING=False` |
| **Finetune + GBP eval** | `N_ITERS=1`, `FINETUNE_STEPS=1000`, `EVAL_USE_PLANNING=True`, ... |
| **E2E with BC masked on failures** | `N_ITERS=1`, `FINETUNE_STEPS=1000`, `FINETUNE_ONLINE_MODE="all"`, `BC_MASK_MODE="failure"` — all episodes enter e2e finetune; BC loss active only on successes, WM loss active on everything. Policy learns from successes; WM learns from all data including failures. All params trainable. |
| **E2E WM-only (BC masked on all)** | `N_ITERS=1`, `FINETUNE_STEPS=1000`, `FINETUNE_ONLINE_MODE="all"`, `BC_MASK_MODE="all"` — BC loss is zeroed on *every* sample (successes, failures, and pretrain replay). Effectively WM-only training but through the e2e pipeline with all params trainable and gradients flowing through the full graph. Unlike `finetune_wm()` which freezes non-WM params, this lets encoder representations shift via WM gradients alone. |
| **Determinism check** | Run BC eval twice — results must be identical |

## Step 0: Configure hyperparameters

The user will describe one or more experiments. For each experiment:

1. Update **only** `POLICY`, `OUTPUT_DIR`, and the config block values requested.
2. If the user specifies a base model path, update `POLICY` and recompute `OUTPUT_DIR = str(Path(POLICY).parent / "self_improvement")`.
3. Leave all other values at their current defaults unless told otherwise.
4. Confirm the final config with the user before proceeding.

If the user does not request changes, skip this step and run with current values.

## Step 1: Switch to experiment branch, commit, and tag

Each experiment gets its own commit on the **user-specified experiment branch** (e.g. `self_improvement_playground`). This keeps `main` clean while preserving a full history of every experiment config.

### One-time setup

If the branch doesn't exist yet, create it from `main`:
```bash
git branch <experiment_branch> main 2>/dev/null || true
```

### Stash, switch, commit, tag

1. **Record the current branch**:
   ```bash
   ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
   ```

2. **Stash uncommitted work**:
   ```bash
   git stash push -m "auto-stash before experiment" --include-untracked 2>/dev/null || true
   ```

3. **Switch to the experiment branch** and merge latest `main`:
   ```bash
   git checkout <experiment_branch>
   git merge main --no-edit
   ```

4. **Stage** the playground script:
   ```bash
   git add src/lerobot/scripts/self_improvement_playground.py
   ```

5. **Commit** with a descriptive slug summarizing the config. The user may provide a description — use it. Otherwise generate one from the hyperparameters (e.g. `"exp: bc-eval-only"`, `"exp: ft1000-naive-lr5e6"`):
   ```bash
   git commit -m "exp: <description>"
   ```

6. **Tag** for easy reference:
   ```bash
   git tag run/<description>
   ```

Note the short commit hash — it is passed to the script as `sys.argv[1]` and used for checkpoint naming and results tracking:
```bash
COMMIT_HASH=$(git rev-parse --short HEAD)
```

## Step 2: Submit and wait for R state

Submit the script on a GPU node:
```bash
sbatch compute_inference.sh python -u src/lerobot/scripts/self_improvement_playground.py $COMMIT_HASH
```

Note the SLURM job ID. Then **poll until the job reaches `R` (running) state**:
```bash
squeue -u $USER -j <jobid>
```

You **must not** edit the playground script or proceed to the next experiment until the job is in `R` state. See "Orchestrating multiple experiments" above.

Once confirmed running, switch back to your original branch:
```bash
git checkout "$ORIGINAL_BRANCH"
git stash pop 2>/dev/null || true
```

## Step 3: Babysit the job (or delegate to a subagent)

If running a single experiment, babysit it yourself. If running multiple experiments, spawn a background babysit subagent (see orchestration section above) and move on to the next experiment.

The babysit procedure (whether you or a subagent):

1. Poll `squeue -u $USER -j <jobid>` every 30–60 seconds. Do **not** read logs while the job is running.
2. Once the job disappears from `squeue`, check the output:
   ```bash
   grep 'EVAL_RESULTS\|EVAL_DIR' slurm_out/Report-<jobid>.out
   ```
3. Expected output:
   ```
   EVAL_RESULTS: 72.0% success
   EVAL_DIR: /path/to/eval/dir
   ```
4. If grep is empty, the run crashed — check `tail -n 50 slurm_out/Report-<jobid>.out` for the traceback and report the error.
5. Extract full metrics from the saved eval JSON:
   ```bash
   cat <EVAL_DIR>/eval_info.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'pc_success={d[\"pc_success\"]}, avg_max_reward={d.get(\"avg_max_reward\",0)}, eval_ep_s={d.get(\"eval_ep_s\",0)}')"
   ```

## Step 4: Log results and report

### Log to results_self_improvement_deterministic.tsv

Append results to `results_self_improvement_deterministic.tsv` (tab-separated). The file already exists with the header row.

Columns:
```
commit	pc_success	avg_max_reward	eval_ep_s	status	description	suggestions
```

1. `commit` — short git hash (7 chars)
2. `pc_success` — success rate, e.g. `72.0` (use `0.0` for crashes)
3. `avg_max_reward` — average max reward, e.g. `0.8217` (use `0.0` for crashes)
4. `eval_ep_s` — episodes per second (use `0.0` for crashes)
5. `status` — `keep`, `discard`, or `crash`
6. `description` — brief explanation: motivation, notable hyperparameters, changes from previous runs
7. `suggestions` — any structural improvements needed in the playground or utils for future experiments (leave empty if none)

Do **not** commit `results_self_improvement_deterministic.tsv` to git.

### Report to user

```
Final: <pc_success>% success | avg_max_reward: <avg_max_reward> | eval_ep_s: <eval_ep_s>
Commit: <hash> | Tag: <tag>
Logged to results_self_improvement_deterministic.tsv
```

## Failure modes and recovery

**Job crashes (GPU error / OOM / NVML)**:
- The eval never ran. Resubmit the job:
  ```bash
  git checkout <experiment_branch>
  sbatch compute_inference.sh python -u src/lerobot/scripts/self_improvement_playground.py $COMMIT_HASH
  ```

**Non-determinism error at runtime**:
- Report to user. Do NOT silence it by relaxing determinism settings.

## Constraints

- The **only** modifications you may make to source files are: `POLICY`, `OUTPUT_DIR`, and hyperparameters in the config block of `src/lerobot/scripts/self_improvement_playground.py`.
- Do **NOT** modify `src/lerobot/scripts/self_improvement_utils.py` or any other source file.
- Do **NOT** modify or overwrite the base checkpoint.
- Do **NOT** cancel any running jobs.
- Only check `squeue` while babysitting — do not read job logs until the job has left the queue.
- Do **NOT** commit `results_self_improvement_deterministic.tsv` to git.
- **Determinism is mandatory.** All evaluation and finetuning runs under fully deterministic settings hardcoded in the playground script. Under **no** circumstance may you add code that overrides, disables, or weakens these settings (e.g. `cudnn.benchmark = True`, `allow_tf32 = True`, or `use_deterministic_algorithms(False)`).
