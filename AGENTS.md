# agents.md — Research Assistant Context for LeRobot

This file provides Claude with the context needed to act as an effective research assistant
for robot learning experiments using the LeRobot codebase.

---

## Project Overview

**LeRobot** (v0.4.4) is a HuggingFace library for real-world robot learning in PyTorch.
This fork is used for behaviour cloning research on robot manipulation tasks.

**Research focus:** Behaviour cloning (BC) policies for robot manipulation, with custom extensions:
- `act_awm` — ACT variant with an Action-World-Model using a pretrained SigLIP2 encoder for future state alignment
- `act_simple` branch — simplified ACT implementation
- RA-BC (Reward-Aligned Behaviour Cloning) — reward-weighted training via `use_rabc` flag in `TrainPipelineConfig`

**Key source layout:**
```
src/lerobot/
  policies/          # Policy implementations (act, act_awm, diffusion, sarm, smolvla, ...)
    act/             # Baseline ACT (Action Chunking Transformer)
    act_awm/         # Custom: ACT + Action-World-Model (research branch)
    diffusion/       # Diffusion policy
    sarm/            # SARM policy
  configs/
    train.py         # TrainPipelineConfig (batch_size, steps, RA-BC params, etc.)
    eval.py          # EvalPipelineConfig
    default.py       # DatasetConfig, EvalConfig, WandBConfig
  scripts/           # Entry-point scripts (train, eval, record, replay, ...)
  datasets/          # LeRobotDataset (Parquet + MP4 format)
  envs/              # Simulation environments (LIBERO, MetaWorld, ...)
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

## Key References

- Upstream repo: https://github.com/huggingface/lerobot
- HF Hub datasets: https://huggingface.co/lerobot
- Documentation: https://huggingface.co/docs/lerobot/index
- ACT paper: https://huggingface.co/papers/2304.13705
