# Finetuning stack (assembly task)

What this branch (`feature/reward-models-port`) adds to LeRobot for human-in-the-loop
finetuning on the MuJoCo assembly task, plus the sibling plugin repos it relies on.
All RL pipelines run through the lerobot actor/learner with configs in `src/lerobot/rl/`.

## Features implemented in this branch

- **Reward models for RL** (`processor/reward_model/`): pluggable reward sources —
  SARM (stage-aware reward model), CNN success classifier, height+gripper heuristic,
  and `teleop_bonus` (step penalty + per-stage advance bonus + terminal success/failure).
  Dispatched via a common base; SARM routes to the `sarm_ext` plugin or upstream base.
- **Stage annotation** (`processor/stage_annotator.py`): gamepad-marked substage
  progression for multi-stage tasks, written into the dataset (sparse/dense subtask names).
- **Sim assembly env** (`envs/sim_assembling.py`, `AssembleBase-v0`): wired into
  `gym_manipulator` with reward-model + stage + reset/terminate plumbing.
- **HIL-SERL extensions**: DSRL learner/actor branch, n-step returns, per-transition
  bootstrap discount; gamepad teleop with yaw, continuous-gripper width, configurable
  stage-advance button, and edge-detected intervention.
- **Continuous gripper** (`processor/gripper_continuous.py`) + observation-processor updates.
- **Eval / replay**: `eval_dsrl.py` (vanilla & DSRL rollouts, video + score logging),
  chunk-policy eval, replay-demo-in-env.
- **Data tooling** (`scripts/`): destale actions, SARM relabel/prepare, ACT↔DP dataset
  merge, action-stats refresh, gripper→continuous conversion, workspace-bounds finder.
- **Configs** (`src/lerobot/rl/`): v5 ACT/DP training, SARM/QC/DSRL/residual HIL-SERL,
  and sim eval/record envs.

## Approaches & repos

| Approach | Repo | Policy key / entry |
|---|---|---|
| This branch (reward models + HIL-SERL + sim glue) | https://github.com/VAlikV/lerobot/tree/feature/reward-models-port | `src/lerobot/rl/`, `processor/reward_model/` |
| **SARM** — Stage-Aware Reward Modeling (`sarm_ext`) | https://github.com/domrachev03/lerobot_policy_sarm | `sarm_ext` |
| **QC** — Q-Chunking RL (`qc_ext`) | https://github.com/domrachev03/lerobot_policy_qc | `qc_ext` |
| **DSRL** — Diffusion Steering via Latent-Space RL (`dsrl_ext`) | https://github.com/domrachev03/lerobot_dsrl | `dsrl_ext` |
| **RDP** — Reactive Diffusion Policy | https://github.com/domrachev03/lerobot_rdp | `rdp_tokenizer`, `rdp_latent_diffusion` |
| **Residual RL** — residual SAC | https://github.com/amazon-far/residual-offpolicy-rl (reference) | SAC `residual_mode` |
| **Sim env** — MuJoCo `AssembleBase-v0` | https://github.com/VAlikV/simulator_for_IL_RL | `--env.type=gym_manipulator` |

## Plugins

Install into the same env as `lerobot` (`uv pip install -e .`); they register on import.
`lerobot_policy_*` auto-discover by prefix; `lerobot_dsrl` loads via
`--policy.discover_packages_path=lerobot_dsrl`. Each repo has its own README + `docs/`.

Integration tests live under `tests/rl/` (kept local, gitignored); run with `MUJOCO_GL=egl`.
