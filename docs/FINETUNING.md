# Finetuning stack (assembly task)

Index of what is implemented across this fork + its sibling plugin repos,
and where to look for each. All RL pipelines run through the lerobot
actor/learner with configs in `src/lerobot/rl/`.

## Approaches & where they live

| Approach | What it is | Repo | Policy key / entry |
|---|---|---|---|
| **IL (ACT / Diffusion)** | Behaviour cloning baselines | `lerobot` (this repo) | `lerobot-train`; configs `act_v5_*`, `dp_v5_*` |
| **HIL-SERL** | Human-in-the-loop SAC, actor+learner over gRPC | `lerobot` (this repo) | `src/lerobot/rl/{actor,learner}.py` |
| **SARM** | Stage-Aware Reward Modeling (reward model for RL) | `lerobot_policy_sarm` | `sarm_ext`; configs `sim_3stage_sarm_v5_*` |
| **QC** | Q-Chunking RL (action-chunked actor-critic) | `lerobot_policy_qc` | `qc_ext`; `sim_qc_chunk5_hilserl_v5_a6000_train.json` |
| **DSRL** | Diffusion Steering via Latent-Space RL (noise-space SAC) | `lerobot_dsrl` | `dsrl_ext`; `sim_dsrl_v5_hilserl_a6000_train.json` |
| **Residual RL** | Residual SAC over a base policy | `lerobot` + `residual-offpolicy-rl` | SAC `residual_mode`; `sim_residual_K_chunk10_v1_train.json` |
| **RDP** | Reactive Diffusion Policy (tokenizer + latent diffusion) | `lerobot_rdp` | `rdp_tokenizer`, `rdp_latent_diffusion` |
| **Sim env** | MuJoCo `AssembleBase-v0` task backend | `simulator_for_IL_RL` | `--env.type=gym_manipulator` |

## Plugins

Plugins install into the same env as `lerobot` (`uv pip install -e .`) and
register on import. `lerobot_policy_*` are auto-discovered by prefix;
`lerobot_dsrl` is loaded explicitly via
`--policy.discover_packages_path=lerobot_dsrl`. Each plugin repo has its own
`README.md` + `docs/` with design notes and run commands.

## Tests

`tests/rl/` is the integration harness (config decode, plugin registration,
SAC actor+learner, QC, DSRL, SARM reward wiring, sim record). Run headless
with `MUJOCO_GL=egl`.

Project design/iteration notes are kept locally under `docs/` (untracked).
