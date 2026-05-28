# HIL-SERL — team training guide

Project-internal notes on running HIL-SERL with LeRobot. For the canonical
tutorial (real-robot setup, gamepad wiring, reward-classifier training) see
[`source/hilserl.mdx`](source/hilserl.mdx). For the sim-side commands index
see [`port/2026-04-22-sim-hilserl-commands.md`](port/2026-04-22-sim-hilserl-commands.md).

This document is the per-policy-variant entry point. Two variants currently
supported on the actor side:

- **Residual SAC** (default, in-tree `lerobot.policies.sac`). Tanh-Gaussian
  actor produces a per-step residual added to a frozen ACT base. Training
  config: `lerobot/src/lerobot/rl/sim_residual_K_chunk10_v1_train.json`.
- **Q-chunking (QC-FQL)** (ext plugin `lerobot_policy_qc`). Chunk-level
  actor + chunk-level critic with γ^h discount and unbiased n-step backup.
  Training config: `lerobot/src/lerobot/rl/sim_qc_chunk5_v1_train.json`.

The two variants share the same actor-process / learner-process / gRPC
infrastructure (`lerobot/src/lerobot/rl/actor.py`, `…/learner.py`) and the
same reward stream (`processor/reward_model/sarm.py`). They differ only in
how `select_action` is dispatched and how the training loss is computed.

---

## Action-chunking variant (QC)

PyTorch port of Li, Zhou, Levine — *Reinforcement Learning with Action
Chunking* (arXiv:2507.07969, NeurIPS 2025). Background notes:
[`qc_notes.md`](qc_notes.md). Design / port-spec:
[`qc_design.md`](qc_design.md).

> **Back-compat guarantee.** The QC port is purely additive. Every change
> in `lerobot/` is gated on `cfg.policy.type == "qc_ext"` (or on a config
> field that SACConfig doesn't have). Existing residual-SAC training
> configs (`sim_residual_K_chunk10_v1_train.json` and friends) parse and
> run bit-identically. Verify with `git diff main -- lerobot/`.

### When to use

- **Use QC when**: the base BC chunk-policy (e.g. ACT) plateaus before
  task completion, especially on long-horizon tasks where the last stage
  (4/5/6) never improves under residual SAC. Our team's residual-SAC runs
  hit a wall around stage 4–5 on the 3-stage assembly task — QC's
  chunk-level Q-function and unbiased n-step backup are architecturally
  better suited for fine-tuning a chunked BC policy.
- **Use residual SAC when**: BC is already strong on all stages and you
  only need a small per-step correction (e.g. compensating for sim/real
  gap). Residual SAC has lower per-step inference cost and a more mature
  training pipeline.
- **Neither variant changes the SARM reward, intervention semantics, or
  episode-success gating** — these remain the same as the existing
  residual-SAC workflow.

### Install

QC lives in an external plugin discovered via the `lerobot_policy_` prefix
scan. Editable install:

```bash
uv pip install -e /home/dom-iva/github.com/orel/lerobot/lerobot_policy_qc
```

Verify the registration succeeded:

```bash
python -c "from lerobot.configs.policies import PreTrainedConfig; \
print('qc_ext' in PreTrainedConfig.get_known_choices())"
```

Should print `True`. If not, the plugin import failed silently — re-run
with `python -c "import lerobot_policy_qc"` to see the import error.

### Training config

Template: [`src/lerobot/rl/sim_qc_chunk5_v1_train.json`](../src/lerobot/rl/sim_qc_chunk5_v1_train.json).
Same overall structure as `sim_residual_K_chunk10_v1_train.json` (env block
with `gym_manipulator`, SARM reward model, gamepad teleop; policy block at
the bottom). QC-specific knobs in the `policy` block:

| Field | Default | Meaning / paper §  |
|---|---|---|
| `type` | `"qc_ext"` | Plugin registration key. Resolves to `QCPolicy`. |
| `horizon_length` | `5` | Chunk length *h*. Paper sweet spot 5–10. `h=50` collapses. |
| `actor_variant` | `"qc_fql"` | v1 supports only `"qc_fql"` (deterministic distilled). `"qc"` (best-of-N) is v2. |
| `flow_steps` | `10` | Euler steps for the flow-matching BC policy `f_ξ`. |
| `alpha_bc` | `100.0` | BC distillation coefficient (paper Table 4). |
| `discount` | `0.95–0.99` | Per-step γ. Critic uses **γ^h** for the bootstrap. |
| `critic_target_update_weight` | `0.005` | τ for Polyak EMA. |
| `chunk_execution` | `"full"` | Roll out the whole chunk open-loop. Other modes deferred. |
| `flush_chunk_on_intervention` | `true` | Empty FIFO on human takeover. |
| `residual_mode` | `false` | v2 switch. Set `false` in v1. |
| `critic.num_critics` | `2` | Ensemble size. K=10 helps marginally; K=2 matches paper. |

All other fields (`vision_encoder_name`, `shared_encoder`,
`image_encoder_hidden_dim`, `actor_learner_config`, `concurrency`, etc.)
have the same meaning as in `configuration_sac.py`. The full reference
table is in `lerobot_policy_qc/README.md`.

### Launch

Two terminals, same pattern as residual SAC:

```bash
# Terminal 1 — learner
python -m lerobot.rl.learner \
  --config_path=src/lerobot/rl/sim_qc_chunk5_v1_train.json

# Terminal 2 — actor (after learner has bound the gRPC port)
python -m lerobot.rl.actor \
  --config_path=src/lerobot/rl/sim_qc_chunk5_v1_train.json
```

> **Status — T6b open.** The learner-process training-loop branch for
> `cfg.policy.type == "qc_ext"` (chunk-aware buffer sampling +
> dispatching `policy.forward(model="critic"|"actor")`) is the last
> integration piece still landing. Until T6b merges, the learner will
> fall through to the SAC code path and produce shape-mismatch errors on
> the first `policy.forward` call. The actor process is already
> wired (see *Actor-side wiring* below). T6b ships before the T8 GPU
> smoke test.

### Actor-side wiring (already landed)

Three small additions in [`lerobot/src/lerobot/rl/actor.py`](../src/lerobot/rl/actor.py):

| Line | What | Why |
|---|---|---|
| 268 | `policy.reset()` once at startup | Clears QC's chunk queue before the first `select_action`. No-op for SAC. |
| 485–488 | `policy.reset()` on intervention rising edge, gated on `getattr(cfg.policy, "flush_chunk_on_intervention", False)` | Discards a stale chunk when the user takes over. Pre-intervention chunk is wrong for the post-intervention state. |
| 606 | `policy.reset()` on episode reset | Mirrors the `_base_policy.reset()` ACT-chunk flush at line 615. |

`SACPolicy.reset()` is a no-op (`modeling_sac.py:127`), so these calls are
free for the SAC path. The `flush_chunk_on_intervention` field is absent
from `SACConfig`, so the `getattr(..., False)` default keeps the branch
unreachable for SAC configs.

### Expected metrics (W&B)

Same actor-side metrics as SAC (`Episodic reward`, `Episode success`,
`Episode max SARM progress`, `Episode intervention`, `Intervention rate`,
`Episode stage advances`). Additionally on the learner side once T6b
lands:

| Key | Watch for |
|---|---|
| `train/loss_critic` | Should decrease then plateau. Spikes after intervention bursts are normal. |
| `train/loss_actor` | Negative (it's `−Q + α·distill + bc_flow`). Decreasing magnitude. |
| `train/loss_bc_flow` | Slow steady decrease as `f_ξ` learns the chunk distribution. |
| `train/loss_distill` | Should decrease faster than `loss_bc_flow` (μ_ψ chases f_ξ-Euler). |
| `train/q_mean`, `q_max`, `q_min` | Sanity-check magnitudes against per-chunk reward scale. Q should not explode. |
| `train/chunk_flush_events` | Per-episode count of intervention-triggered flushes. High values = noisy teleop or unstable user. |

### Troubleshooting

- **"`predict_action_chunk` raised NotImplementedError"**: you set
  `actor_variant="qc"` or `"qc_rlpd"`. v1 is QC-FQL only. Use
  `"qc_fql"`.
- **Chunk queue stuck after intervention**: confirm
  `flush_chunk_on_intervention: true` in the policy block. Without it
  the policy resumes the pre-intervention chunk from a stale state.
- **Q-values explode after ~5k steps**: γ^h with high γ and large h can
  push the discount close to 1. Try `discount=0.95` (config default) or
  reduce `horizon_length` to 3.
- **"target Q computed at s_{t+1}" looking sensible but loss diverges**:
  the bootstrap state must be **s_{t+h}**, not s_{t+1}. If you wrote a
  custom buffer or hand-built a batch for debugging, the `state_at_h`
  field needs to advance *h* steps. See `modeling_qc.py:329`.
- **Loss is finite but the policy never moves the gripper**: SAC has a
  separate discrete gripper head; QC treats the gripper as a continuous
  action dim. Confirm your `action.shape = [5]` (xyz, yaw, gripper),
  `action.min = [-1, ..., -1]`, `action.max = [1, ..., 1]`. Check the
  config: `num_discrete_actions` doesn't exist in `QCConfig`.
- **`PreTrainedConfig.from_pretrained` fails with `Unknown policy type
  'qc_ext'`**: the plugin didn't import. Re-run `uv pip install -e
  /path/to/lerobot_policy_qc` and verify with the
  `'qc_ext' in PreTrainedConfig.get_known_choices()` check above.
- **`InterventionActionProcessorStep` keeps the teleop action through
  the chunk boundary**: it should — the intervention overrides whichever
  policy action was queued. QC's flush handles the *next* chunk. If you
  want the policy to immediately resume after release, that's already
  the case (queue is empty → re-sample on next `select_action`).

### Cross-references

- Paper notes + reference-repo file index: [`qc_notes.md`](qc_notes.md)
- Plugin port-spec (config schema, loss equations, integration points):
  [`qc_design.md`](qc_design.md)
- Plugin README (full per-field config table, comparison vs residual
  SAC): [`../../lerobot_policy_qc/README.md`](../../lerobot_policy_qc/README.md)
- Canonical HIL-SERL tutorial: [`source/hilserl.mdx`](source/hilserl.mdx)
- Sim env wiring + commands: [`port/2026-04-22-sim-hilserl-commands.md`](port/2026-04-22-sim-hilserl-commands.md)
