# Residual HIL-SERL design (extending existing SAC)

## Goal

Train residual SAC on top of frozen ACT v11. Both vanilla & residual hil-serl available via cfg flag.

## Strategy: minimal-diff flag toggle

NO new parallel module. Extend SACConfig + SACPolicy + tiny env wrapper.

## Diff surface

| file | change |
|---|---|
| `policies/sac/configuration_sac.py` | +5 fields: `residual_mode`, `base_policy_path`, `base_policy_type` ('act'/'diffusion'), `residual_action_scale`, `freeze_base_policy`. validate adds `observation.base_action` to input_features when residual_mode. |
| `policies/sac/modeling_sac.py` | +`_load_base_policy()` (called from `__init__` if residual). +`select_action`: if residual, compute a_b → a_full=clip(a_b + scale*a_r, -1, 1). +compute_loss_critic/actor: combine via `obs["observation.base_action"]`. Policy.forward: scale residual output by cfg.residual_action_scale. |
| `rl/actor.py` | tiny wrapper: when residual_mode, on reset/step augment obs dict with `observation.base_action` from frozen base. Pass full action to env. |
| `rl/learner.py` | unchanged (forward_batch flows through; obs dict carries new key automatically). |

Replay buffer untouched (dict-of-tensors, schema-agnostic).

## Residual semantics

- a_b = base_policy.select_action(obs) (ACT chunked queue, frozen)
- a_r = scale * tanh(actor.mu(obs ⊕ a_b))   [scale e.g. 0.1]
- a_full = clip(a_b + a_r, -1, 1)
- env.step(a_full)
- transition: action=a_full, state contains observation.base_action=a_b, next_state contains observation.base_action=a_b'

## Loss formulas (mirrored from paper)

```python
# critic target
with no_grad:
    a_r_next = actor(next_obs).sample()  # already scale-clipped via Policy.forward
    a_b_next = next_obs["observation.base_action"]
    a_full_next = clamp(a_b_next + a_r_next, -1, 1)
    q_target = critic_target(next_obs, a_full_next).min(ensemble)
    y = r + gamma*(1-d)*q_target

# critic loss
q = critic(obs, action_stored_full)  # action_stored = a_b + a_r at rollout time
L_critic = MSE(q, y)

# actor loss
a_r_pred = actor(obs).rsample()
a_b = obs["observation.base_action"]
a_full = clamp(a_b + a_r_pred, -1, 1)
L_actor = -Q(obs, a_full).min(ensemble) + alpha*log_prob   # SAC entropy term
```

## Config additions (proposed)

```python
# in SACConfig
residual_mode: bool = False
base_policy_path: str | None = None
base_policy_type: str = "act"  # or "diffusion"
residual_action_scale: float = 0.1
freeze_base_policy: bool = True
```

## Encoder: how base_action enters obs

Inject `observation.base_action` as an OBS_STATE-class feature with shape=(action_dim,). The existing `SACObservationEncoder` already handles state-vector inputs. So encoder reads it transparently without arch change. Validate_features ensures it's registered.

## Env wrapper (in actor.py)

```python
class _BasePolicyAugmenter:
    """Wraps a gym env. Queries frozen base policy on every obs → adds observation.base_action."""
    def __init__(self, env, base_policy):
        self.env = env
        self.base = base_policy.eval()
    def reset(self, **kw):
        obs, info = self.env.reset(**kw); self.base.reset()
        obs["observation.base_action"] = self.base.select_action(obs)
        return obs, info
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        obs["observation.base_action"] = self.base.select_action(obs)
        if term: self.base.reset()
        return obs, r, term, trunc, info
    def __getattr__(self, n): return getattr(self.env, n)
```

ACT chunk queue handled inside ACTPolicy.select_action (existing).

## Phase plan (paper Algorithm 1)

| phase | what |
|---|---|
| **warmup** (skip OR tiny) | optional uniform noise residual to populate buffer. Paper: yes. We may skip — actor's initial gaussian std already explores, and v11 base provides good prior. |
| **online** | normal hil-serl actor/learner loop. |

Add small toggle: if `cfg.residual_warmup_steps > 0`, force `a_r = U(-noise, noise)` for that many initial steps.

## Hyperparams initial (mostly paper defaults adapted)

| param | val | note |
|---|---|---|
| residual_action_scale | 0.1 | safety |
| critic_lr | 3e-4 | default |
| actor_lr | 3e-4 | default |
| temperature_init | 0.01 | LOW — we want little entropy for fine residual |
| target_entropy | -5 | num actions / 2 = 2.5 (or even tighter) |
| critic.layer_norm | TRUE | paper: crucial |
| num_critics | 4 | REDQ-like |
| num_subsample_critics | 2 | min subset |
| utd_ratio | 1 | start; raise if needed |
| use_backup_entropy | True | sac default |
| discount | 0.99 | default |
| online_step_before_learning | 200 | small warmup |
| policy_update_freq | 2 | TD3 delayed actor |
| critic_target_update_weight | 0.005 | polyak τ |
| online_buffer_capacity | 100k | sim, ample |
| offline_buffer_capacity | 50k | demos |
| dataset_repo_id | local/sim_assemble_actdp_combined_destale_tail30 | for offline 50/50 mix (action stored = teleop demo, base_action computed at training-time… see "Demo a_b" below) |

## Demo a_b problem

Paper uses demos in 50/50 sampling. The offline buffer must contain `observation.base_action` for each demo frame. Two choices:

1. **Pre-compute**: run base_policy over demo trajectories ONCE at training start, dump a_b column into a dataset_index→a_b lookup. Inject into batches as they're sampled.
2. **On-the-fly**: skip offline buffer use entirely for v0 (paper showed it helps but works without).

V0: choice **2** (skip demos for first run). If converges slow, add #1 (pre-compute pass).

## Intervention support (paper has none)

Teleop intervention overrides full action. Buffer entry:
- action = teleop_full
- observation.base_action = a_b (whatever base would have done at this s)
- implied residual = teleop_full - a_b

**Critic update**: standard off-policy. Q learns to predict return from (s, teleop_full). Identical math, no change.

**Actor update**: standard SAC max Q(s, a_b + π(s,a_b)). Intervention transitions contribute only via critic — actor never sees teleop residual directly.

**Optional BC anchor on intervention frames**: add `λ * MSE(a_r_pred, teleop_full - a_b)` for intervention transitions. Forces actor toward human corrections. Skip for v0.

## Risks / fallbacks

1. **ACT chunk queue + sampled batches**: at training time, the actor net sees obs with a_b already attached (correct from rollout). At inference time same. No issue.
2. **Action norm mismatch**: ACT outputs in normed [-1,1] (MIN_MAX). SAC actor outputs tanh-squash [-1,1] residual. Both space-consistent. ✓
3. **Stage1 regression risk**: residual scale 0.1 keeps a near base. If residual scale too big (0.5+) early, may break v11's stage1 competence. Monitor first 5 min stage1 reach rate.
4. **Reward density**: SARM dense vs paper's sparse. Should help, not hurt. May need reward normalization (clip to [0,1] for stable Q targets). Add `clip_q_target_to_reward_range` flag (resfit has it).

## V0 implementation order

1. cfg fields (5 lines)
2. validate_features auto-adds base_action key
3. _load_base_policy + freeze
4. select_action residual path  
5. compute_loss_critic + actor: combine via obs.base_action
6. _BasePolicyAugmenter wrapper
7. cfg JSON for run
8. launch sim non-headless 10-15min sanity

If signal: long-run 30-40min.

## Eval criteria (per user)

- consistency reaching SARM>=0.90 (currently v11: ~80% eps, terminal phase)
- success rate at SARM>=0.97 (currently v11: 0% — that's the gap to close)

Both metrics come from gym_manipulator's reward stream (SARM dense). Log per-episode max & success indicator.
