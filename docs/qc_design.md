# `lerobot_policy_qc` — Plugin Port Design (T2 spec)

**Companion doc:** `qc_notes.md` (paper summary + reference-repo index).
**Author:** qc-researcher (team `qc`), 2026-05-20.
**Read-only research phase.** This doc proposes; T3+ implements. No code changes are landed yet.

**Goal:** Port Q-chunking RL (Li, Zhou, Levine 2025) into lerobot's HIL-SERL stack as an external plugin `lerobot_policy_qc`, mirroring the existing `lerobot_policy_sarm` pattern.

**v1 target: QC-FQL** — deterministic distilled actor μ_ψ(s, z), one forward pass per chunk, single-action `select_action` interface that matches lerobot's existing HIL-SERL infra. (Lead-confirmed 2026-05-20.) The non-distilled **QC (best-of-N=32) variant is deferred to v2** as a future extension — extra cost (32 × 10 = 320 forward passes per chunk) risks breaking HIL-SERL's actor FPS gate, and the success-rate delta vs QC-FQL is statistically zero on aggregated benchmarks (paper Table 1). All v1 code paths assume `actor_variant="qc_fql"`; the QC variant config switch is kept in `QCConfig` only as a placeholder enum value with a `NotImplementedError` guard.

---

## § Plugin file layout

Mirror `lerobot_policy_sarm/`. Reference structure (existing, already lands in editable install via `pyproject.toml`):

```
lerobot_policy_qc/
├── pyproject.toml                   # hatchling, deps: torch, lerobot
├── README.md
└── src/
    └── lerobot_policy_qc/
        ├── __init__.py              # exports Config, Policy, processor factory
        ├── configuration_qc.py      # @PreTrainedConfig.register_subclass("qc_ext")
        ├── modeling_qc.py           # QCPolicy(PreTrainedPolicy)
        ├── processor_qc.py          # make_qc_ext_pre_post_processors
        ├── qc_utils.py              # dataset chunk sampler, BC flow-matching helpers
        └── networks/                # explicit subpkg (chunk-aware nets are non-trivial)
            ├── __init__.py
            ├── flow_actor.py        # ActorVectorField (PyTorch) + Euler integrator
            ├── chunk_critic.py      # Q(s, a_{t:t+h}) ensemble
            └── one_step_actor.py    # μ_ψ(s, z) noise-conditioned distilled policy
```

**Naming key:** registration string `qc_ext`, mirroring `sarm_ext`. Policy class `QCPolicy`, config class `QCConfig`. Module suffix convention (`configuration_qc.py`, `modeling_qc.py`, `processor_qc.py`) is **required** by lerobot's `get_policy_class` (factory.py:564) and `_make_processors_from_policy_config` (factory.py:593) — they swap `configuration_` ↔ `modeling_` / `processor_` in the module name.

**`pyproject.toml`** minimal, copy `lerobot_policy_sarm/pyproject.toml`. Deps: `torch`, `numpy`, `lerobot` (declared explicitly so installer warns if it's missing).

**Editable install** during dev: `uv pip install -e /path/to/lerobot_policy_qc/`.

---

## § Config class

**Decision: subclass `PreTrainedConfig` directly (NOT subclass `SACConfig`).** Justification:

1. `SACConfig` has fields that don't apply (`temperature_init`, `target_entropy`, `use_backup_entropy`, `num_discrete_actions`, `actor_lr` separate from `critic_lr`, etc.) and is missing fields we need (`horizon_length`, `bc_alpha`, `actor_type`, `flow_steps`, `actor_num_samples`).
2. Inheriting would force us to keep all SAC's `residual_mode` plumbing as live code paths even when not used; cleaner to copy the *useful* parts (`residual_mode`, `base_policy_path`, `freeze_base_policy`, `residual_action_scale`) explicitly.
3. lerobot's draccus-based config system uses `@register_subclass("qc_ext")` for discovery; the choice key is namespace-local, so inheritance gives no factory benefit.

Skeleton (`configuration_qc.py`):

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


@dataclass
class QCFlowActorConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    use_layer_norm: bool = False
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64


@dataclass
class QCCriticConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    use_layer_norm: bool = True
    num_critics: int = 2                       # paper default; K=10 marginally helps


@PreTrainedConfig.register_subclass("qc_ext")
@dataclass
class QCConfig(PreTrainedConfig):
    # ---- chunking ----
    horizon_length: int = 5                    # h; paper sweet spot is 5–10
    actor_variant: str = "qc_fql"              # "qc_fql" | "qc" (best-of-N) | "qc_rlpd"
    actor_num_samples: int = 32                # for actor_variant="qc"
    flow_steps: int = 10                       # Euler integration steps for f_ξ
    alpha_bc: float = 100.0                    # BC distill coef for QC-FQL
    bc_alpha_rlpd: float = 0.01                # entropy-style BC weight for QC-RLPD

    # ---- RL ----
    discount: float = 0.99
    critic_target_update_weight: float = 0.005
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    grad_clip_norm: float = 5.0
    utd_ratio: int = 1
    online_steps: int = 200_000
    online_buffer_capacity: int = 50_000
    offline_buffer_capacity: int = 1
    online_step_before_learning: int = 1_000
    policy_update_freq: int = 1

    # ---- nets ----
    flow_actor: QCFlowActorConfig = field(default_factory=QCFlowActorConfig)
    critic: QCCriticConfig = field(default_factory=QCCriticConfig)
    shared_encoder: bool = True
    vision_encoder_name: str = "helper2424/resnet10"
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 64

    # ---- residual mode (mirrors SAC) ----
    residual_mode: bool = False
    base_policy_path: str | None = None
    base_policy_type: str = "act"
    residual_action_scale: float | list[float] = 0.3
    freeze_base_policy: bool = True

    # ---- HIL-SERL plumbing (mirrors SAC) ----
    storage_device: str = "cpu"
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # ---- chunk env-step semantics ----
    chunk_execution: str = "full"              # "full" | "single" | "mpc"  (see § Action-chunk env-step semantics)
    flush_chunk_on_intervention: bool = True
```

**`normalization_mapping`, `input_features`, `output_features`** identical schema to SAC. Action space: `output_features[ACTION].shape = [A]` — the *per-step* action dim. Chunk dim is implicit (`= A * horizon_length`).

---

## § Policy class structure

`modeling_qc.py::QCPolicy(PreTrainedPolicy)`. PyTorch port — JAX/Flax constructs in the reference repo map as follows:

| Reference (JAX, `acfql.py`) | PyTorch (`modeling_qc.py`) |
|---|---|
| `ACFQLAgent.network.select('critic')` ensemble via `nn.vmap` | `nn.ModuleList([ChunkCritic(...) for _ in K])` + `torch.stack` over outputs |
| `ACFQLAgent.network.select('target_critic')` (EMA params) | `copy.deepcopy(critic_ensemble)`; manual Polyak in `update_target_networks` |
| `ActorVectorField` flow field $f_\xi(s, x, t)$ | `FlowActor` PyTorch module — MLP over `cat([encoded_s, x, t_emb])` |
| `actor_onestep_flow` $\mu_\psi(s,z)$ | `OneStepActor` PyTorch module — MLP over `cat([encoded_s, z])` |
| `compute_flow_actions` (Euler loop) | `FlowActor.integrate(s, z, T)` — pure Python `for i in range(T): vels = self(...); a = a + vels/T` |
| `sample_actions` (best-of-N branch) | `QCPolicy.sample_chunk` — samples N noises, integrates flow, scores via critic, argmax |
| `target_update` (`τ`-weighted EMA) | `QCPolicy.update_target_networks` (already present in SAC; copy verbatim) |

Skeleton:

```python
class QCPolicy(PreTrainedPolicy):
    config_class = QCConfig
    name = "qc_ext"

    def __init__(self, config: QCConfig | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        A = config.output_features[ACTION].shape[0]
        h = config.horizon_length
        self._A = A
        self._h = h
        self._chunk_dim = A * h

        self._init_encoders()                       # SAC-style shared encoder
        self._init_critics(self._chunk_dim)         # ensemble of Q(s, flat_chunk)
        self._init_flow_actor(self._chunk_dim)      # f_ξ
        if config.actor_variant in ("qc_fql", "qc_rlpd"):
            self._init_one_step_actor(self._chunk_dim)  # μ_ψ
        # QC variant uses *only* f_ξ (no μ_ψ); best-of-N at inference

        # Residual base-policy mirror of SAC
        object.__setattr__(self, "_base_policy", None)
        if getattr(self.config, "residual_mode", False):
            object.__setattr__(self, "_base_policy", self._load_base_policy())

        # internal action queue (only for actor process — learner doesn't use it)
        self._chunk_queue: list[torch.Tensor] = []

    def _load_base_policy(self):
        # COPY VERBATIM from lerobot/src/lerobot/policies/sac/modeling_sac.py:79-107.
        # No changes — same ACT/diffusion loading, same preprocessor attachment.
        ...

    @torch.no_grad()
    def predict_action_chunk(self, batch) -> Tensor:
        """Returns (B, h, A) chunk. The chunk-level interface."""
        s = self._encode(batch)
        if self.config.actor_variant == "qc_fql":
            z = torch.randn(s.shape[0], self._chunk_dim, device=s.device)
            chunk = self.one_step_actor(s, z)                # (B, h*A)
        else:  # qc best-of-N
            chunk = self._sample_best_of_n(s)
        chunk = chunk.clamp(-1.0, 1.0).view(-1, self._h, self._A)
        return chunk

    @torch.no_grad()
    def select_action(self, batch) -> Tensor:
        """Per-env-step interface. Manages the internal chunk queue.

        Called by lerobot/src/lerobot/rl/actor.py:331 — the existing SAC entry
        point. We refill the queue at chunk boundaries; otherwise just pop.
        """
        if len(self._chunk_queue) == 0:
            chunk = self.predict_action_chunk(batch)         # (1, h, A)
            self._chunk_queue = list(chunk[0])               # list of (A,)
        action = self._chunk_queue.pop(0).unsqueeze(0)       # (1, A)

        if getattr(self.config, "residual_mode", False):
            base_action = batch["observation.base_action"]
            action = (base_action + action * self.residual_scale_buf).clamp(-1.0, 1.0)
        return action

    def reset(self):
        """Called by actor.py on episode end. Flush chunk queue.
        Also called when flush_chunk_on_intervention triggers."""
        self._chunk_queue = []

    def forward(self, batch, model: str = "critic"):
        """Learner-side loss computation. Three branches: critic, actor, flow_bc."""
        ...
```

### Critic loss (paper Eq. 4 / 11; ref: `acfql.py::critic_loss`)

```python
def compute_loss_critic(self, batch):
    # batch fields (h-aware): see § Dataset transitions
    s        = batch["state"]                                   # (B, *)
    chunk    = batch[ACTION].reshape(batch[ACTION].size(0), -1) # (B, h*A)
    rewards  = batch["reward_chunk"]                            # (B, h)  per-step
    mask_h   = batch["mask_h"]                                  # (B,) — 1 - terminated_at_h
    valid_h  = batch["valid_h"]                                 # (B,)  drop chunks crossing terminal
    s_next_h = batch["state_at_h"]                              # (B, *)

    with torch.no_grad():
        next_chunk_star = self._sample_chunk_for_target(s_next_h)   # (B, h*A)
        next_q = self.critic_target(s_next_h, next_chunk_star)      # (K, B)
        next_q = next_q.mean(0) if cfg.q_agg == "mean" else next_q.amin(0)
        R_h = (rewards * self._discount_pows).sum(dim=1)            # γ-weighted h-step sum
        target = R_h + (cfg.discount ** self._h) * mask_h * next_q  # (B,)

    q = self.critic_ensemble(s, chunk)                              # (K, B)
    loss = ((q - target.unsqueeze(0)) ** 2 * valid_h.unsqueeze(0)).mean()
    return loss
```

Three critical correctness points (easy to bug):
1. `target = R_h + γ^h · mask_h · Q_target(s_{t+h}, π(s_{t+h}))`. **The discount is `γ^h`**, not `γ`. (`acfql.py:41` confirms.)
2. The bootstrap state is `s_{t+h}`, NOT `s_{t+1}`. The dataset must emit the state h steps ahead.
3. `valid_h` masks chunks that cross episode boundaries (where padding repeats the terminal frame). Match `acfql.py` `valid` semantics: `valid[i] = 1 - terminals[i-1]`.

### Actor loss (paper Eq. 13/14; ref: `acfql.py::actor_loss`)

```python
def compute_loss_actor(self, batch):
    s     = batch["state"]
    chunk = batch[ACTION].reshape(B, self._chunk_dim)

    # (A) BC flow-matching loss for f_ξ (always trained, even in QC variant)
    z0 = torch.randn_like(chunk)
    u  = torch.rand(B, 1, device=s.device)
    x_t = (1 - u) * z0 + u * chunk
    vel_target = chunk - z0
    vel_pred = self.flow_actor(s, x_t, u)
    valid_chunk = batch["valid_chunk"].view(B, self._h, 1)            # (B, h, 1)
    bc_flow_loss = (((vel_pred - vel_target).view(B, self._h, self._A)) ** 2 * valid_chunk).mean()

    if self.config.actor_variant == "qc_fql":
        # (B) Distillation loss for μ_ψ (Eq. 13)
        with torch.no_grad():
            target_chunk = self.flow_actor.integrate(s, z0, self.config.flow_steps)
        actor_chunk = self.one_step_actor(s, z0)
        distill_loss = ((actor_chunk - target_chunk) ** 2).mean()

        # (C) Q-max loss (Eq. 14)
        actor_chunk_clip = actor_chunk.clamp(-1, 1)
        q = self.critic_ensemble(s, actor_chunk_clip).mean(0)
        q_loss = -q.mean()

        actor_loss = bc_flow_loss + self.config.alpha_bc * distill_loss + q_loss
    else:
        actor_loss = bc_flow_loss
    return actor_loss
```

---

## § Factory registration

**No core lerobot edit required for discovery** — `@PreTrainedConfig.register_subclass("qc_ext")` registers `QCConfig` in draccus's choice registry at import time, exactly like SARM. Then `make_policy(cfg)` at factory.py:409 takes `cfg.type == "qc_ext"`, calls `get_policy_class("qc_ext")` (factory.py:540-566) which:
1. Looks up the registered config class `QCConfig`.
2. Strips the `Config` suffix → `QC`.
3. Builds class name `QCPolicy`.
4. Replaces `configuration_` ↔ `modeling_` in `QCConfig.__module__` → `lerobot_policy_qc.modeling_qc`.
5. `importlib.import_module(...)` + `getattr` → returns `QCPolicy`.

The same pattern handles the processor side: `_make_processors_from_policy_config` (factory.py:580-595) replaces `configuration_` ↔ `processor_` and looks up `make_qc_ext_pre_post_processors`.

**The auto-discovery requires that the plugin package gets imported once** before `make_policy` is called. This is what `lerobot_policy_sarm/__init__.py` does — by `from lerobot_policy_sarm.configuration_sarm import SARMConfig` at top of `__init__.py`, importing the plugin pkg side-effects `register_subclass`. The plugin is imported via `lerobot_train.py`'s plugin-discovery scan (greps installed packages with `lerobot_policy_` prefix). **Confirmed pattern: SARM works the same way today.** Our `__init__.py`:

```python
# lerobot_policy_qc/src/lerobot_policy_qc/__init__.py
try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot_policy_qc requires lerobot") from e

from lerobot_policy_qc.configuration_qc import QCConfig
from lerobot_policy_qc.modeling_qc import QCPolicy
from lerobot_policy_qc.processor_qc import make_qc_ext_pre_post_processors

__all__ = ["QCConfig", "QCPolicy", "make_qc_ext_pre_post_processors"]
```

---

## § Integration touchpoints (lerobot/ core edits)

**Target: ≤ 100 LoC additive across `lerobot/`.** Based on the SARM ext port (2026-04-25 plan doc) the actual core touch surface is small. SARM needed 3 edits totaling ~8 LoC. QC needs more because it interacts with the *action* path, not just the *reward* path.

| File | Lines / function | Change | LoC |
|---|---|---|---|
| `lerobot/src/lerobot/scripts/lerobot_train.py` | line 259 area, `if cfg.policy.type == "sarm" or cfg.policy.type.startswith("sarm_")` block | **No change** — QC doesn't need `dataset_meta` for progress norm. SARM-specific. | 0 |
| `lerobot/src/lerobot/policies/factory.py` | `_make_processors_from_policy_config` line 580 | **No change** — generic dynamic-import path handles QC's `processor_qc.make_qc_ext_pre_post_processors` automatically. | 0 |
| `lerobot/src/lerobot/policies/factory.py` | `make_policy` line 409 — currently does `if cfg.type == "vqbet" and cfg.device == "mps": raise` etc. | **Possibly add** chunk-aware check: warn if `cfg.type == "qc_ext"` and `env_cfg.fps` is much lower than `1/policy_fps × horizon_length`. Cosmetic. Or skip — let actor.py handle it. | 0–5 |
| `lerobot/src/lerobot/rl/actor.py` | `act_with_policy` env loop, around line 331 (`action = policy.select_action(batch=observation)`) | **No change** — `QCPolicy.select_action` is a single-action interface that internally manages a chunk queue. The env loop sees one action per env-step exactly like SAC. The chunk semantics are encapsulated inside `QCPolicy`. | 0 |
| `lerobot/src/lerobot/rl/actor.py` | episode-reset block around line 588–593 (currently flushes ACT chunk queue: `_base_policy.reset()`) | **Add 1 line:** `policy.reset()`. (SACPolicy.reset() is already a no-op pass at modeling_sac.py:127; calling `policy.reset()` on episode reset is currently absent from the actor loop in the non-residual case. We need it for QC to flush its chunk queue at episode boundaries.) | +1 |
| `lerobot/src/lerobot/rl/actor.py` | intervention detection around line 470 (`intervention_info.get(TeleopEvents.IS_INTERVENTION)`) | **Add 1 line:** `if intervention_info.get(IS_INTERVENTION) and getattr(cfg.policy, "flush_chunk_on_intervention", False): policy.reset()`. | +1 |
| `lerobot/src/lerobot/rl/learner.py` | learner training loop — currently samples 1-step transitions, calls `policy.forward(model="critic")` etc. | **Conditional branch:** if `cfg.policy.type == "qc_ext"`, sample h-step sequences from buffer instead of 1-step. QC's critic/actor loss needs `state, chunk, reward_chunk, state_at_h, valid_h, mask_h`. Add a helper `_sample_qc_sequence(buffer, h)`. | +30–50 |
| `lerobot/src/lerobot/rl/transition_buffer.py` (or equivalent replay-buffer file) | Replay buffer currently stores transitions; needs an indexed-sequence sampler analogous to `~/scratch/qc/utils/datasets.py::Dataset.sample_sequence` | **Add method:** `sample_chunk_sequence(batch_size, horizon, discount) -> dict` returning the chunk-aware fields. ~50 LoC, well-isolated. | +50 |
| `lerobot/src/lerobot/scripts/lerobot_train.py` | offline pretrain loop (not HIL-SERL; standard offline RL pretrain) | If `cfg.policy.type == "qc_ext"`, switch dataloader to chunk-sequence mode. Same `sample_chunk_sequence` helper. | +10 |

**Estimated total: ~92 LoC across 3 files** (`actor.py: +2`, `learner.py: +40`, `transition_buffer.py: +50`, `lerobot_train.py: +10`, `factory.py: 0–5`). Comfortably under the 100-LoC target.

**Better still — alternative:** put the chunk-sequence sampler in the plugin (`lerobot_policy_qc.qc_utils.sample_chunk_sequence`) and have `learner.py`'s QC branch import it lazily. That cuts core LoC to ~30. Recommendation: do that.

---

## § Back-compat guard

The user's residual SAC training configs (`lerobot/src/lerobot/rl/sim_residual_K_chunk10_v1_train.json` and friends) must keep parsing + running identically. Touch-by-touch verification:

| Touchpoint | Risk | Guard |
|---|---|---|
| `actor.py: policy.reset()` on episode end | None — SAC's `reset()` is already a no-op pass (`modeling_sac.py:127-129`). Calling it is a free op. | Add unconditionally. |
| `actor.py: policy.reset()` on intervention | Only fires if `getattr(cfg.policy, "flush_chunk_on_intervention", False)` — SAC config doesn't have this field, so `getattr` returns `False`. No-op for SAC. | `getattr(..., default=False)`. |
| `learner.py: if cfg.policy.type == "qc_ext"` branch | None — string compare, default path unchanged. | Explicit type check, not `isinstance`. |
| `transition_buffer.py: sample_chunk_sequence` | None — pure additive method on the buffer class; existing `sample(batch_size)` untouched. | New method, no override. |
| `lerobot_train.py: chunk-mode dataloader` | None — gated by `cfg.policy.type == "qc_ext"`. | Type-check. |
| `factory.py` warning | None if we skip; otherwise pure logging. | Skip; YAGNI. |

**Tested as part of T7 (integration tests):** load `sim_residual_K_chunk10_v1_train.json` after every edit; confirm parse → instantiate → 5-step training does NOT diverge from baseline. (T7 is lead-owned; flagged here.)

---

## § SARM / intervention reuse

**Goal: zero changes to `lerobot/src/lerobot/processor/reward_model/sarm.py`.**

The SARM reward step (`SARMRewardProcessorStep` in that file) operates on the per-step transition. It:
1. Takes the current `observation.images.*` + state + task,
2. Pushes onto its ring buffer of size `n_obs_steps`,
3. Calls `model.forward(...)` to produce `progress ∈ [0, 1]`,
4. Writes `transition[TransitionKey.REWARD] = progress_delta` (potential-based) or `progress` (raw), and stamps `info['sarm_progress']`.

For QC this is **structurally compatible without modification**:

- Per-env-step reward stream is unchanged. SARM writes one scalar per env-step exactly as today.
- QC consumes the reward stream at training time by accumulating `h` rewards via $\gamma$-discount (in `sample_chunk_sequence`).
- Stage-advance bonus (`TeleopEvents.STAGE_ADVANCE` → +bonus reward) lands as a per-step reward; the chunk-cumulative `R^h_t` sees it. No special handling needed.
- Intervention flag (`TeleopEvents.IS_INTERVENTION`) is independent of the reward stream; it controls actor-side chunk flushing (see touchpoint above) and acts as a BC-anchor weight in the learner — these are *additional* signals, not modifications to SARM.

**Concrete check.** I traced `processor/reward_model/sarm.py` lines 141, 228, 573, 656: the reward model dispatches on `config.type == "sarm_ext"` and writes `info["sarm_progress"]`; none of this depends on action chunking. **No edit needed.**

---

## § Action-chunk env-step semantics

**The decision:** start with **`chunk_execution="full"`** (paper-style: roll out the full chunk open-loop before re-planning), with `flush_chunk_on_intervention=True` and chunk flush on episode reset. Reasons:

1. **Paper prescribes it.** `~/scratch/qc/main.py:211-231`:
   ```python
   if len(action_queue) == 0:
       action = agent.sample_actions(observations=ob, rng=key)
       action_chunk = np.array(action).reshape(-1, action_dim)
       for action in action_chunk:
           action_queue.append(action)
   action = action_queue.pop(0)
   ...
   if done:
       ob, _ = env.reset()
       action_queue = []  # flush
   ```
   "during online rl, the action chunk is executed fully" (main.py comment line 178). The unbiased $n$-step backup (Prop. A.1) holds *exactly* under this open-loop assumption — sampling each env-step is what the critic learns to value.

2. **Inference latency win.** QC-FQL: one forward pass per `h` env-steps instead of one per env-step. At `h=5` that's ~5× fewer policy calls — a meaningful win given the actor process is the latency-critical path in HIL-SERL (matters for the FPS gate in `actor.py: log_policy_frequency_issue`).

3. **Intervention safety.** Human intervention mid-chunk is the failure case. We handle it by setting `flush_chunk_on_intervention=True`: as soon as the actor sees `IS_INTERVENTION=True` in `intervention_info`, it calls `policy.reset()`, which empties the chunk queue. The next env-step re-plans from the human-corrected state. This is a 1-line addition in `actor.py` (see § Integration touchpoints).

4. **Stage advance compatibility.** Stage advance is a SARM/teleop event that *adds reward*. It doesn't change the optimal action stream, so we don't flush on it. The chunk continues; the critic credits the spike via $R^h_t$.

**Rejected alternatives:**
- `chunk_execution="single"` (slice `chunk[0]` only, re-plan every step). This breaks the unbiased $n$-step property because the off-policy data we collect no longer matches the chunk policy that the critic models. Paper specifically argues against this (§ 4.1). Also kills the latency win.
- `chunk_execution="mpc"` (re-plan every step, but use $h$-ahead lookahead for the *critic target* only — Tian et al. 2025, "Chunking the critic"). Worth a comparative ablation later, but adds complexity; not for T3.

**Config knob still exposed.** `chunk_execution: str = "full"` in `QCConfig` so we can A/B later without code change.

---

## § HIL-SERL chunk semantics (intervention, truncation, termination)

This section formalizes how the chunked policy interacts with the existing HIL-SERL teleop / intervention / termination flow. **Rule of thumb: the chunk is *truncated on intervention*, never paused.** Lead-decided 2026-05-20.

### Three-way action source per env-step

Every env-step, the executed action comes from one of three sources, in priority order:

1. **Teleop override** — if `info[TeleopEvents.IS_INTERVENTION]` is True at this step. `InterventionActionProcessorStep` (`lerobot/src/lerobot/processor/hil_processor.py:441`) already overrides `transition[ACTION] = teleop_action` regardless of what the policy produced. **No change** to this step.
2. **Policy chunk action** — the next entry popped from `QCPolicy._chunk_queue`.
3. **Random warm-up** — during `interaction_step < online_step_before_learning`, lerobot already injects uniform random actions; QC inherits this unchanged.

### Intervention enter / exit lifecycle

**Enter intervention** (rising edge of `IS_INTERVENTION`):
- `InterventionActionProcessorStep` already substitutes the action. The policy's chunk queue **must be flushed** so the next post-intervention env-step re-plans from the current obs, not from a stale chunk produced before the human took over.
- Mechanism: `actor.py` polls `intervention_info.get(TeleopEvents.IS_INTERVENTION)` (currently at line ~470). On rising edge → `policy.reset()` (which empties `_chunk_queue`).
- Implementation: add 2 LoC in `actor.py` after intervention detection — see Integration touchpoints table.

**Exit intervention** (falling edge): no special handling. Chunk queue is already empty (flushed on enter); next `select_action` call sees empty queue → re-plans → fills queue from current state. Natural recovery.

### What gets stored to the replay buffer

Per env-step, store a transition `(s_t, a_executed_t, r_t, s_{t+1}, done_t, intervention_t)`. The *executed action* (whichever of teleop / policy / random) is what the buffer sees — same convention as SAC today.

For QC's chunk-aware sampler, given a query index `t`:
- `state` ← `s_t`
- `action_chunk` ← stack of `a_executed_{t}, a_executed_{t+1}, ..., a_executed_{t+h-1}` (these may be a mixture of policy + teleop frames if the user intervened mid-chunk).
- `reward_chunk` ← `(r_t, ..., r_{t+h-1})` — real env rewards regardless of who acted.
- `state_at_h` ← `s_{t+h}` for the bootstrap target.
- `mask_h` ← 0 if any frame in [t, t+h) terminates; 1 otherwise.
- `valid_h` ← 1 only if the chunk fits inside one episode (no boundary crossing).

**Why this is unbiased:** Proposition A.1 (paper Appendix A) says the $h$-step return $\sum_{t'} \gamma^{t'} r_{t+t'} + \gamma^h Q(s_{t+h}, a^\star_{t+h:t+2h})$ is an unbiased estimator of $Q^\pi(s_t, \mathbf a_{t:t+h})$ **provided the bootstrap state $s_{t+h}$ is reached by executing the chunk that the critic was given**. That holds here: the critic sees the *actually executed* chunk (policy + teleop mixture), and `state_at_h` was the actual reached state. So the critic learns "what is the value of executing this chunk from this state", including teleop chunks. This is structurally similar to off-policy SAC seeing replay-buffer actions from a stale policy — the critic doesn't know who produced the action, only what it was.

**Intervention as a BC signal:** teleop chunks land in the buffer with `intervention_t=True`. The flow-matching BC loss for $f_\xi$ can up-weight these frames (the human's chunk is a high-quality BC anchor — same role intervention transitions play in current SAC HIL-SERL). v1 ships with uniform weighting; v2 considers an intervention-weighted BC term (config knob: `bc_intervention_weight` ≥ 1.0).

### Episode termination mid-chunk

Termination sources: Triangle (manual SUCCESS), Cross (rerecord/discard), env timeout (`truncated`), or natural env-`done`. All flow through `new_transition[DONE]` / `[TRUNCATED]`.

Chunk-sampler edge case: query index `t` where `t+h > terminal_idx`.

- **Discard option** (chosen): set `valid_h = 0` for any chunk window that crosses a terminal. The flow-matching BC and the actor's Q-max losses both mask on `valid_chunk` (per-frame) or `valid_h` (per-chunk). Same convention as `~/scratch/qc/utils/datasets.py::Dataset.sample_sequence` (computes a per-frame `valid` mask exactly this way, lines 117-129).

For chunks that *end exactly at* a terminal (last in-episode frame), behavior:
- `mask_h = 1 - terminated_at_h`, where `terminated_at_h` is the terminal flag at frame `t+h-1`. So if the chunk ends on the terminal step, `mask_h = 0` ⇒ no bootstrap ⇒ Q-target = $R^h_t + 0$. Standard.
- If the chunk's *last* frame is terminal but `state_at_h` would index past the episode, we instead clamp `state_at_h = s_{terminal_idx}` and set `mask_h = 0`. The clamp is harmless because `mask_h = 0` zeroes the bootstrap term anyway. Matches `acfql.py:40-41` semantics (`masks[..., -1] * next_q`).

### Triangle / Cross / stage-advance interplay

- **Triangle (SUCCESS)** — manual success flag. Already gates whether episode transitions get pushed to the learner (actor.py:475-477). With chunks, the *whole episode's* transitions are stored or discarded as a unit; nothing changes structurally. Stage-advance reward bonuses already in SARM compose naturally with $R^h_t$.
- **Cross (RERECORD / discard)** — episode dropped. Same.
- **Stage advance** — fires a +bonus reward at one frame. Gets absorbed into that frame's $r_t$, which then gets discounted into $R^h_t$ for whichever chunk contains it. No chunk flush.

### Implementation summary (HIL-SERL specific)

| Behaviour | Mechanism | Code site |
|---|---|---|
| Intervention overrides policy action per-step | Existing `InterventionActionProcessorStep.__call__` | `hil_processor.py:458` — **no change** |
| Flush chunk queue on intervention rising edge | `policy.reset()` in actor loop | `actor.py` line ~471 — **+2 LoC** |
| Flush chunk queue on episode reset | `policy.reset()` in actor's reset block | `actor.py` line ~588 — **+1 LoC** |
| Chunk sampler masks chunks crossing terminal | `valid_h` field in `sample_chunk_sequence` | plugin-local — **+5 LoC** |
| Critic bootstrap zeroed at chunk-terminal frame | `mask_h * next_q` in `compute_loss_critic` | plugin-local — already in skeleton |
| Teleop-mixed chunks land in buffer as-is | Already the case (buffer stores executed action) | `actor.py:466` — **no change** |

---

## § Non-change audit — files explicitly NOT modified

For T7 back-compat verification. This list is as important as the change list — every entry here is a guarantee that an existing residual-SAC training run (e.g. `sim_residual_K_chunk10_v1_train.json`) keeps parsing and running identically.

### `lerobot/src/lerobot/policies/sac/` — entire directory untouched

| File | Why unchanged |
|---|---|
| `modeling_sac.py` | `SACPolicy` is the existing residual-RL policy. QC is a separate plugin with its own `QCPolicy`. Zero edits. `SACPolicy.reset()` (line 127) stays a no-op `pass`; calling `policy.reset()` from `actor.py` is therefore safe for SAC. |
| `configuration_sac.py` | All existing `SACConfig` fields preserved: `residual_mode`, `base_policy_path`, `base_policy_type`, `residual_action_scale`, `freeze_base_policy`, `temperature_init`, `target_entropy`, `num_discrete_actions`, `bc_loss_weight`, `critic_target_update_weight`, etc. No fields added, removed, renamed, or defaulted-differently. |
| `processor_sac.py` | SAC's preprocessor pipeline untouched. |
| `reward_model/` (the SAC-internal reward_classifier) | Unrelated to QC; untouched. |

### `lerobot_policy_sarm/` — entire ext repo untouched

| File | Why unchanged |
|---|---|
| `configuration_sarm.py`, `modeling_sarm.py`, `processor_sarm.py`, `sarm_utils.py` | SARM is the reward model, orthogonal to the policy. QC consumes SARM's per-step reward stream through the existing transition pipeline; SARM doesn't need to know about chunking. |
| `__init__.py` | Plugin entry points unchanged. |
| `pyproject.toml` | Independent installable. |

### `lerobot/src/lerobot/processor/reward_model/sarm.py` — untouched

The reward-model dispatch step (`SARMRewardProcessorStep`) writes one scalar reward per env-step into `transition[REWARD]` and stamps `info['sarm_progress']`. QC consumes this stream by accumulating $h$ rewards via $\gamma$-discount inside `sample_chunk_sequence` — entirely on the learner side. **No edit to sarm.py.**

The existing dispatch on `config.type == "sarm_ext"` (lines 141, 144, 656) is QC-agnostic.

### `lerobot/src/lerobot/processor/hil_processor.py` — untouched

- `InterventionActionProcessorStep` (line 441) — already overrides `transition[ACTION]` with `teleop_action` when `IS_INTERVENTION`. QC's chunk-flush semantics are layered *on top* in `actor.py`, not inside this step. This step keeps its current single-action behaviour.
- `AddTeleopActionAsComplementaryDataStep`, `AddTeleopEventsAsInfoStep`, `GripperPenaltyStep` — all untouched.

### `lerobot/src/lerobot/policies/factory.py` — touched in 0 lines (or ≤5 cosmetic)

- `make_policy` (line 409) handles `policy_cls = get_policy_class(cfg.type)` generically. `cfg.type == "qc_ext"` resolves through the existing draccus `register_subclass` machinery — no `elif policy_type == "qc"` block needed (unlike the older hardcoded entries at lines 159-185, which are legacy and don't need a new branch for plugin types).
- `_make_processors_from_policy_config` (line 580) handles `processor_qc` lookup via the existing `configuration_` ↔ `processor_` module-name swap.
- `make_pre_post_processors` (line 214) — generic `from_pretrained` path. Untouched.
- The hardcoded `elif isinstance(policy_cfg, SARMConfig):` branch at line 359 is for the **original in-tree SARM** (not the ext plugin). It stays. The ext plugin (`sarm_ext`) bypasses this branch by going through the generic dynamic-import path. QC inherits the same: no `isinstance(policy_cfg, QCConfig)` branch added.

### `lerobot/src/lerobot/rl/learner.py` — only +30–40 LoC, type-gated

All existing learner code paths (SAC critic/actor/temp update, transition consumption, gRPC parameter push) are preserved unchanged. The QC addition is a single `if cfg.policy.type == "qc_ext":` branch that calls `sample_chunk_sequence` from the plugin and dispatches `policy.forward(model="critic"|"actor"|"flow_bc")`. Outside this branch, learner behaviour is bit-identical.

### `lerobot/src/lerobot/rl/actor.py` — only +2–3 LoC, default-gated

- `policy.reset()` call on episode reset (around line 588): safe because SAC's `reset()` is a no-op (`modeling_sac.py:127`). Pure addition.
- `policy.reset()` call on intervention rising edge: gated on `getattr(cfg.policy, "flush_chunk_on_intervention", False)`; SAC config doesn't have this field, so `getattr` returns `False`, branch is never taken.
- The rest of the actor loop — `select_action`, transition packing, gRPC send, episode-end stats, gripper diagnostics, action-arrow viz — completely unchanged.

### `lerobot/src/lerobot/scripts/lerobot_train.py` — only +5–10 LoC, type-gated

The `if cfg.policy.type == "sarm" or cfg.policy.type.startswith("sarm_")` block at line 259 stays. We add a parallel branch for `cfg.policy.type == "qc_ext"` to switch the offline-pretrain dataloader to chunk-sequence mode. Outside this branch (i.e. for every existing config including all `sim_residual_*` configs and all `sarm_ext` configs), the dataloader behaviour is unchanged.

### All existing JSON training configs — untouched, all parse and run

| Config | Status |
|---|---|
| `lerobot/src/lerobot/rl/sim_residual_K_chunk10_v1_train.json` | Parses (it's `type=sac`, `residual_mode=true`, untouched fields). Runs the existing residual-SAC code path. No QC code is reached. |
| All `act_*_train.json` (ACT BC training) | Untouched; no policy.type swap, no field additions. |
| All `sim_assemble_sarm_ext_*_train.json` (SARM training) | Untouched. |

### Existing test suite — untouched

Any tests under `lerobot/tests/` that exercise SAC, ACT, diffusion, SARM, processor pipelines remain valid. QC adds new tests under `lerobot/tests/policies/` and inside the plugin's own `tests/` subdir (T4 + T7). No existing test is modified.

### Summary table — change vs no-change LoC count

| Category | LoC |
|---|---|
| **Plugin** `lerobot_policy_qc/` (new repo, ~ ext-pkg) | ~600–800 LoC (config + modeling + processor + networks + utils + tests) — not counted against lerobot/ |
| **lerobot/ changes** (additive, all type-gated) | ~80–95 LoC across 3 files: `actor.py +3`, `learner.py +40`, `lerobot_train.py +10`, `transition_buffer.py +0` (sequence sampler lives in plugin), `factory.py +0` |
| **lerobot/ non-changes** | Everything else — full SAC/ACT/diffusion/SARM stacks bit-identical. |

**Audit cmd for T7:** `git diff main -- lerobot/` should show changes only in the 3 files listed above and no others.

---

## § Open questions (please decide before T3 scaffold)

1. ~~**Best-of-N at inference for the QC (non-FQL) variant — keep it, or drop it from v1?**~~ **RESOLVED 2026-05-20 (lead):** v1 ships QC-FQL only. QC best-of-N variant deferred to v2 as a future extension. Config keeps a `QCVariant.QC_BEST_OF_N` enum value as a placeholder with a `NotImplementedError` guard so v2 wiring is a single-PR addition.

2. **`f_ξ` initialization — train from scratch online, or warm-start from frozen ACT?**
   - Reference paper trains $f_\xi$ from scratch via flow matching on offline data.
   - Lerobot already has ACT checkpoints (`outputs/act_v2_full_6stg_bc_chunk10_v11`) that *are* good chunk-distribution models — could distill ACT into $f_\xi$ via a one-shot flow-matching distillation step, skipping most of the offline-pretrain time.
   - Pure speculation, no paper support, but technically clean. **Flag for team-lead.**

3. **Discrete gripper action handling.**
   - SAC has a separate `discrete_critic` for the gripper dim (`modeling_sac.py:255-272`).
   - QC paper treats action as fully continuous (all 5 dims of the OGBench cube tasks: xyz, yaw, continuous gripper).
   - lerobot residual-SAC configs already use continuous gripper (`action.shape = [5]`, gripper as last dim in `[-1, 1]`).
   - **Recommendation:** QC port also goes fully continuous, no discrete head. Matches our existing config; matches the paper.

4. **Residual vs full chunk output.**
   - Option A: QC outputs a full chunk; ACT base is unused in QC mode.
   - Option B: QC outputs a residual chunk; combined with ACT chunk for the final action (mirrors current SAC residual_mode).
   - Paper does (A). Existing HIL-SERL plumbing favors (B) since `observation.base_action` is already populated per-step (`actor.py:286-289`).
   - **Recommendation:** ship (A) in v1 (cleaner port, fewer touchpoints). Add `residual_mode=True` for (B) as a v2 config switch — wiring is the same shape as SAC's residual_mode and the `_load_base_policy` code copies verbatim from `modeling_sac.py:79-107`.

5. **Critic ensemble size $K$.**
   - Paper default for QC/QC-FQL: $K=2$.
   - lerobot SAC default: $K=2$ (`num_critics: 2`).
   - Match paper + lerobot. No question.

6. **Replay buffer schema.** QC needs `state, action, reward, next_state, terminal, mask, valid` per-step plus the ability to query an aligned $h$-step window. Existing lerobot `TransitionBuffer` stores tuples; we need an indexed buffer. Probably worth a dedicated `QCReplayBuffer` class in the plugin. **Question for lead:** acceptable to add a *plugin-local* replay buffer that the learner pulls from? Cleaner than editing lerobot's buffer.

---

## § Implementation order (T3 → T5)

1. **T3** scaffold: pyproject, `__init__.py`, empty config/modeling/processor with the registration decorator + smoke test that `make_policy(QCConfig())` succeeds and instantiates a `QCPolicy` (no real loss computation yet).
2. **T4** unit tests in plugin: `test_chunk_critic_shape`, `test_flow_actor_integrate`, `test_critic_loss_value_at_known_input`, `test_actor_loss_finite`, `test_chunk_queue_pop_and_flush`. Stub real losses.
3. **T5** real impl: flow-matching loss, distillation loss, critic loss with proper `γ^h` discount + `s_{t+h}` bootstrap, target EMA update, `_sample_best_of_n` for QC variant.
4. **T6 (lead)** lerobot core touchpoints (`actor.py +2 LoC`, `learner.py +40 LoC`, plugin-import wiring), JSON cfg template `sim_qc_chunk5_v1_train.json` mirroring `sim_residual_K_chunk10_v1_train.json`.
5. **T7 (lead)** integration tests: residual SAC config still parses; QC config trains 50 steps on CPU; chunk-flush smoke.
6. **T8 (lead, GPU-gated)** convergence smoke test on sim_assemble.

Hand-off to T3 is contingent on lead approval of the 6 open questions above.
