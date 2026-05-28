# Q-Chunking (QC) — Paper Notes & Reference-Repo Index

**Paper:** Li, Zhou, Levine — *Reinforcement Learning with Action Chunking* — arXiv 2507.07969v4 (NeurIPS 2025).
**Code (canonical):** https://github.com/ColinQiyangLi/qc (cloned at `~/scratch/qc/`).
**Built on:** [FQL](https://github.com/seohongpark/fql) (Park, Li, Levine 2025) + [RLPD](https://github.com/ikostrikov/rlpd) (Ball et al. 2023).
**Backend:** JAX / Flax (not PyTorch). All loss eqs are JAX impls — port to PyTorch.

Author: qc-researcher (team `qc`), 2026-05-20. Read-only research phase — do NOT touch lerobot/ or lerobot_policy_sarm/ sources yet.

---

## § Algorithm summary

### Setting
Offline-to-online RL. MDP $(\mathcal S, \mathcal A, \rho, T, r, \gamma)$. Prior offline dataset $\mathcal D$ of transitions. Goal: pretrain on $\mathcal D$, then continue online with environment interaction. Sparse, long-horizon manipulation is the target regime.

### Core idea
Run actor-critic RL on a **temporally extended action space** of length $h$ (action chunk):
- **QC policy** $\pi_\psi(\mathbf a_{t:t+h} \mid s_t)$ outputs an *$h$-step chunk*.
- **QC critic** $Q_\theta(s_t, \mathbf a_{t:t+h})$ takes a state plus the *full chunk* as input.

This buys two wins simultaneously:

1. **Unbiased $n$-step backups** (Prop. A.1). Standard $n$-step TD with $n>1$ has off-policy bias because $r_{t+1\dots t+n-1}$ depends on the *behavior* policy, not the current policy. QC dodges this because the critic *takes the whole chunk as input*, so the reward sum and the bootstrap target match the same chunk policy exactly.
2. **Temporally-coherent exploration**. Chunking forces the policy to commit to coherent multi-step intentions instead of jittery 1-step actions, which is critical when offline data is non-Markovian (human teleop, mixture-of-skills). Empirically: QC trajectories show fewer pauses, more diverse end-effector coverage than 1-step baselines (Fig. 5).

### Critic loss (Eq. 4, 11)
$$L(\theta) = \mathbb E\Big[\big(Q_\theta(s_t,\mathbf a_{t:t+h}) - \underbrace{\sum_{t'=0}^{h-1}\gamma^{t'}r_{t+t'}}_{R^h_t}-\gamma^h Q_{\bar\theta}(s_{t+h},\mathbf a^\star_{t+h:t+2h})\big)^2\Big]$$
- $\bar\theta$ = EMA target.
- $\gamma^h$ discount (chunk-level) — NOT $\gamma$.
- $\mathbf a^\star_{t+h:t+2h}$ comes from $\pi_\psi(\cdot|s_{t+h})$ — best-of-$N$ in QC, deterministic distill in QC-FQL.

### Policy loss
Two implementations:

**QC (best-of-$N$, implicit KL).** Eq. 9 + 10. Sample $N$ chunks $\{\mathbf a^i\}$ from a separately-trained flow-matching BC policy $f_\xi(\cdot|s)$; pick $\mathbf a^\star = \arg\max_i Q(s,\mathbf a^i)$. KL constraint $D_{\text{KL}}(\pi_\psi\|f_\xi) \le \varepsilon$ enforced *implicitly* via the closed-form best-of-$N$ KL bound $\log N - (N-1)/N$. No explicit policy weights — $\pi_\psi$ is the implicit policy defined by max over BC samples.

**QC-FQL (explicit $W_2$, distill).** Eq. 13–15. A noise-conditioned one-step policy $\mu_\psi(s,z): \mathcal S \times \mathbb R^{Ah}\to\mathbb R^{Ah}$ outputs the full chunk in one forward pass. Trained with:
- A distillation loss (BC) that upper-bounds $W_2(\pi_\psi,f_\xi)$: $\|\mathbf z^1 - \mu_\psi(s,\mathbf z^0)\|_2^2$ where $\mathbf z^1$ is the result of Euler-integrating $f_\xi$ from noise $\mathbf z^0$.
- A Q-maximization term $-Q_\theta(s,\mu_\psi(s,\mathbf z^0))$.
- Total: $L(\psi) = \alpha\cdot\text{distill} - Q$. $\alpha$ controls regularization strength.

**Behavior policy $f_\xi$** — flow-matching velocity field (rectified flow). Trained with standard FM loss (Eq. 22): $\|f_\xi(s, u\mathbf a + (1-u)\mathbf z, u) - (\mathbf a - \mathbf z)\|_2^2$, $u\sim U(0,1)$, $\mathbf z\sim \mathcal N$. Integrated with $T=10$ Euler steps to sample.

### Algorithm 1 (QC training loop, paper p.19)
```
For every env step t do
    if t mod h == 0:           # only sample a new chunk every h steps
        sample N chunks a^1..a^N ~ f_ξ(·|s_t)
        a*_{t:t+h} = argmax_i Q_θ(s_t, a^i)
    act with a*_t and receive (s_{t+1}, r_t)
    D ← D ∪ {(s_t, a_t, s_{t+1}, r_t)}
    update f_ξ via flow-matching loss
    update Q_θ via Eq. 11
```
Same code path for offline (skip env interaction) and online (with interaction). Behavior cloning policy is updated continually — even online — which is the *opposite* of frozen-skill HRL.

### Why it works (Sec. 4, ablations)
- **Action chunk size**: $h=5$ is the sweet spot (Fig. 6). $h=25$ trains faster early but plateaus at the same asymptote. $h=50$ fails entirely (policy must predict 250-d action vector at every step → too non-Markov for the network).
- **Critic ensemble**: $K=2$ is enough. $K=10$ marginally better, $5\times$ slower.
- **UTD ratio**: 1 is fine. UTD=5 doesn't help (Fig. 6 right).
- **Flow > Gaussian.** Gaussian policies on a chunked action space fail catastrophically (Fig. 2). The mixture-of-skills / non-Markovian nature of offline data requires an expressive BC distribution — flow matching captures it; a unimodal Gaussian collapses.

---

## § Hyperparameters (Table 3 + 4 + 5 + 7)

| Param | Symbol | Value |
|---|---|---|
| Batch size | $M$ | 256 |
| Discount | $\gamma$ | 0.99 |
| Optimizer | – | Adam |
| Learning rate | – | $3\times 10^{-4}$ |
| Target update rate | $\tau$ | $5\times 10^{-3}$ (Polyak EMA) |
| Critic ensemble | $K$ | **2** (QC, QC-FQL, BFN) — paper default for chunked methods; **10** for RLPD baselines |
| UTD | – | 1 (paper: ≥5 doesn't help QC) |
| Flow integration steps | $T$ | 10 (Euler) |
| Offline steps | – | $10^6$ |
| Online steps | – | $10^6$ |
| Net width × depth | – | 512 × 4 |
| **Chunk length** | $h$ | **5** default; ablated 1,5,10,15,25,50; $h=10$ peak; $h=50$ collapse |
| BC reg coef | $\alpha$ | QC-FQL: 100–300 OGBench; 10000 robomimic. QC-RLPD: 0.01. |
| Best-of-$N$ samples | $N$ | **QC**: 32. BFN: 4. Tuned per domain (Table 5). |
| Init temperature (RLPD) | – | 1.0; target entropy multiplier 0.5 |
| Behavior policy | $f_\xi$ | Flow matching velocity field, 4-layer MLP width 512, no LN |
| Param count | – | QC ≈ 4.2M, QC-FQL ≈ 5.0M, RLPD ≈ 17.2M (uses $K=10$) |

Reward / mask convention (matches code):
- Rewards: 0 on success, $-1$ otherwise (sparse). Robomimic ${\to}$ shifted by $-1$ to match.
- `masks = 1 - terminated` (no bootstrap if terminated).
- Per-chunk cumulative reward = $\sum_{t'=0}^{h-1}\gamma^{t'} r_{t+t'}$; vectorized in `Dataset.sample_sequence`.

---

## § Implementation index (paper concept → code)

Repo root: `~/scratch/qc/`.

| Paper § / concept | File:Class.method | Notes |
|---|---|---|
| **§4.1** — Q-chunking policy $\pi_\psi(\mathbf a_{t:t+h}\|s)$ | `agents/acfql.py::ACFQLAgent` (QC-FQL); `agents/acrlpd.py::ACRLPDAgent` (QC-RLPD/SAC variant) | QC-FQL is the production variant for Q-chunking; QC-RLPD is the SAC port |
| **§4.1** — Q-chunking critic $Q_\theta(s, \mathbf a_{t:t+h})$ | `acfql.py::ACFQLAgent.critic_loss` lines 22-52 | flattens chunk via `jnp.reshape(actions,(B,-1))` — critic just takes the concatenated $A\cdot h$ vector |
| **Eq. 4 / Eq. 11** — chunk TD loss | `acfql.py::ACFQLAgent.critic_loss` lines 40-45 | `target_q = R^h + γ^h · mask · Q_target(s_{t+h}, π(s_{t+h}))` |
| **§4.2 / Eq. 22** — flow-matching BC loss for $f_\xi$ | `acfql.py::ACFQLAgent.actor_loss` lines 60-81 | `x_t = (1-t)*x_0 + t*x_1; vel = x_1 - x_0; pred = actor_bc_flow(...)` |
| **Eq. 13** — distillation loss for one-step policy $\mu_\psi$ | `acfql.py::ACFQLAgent.actor_loss` lines 83-99 | `distill_loss = MSE(actor_actions, target_flow_actions)` |
| **Eq. 14 + Q max** — actor Q loss | `acfql.py::ACFQLAgent.actor_loss` lines 91-99 | `q_loss = -Q(s, μ_ψ(s,z)).mean()` |
| **Algorithm 3** — Euler integration of flow ODE | `acfql.py::ACFQLAgent.compute_flow_actions` lines 208-223 | 10-step Euler from $z\sim\mathcal N$ |
| **Algo 1 line "best-of-N"** — chunk sampling at action time | `acfql.py::ACFQLAgent.sample_actions` lines 180-205 | `actor_type=="best-of-n"`: sample $N$ chunks via flow, pick argmax-$Q$; `"distill-ddpg"` is QC-FQL deterministic |
| Flow-matching velocity field $f_\xi$ | `utils/networks.py::ActorVectorField` (state-of-art-style MLP with optional Fourier time embedding) | 4×512 MLP, output dim = $A\cdot h$ |
| One-step distilled policy $\mu_\psi(s,z)$ | `utils/networks.py::ActorVectorField` (reused) | Same arch as $f_\xi$ but no `times` input |
| Ensemble of critics | `utils/networks.py::Value` + `ensemblize` | `nn.vmap` over $K$ copies |
| **§4.3** — RLPD/SAC chunked variant (Tanh-Gaussian actor) | `agents/acrlpd.py::ACRLPDAgent` | Used for QC-RLPD only; SAC-style entropy-tuned actor + critic ensemble (default $K=10$). Has explicit `bc_alpha` (Eq. 29) |
| Tanh-squashed Gaussian | `rlpd_distributions/tanh_normal.py` | Vendored from RLPD repo |
| Target net EMA | `acfql.py::ACFQLAgent.target_update` lines 129-136 | `θ_target ← τ·θ + (1-τ)·θ_target` |
| Chunk-aware dataset sampling | `utils/datasets.py::Dataset.sample_sequence` lines 92-154 | computes `valid` mask + per-chunk cumulative reward; pads at episode ends; key shape: `actions=(B,h,A)`, `rewards=(B,h)`, `valid=(B,h)`, `next_observations=(B,h,…)` |
| Replay buffer | `utils/datasets.py::ReplayBuffer` lines 177-237 | extends `Dataset`, supports `add_transition`; same `sample_sequence` interface |
| Offline + online training loop | `main.py` lines 67-330 | for QC (offline + online); single training step is `agent.update(batch)` |
| Online-only training loop | `main_online.py` lines 67-307 | for RLPD-style from-scratch online RL (QC-RLPD) |
| Open-loop chunk execution (env interaction) | `main.py` lines 211-231 | `action_queue` is filled once per chunk; popped one action per env-step; flushed on episode reset |
| Eval rollout | `evaluation.py::evaluate` | "during eval, the action chunk is executed fully" (comment, main.py:193) — same queue pattern |
| Config defaults (QC-FQL) | `agents/acfql.py::get_config` lines 315-345 | the canonical hyperparameter set |
| Config defaults (QC-RLPD) | `agents/acrlpd.py::get_config` lines 237-260 | |
| Agent registry | `agents/__init__.py` | `agents = {'acfql': ACFQLAgent, 'acrlpd': ACRLPDAgent}` |

---

## § Distinctions from SAC (lerobot's current RL policy)

| Aspect | SAC (`lerobot/policies/sac/modeling_sac.py`) | QC / QC-FQL |
|---|---|---|
| Policy output | $a_t \in \mathbb R^A$ (single action) | $\mathbf a_{t:t+h} \in \mathbb R^{Ah}$ (chunk) |
| Policy parameterization | Tanh-Gaussian | Flow-matching $f_\xi$ + one-step distillation $\mu_\psi$ (QC-FQL), OR best-of-$N$ from $f_\xi$ (QC) |
| Critic input | $(s, a) \in \mathcal S \times \mathcal A$ | $(s, \mathbf a) \in \mathcal S \times \mathcal A^h$ — flattened chunk concatenated to state |
| TD discount | $\gamma$ | $\gamma^h$ (chunk-level discount — `discount**horizon_length` in code) |
| Reward target | $r_t + \gamma\,Q_\bar\theta(s_{t+1},\pi(s_{t+1}))$ | $R^h_t + \gamma^h\,Q_\bar\theta(s_{t+h},\pi(s_{t+h}))$, $R^h_t = \sum_{t'=0}^{h-1}\gamma^{t'}r_{t+t'}$ |
| Bootstrap state offset | $s_{t+1}$ | $s_{t+h}$ |
| Entropy regularization | yes (auto-tuned $\alpha$) | **no** (QC, QC-FQL) — replaced by behavior-cloning constraint (KL or $W_2$); QC-RLPD keeps SAC entropy + adds BC term |
| Env-step / policy-step ratio | 1:1 | $h$:1 (one policy call → $h$ env steps via FIFO queue) |
| BC term in actor loss | optional (offline RL) | central. $\alpha\cdot\|\mu_\psi - \text{FlowEuler}(f_\xi)\|^2$ in QC-FQL; implicit via best-of-$N$ in QC |
| Target networks | yes, EMA $\tau$ | same |
| Critic ensemble size | 2 (lerobot default) | 2 (QC/QC-FQL); 10 (QC-RLPD/RLPD); paper says K=10 helps marginally but cost is high |
| Episodic vs continuous | continuous | continuous (offline-to-online); no episodic chunk-restart semantics in the algo, but the env loop *does* flush the chunk queue on `done` |

**Key port subtleties:**
- The "n-step backup" in QC is NOT the usual biased $\sum\gamma^{t'}r$ + $Q(s_{t+n},a_{t+n})$ ; it's an *unbiased* $h$-step return because the critic sees the chunk that produced those rewards. Don't accidentally implement biased $n$-step (Prop. A.1 is the whole point).
- "Action chunk" in QC = "predict + roll out $h$ actions open-loop". In lerobot ACT/diffusion-policy terminology this matches `predict_action_chunk`. In SAC terminology there is *no* chunking — `select_action` returns one action and `predict_action_chunk` raises `NotImplementedError` (modeling_sac.py:132-134).
- Best-of-$N$ chunk selection happens at **every chunk boundary**, not every env step. Cost: $N$ flow-Euler integrations per chunk (32 × 10 steps = 320 forward passes); ~3-4× wall time of QC-FQL per online step (Fig. 17: 15.07 ms vs 11.33 ms).

---

## § HIL-SERL hook points (speculative, to firm up in T2)

HIL-SERL = lerobot's actor-learner setup. Current SAC residual mode (modeling_sac.py + rl/actor.py) supports: (a) frozen ACT base + learned residual SAC actor, (b) SARM reward model (lerobot_policy_sarm) producing stage-graded rewards, (c) human intervention transitions injected into the replay buffer with `is_intervention` flags.

Mapping to QC:

| HIL-SERL piece | Current SAC integration | Expected QC integration |
|---|---|---|
| **Frozen ACT base** | `SACPolicy._load_base_policy` (modeling_sac.py:79-107) loads ACT, freezes, attaches `_pre_processor` | QC policy can do the same: load base ACT, freeze, predict chunk-of-$h$, use as "behavior prior". **Open question**: does $f_\xi$ replace ACT (learned from scratch), or do we *condition* $f_\xi$ on a frozen ACT chunk as a residual? The paper trains $f_\xi$ from scratch via flow matching — but for HIL-SERL we already have ACT; reusing it as $f_\xi$ skips offline pre-training. |
| **Residual mode (base + residual)** | `select_action` (modeling_sac.py:137-168): `action = clip(base_action + residual_scale * sac_residual)` | Two options: (i) QC outputs a *full chunk*, ignores ACT (clean port). (ii) QC outputs a *residual chunk*, added to ACT's chunk (mirrors current residual_mode). Probably (i) for cleanliness, but (ii) is a smaller break from current actor loop. |
| **SARM stage rewards** | `lerobot_policy_sarm.processor_sarm.SARMEncodingProcessorStep` injects per-step reward via processor pipeline | Should be a *no-op change*. The reward stream is per-env-step. QC's `Dataset.sample_sequence` already aggregates $h$ rewards via $\gamma$-discount. SARM produces 1 scalar per env-step; QC consumes $h$ scalars per training transition. Should compose. |
| **Human-intervention transitions** | `is_intervention` mask on transitions; HIL-SERL prioritizes intervention chunks in actor loss as BC anchors | This **is** the QC BC term, structurally. In QC, all offline data acts as the BC target via $f_\xi$. Intervention transitions become BC anchors with extra weight — exactly the role of the offline dataset in QC. |
| **`actor.py` env loop** | one `policy.select_action(obs)` per env-step (rl/actor.py:331) | Open-loop chunk: one `policy.select_action` (returns chunk) every $h$ env-steps. Need a chunk-queue identical to QC's `main.py:211-231`. **This is the biggest actor.py touch.** ACT already has its own chunk queue in residual mode — pattern is partly there. |
| **Triangle / Cross / stage advance** | These are SARM-level (per-step) and don't intersect QC's chunking | Mostly orthogonal. Stage transitions happen within a chunk; the critic still operates on the full $h$ rewards as a sum. |
| **Learner-side updates** | `learner.py` consumes transitions, does SAC critic + actor + temp updates | QC has 2 (QC) or 3 (QC-FQL) loss terms instead of SAC's 3 (critic/actor/temp). No temperature parameter in QC; instead $\alpha$ (BC coef) is a fixed config. |
| **gRPC parameter sync** | Pushes actor weights every N seconds | QC has 2-3 networks ($f_\xi$, $\mu_\psi$, $Q_\theta$). Need to push $\mu_\psi$ (acting policy, QC-FQL) or $f_\xi$+$Q_\theta$ (QC best-of-$N$). |

**Top design unknowns for T2:**
1. **QC variant**: QC-FQL (deterministic distilled actor, ~3× faster online) vs QC best-of-$N$ (better sample efficiency in paper). HIL-SERL needs real-time inference — QC-FQL wins on latency. Recommendation: **start with QC-FQL** as the first port.
2. **Action-chunk env-step semantics**: paper's `main.py` *executes the chunk fully* (no replanning). Real HIL-SERL has stochastic perturbations + human interventions mid-chunk. Need a story for: (a) chunk-flush on intervention, (b) chunk-flush on stage advance (?), (c) MPC-style chunk re-sampling vs full-open-loop.
3. **ACT-as-$f_\xi$ vs train-$f_\xi$-from-scratch**: lerobot already has ACT trained on demos. Letting QC consume the existing ACT chunk-distribution skips most offline pre-training cost. But ACT is a transformer/decoder not a flow field — would need a wrapper or distillation step.
4. **Discrete gripper action**: SAC modeling has a separate `discrete_critic` for the gripper. QC paper doesn't handle this — would need a hybrid actor.

---

## § Quick recap of evaluation results (Table 1 + Fig. 3)

OGBench long-horizon sparse tasks, 25 tasks aggregated, online success rate after 1M offline + 1M online:

- **QC: 86%** (best by 27pt on hardest domain cube-quadruple — 73% vs ~14% for FQL).
- QC-FQL: 86% (statistically tied with QC; faster online).
- QC-IFQL: 59%.
- FQL (1-step baseline): 67%.
- BFN (1-step best-of-N, no chunking): 63%.
- RLPD: 67%.
- RLPD-AC (chunked Gaussian actor): 61% (worse than non-chunked RLPD! Gaussian + chunking is broken without flow).
- IQL (offline): 12%.

Robomimic (lift/can/square): QC dominates from offline phase. Square (precision insertion) is the differentiator: QC ~90% vs FQL/BFN ~30% at 1M online steps.

**Take-home**: QC-FQL is the production sweet-spot (matches QC accuracy, faster). The unambiguous lift in performance comes from (1) chunked critic, (2) flow-matching BC, (3) the unbiased $n$-step backup — in that order.
