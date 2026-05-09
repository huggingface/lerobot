# ResFiT paper notes (2509.19301v2)

ref impl: `../residual-offpolicy-rl/resfit/`

## Core idea

Pretrain BC π_base on demos. **Freeze**. Learn small residual π_θ(s, a_b) via off-policy RL. Full action a = a_b + π_θ(s, a_b). Critic Q(s, a) sees full action.

Action-chunking-agnostic. Single-step residual on top of chunked base.

## Algorithm 1 (recap)

```
init residual policy π_θ, Q ensemble Q_φ_1..N, encoder f_ω, empty D_online
warmup: for k steps: a_t = ε + a_b_t (ε ~ U(-noise, noise)). store (s, a_b, a, s', a_b', r, d).
loop:
  a_b ~ π_base(s); a_r ~ π_θ(s, a_b); a = a_b + a_r
  step env. store transition.
  UTD times:
    sample B 50/50 offline∪online
    a' = a_b' + π_θ'(s', a_b')
    y = r + γ(1-d) min_subset Q_φ'_i(s', a')
    update Q_φ_i (MSE)
    polyak target φ' ← ρφ' + (1-ρ)φ
  actor: max E[Q_φ(s, a_b + π_θ(s, a_b))] (mean ensemble)
  polyak θ' (TD3 delayed if cfg)
```

## Crucial design decisions (ablations confirmed)

| design | value/notes | why |
|---|---|---|
| **off-policy** (vs PPO) | DDPG-style, det actor + Gaussian noise | 200x sample-eff vs PPO |
| **UTD>1** | 4-8 (saturates) | sample efficiency under sparse reward |
| **n-step returns** | n=3 typical | sparse reward, long horizon |
| **LayerNorm in critic MLP** | YES — paper says **crucial** | mitigates Q over-estimation w/o explicit constraint |
| **Polyak τ** | 0.005 | TD3-like |
| **Target action smoothing** | gaussian noise on target action | TD3 stability |
| **Delayed actor updates** | 2-8 critic updates per actor | TD3 stability |
| **REDQ subset min** | random 2 from N=10 ens | reduce over-estimation bias |
| **DrQ image aug** | random shift pad=4 | overfit prevention |
| **Symmetric sampling** | 50% offline demos + 50% online buffer per batch | demos seed critic + stability |
| **Sparse binary reward** | 0/1 | works for high-DoF |
| **Action squash** | tanh(mu) * action_scale, then clamp(a_b+a_r, -1, 1) | bound residual magnitude |
| **action_scale (residual)** | small (paper: ~0.1-0.3 in normed [-1,1]) | safety + stability |
| **demos in BOTH phases** | freeze offline, mix online | critic generalization |
| **encoder** | shallow ViT + DrQ aug | vision side; can sub in ResNet |

## Residual MDP math

Standard: Q*(s, a) = E[r + γ max_a' Q*(s', a')]

Residual reparam: π_θ(s, a_b) → action delta. Full a = a_b + π_θ(s, a_b).

Loss critic:
```
L(φ) = E[(Q_φ(s,a) - (r + γ(1-d) Q_φ'(s', a_b' + π_θ(s', a_b'))))²]
```

Loss actor (det policy gradient):
```
L(θ) = -E[Q_φ(s, a_b + π_θ(s, a_b))]
```

## Real-world results (relevant to us)

ACT base, 1000 demos. WoollyBallPnP: BC=14% → ResFiT=64%. PackageHandover: BC=23% → 64%. ~15min real-world RL data.

For our task (5D action, sim, ACT v11 base @ 40% thr=0.95): we expect similar boost into 60-70%+.

## Key takeaways for our use-case

| our setup | adaptation |
|---|---|
| 5D action (delta_xyz_yaw + gripper) | residual outputs same 5D, scale=0.1 (ie ~10% of normed range) |
| ACT chunked base, n_action_steps=10 | use ACT's `select_action` in actor process — it manages chunk queue. Store a_b at sample time in transition. |
| SARM dense reward (not 0/1) | works directly — paper notes binary reward sufficed but dense is fine. Cap target Q via reward range maybe (clip 0..1). |
| 76 demos available (destale_tail30) | use as offline buffer with 50/50 sampling |
| sim env (gym_manipulator), CPU constrained | non-headless OK; UTD=1-2 to start (paper sweep showed UTD=4 ok but UTD=1 also passes) |
| existing hil-serl SAC infra | extend SACPolicy + actor.py + learner.py minimally. Flag-toggle. |
| **interventions (we have, paper doesn't)** | covered separately in design md |
