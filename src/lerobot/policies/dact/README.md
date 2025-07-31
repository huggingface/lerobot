# Summary: Hybridizing ACT and Action Diffusion for Chunked Style-Conditioned Action Generation

## Goal
Combine the structured chunking + style latent of ACT with the expressive iterative refinement of action diffusion to get:
- **Smooth**, **mode-aware**, **high-fidelity** action chunks.
- **Controlled diversity** via an explicit style latent \( z \) that is context-aligned.
- **Baselines** for comparison: (a) diffusion conditioned only on observation (like ACT’s inference with \( z=0 \)), (b) deterministic style mapping \( z=f(\text{obs}) \), and (c) full hybrid with learned conditional prior \( p(z|\text{obs}) \) feeding a diffusion head plus temporal ensembling.

## Background (for LLM)

### ACT (Action Chunking with Transformers)
- Predicts overlapping action chunks.
- Uses a CVAE: encoder \( q(z|a,\text{obs}) \) during training to infer chunk-level style latent \( z \) from ground-truth chunk \( a \) and current observation.
- Decoder/policy takes fixed \( z=\mathbf{0} \) (mean of prior) at inference and conditions on current observation (images + joint states) to output chunks. Temporal ensembling smooths overlapping chunks.

### Action Diffusion
- Models target action chunks via a diffusion (forward noise + learned reverse denoising) process.
- Conditioning is typically **direct**: the denoiser receives noisy chunk \( a_t \) and context embeddings (history, observation, goal) without an explicit latent \( z \).  
- Sampling is iterative: starting from noise, reverse diffusion produces chunk.

## Hybrid Design Variants

### Shared Infrastructure
- **Chunking & Ensembling:** Same as ACT — overlapping chunks predicted and fused to get per-timestep actions.  
- **Observation:** Can include current proprioception, images, short history if needed.  
- **Style Latent \( z \):** Chunk-level latent capturing modality/style of an action chunk.

---

### Variant A: Diffusion conditioned only on observation (ACT-style inference)
- **No explicit style latent at inference**: set \( z=\mathbf{0} \) or omit \( z \) entirely.
- **Denoiser input:** noisy chunk \( a_t \), timestep \( t \), and observation embedding.  
- **Training:** Standard conditional diffusion loss; no latent / prior machinery.  
- **Inference:** Run reverse diffusion given observation to get chunk; ensemble overlaps.

**Purpose:** Baseline to measure what expressive diffusion + chunking/ensembling gives without explicit style abstraction.

---

### Variant B: Deterministic style latent \( z = f(\text{obs}) \)
- **Latent:** Single deterministic mapping from observation to style code. No posterior/prior stochasticity.  
- **Denoiser conditioning:** \( z=f(\text{obs}) \) plus observation (optionally).  
- **Training:** Learn \( f \) jointly so that diffusion conditioned on \( z=f(\text{obs}) \) reconstructs true chunks.  
- **Inference:** Compute \( z=f(\text{obs}) \) and run diffusion.

**Purpose:** Intermediate variant; introduces explicit style abstraction without stochastic sampling. Tests whether representing chunk style separately helps.

---

### Variant C: Full hybrid — Style-conditioned diffusion with conditional prior (recommended core innovation)
- **Posterior encoder** \( q(z|a,\text{obs}) \): infers latent from ground-truth chunk and observation during training.  
- **Conditional prior** \( p(z|\text{obs}) \): predicts distribution over \( z \) given observation alone; trained via KL to match posterior.  
- **Diffusion denoiser:** iterative denoising of action chunk \( a_t \), conditioned on both \( z \) (sampled from posterior during training, from prior at inference) and the current observation.  
- **Temporal ensembling:** Overlapping chunk fusion as in ACT.

**Training loss:**
\[
\mathcal{L} = \mathbb{E}_{a_0,\text{obs}} \Big[
\mathbb{E}_{t, z \sim q(z|a_0,\text{obs}), \epsilon} \left\| \epsilon - \epsilon_\theta(a_t, t \mid z, \text{obs}) \right\|^2
+ \beta\, \mathrm{KL}(q(z \mid a_0,\text{obs}) \,\|\, p(z \mid \text{obs}))
\Big]
\]
- \( \beta \) can be annealed or use free bits to avoid posterior collapse.

**Inference:**
1. Given observation, sample \( z \sim p(z|\text{obs}) \).  
2. Run reverse diffusion conditioned on \( z \) and observation.  
3. Temporal ensemble overlapping chunks.

**Purpose:** Structured, context-aware multimodal generation with controlled diversity via \( z \).

---

## Architectural Responsibilities (for LLM to implement)

1. **Encoder / Posterior network**  
   - Input: ground-truth chunk \( a \), observation.  
   - Output: parameters of posterior Gaussian \( q(z|a,\text{obs}) \).

2. **Conditional prior network**  
   - Input: observation.  
   - Output: parameters of prior Gaussian \( p(z|\text{obs}) \).

3. **Style code sampling / reparameterization**  
   - Sample \( z \) from posterior during training, from prior during inference.  

4. **Diffusion denoiser (head)**  
   - Inputs: noisy chunk \( a_t \), timestep \( t \), style latent \( z \), observation.  
   - Output: predicted noise \( \epsilon_\theta \) for denoising objective.  
   - Fusion: keep \( z \) and observation conditioning distinct (e.g., \( z \) via FiLM or global bias, observation via cross-attention or embedding concatenation).

5. **Chunking + Temporal Ensembler**  
   - Manage overlapping chunk outputs, blend for per-timestep control signal.

6. **Training loop variants**  
   - Variant A: only diffusion loss.  
   - Variant B: deterministic \( z=f(\text{obs}) \) (no KL).  
   - Variant C: diffusion + KL alignment with conditional prior.

7. **Inference procedures** for all three variants.

---

## Implementation Plan

### Step 1: Baselines
- Implement Variant A (observation-only diffusion + ensembling).  
- Implement ACT-style chunking integration if not already shared.  
- Validate quality & smoothness; collect metrics.

### Step 2: Deterministic style latent (Variant B)
- Add a small network \( f(\text{obs}) \) producing \( z \).  
- Condition diffusion on \( z \) (and optionally obs again).  
- Train; compare to Variant A.

### Step 3: Full hybrid (Variant C)
- Implement posterior encoder \( q \) and conditional prior \( p \).  
- Add KL loss with annealing.  
- Plug \( z \) into diffusion head as described.  
- At inference sample \( z \sim p(z|\text{obs}) \) and decode.

### Step 4: Ablation & comparison
- Evaluate: ACT original (CVAE + deterministic \( z=0 \)), Variant A, B, C.  
- Metrics: chunk fidelity, smoothness (e.g., action difference norms), success on downstream task (e.g., docking), diversity/control via \( z \), sample efficiency.

---

## Training / Inference Pseudocode (condensed)

```python
# === Training for Variant C ===
for (obs, true_chunk) in data:
    # Posterior and prior
    q_mu, q_sigma = posterior_encoder(true_chunk, obs)
    p_mu, p_sigma = prior_network(obs)
    z = sample(q_mu, q_sigma)  # reparameterize

    # Diffusion forward noising
    t = sample_timestep()
    a_t, noise = forward_diffusion(true_chunk, t)

    # Denoiser prediction conditioned on z and obs
    pred_noise = diffusion_denoiser(a_t, t, z, obs)

    diff_loss = mse(pred_noise, noise)
    kl = KL(q_mu, q_sigma, p_mu, p_sigma)
    loss = diff_loss + beta * kl

    optimize(loss)

# === Inference for Variant C ===
z = sample_from_prior(obs)             # z ~ p(z|obs)
a_T = noise_init()
chunk = reverse_diffusion(a_T, z, obs) # iterative denoising
final_action = temporal_ensemble_overlapping_chunks(chunk)
