# Hybridizing ACT and Action Diffusion for Chunked Style-Conditioned Action Generation

## Goal

Combine the structured chunking and style latent of ACT with the expressive iterative refinement of action diffusion to achieve:

- **Smooth**, **mode-aware**, **high-fidelity** action chunks.
- **Controlled diversity** via an explicit style latent $z$ that is context-aligned.
- **Baselines** for comparison:
  - (A) Diffusion conditioned only on observation (like ACT’s inference with $z = 0$),
  - (B) Deterministic style mapping $z = f(\text{obs})$,
  - (C) Full hybrid with learned conditional prior $p(z|\text{obs})$ feeding a diffusion head plus temporal ensembling.

---

## Background (for LLM)

### ACT (Action Chunking with Transformers)

- Predicts overlapping action chunks.
- Uses a CVAE:
  - Encoder: $q(z \mid a, \text{obs})$ infers chunk-level style latent $z$ from ground-truth chunk $a$ and current observation.
  - Decoder: takes fixed $z = \mathbf{0}$ (mean of prior) at inference, conditions on current observation (images + joint states) to output action chunks.
- **Temporal ensembling** smooths overlapping chunks into per-timestep actions.

### Action Diffusion

- Models target action chunks via a diffusion process (forward noise + learned reverse denoising).
- Conditioning is typically **direct**: the denoiser receives:
  - Noisy chunk $a_t$,
  - Timestep $t$,
  - Context embeddings (history, observation, goal),
  - **No explicit latent $z$**.
- Sampling is **iterative**: starting from noise, reverse diffusion produces an action chunk.

---

## Hybrid Design Variants

### Shared Infrastructure

- **Chunking & Ensembling**: Same as ACT — overlapping chunks predicted and fused.
- **Observation**: Can include proprioception, images, short history.
- **Style Latent $z$**: Chunk-level latent capturing modality/style.

---

### Variant A: Diffusion Conditioned Only on Observation (ACT-style inference)

- **No explicit $z$ at inference**: set $z = \mathbf{0}$ or omit entirely.
- **Denoiser input**: noisy chunk $a_t$, timestep $t$, observation embedding.
- **Training**: Standard conditional diffusion loss.
- **Inference**: Run reverse diffusion given observation.

**Purpose**: Baseline to assess effect of diffusion + chunking/ensembling without latent style.

---

### Variant B: Deterministic Style Latent ($z = f(\text{obs})$)

- **Latent**: Deterministic mapping from observation to style code.
- **Denoiser conditioning**: $z = f(\text{obs})$, optionally add observation embedding.
- **Training**: Learn $f$ jointly with diffusion denoiser.
- **Inference**: Compute $z = f(\text{obs})$ and run reverse diffusion.

**Purpose**: Tests whether explicit style abstraction helps, without posterior/prior sampling.

---

### Variant C: Full Hybrid (Style-Conditioned Diffusion + Conditional Prior)

**Training Components**:

- **Posterior encoder**: $q(z \mid a, \text{obs})$
- **Conditional prior**: $p(z \mid \text{obs})$
- **Diffusion denoiser**: conditioned on $z$ and $\text{obs}$

**Loss Function**:

$\mathcal{L} = \mathbb{E}_{a_0, \text{obs}} \left[\mathbb{E}_{t, z \sim q(z \mid a_0, \text{obs}), \epsilon} \left\| \epsilon - \epsilon_\theta(a_t, t \mid z, \text{obs}) \right\|^2 + \beta \cdot \mathrm{KL}(q(z \mid a_0, \text{obs}) \, \| \, p(z \mid \text{obs})) \right]$

- $\beta$ can be annealed or use free-bits to avoid posterior collapse.

**Inference Procedure**:

1. Given observation, sample $z \sim p(z \mid \text{obs})$
2. Run reverse diffusion conditioned on $z$ and observation
3. Temporal ensemble overlapping chunks

**Purpose**: Structured, context-aware, multimodal generation with controlled diversity via $z$.

---

## Architectural Responsibilities

1. **Encoder / Posterior network**
   - Input: ground-truth chunk $a$, observation
   - Output: parameters of $q(z \mid a, \text{obs})$

2. **Conditional prior network**
   - Input: observation
   - Output: parameters of $p(z \mid \text{obs})$

3. **Style code sampling / reparameterization**
   - Sample $z$ from posterior (training) or prior (inference)

4. **Diffusion denoiser (head)**
   - Inputs: $a_t$, $t$, $z$, observation
   - Output: predicted noise $\epsilon_\theta$
   - Conditioning:
     - $z$: via FiLM or additive bias
     - Observation: via cross-attention or token embedding

5. **Chunking + Temporal Ensembler**
   - Blend overlapping chunks into smooth per-timestep actions

6. **Training Loop Variants**
   - Variant A: Only diffusion loss
   - Variant B: Deterministic $z = f(\text{obs})$
   - Variant C: Diffusion + KL loss

7. **Inference Procedures**
   - Implement for A, B, and C.

---

## Implementation Plan

### Step 1: Baselines

- Implement Variant A (observation-only diffusion)
- Integrate ACT-style chunking
- Collect metrics (smoothness, fidelity)

### Step 2: Deterministic Style Latent (Variant B)

- Add network $f(\text{obs}) \rightarrow z$
- Train and evaluate against Variant A

### Step 3: Full Hybrid (Variant C)

- Implement encoder $q$ and prior $p$
- Add KL loss with annealing
- Sample $z \sim p(z \mid \text{obs})$ at inference

### Step 4: Ablation and Comparison

- Compare: ACT (original), Variant A, B, and C
- Metrics:
  - Chunk fidelity
  - Smoothness ($\| a_{t+1} - a_t \|$ norms)
  - Task success (e.g., docking)
  - Diversity via $z$
  - Sample efficiency

---

## Training / Inference Pseudocode

### Variant C — Training

```python
for obs, true_chunk in data:
    q_mu, q_sigma = posterior_encoder(true_chunk, obs)
    p_mu, p_sigma = prior_network(obs)
    z = sample(q_mu, q_sigma)  # reparameterization trick

    t = sample_timestep()
    a_t, noise = forward_diffusion(true_chunk, t)

    pred_noise = diffusion_denoiser(a_t, t, z, obs)

    diff_loss = mse(pred_noise, noise)
    kl_loss = kl_divergence(q_mu, q_sigma, p_mu, p_sigma)
    loss = diff_loss + beta * kl_loss

    loss.backward()
    optimizer.step()
