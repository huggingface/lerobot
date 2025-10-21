# Real-Time Action Chunking (RTC) for SmolVLA

This directory contains the implementation of **Real-Time Action Chunking** for SmolVLA policies, based on the paper:

> **Real-Time Execution of Action Chunking Flow Policies**
> [https://arxiv.org/pdf/2506.07339](https://arxiv.org/pdf/2506.07339)

---

## Overview

Real-Time Action Chunking (RTC) is a **training-free inference-time wrapper** that enables flow-matching VLA policies to execute actions with minimal latency while maintaining temporal consistency across chunk boundaries.

### The Problem

Action-chunking policies output a _chunk_ of H future actions `A_t = [a_t, ..., a_{t+H-1}]` per inference, and the controller executes only the first `s` actions before requesting a new chunk. While this improves temporal consistency, it introduces two critical challenges:

1. **Inference Latency**: If inference takes `d` controller ticks, the robot must either:
   - Pause and wait (causing jerky motion)
   - Switch to the next chunk at a boundary that can be discontinuous

2. **Temporal Inconsistency**: Traditional chunking can produce discontinuities at chunk boundaries, leading to non-smooth trajectories.

### The RTC Solution

RTC solves both problems by:

1. **Asynchronous Generation**: Starting the generation of the _next_ chunk while executing the _current_ one
2. **Guided Inpainting**: Constraining the next chunk to agree with the portion of the current chunk that will be executed during inference
3. **Soft Masking**: Using a differentiable mask `W` that transitions smoothly from full guidance to no guidance across the action horizon

#### Key Insight

Treat next-chunk generation as an **inpainting problem**:

- **Freeze** the first `d` steps to match already-committed actions (guidance weight = 1.0)
- Apply a **soft mask** in the overlap region `d ≤ i < H-s` that decays exponentially
- Use **no guidance** for the final `s` steps to maintain reactivity

This is realized through **guided denoising** using Vector-Jacobian Product (VJP) guidance with a clipped guidance weight.

---

## Algorithm Overview

RTC implements **Algorithm 1** from the paper using two asynchronous threads:

### Controller Thread (`GETACTION`)

Called every control period `Δt`:

- Returns the next action from the current chunk `A_cur`
- Provides the latest observation `o_t` to the inference thread

### Background Inference Thread (`INFERENCELOOP`)

Continuously running:

1. **Wait** until at least `s_min` actions have been executed since the last inference start
2. **Build** `A_prev` by dropping the `s` already-executed actions from `A_cur`: `A_prev = A_cur[s:H]`
3. **Estimate** the next delay `d` conservatively as `max(Q)` where `Q` is a history of observed delays
4. **Run** `GUIDEDINFERENCE` with guided inpainting:
   - Right-pad `A_prev` to length `H`
   - Construct the soft mask `W` (Equation 5 from the paper)
   - Iterate `n` denoising steps with ΠGDM guidance (Equations 2-4)
5. **Swap** to the new chunk `A_new` as soon as it's ready
6. **Record** the observed delay and add to buffer `Q`

---

## Technical Details

### Guided Denoising with VJP

The core innovation is **guided denoising** using the following equations from the paper:

#### Equation 1: Euler Integration Step

```
A ← A + (1/n) [v_π + w * guidance]
```

#### Equation 2: Guidance Computation via VJP

```
guidance = J^T [W ⊙ (A_prev - f(A))]
```

where `J = ∂f/∂A` is computed efficiently using PyTorch's VJP.

#### Equation 3: Denoising Function

```
f(A) = A + (1 - τ) v_π(A, o, τ)
```

#### Equation 4: Time-Dependent Scaling

```
r_τ² = (1 - τ)² / (τ² + (1 - τ)²)
```

#### Equation 5: Soft Mask

```
W_i = {
    1                           if i < d
    c_i * exp(c_i - 1) / (e-1)  if d ≤ i < H-s,  where c_i = (H-s-i)/(H-s-d+1)
    0                           if i ≥ H-s
}
```

### Guidance Weight Clipping

The guidance weight is clipped to prevent excessive guidance at extreme time steps:

```python
w = min(β, (1 - τ) / (τ * r_τ²))
```

---

## Worked Example

Let's walk through a concrete example with specific numbers:

### Setup

- **Chunk Horizon (H)**: `H = 10` actions
- **Min Execution Steps (s_min)**: `s_min = 4`
- **Inference Delay (d)**: `d = 3` ticks

### Cycle 1: The First Chunk

**T=0:** Initial inference generates `A_cur = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]`

**T=1-3:** Controller executes `a0`, `a1`, `a2` (s_count < s_min, waiting)

**T=4:**

- Controller executes `a3` → `s_count = 4 ≥ s_min`
- **Inference Starts!**
  - Build `A_prev = A_cur[4:10] = [a4, a5, a6, a7, a8, a9]`
  - Right-pad to length H: `Y = [a4, a5, a6, a7, a8, a9, ?, ?, ?, ?]`
  - Compute mask `W`:
    - `W[0:3] = [1, 1, 1]` (frozen, i < d)
    - `W[3:6] = [0.34, 0.18, 0.07]` (soft guidance, d ≤ i < H-s)
    - `W[6:10] = [0, 0, 0, 0]` (no guidance, i ≥ H-s)

**T=5-6:** Controller executes `a4`, `a5` (inference running in background)

**T=7:**

- Controller executes `a6`
- **Inference Completes!** (d_obs = 3 ticks)
- Generated `A_new = [a4, a5, a6, a7', a8', a9', b6, b7, b8, b9]`
- Note: First 3 actions match exactly what was executed during T=5,6,7
- **Swap:** `A_cur ← A_new`, `t_idx = 3`, `s_count = 0`

### Cycle 2: The Second (Inpainted) Chunk

**T=8-10:** Controller executes `a7'`, `a8'`, `a9'`

**T=11:**

- Controller executes `b6` → `s_count = 4 ≥ s_min`
- **Next Inference Starts!**
- Build `A_prev = [b7, b8, b9]`
- Process repeats...

### Executed Action Sequence

The complete sequence is perfectly **continuous** with no pauses or jerks:

```
a0 → a1 → a2 → a3 → a4 → a5 → a6 [Swap] → a7' → a8' → a9' → b6 → b7 → b8 → b9 [Swap] → b10' → ...
```

The chunk boundaries are seamless because the frozen portion of each new chunk exactly matches the actions being executed during inference.

---
