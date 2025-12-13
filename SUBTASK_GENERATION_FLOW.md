# Subtask Token Generation Flow Diagram

## Overview
This document provides visual representations of how subtask tokens are processed during training and inference in the PI05 model.

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING FORWARD PASS                    │
└─────────────────────────────────────────────────────────────┘

Input Batch:
├─ images              (observations)
├─ high_level_task     (user prompt tokens)
├─ tokens              (instruction prompt with state)
├─ subtask_tokens      (ground truth subtask - TARGET)
└─ actions             (ground truth actions)

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 1: Decode & Print Ground Truth Subtask Tokens        │
│  ---------------------------------------------------------- │
│  • Extract valid tokens (remove padding)                    │
│  • Decode using tokenizer                                   │
│  • Print: "[Training] Ground truth subtask {i}: {text}"    │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 2: Prefix-Only Forward Pass                          │
│  ---------------------------------------------------------- │
│  Embed Prefix:                                              │
│    • images → image embeddings                              │
│    • high_level_task → language embeddings                  │
│    • tokens → language embeddings                           │
│    • subtask_tokens → language embeddings (with causal mask)│
│                                                             │
│  Forward through transformer (prefix only)                  │
│    → Get prefix_out                                         │
│                                                             │
│  Apply LM Head:                                             │
│    prefix_out → logits                                      │
│                                                             │
│  Extract subtask logits:                                    │
│    logits[start_index:end_index]                           │
│                                                             │
│  Compute Cross-Entropy Loss:                                │
│    CE(predicted_logits, subtask_tokens) → subtask_loss     │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 3: Full Forward Pass (Prefix + Suffix)               │
│  ---------------------------------------------------------- │
│  Add noisy actions to prefix:                               │
│    x_t = time * noise + (1-time) * actions                 │
│                                                             │
│  Forward through transformer:                               │
│    [prefix_embs, action_embs] → [prefix_out, suffix_out]  │
│                                                             │
│  Predict velocity field:                                    │
│    suffix_out → v_t                                        │
│                                                             │
│  Compute Flow Matching Loss:                                │
│    MSE(u_t, v_t) → flow_loss                              │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 4: Combined Loss                                      │
│  ---------------------------------------------------------- │
│  total_loss = 10 * flow_loss + subtask_loss               │
└─────────────────────────────────────────────────────────────┘

Output: loss, {flow_loss, subtask_loss, loss}
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE FORWARD PASS                    │
└─────────────────────────────────────────────────────────────┘

Input Batch:
├─ images              (observations)
├─ high_level_task     (user prompt tokens)
└─ tokens              (instruction prompt with state)

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 1: Autoregressive Subtask Token Generation           │
│  ---------------------------------------------------------- │
│                                                             │
│  Initialize:                                                │
│    prefix_embs = [images, high_level_task, tokens]         │
│    generated_tokens = []                                    │
│                                                             │
│  For t in range(max_subtask_tokens):                       │
│    ┌───────────────────────────────────────────┐          │
│    │ Forward Pass:                              │          │
│    │   prefix_embs → transformer → prefix_out   │          │
│    │                                             │          │
│    │ Apply LM Head:                              │          │
│    │   prefix_out → logits                      │          │
│    │                                             │          │
│    │ Greedy Decode:                              │          │
│    │   next_token = argmax(logits[-1])         │          │
│    │                                             │          │
│    │ Store Token:                                │          │
│    │   generated_tokens.append(next_token)      │          │
│    │                                             │          │
│    │ Check EOS:                                  │          │
│    │   if next_token == EOS: break              │          │
│    │                                             │          │
│    │ Embed & Append:                             │          │
│    │   next_emb = embed(next_token)             │          │
│    │   prefix_embs = concat(prefix_embs, next_emb)│        │
│    │   Update masks (causal attention)          │          │
│    └───────────────────────────────────────────┘          │
│                                                             │
│  Return: generated_tokens                                   │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 2: Decode & Print Generated Subtask Tokens           │
│  ---------------------------------------------------------- │
│  • Remove padding tokens (value = 0)                        │
│  • Decode using tokenizer                                   │
│  • Print: "[Inference] Generated subtask {i}: {text}"      │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 3: Embed Prefix (without subtask tokens)             │
│  ---------------------------------------------------------- │
│  prefix_embs = [images, high_level_task, tokens]           │
│  (Note: subtask_tokens = None during inference)            │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 4: Cache KV for Prefix                                │
│  ---------------------------------------------------------- │
│  Forward pass through transformer with use_cache=True       │
│    → past_key_values                                        │
└─────────────────────────────────────────────────────────────┘

                          ↓

┌─────────────────────────────────────────────────────────────┐
│  Step 5: Flow Matching Denoising Loop                       │
│  ---------------------------------------------------------- │
│  Initialize:                                                │
│    x_t = noise  (random action sequence)                   │
│    time = 1.0                                              │
│    dt = -1.0 / num_steps                                   │
│                                                             │
│  While time >= -dt/2:                                      │
│    ┌───────────────────────────────────────────┐          │
│    │ Denoise Step:                              │          │
│    │   • Embed x_t and time                     │          │
│    │   • Forward through transformer            │          │
│    │     (uses cached past_key_values)          │          │
│    │   • Predict velocity: v_t                  │          │
│    │                                             │          │
│    │ Euler Step:                                 │          │
│    │   x_t = x_t + dt * v_t                     │          │
│    │   time = time + dt                         │          │
│    └───────────────────────────────────────────┘          │
│                                                             │
│  Return: x_t (final denoised actions)                      │
└─────────────────────────────────────────────────────────────┘

Output: actions (predicted action chunk)
```

## Key Differences Between Training and Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Subtask Tokens | **Ground truth provided** | **Generated autoregressively** |
| Subtask Processing | Decode & print for monitoring | Generate → decode → print |
| Forward Passes | 2 passes (prefix-only, then full) | Multiple passes (1 per token + denoising) |
| Loss Computation | Subtask loss + flow loss | No loss (inference only) |
| Subtask Usage | Used for loss calculation | Generated but not used in action prediction |

## Autoregressive Generation Detail

```
Time step t=0:
┌─────────────────────────────────────────────────────┐
│ Prefix: [IMG] [IMG] [TASK] [STATE] →               │
│   → Forward → LM Head → "pick" (token: 1234)       │
└─────────────────────────────────────────────────────┘

Time step t=1:
┌─────────────────────────────────────────────────────┐
│ Prefix: [IMG] [IMG] [TASK] [STATE] [pick] →        │
│   → Forward → LM Head → "up" (token: 5678)         │
└─────────────────────────────────────────────────────┘

Time step t=2:
┌─────────────────────────────────────────────────────┐
│ Prefix: [IMG] [IMG] [TASK] [STATE] [pick] [up] →   │
│   → Forward → LM Head → "the" (token: 9012)        │
└─────────────────────────────────────────────────────┘

... continues until EOS or max_length ...

Final Result: "pick up the red block"
```

## Attention Masking Pattern

```
During Subtask Generation:

Position:    0    1    2    3    4    5    6    (token positions)
Token:     [IMG] [IMG] [TASK] [T1] [T2] [T3]    (T* = generated tokens)
Mask Type:   0    0     0     1    1    1      (0=full attn, 1=causal)

Attention Pattern:
  [IMG] can attend to: [IMG] (itself)
  [IMG] can attend to: [IMG], [IMG] (all previous)
  [TASK] can attend to: [IMG], [IMG], [TASK] (all previous)
  [T1] can attend to: [IMG], [IMG], [TASK] (causal: only previous)
  [T2] can attend to: [IMG], [IMG], [TASK], [T1] (causal)
  [T3] can attend to: [IMG], [IMG], [TASK], [T1], [T2] (causal)
```

## Benefits of This Approach

1. **Training:**
   - Model learns to predict subtask tokens given observations
   - Joint training of subtask prediction and action prediction
   - Ground truth subtasks are visible for debugging

2. **Inference:**
   - Model can generate interpretable subtask descriptions
   - Autoregressive generation ensures coherent subtask text
   - Provides insight into model's reasoning process
   - Can be used for hierarchical planning

## Implementation Notes

- **Greedy Decoding:** Always selects the most likely token (argmax)
- **No KV Cache:** Each generation step performs full forward pass (can be optimized)
- **Max Length:** Limited to 50 tokens (configurable)
- **EOS Handling:** Stops early if all batches generate EOS token
- **Padding Handling:** Filters out padding tokens before decoding






