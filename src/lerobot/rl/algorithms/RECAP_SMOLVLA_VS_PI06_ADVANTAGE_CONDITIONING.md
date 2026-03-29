# RECAP Advantage Conditioning: SmolVLA vs pi-0.6

This document explains why applying RECAP advantage conditioning to SmolVLA
requires a different approach than the one described in the original paper
(which targets pi-0.6), and how the SmolStar06 implementation solves this.

---

## The core problem

The RECAP paper conditions the policy on a binarized advantage indicator
("Advantage: positive" or "Advantage: negative") so that at inference time,
conditioning on "positive" produces actions that are better than the average
behavior in the training data. The specific mechanism the paper uses —
appending advantage text tokens to the prompt — works for pi-0.6 but
**does not work** for SmolVLA due to fundamental architectural differences.

---

## Architecture comparison

### pi-0.6: single-stream

pi-0.6 processes everything in a **single causal token sequence**. The VLM
backbone, sub-task predictor, and action expert all operate on the same
sequence of tokens:

```
[images] [language] [sub-task text] [Advantage: positive] [action tokens]
```

The action expert generates actions autoregressively within this sequence.
When it starts producing action tokens, the advantage text is the **most
recent context** — it sits right before the action positions in the causal
attention mask. The expert cannot avoid attending to it.

Key properties:
- The advantage tokens are **positionally adjacent** to the action tokens.
- Standard causal attention guarantees the expert sees them.
- The gradient from the action loss flows directly back through the
  attention weights to the advantage token embeddings.
- At inference, you literally append `"Advantage: positive"` to the prompt.

### SmolVLA: two-stream

SmolVLA separates processing into **two parallel transformer stacks** that
run layer-by-layer in lockstep:

| | Stream 0 (VLM) | Stream 1 (Action Expert) |
|---|---|---|
| **Input** | Images + language + state | Noisy actions + flow timestep |
| **Embedding** | `embed_prefix()` | `embed_suffix()` |
| **Output** | VLM hidden states | Predicted velocity for flow matching |

The two streams are connected via attention:
- **Cross-attention layers** (odd layers): the expert generates queries from
  its hidden states, and keys/values come from the VLM's output re-projected
  through the expert's own K/V projections.
- **Joint-attention layers** (even layers): both streams' Q/K/V are
  concatenated and run through a single attention operation.

There is **no shared token sequence**. Language tokens are embedded and
processed exclusively by stream 0. The expert in stream 1 learns about
language only through attention to VLM representations — it never directly
sees language token IDs.

---

## Why text-based conditioning fails for SmolVLA

If you append `"Advantage: positive"` as text tokens to the language prompt
(as the paper describes), the signal path is:

```
"Advantage: positive" tokens
    → VLM text embedding table
    → embed_prefix (becomes part of stream 0 input)
    → N layers of VLM transformer processing
    → VLM produces K/V states at each layer
    → Expert cross-attends to VLM K/V (re-projected through expert's K/V weights)
    → Expert hidden states
    → action_out_proj → predicted velocity
    → MSE loss
```

This fails for several reasons:

1. **Dilution**: The advantage is 4 tokens among 200+ prefix tokens (images
   produce many visual tokens after the vision encoder and connector). The
   VLM has no reason to give these 4 tokens special treatment in its internal
   representations.

2. **No direct training signal**: The VLM's primary objective is processing
   language and vision. There is no loss term that explicitly encourages the
   VLM to amplify the advantage signal in its key/value states for the
   expert's benefit.

3. **Weak gradient path**: The gradient from the flow-matching MSE loss must
   backpropagate through `action_out_proj` → expert layers → cross-attention
   Q/K/V projections → VLM K/V states → VLM layers → token embeddings. By
   the time it reaches the advantage token embeddings, the signal is
   negligible.

4. **Empirical confirmation**: Training with text-based advantage tokens
   produces `cond_acc=0.0` and `cond_gap=0.0` — the model generates
   identical flow-matching loss whether given the correct or flipped
   advantage label.

---

## SmolStar06 solution: expert-side advantage embedding

SmolStar06 injects the advantage signal directly into the action expert's
input pathway, bypassing the VLM entirely.

### The embedding

```python
self.advantage_embedding = nn.Embedding(2, expert_hidden_size)
# index 0 → negative advantage (learned dense vector)
# index 1 → positive advantage (learned dense vector)
# zero-initialized so it starts neutral
```

### Where it's injected

The advantage embedding is added to the suffix embeddings produced by
`embed_suffix()`, which is the action expert's input:

```python
# In _forward_with_advantage():
suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, time)

adv_emb = self.advantage_embedding(advantage_indicator.long())  # [B, expert_hidden]
adv_emb = adv_emb.unsqueeze(1).expand_as(suffix_embs)           # [B, chunk_size, expert_hidden]
suffix_embs = suffix_embs + adv_emb                              # direct additive conditioning
```

### The gradient path

```
advantage_embedding weights
    → lookup → dense vector
    → added to suffix_embs
    → expert transformer layers (2-3 layers)
    → action_out_proj → predicted velocity v_t
    → MSE loss against target u_t
```

The gradient flows directly from the MSE loss through a few expert layers
back to the embedding weights. The model **cannot** ignore the advantage
signal — it is literally part of the expert's input at every position in the
action chunk.

### What stays the same

- The VLM prefix (images, language, state) is processed identically to base
  SmolVLA — no advantage tokens in the language.
- The expert still receives all language and visual information through
  cross-attention to the VLM's K/V states, exactly as before.
- The advantage embedding is a **separate channel** that does not compete
  with language or image tokens.

---

## Training comparison

| Aspect | pi-0.6 (paper) | SmolVLA (SmolStar06) |
|--------|---------------|---------------------|
| **Advantage representation** | Text tokens: `"Advantage: positive"` | Learned embedding: `nn.Embedding(2, expert_hidden_size)` |
| **Injection point** | Language prompt (before action tokens in shared sequence) | `embed_suffix` (action expert input, stream 1) |
| **VLM involvement** | Advantage tokens processed by VLM as part of the causal sequence | VLM never sees advantage signal |
| **Gradient path** | Through shared VLM+expert layers (short, since tokens are adjacent to actions) | Through expert layers only (2-3 layers from loss to embedding) |
| **Dropout** | 30% of training steps: advantage text omitted from prompt | 30% of training steps: advantage embedding zeroed out |
| **Training objective** | Eq. 3: conditional + unconditional log-likelihood | Single pass per sample: either conditional or unconditional (per-sample dropout) |

---

## Inference comparison

| Aspect | pi-0.6 (paper) | SmolVLA (SmolStar06) |
|--------|---------------|---------------------|
| **Standard (`beta=1`)** | Append `"Advantage: positive"` to prompt, sample normally | Add positive advantage embedding (index 1) to every `embed_suffix` call |
| **CFG (`beta>1`)** | Two forward passes per denoising step: one with `"positive"` text, one without; interpolate flow vectors | Two `embed_suffix` passes per denoising step: one with positive embedding, one without (zeroed); interpolate flow vectors. **Only one VLM prefix forward** — shared KV cache since language tokens are identical |
| **Value network needed?** | No | No |

The CFG implementation is more efficient in SmolVLA because the VLM prefix
computation (the expensive part) only runs once — the two denoising passes
differ only in the suffix embeddings, which are cheap.

---

## Summary

The RECAP paper's approach of adding advantage text to the prompt is
architecturally appropriate for pi-0.6's single-stream design, where the
advantage tokens are causally adjacent to the action outputs. SmolVLA's
two-stream architecture routes language tokens through a separate VLM
stream that is only connected to the action expert via cross-attention,
making text-based conditioning ineffective.

SmolStar06 solves this by injecting the advantage as a learned embedding
directly into the action expert's input pathway (`embed_suffix`). This
preserves the same semantics — the model learns to produce different action
distributions for positive vs negative advantage — but uses the
architecturally correct mechanism for a two-stream model.

The key insight: the advantage is not linguistic information that needs
language understanding. It is a binary control signal that should modulate
action generation directly. In pi-0.6, text tokens happen to work because
the sequence architecture places them adjacent to actions. In SmolVLA, a
learned embedding in the expert's input is the natural equivalent.
