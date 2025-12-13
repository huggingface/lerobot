# Subtask Token Generation - Quick Reference

## What Was Done

Added **autoregressive subtask token generation** to PI05 model with decoding and printing during both training and inference.

## Key Features

✅ **Training:** Prints ground truth subtask tokens for monitoring  
✅ **Inference:** Generates and prints predicted subtask tokens using next token prediction  
✅ **Autoregressive:** Each token conditioned on previous tokens  
✅ **Greedy Decoding:** Selects most likely token at each step  

## Implementation Location

**File:** `src/lerobot/policies/pi05/modeling_pi05.py`

**New Method:** `_generate_subtask_tokens()` (lines 844-914)
- Autoregressive token generation
- Uses PaliGemma language model head
- Greedy decoding with early stopping

**Modified Methods:**
- `sample_actions()` - Calls generation and prints during inference
- `predict_action_chunk()` - Passes tokenizer to enable generation
- `forward()` - Prints ground truth tokens during training
- `__init__()` - Loads tokenizer

## Console Output Examples

### Training:
```
[Training] Ground truth subtask 0: pick up the red block
[Training] Ground truth subtask 1: place in blue container
```

### Inference:
```
[Inference] Generated subtask 0: grasp the object
[Inference] Generated subtask 1: move to target location
```

## How to Use

### No Code Changes Required!

The implementation is automatic:

1. **Training:** Just run your training script
   - Subtasks will be printed to console automatically
   
2. **Inference:** Just run your inference script  
   - Subtasks will be generated and printed automatically

### To Disable (if needed):

To disable subtask generation during inference for better performance:

```python
# In the model code, set tokenizer to None temporarily
policy.tokenizer = None
actions = policy.predict_action_chunk(batch)
```

## Technical Specs

| Property | Value |
|----------|-------|
| **Generation Method** | Autoregressive (sequential) |
| **Decoding Strategy** | Greedy (argmax) |
| **Max Tokens** | 50 (configurable) |
| **Tokenizer** | google/paligemma-3b-pt-224 |
| **Attention** | Causal masking for generated tokens |
| **Performance Cost** | ~50 extra forward passes per inference |

## Architecture Flow

```
Training:    Ground Truth Tokens → Decode → Print → Loss Computation
Inference:   Observations → Generate Tokens → Decode → Print → Action Prediction
```

## Method: `_generate_subtask_tokens()`

**Purpose:** Generate subtask tokens autoregressively

**Algorithm:**
```python
1. Start with prefix = [images, high-level task, state]
2. For each position (up to max_length):
   a. Forward pass → get logits
   b. Apply LM head → token probabilities
   c. Select best token (greedy)
   d. Embed token
   e. Append to prefix
   f. Update masks (causal attention)
3. Stop when EOS or max length reached
4. Return generated tokens
```

**Key Parameters:**
- `images` - Visual observations
- `img_masks` - Image padding masks
- `tokens` - Instruction tokens with state
- `masks` - Token attention masks
- `tokenizer` - For EOS detection
- `max_length` - Maximum tokens to generate (default: 50)
- `device` - Computation device

## Files Created

📄 `SUMMARY.md` - Comprehensive summary  
📄 `SUBTASK_GENERATION_CHANGES.md` - Detailed technical docs  
📄 `SUBTASK_GENERATION_FLOW.md` - Visual flow diagrams  
📄 `QUICK_REFERENCE.md` - This file  
📄 `examples/dataset/test_subtask_generation.py` - Test script  

## Quick Test

```bash
# Test that tokenizer loads correctly
python examples/dataset/test_subtask_generation.py

# Run training to see ground truth subtasks
python your_training_script.py

# Run inference to see generated subtasks
python your_inference_script.py
```

## Troubleshooting

### No subtask output during inference?
- Check that tokenizer loaded: `print(policy.tokenizer)`
- Should see: `PaliGemmaTokenizerFast(name_or_path='google/paligemma-3b-pt-224'...)`

### Tokenizer failed to load?
- Check internet connection (first run downloads tokenizer)
- Check transformers library installed: `pip install transformers`

### Performance too slow during inference?
- Disable subtask generation by setting `policy.tokenizer = None`
- Or implement KV caching for faster generation (future optimization)

## Integration Points

The implementation integrates seamlessly with existing code:

- **Training Loop:** No changes needed, prints happen automatically
- **Inference Loop:** No changes needed, generation happens automatically  
- **Data Processing:** Uses existing tokenizer from processor
- **Loss Computation:** Already implemented in training forward pass

## Future Enhancements

Possible improvements (not yet implemented):

- [ ] KV caching for faster generation
- [ ] Temperature/top-k/top-p sampling
- [ ] Beam search for better quality
- [ ] Optional flag to enable/disable printing
- [ ] Save generated subtasks to file
- [ ] Compute subtask prediction accuracy metrics
- [ ] Use generated subtasks in action prediction (hierarchical)

## Code Snippet - How Autoregressive Generation Works

```python
# Simplified pseudocode
generated_tokens = []
prefix = [images, high_level_task, state]

for t in range(max_length):
    # Forward pass
    logits = model(prefix)
    
    # Greedy decode
    next_token = argmax(logits[-1])
    
    # Store
    generated_tokens.append(next_token)
    
    # Stop if EOS
    if next_token == EOS:
        break
    
    # Append for next iteration
    prefix = prefix + [next_token]

return generated_tokens
```

## Questions?

See the detailed documentation files:
- `SUBTASK_GENERATION_CHANGES.md` - Full technical details
- `SUBTASK_GENERATION_FLOW.md` - Visual flow diagrams
- `SUMMARY.md` - Complete overview

---

**Implementation Status:** ✅ Complete and Ready to Use






