# Subtask Token Generation and Decoding Implementation

## Overview
This document describes the implementation of subtask token generation and decoding in the PI05 model. The implementation enables the model to generate subtask tokens autoregressively during inference and decode them to human-readable text during both training and inference.

## Changes Made

### 1. Added Autoregressive Subtask Token Generation (`modeling_pi05.py`)

#### New Method: `_generate_subtask_tokens()`
**Location:** Lines 844-914

**Purpose:** Generates subtask tokens autoregressively using next token prediction during inference.

**How it works:**
1. Embeds the prefix (images + high-level task tokens + state)
2. Iteratively generates tokens one at a time:
   - Forward pass through the model to get logits
   - Apply language model head to get token probabilities
   - Use greedy decoding to select the most likely next token
   - Embed the generated token and append to prefix
   - Update attention masks for causal attention
3. Stops when EOS token is generated or max length is reached
4. Returns tensor of generated tokens

**Key Features:**
- Uses `@torch.no_grad()` decorator for inference efficiency
- Implements greedy decoding (selects highest probability token)
- Uses causal attention masking for generated tokens
- Supports early stopping with EOS token detection

### 2. Updated `sample_actions()` Method
**Location:** Lines 916-1020

**Changes:**
- Added `tokenizer` parameter (optional)
- Added `max_subtask_tokens` parameter (default: 50)
- Calls `_generate_subtask_tokens()` if tokenizer is provided
- Decodes and prints generated subtask tokens during inference

**Output Format:**
```
[Inference] Generated subtask {batch_idx}: {decoded_text}
```

### 3. Updated `PI05Policy.__init__()` Method
**Location:** Lines 1066-1099

**Changes:**
- Added tokenizer loading using `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")`
- Stores tokenizer as `self.tokenizer`
- Includes error handling with warning if tokenizer fails to load

### 4. Updated `predict_action_chunk()` Method
**Location:** Lines 1387-1409

**Changes:**
- Passes `self.tokenizer` to `model.sample_actions()`
- Enables subtask generation during inference

### 5. Updated `forward()` Method (Training)
**Location:** Lines 1411-1445

**Changes:**
- Added ground truth subtask token decoding during training
- Prints decoded subtask tokens when in training mode
- Uses subtask masks to filter out padding tokens

**Output Format:**
```
[Training] Ground truth subtask {batch_idx}: {decoded_text}
```

## Technical Details

### Autoregressive Generation Process

The generation process follows these steps:

```
1. Start with prefix: [images, high-level task, state]
2. For each generation step (up to max_subtask_tokens):
   a. Create attention masks (causal for generated tokens)
   b. Forward pass through transformer
   c. Apply language model head → logits
   d. Greedy decode: select argmax(logits)
   e. Embed selected token
   f. Append to prefix embeddings
   g. Update masks
3. Stop when EOS token or max length reached
4. Return generated token sequence
```

### Attention Masking

- **Prefix tokens (images + high-level task):** Can attend to all previous tokens (attention mask = 0)
- **Generated subtask tokens:** Use causal attention, can only attend to previous tokens (attention mask = 1)

### Tokenizer

- **Model:** `google/paligemma-3b-pt-224`
- **Type:** PaliGemma tokenizer (based on SentencePiece)
- **Usage:** Decodes token IDs to human-readable text
- **Special tokens:** Automatically filtered with `skip_special_tokens=True`

## Usage Examples

### During Training

The model automatically prints ground truth subtask tokens:

```python
# Training batch contains subtask_tokens
loss, loss_dict = policy.forward(batch)

# Output (automatically printed):
# [Training] Ground truth subtask 0: pick up the red block
# [Training] Ground truth subtask 1: move to the blue container
```

### During Inference

The model generates and prints predicted subtask tokens:

```python
# Inference - tokenizer is automatically passed
actions = policy.predict_action_chunk(batch)

# Output (automatically printed):
# [Inference] Generated subtask 0: grasp the object
# [Inference] Generated subtask 1: place in target location
```

## Benefits

1. **Transparency:** See what subtasks the model predicts during inference
2. **Debugging:** Verify that subtask prediction is working correctly
3. **Interpretability:** Understand the model's reasoning process
4. **Monitoring:** Track subtask generation quality during training

## Performance Considerations

- **Training:** Minimal overhead (only decoding for printing, no generation)
- **Inference:** Additional computational cost due to autoregressive generation
  - Each token requires a forward pass through the transformer
  - For max_subtask_tokens=50, up to 50 forward passes
  - Can be disabled by not passing tokenizer (for production deployments)

## Future Enhancements

Possible improvements to consider:

1. **Sampling strategies:** Add temperature, top-k, top-p sampling
2. **Beam search:** Generate multiple candidates and select best
3. **Caching:** Use KV-cache to speed up autoregressive generation
4. **Logging:** Redirect prints to logger instead of console
5. **Metrics:** Track subtask prediction accuracy during training
6. **Optional flag:** Add config option to enable/disable printing

## Testing

To test the implementation, run:

```bash
python examples/dataset/test_subtask_generation.py
```

This will demonstrate the subtask generation features and verify the tokenizer is loaded correctly.

## Related Files

- `src/lerobot/policies/pi05/modeling_pi05.py` - Main implementation
- `src/lerobot/policies/pi05/processor_pi05.py` - Subtask token preprocessing
- `src/lerobot/policies/pi05/configuration_pi05.py` - Configuration
- `examples/dataset/test_subtask_generation.py` - Test script






