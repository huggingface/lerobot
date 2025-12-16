#!/usr/bin/env python

"""
Example demonstrating how to use the ActionTokenizerProcessorStep to tokenize actions.

This example shows how to:
1. Load a dataset with action data
2. Apply the action tokenizer processor to tokenize actions with proper padding/truncation
3. Access both the tokenized actions and the attention mask
4. Decode tokenized actions back to their original form
"""

import torch
from transformers import AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep
from lerobot.utils.constants import ACTION_TOKEN_MASK

# Define delta timestamps for the dataset
delta_timestamps = {
    'action': [
        0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333,
        0.16666666666666666, 0.2, 0.23333333333333334, 0.26666666666666666, 0.3,
        0.3333333333333333, 0.36666666666666664, 0.4, 0.43333333333333335,
        0.4666666666666667, 0.5, 0.5333333333333333, 0.5666666666666667, 0.6,
        0.6333333333333333, 0.6666666666666666, 0.7, 0.7333333333333333,
        0.7666666666666667, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9,
        0.9333333333333333, 0.9666666666666667, 1.0, 1.0333333333333334,
        1.0666666666666667, 1.1, 1.1333333333333333, 1.1666666666666667, 1.2,
        1.2333333333333334, 1.2666666666666666, 1.3, 1.3333333333333333,
        1.3666666666666667, 1.4, 1.4333333333333333, 1.4666666666666666, 1.5,
        1.5333333333333334, 1.5666666666666667, 1.6, 1.6333333333333333
    ]
}

# Load the dataset
print("Loading dataset...")
dataset = LeRobotDataset(
    repo_id="local",
    root="/fsx/jade_choghari/outputs/pgen_annotations1",
    delta_timestamps=delta_timestamps
)

# Create a dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=4,
    shuffle=True,
)

# Get a batch of data
batch = next(iter(dataloader))
action_data = batch["action"]  # Shape: (batch_size, action_horizon, action_dim)

print(f"\nOriginal action shape: {action_data.shape}")
print(f"Original action data (first sample, first timestep):\n{action_data[0, 0]}")

# Method 1: Using the tokenizer directly (as in fast_tokenize.py)
print("\n" + "="*80)
print("Method 1: Direct tokenizer usage")
print("="*80)

tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize directly
tokens = tokenizer(action_data)
print(f"\nDirect tokenization result type: {type(tokens)}")
print(f"Tokens shape/length: {tokens.shape if isinstance(tokens, torch.Tensor) else len(tokens)}")

# Decode
decoded_actions = tokenizer.decode(tokens)
print(f"Decoded actions shape: {decoded_actions.shape}")
reconstruction_error = torch.abs(action_data - decoded_actions).mean()
print(f"Mean absolute reconstruction error: {reconstruction_error.item():.6f}")

# Method 2: Using the ActionTokenizerProcessorStep with proper padding/truncation
print("\n" + "="*80)
print("Method 2: Using ActionTokenizerProcessorStep (with padding & mask)")
print("="*80)

# Create the action tokenizer processor step
action_tokenizer_processor = ActionTokenizerProcessorStep(
    tokenizer_name="physical-intelligence/fast",
    trust_remote_code=True,
    max_action_tokens=32,  # Maximum number of tokens per action
)

# Create a transition with the action data
transition = {
    TransitionKey.ACTION: action_data,
    TransitionKey.OBSERVATION: {},  # Empty for this example
}

# Apply the processor
processed_transition = action_tokenizer_processor(transition)

# Extract tokenized actions and mask
tokenized_actions = processed_transition[TransitionKey.ACTION]
complementary_data = processed_transition[TransitionKey.COMPLEMENTARY_DATA]
action_mask = complementary_data[ACTION_TOKEN_MASK]

print(f"\nTokenized actions shape: {tokenized_actions.shape}")  # (batch_size, max_action_tokens)
print(f"Action mask shape: {action_mask.shape}")  # (batch_size, max_action_tokens)
print(f"Tokenized actions dtype: {tokenized_actions.dtype}")
print(f"Action mask dtype: {action_mask.dtype}")

# Show token statistics
print(f"\nFirst sample tokens: {tokenized_actions[0]}")
print(f"First sample mask: {action_mask[0]}")
num_real_tokens = action_mask[0].sum().item()
print(f"Number of real tokens (non-padding): {num_real_tokens}")
print(f"Number of padding tokens: {action_mask.shape[1] - num_real_tokens}")

# Decode using the mask
print("\nDecoding tokenized actions...")
decoded_with_processor = tokenizer.decode(tokenized_actions)
print(f"Decoded actions shape: {decoded_with_processor.shape}")

# Calculate reconstruction error
reconstruction_error_processor = torch.abs(action_data - decoded_with_processor).mean()
print(f"Mean absolute reconstruction error: {reconstruction_error_processor.item():.6f}")

# Show that masking works correctly
print("\n" + "="*80)
print("Mask demonstration")
print("="*80)
for i in range(min(4, tokenized_actions.shape[0])):
    mask_i = action_mask[i]
    num_real = mask_i.sum().item()
    print(f"Sample {i}: {num_real} real tokens, {len(mask_i) - num_real} padding tokens")

print("\n" + "="*80)
print("Action tokenization example completed successfully!")
print("="*80)

