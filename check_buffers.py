#!/usr/bin/env python
"""Simple script to check buffer naming in the transformed model."""

from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# Load the model with strict=False to see what buffers we have
print("Loading model...")
policy = PI0Policy.from_pretrained("pepijn223/pi0_libero_lerobot", strict=False)

# Check what buffer keys exist
state_dict = policy.state_dict()
buffer_keys = [k for k in state_dict.keys() if "buffer" in k]
normalize_keys = [k for k in state_dict.keys() if "normalize" in k]

print("\nAll buffer keys:")
for key in buffer_keys:
    print(f"  {key}")

print("\nAll normalize keys:")
for key in normalize_keys:
    print(f"  {key}")

print("\nAll keys (first 20):")
for i, key in enumerate(state_dict.keys()):
    if i < 20:
        print(f"  {key}")
