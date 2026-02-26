#!/usr/bin/env python3
"""Test script to verify X-VLA loads and runs inference on MPS."""

import torch
import time
import logging
import os

# Enable verbose logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Enable HuggingFace download progress
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Use standard download with progress

def main():
    print("=" * 60)
    print("X-VLA Inference Test")
    print("=" * 60)
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU (this will be slow)")
    
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Load the policy - let it load config from the pretrained model
    print("Loading X-VLA policy from 'lerobot/xvla-base'...")
    print("(This may take a while on first run - downloading ~3GB model)")
    print()
    start = time.time()
    
    print("[1/5] Importing XVLAPolicy...")
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from transformers import AutoTokenizer
    
    print("[2/5] Downloading/loading model from HuggingFace Hub...")
    policy = XVLAPolicy.from_pretrained("lerobot/xvla-base")
    
    print(f"[3/5] Moving model to {device}...")
    policy.to(device)
    
    print("[4/5] Setting eval mode...")
    policy.eval()
    
    print("[5/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name)
    
    load_time = time.time() - start
    print(f"✅ Policy loaded in {load_time:.1f}s")
    print()
    
    # Print model info
    print("Model configuration:")
    print(f"   Input features: {list(policy.config.input_features.keys())}")
    print(f"   Output features: {list(policy.config.output_features.keys())}")
    print(f"   Action mode: {policy.config.action_mode}")
    print(f"   Tokenizer: {policy.config.tokenizer_name}")
    print()
    
    # Create dummy observation based on the model's expected inputs
    print("Creating dummy observation...")
    observation = {}
    
    for key, feature in policy.config.input_features.items():
        # PolicyFeature is an object, not a dict
        shape = feature.shape if hasattr(feature, 'shape') else []
        dtype = feature.dtype if hasattr(feature, 'dtype') else "float32"
        
        if dtype == "image" or "image" in key:
            # Image features
            observation[key] = torch.rand(1, *shape, device=device)
            print(f"   {key}: image {list(observation[key].shape)}")
        else:
            # State features
            observation[key] = torch.rand(1, *shape, device=device)
            print(f"   {key}: state {list(observation[key].shape)}")
    
    # Add language tokens (required for VLA models)
    task_description = "pick up the red block"
    print(f"   Task: '{task_description}'")
    
    # Use config's tokenizer settings, but cap at 64 to stay within model's sequence limit
    max_len = min(policy.config.tokenizer_max_length, 64)
    print(f"   Tokenizer max_length: {max_len}")
    
    tokens = tokenizer(
        task_description,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    observation["observation.language.tokens"] = tokens["input_ids"].to(device)
    print(f"   observation.language.tokens: {list(observation['observation.language.tokens'].shape)}")
    
    print()
    
    # Run inference
    print("Running inference...")
    start = time.time()
    
    with torch.no_grad():
        policy.reset()
        
        try:
            action = policy.select_action(observation)
            inference_time = time.time() - start
            
            print()
            print(f"✅ Inference successful!")
            print(f"   Time: {inference_time*1000:.1f}ms")
            print(f"   Action shape: {action.shape}")
            print(f"   Action values: {action.cpu().numpy()}")
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
