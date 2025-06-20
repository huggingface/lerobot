#!/usr/bin/env python

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def debug_model_forward():
    """Debug the model's forward pass and loss computation."""

    print("🔍 Debugging SmolVLA model forward pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)

    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )

    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset("a6047425318/green-marker-part2-ep0-debug")

    # Create small batch
    batch_size = 2
    indices = list(range(min(batch_size, len(dataset))))
    batch = {}

    for i in indices:
        sample = dataset[i]
        for key, value in sample.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(value)

    # Stack tensors
    for key, values in batch.items():
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values).to(device)
        elif isinstance(values[0], str):
            batch[key] = values  # Keep as list for text
        else:
            batch[key] = torch.tensor(values).to(device)

    print(f"Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Test forward pass
    print("\n🚀 Testing forward pass...")
    policy.train()

    try:
        loss, output_dict = policy.forward(batch)
        print("✅ Forward pass successful!")
        print(f"Loss: {loss.item():.6f}")
        print(f"Output keys: {list(output_dict.keys()) if output_dict else 'None'}")

        # Test backward pass
        print("\n🔄 Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful!")

        # Check gradients
        grad_norms = []
        for name, param in policy.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm > 0:
                    print(f"  {name}: grad_norm={grad_norm:.6f}")

        if grad_norms:
            print(f"Average gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")
            print(f"Max gradient norm: {max(grad_norms):.6f}")
        else:
            print("⚠️  No non-zero gradients found!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_model_forward()
