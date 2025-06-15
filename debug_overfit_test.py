#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def overfit_single_batch():
    """Test overfitting on a single batch to debug training setup."""

    print("🔥 Testing single batch overfitting...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model with unfrozen settings
    print("Loading model...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

    # CRITICAL: Unfreeze all components for debugging
    policy.config.freeze_vision_encoder = False
    policy.config.train_expert_only = False

    # Rebuild model with new settings
    from lerobot.common.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    policy.model = SmolVLMWithExpertModel(policy.config)

    policy.to(device)
    policy.train()

    # Check trainable parameters
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(
        f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )

    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset("a6047425318/green-marker-part2-ep0-debug")
    print(f"Dataset size: {len(dataset)} frames")

    # Create single batch (key for overfitting test)
    batch_size = 2
    indices = list(range(batch_size))
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
            batch[key] = values
        else:
            batch[key] = torch.tensor(values).to(device)

    print(f"Batch created with keys: {list(batch.keys())}")

    # Setup optimizer with higher learning rate for overfitting
    optimizer = AdamW(
        policy.parameters(),
        lr=0.01,  # High LR for overfitting
        betas=(0.9, 0.95),
        weight_decay=0.0,  # No weight decay for overfitting
    )

    # Training loop
    losses = []
    num_steps = 500

    print(f"\n🚀 Starting overfitting test for {num_steps} steps...")

    with tqdm(range(num_steps), desc="Overfitting") as pbar:
        for step in pbar:
            optimizer.zero_grad()

            try:
                loss, output_dict = policy.forward(batch)
                loss.backward()

                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

                optimizer.step()

                losses.append(loss.item())

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.6f}", "grad_norm": f"{grad_norm:.4f}"})

                # Print detailed info every 50 steps
                if step % 50 == 0:
                    print(f"\nStep {step}:")
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Grad norm: {grad_norm:.6f}")
                    if output_dict:
                        for key, value in output_dict.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.6f}")

            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

    # Analyze results
    print("\n📊 Results Analysis:")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Overfitting Test - Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("overfit_loss_curve.png")
    print("📈 Loss curve saved as 'overfit_loss_curve.png'")

    # Verdict
    if losses[-1] < 0.01:
        print("✅ SUCCESS: Model can overfit (loss < 0.01)")
    elif losses[-1] < losses[0] * 0.1:
        print("⚠️  PARTIAL: Model is learning but slowly")
    else:
        print("❌ FAILURE: Model is not overfitting properly")
        print("   Potential issues:")
        print("   - Parts of model still frozen")
        print("   - Learning rate too low")
        print("   - Gradient issues")
        print("   - Data preprocessing problems")


if __name__ == "__main__":
    overfit_single_batch()
