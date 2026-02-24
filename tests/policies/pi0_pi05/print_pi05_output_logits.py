#!/usr/bin/env python
"""Print PI05 output logits on deterministic dummy data. Run locally to test your PI05 impl."""

import torch

from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
from lerobot.policies.pi0 import PI0Policy, make_pi0_pre_post_processors

from lerobot.utils.random_utils import set_seed

# Deterministic dummy data
SEED = 42
BATCH_SIZE = 1
PRETRAINED = "lerobot/pi0_libero_finetuned"


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = PI0Policy.from_pretrained(pretrained_name_or_path=PRETRAINED, strict=True)
    policy.config.device = device
    policy.to(device)
    config = policy.config
    state_dim = config.input_features["observation.state"].shape[0]
    action_dim = config.output_features["action"].shape[0]
    img_shape = (3, 224, 224)

    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
            "min": torch.zeros(state_dim),
            "max": torch.ones(state_dim),
            "q01": torch.zeros(state_dim),
            "q99": torch.ones(state_dim),
        },
        "action": {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
            "min": torch.zeros(action_dim),
            "max": torch.ones(action_dim),
            "q01": torch.zeros(action_dim),
            "q99": torch.ones(action_dim),
        },
        "observation.images.image": {
            "mean": torch.zeros(*img_shape),
            "std": torch.ones(*img_shape),
            "q01": torch.zeros(*img_shape),
            "q99": torch.ones(*img_shape),
        },
        "observation.images.image2": {
            "mean": torch.zeros(*img_shape),
            "std": torch.ones(*img_shape),
            "q01": torch.zeros(*img_shape),
            "q99": torch.ones(*img_shape),
        },
    }

    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=dataset_stats)

    # Deterministic dummy batch (fixed seed above)
    chunk_size = config.chunk_size
    batch = {
        "observation.state": torch.randn(BATCH_SIZE, state_dim, dtype=torch.float32, device=device),
        "action": torch.randn(BATCH_SIZE, chunk_size, action_dim, dtype=torch.float32, device=device),
        "observation.images.image": torch.rand(
            BATCH_SIZE, *img_shape, dtype=torch.float32, device=device
        ),
        "observation.images.image2": torch.rand(
            BATCH_SIZE, *img_shape, dtype=torch.float32, device=device
        ),
        "task": ["Pick up the object"] * BATCH_SIZE,
    }
    
    batch = preprocessor(batch)

    policy.eval()
    with torch.no_grad():
        # output = policy.predict_action_chunk(batch)

        # now forward with loss
        loss, loss_dict = policy.forward(batch)
        print(f"Loss: {loss.item():.6f}")
        print(f"Loss dict: {loss_dict}")

    print(f"Model: {PRETRAINED}")
    print("PI05 output logits (action chunk):")
    # print(f"  shape: {output.shape}")
    # print(f"  dtype: {output.dtype}")
    # print(f"  min: {output.min().item():.6f}, max: {output.max().item():.6f}, mean: {output.mean().item():.6f}")
    # print("  values [0, 0, :]:", output[0, 0, :].cpu().tolist())
    # print("  full tensor:\n", output.cpu())
# shape: torch.Size([1, 50, 7])
#   dtype: torch.float32
#   min: -1.414879, max: 1.858794, mean: -0.170034
#   values [0, 0, :]: [-1.4148787260055542, -0.06444520503282547, -0.7634842991828918, 0.18793275952339172, -0.2568123936653137, 1.8587939739227295, -0.015481989830732346]

if __name__ == "__main__":
    main()
