#!/usr/bin/env python
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from mock_components import MockBackbone, mock_batch, mock_policy

from lerobot.policies.cig_vla.modeling_cig_vla import CIGVLAPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-backbone", action="store_true")
    parser.add_argument("--real-qwen", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.real_qwen:
        raise SystemExit("Real Qwen smoke requires completed 2B readiness validation")
    torch.manual_seed(0)
    policy = mock_policy()
    batch = mock_batch()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-3)
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite full-policy loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    checkpoint = Path(tempfile.mkdtemp(prefix="cig_smoke_checkpoint_"))
    policy.save_pretrained(checkpoint)
    with patch(
        "lerobot.policies.cig_vla.modeling_interaction_cig_vla.Qwen3VLGroundingBackbone", MockBackbone
    ):
        loaded = CIGVLAPolicy.from_pretrained(checkpoint, strict=True)
    inference_batch = mock_batch(batch_size=1, labels=False)
    chunk = loaded.predict_action_chunk(inference_batch)
    if chunk.shape != (1, 4, 7) or not torch.isfinite(chunk).all():
        raise SystemExit("full-policy inference failed")
    print(
        f"Full-policy mock smoke: PASS steps={args.steps} initial={losses[0]:.6f} "
        f"final={losses[-1]:.6f} checkpoint={checkpoint}"
    )


if __name__ == "__main__":
    main()
