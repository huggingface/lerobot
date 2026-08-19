import argparse

import torch

from lerobot.policies.cig_vla.interaction_head import InteractionGeometryHead
from lerobot.policies.cig_vla.trajectory_geometry import TrajectoryGeometryTargetBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-backbone", action="store_true")
    parser.add_argument("--real-qwen", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    if args.real_qwen:
        raise SystemExit("Run real-Qwen readiness before real interaction-head overfit")
    torch.manual_seed(0)
    head = InteractionGeometryHead(16, 8, 16, 4, 1)
    memory = torch.randn(8, 5, 16)
    state = torch.randn(8, 8)
    mask = torch.ones(8, 5, dtype=torch.bool)
    actions = torch.randn(8, 4, 7) * 0.1
    stats = {"actions": {"mean": torch.zeros(7), "std": torch.ones(7)}}
    target = TrajectoryGeometryTargetBuilder().build(actions, state, stats, None)

    def loss():
        output = head(memory, mask, state)
        return (
            (output.translation_goal - target.translation_goal).square().mean()
            + (output.approach_direction - target.approach_direction).square().mean()
            + (output.translation_magnitude - target.translation_magnitude).square().mean()
            + (output.gripper_transition - target.gripper_transition).square().mean()
        )

    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    initial = loss().item()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
    final = loss().item()
    print(f"Stage A trajectory interaction overfit: initial={initial:.6f} final={final:.6f}")
    if not final < initial * 0.5:
        raise SystemExit("interaction geometry overfit threshold failed")


if __name__ == "__main__":
    main()
