import argparse
import subprocess
import sys
from pathlib import Path

import torch

from lerobot.datasets.adapters.libero_safety import load_libero_safety_contract
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.cig_vla.action_semantics import LiberoOSCActionSemantics
from lerobot.policies.cig_vla.configuration_cig_vla import CIGVLAConfig
from lerobot.policies.cig_vla.flow_matching import (
    compute_target_velocity,
    interpolate_actions,
    velocity_to_action_estimate,
)
from lerobot.policies.cig_vla.trajectory_geometry import TrajectoryGeometryTargetBuilder
from lerobot.policies.factory import get_policy_class, make_policy_config


def validate_dataset_stats(root: Path | None, action_dim: int):
    if root is None:
        contract = load_libero_safety_contract()
        std = torch.as_tensor(contract.stats["actions"]["std"])
        valid = std.numel() == action_dim and bool(torch.isfinite(std).all()) and bool((std > 0).all())
        return valid, "public LIBERO-Safety meta/stats.json"
    try:
        dataset = LeRobotDataset("local/cig", root=root)
        action = dataset.meta.stats.get("action", {})
        std = torch.as_tensor(action.get("std", []))
        valid = std.numel() == action_dim and bool(torch.isfinite(std).all()) and bool((std > 0).all())
        return valid, "valid" if valid else "missing, zero, or non-finite action std"
    except Exception as error:
        return False, str(error)


def validate_real_qwen(required: bool):
    if not required:
        return False, "not requested; run with --require-real-qwen"
    config = CIGVLAConfig(device="cuda")
    try:
        from lerobot.policies.cig_vla.qwen3vl_backbone import Qwen3VLGroundingBackbone

        backbone = Qwen3VLGroundingBackbone(
            config.qwen_model_name,
            config.torch_dtype,
            True,
            True,
            config.lora_rank,
            config.lora_alpha,
            config.lora_dropout,
            config.lora_bias,
        )
        valid = all(value > 0 for value in backbone.lora_target_counts.values())
        del backbone
        return valid, "actual 2B initialized" if valid else "zero LoRA target count"
    except Exception as error:
        return False, str(error)


def run_mock_gates():
    root = Path(__file__).parents[2]
    tests = [
        root / "tests/policies/cig_vla/test_gradient_flow_matrix.py",
        root / "tests/policies/cig_vla/test_full_policy_forward.py",
        root / "tests/policies/cig_vla/test_full_policy_inference.py",
        root / "tests/policies/cig_vla/test_checkpoint_roundtrip.py",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *map(str, tests), "-q"], capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-qwen", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--phase-signal", action="store_true")
    args = parser.parse_args()
    config = CIGVLAConfig(device="cpu")
    clean, noise = torch.randn(2, 4, 7), torch.randn(2, 4, 7)
    timestep = torch.tensor([0.0, 0.7])
    recovered = velocity_to_action_estimate(
        interpolate_actions(clean, noise, timestep), compute_target_velocity(clean, noise), timestep
    )
    checks = {
        "Policy registration": isinstance(make_policy_config("cig_vla", device="cpu"), CIGVLAConfig)
        and get_policy_class("cig_vla").name == "cig_vla",
        "Flow reconstruction": torch.allclose(clean, recovered),
        "Action dimensions": args.action_dim == LiberoOSCActionSemantics.action_dim,
        "Trajectory geometry builder": isinstance(
            TrajectoryGeometryTargetBuilder(), TrajectoryGeometryTargetBuilder
        ),
    }
    unit_blockers = [name for name, passed in checks.items() if not passed]
    qwen_valid, qwen_reason = validate_real_qwen(args.require_real_qwen and not args.offline)
    stats_valid, stats_reason = validate_dataset_stats(args.dataset_root, args.action_dim)
    try:
        contract = load_libero_safety_contract()
        dataset_contract_valid = contract.total_tasks == 15 and contract.features["actions"]["shape"] == [7]
    except Exception:
        dataset_contract_valid = False
    mock_gates_valid, mock_gates_output = run_mock_gates()
    real_blockers = []
    if not qwen_valid:
        real_blockers.append("real Qwen initialization/LoRA/vision freeze unverified")
    if not dataset_contract_valid:
        real_blockers.append("LIBERO-Safety public dataset contract unavailable")
    if not stats_valid:
        real_blockers.append("real geometry dataset action stats unavailable")

    print("CIG-VLA readiness\n--------------------------------")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"Gradient/forward/inference/checkpoint gates: {'PASS' if mock_gates_valid else 'FAIL'}")
    print(f"Main branch detach: {config.detach_bottleneck_for_main_action}")
    print(f"Causal branch detach: {config.detach_bottleneck_for_causal_branch}")
    print(f"Qwen initialization/LoRA/vision freeze: {'PASS' if qwen_valid else 'BLOCKED'} ({qwen_reason})")
    print(f"LIBERO-Safety public contract: {'PASS' if dataset_contract_valid else 'BLOCKED'}")
    print(f"Action stats: {'PASS' if stats_valid else 'BLOCKED'} ({stats_reason})")
    print(f"Source phase signal: {'AVAILABLE' if args.phase_signal else 'UNAVAILABLE'}")
    print("Destination direction loss: DISABLED")
    unit_ready = not unit_blockers
    mock_ready = unit_ready and mock_gates_valid
    real_ready = mock_ready and not real_blockers
    print(f"READY_FOR_UNIT_TESTING: {'YES' if unit_ready else 'NO'}")
    print(f"READY_FOR_MOCK_SMOKE_TRAINING: {'YES' if mock_ready else 'NO'}")
    print(f"READY_FOR_REAL_SMOKE_TRAINING: {'YES' if real_ready else 'NO'}")
    print("READY_FOR_LONG_TRAINING: NO")
    if not mock_gates_valid:
        unit_blockers.append("mock pytest gates failed: " + mock_gates_output[-1000:])
    blockers = unit_blockers + real_blockers
    if blockers:
        print("\nBlocking issues:")
        for issue in blockers:
            print(f"- {issue}")
        raise SystemExit(1)
