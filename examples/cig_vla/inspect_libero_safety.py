import argparse

from lerobot.datasets.adapters.libero_safety import load_libero_safety_contract, task_split


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the public LIBERO-Safety contract without generating object labels"
    )
    parser.add_argument("--repo-id", default="LIBERO-Safety/libero_safety")
    parser.add_argument("--revision")
    args = parser.parse_args()
    contract = load_libero_safety_contract(args.repo_id, args.revision)
    split = task_split(contract)
    print(f"repo_id: {args.repo_id}")
    print(f"codebase_version: {contract.codebase_version}")
    print(f"episodes: {contract.total_episodes}")
    print(f"frames: {contract.total_frames}")
    print(f"tasks: {contract.total_tasks}")
    print(f"fps: {contract.fps}")
    print(f"image keys: {[key for key in contract.features if 'image' in key]}")
    print(f"state shape: {contract.features['observation.state']['shape']}")
    print(f"action shape: {contract.features['actions']['shape']}")
    print(f"train task indices: {split['train']}")
    print(f"validation task indices: {split['validation']}")
    print("Object/safety labels: ABSENT (not generated)")


if __name__ == "__main__":
    main()
