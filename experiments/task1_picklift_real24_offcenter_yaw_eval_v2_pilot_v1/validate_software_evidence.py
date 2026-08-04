from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalv2_pilot import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate frozen software-only pilot evidence.")
    parser.add_argument("--software-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.software_root
    hashes_path = root / "hashes.sha256"
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", maxsplit=1)
        path = root / filename
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen software evidence hash mismatch: {filename}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dry_run = json.loads((root / "dry_run.json").read_text(encoding="utf-8"))
    expected_hardware_access = {
        "serial": False,
        "camera": False,
        "robot": False,
        "torque": False,
        "rollout": False,
    }
    if manifest["hardware_access"] != expected_hardware_access:
        raise RuntimeError("Software manifest claims unexpected hardware access.")
    if dry_run["hardware_access"] != expected_hardware_access:
        raise RuntimeError("Dry-run claims unexpected hardware access.")
    fake = dry_run["fake_protocol"]
    if fake["trials_exercised"] != 12:
        raise RuntimeError("Dry-run does not contain all 12 frozen poses.")
    if fake["policy_reset_calls"] != {"real24_only": 12}:
        raise RuntimeError("Policy reset count differs from the frozen pilot.")
    if fake["all_pre_action_frames_before_policy_send"] is not True:
        raise RuntimeError("Pre-action frame contract was not verified.")
    if (root.parent / "trials").exists():
        raise RuntimeError("Software-only preparation unexpectedly created trial rollout evidence.")
    print("independent_validation=passed")
    print(f"software_root={root}")
    print(f"validated_hash_rows={len(hashes_path.read_text(encoding='utf-8').splitlines())}")
    print("trial_evidence_directory_exists=false")


if __name__ == "__main__":
    main()
