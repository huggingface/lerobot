from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_questsim24_act_v2/training_result_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Mixed v2 results")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.evidence_root / "manifest.json"
    hashes_path = args.evidence_root / "hashes.json"
    summary_path = args.evidence_root / "run_summary.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = json.loads(hashes_path.read_text())
    summary = json.loads(summary_path.read_text())

    if sha256_file(hashes_path) != manifest["hashes_sha256"]:
        raise RuntimeError("hashes.json identity mismatch")
    if sha256_file(summary_path) != manifest["run_summary_sha256"]:
        raise RuntimeError("run_summary.json identity mismatch")
    verified_entries = 0
    for name, entry in hashes["entries"].items():
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Byte count mismatch for {name}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        verified_entries += 1
    if manifest["selected_model_sha256"] != summary["full_training"]["checkpoints"]["100000"]:
        raise RuntimeError("Selected model identity mismatch")
    if any(summary["boundaries"].values()):
        raise RuntimeError("Offline-only boundary evidence is not all false")

    result = {
        "schema": "task1_picklift_mixed_v2_training_result_verification_v1",
        "status": "pass",
        "evidence_root": str(args.evidence_root),
        "manifest_sha256": sha256_file(manifest_path),
        "hashes_sha256": sha256_file(hashes_path),
        "run_summary_sha256": sha256_file(summary_path),
        "verified_hash_entries": verified_entries,
        "selected_model_sha256": manifest["selected_model_sha256"],
        "offline_only_boundaries_all_false": True,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
