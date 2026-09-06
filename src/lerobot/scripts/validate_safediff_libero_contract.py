"""Compare ordinary LIBERO and LIBERO-Safety dataset/action contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from lerobot.policies.safediff_vla.libero_contracts import (
    LIBERO_ACTION_CONTRACT,
    LIBERO_SAFETY_ACTION_CONTRACT,
    compare_dataset_contracts,
)


def _load_info(repo_id: str, revision: str | None, cache_dir: Path | None) -> dict:
    path = hf_hub_download(
        repo_id, "meta/info.json", repo_type="dataset", revision=revision, cache_dir=cache_dir
    )
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default="lerobot/libero")
    parser.add_argument("--target-repo", default="LIBERO-Safety/libero_safety")
    parser.add_argument("--source-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print only JSON")
    args = parser.parse_args()
    report = compare_dataset_contracts(
        _load_info(args.source_repo, args.source_revision, args.cache_dir),
        _load_info(args.target_repo, args.target_revision, args.cache_dir),
        LIBERO_ACTION_CONTRACT,
        LIBERO_SAFETY_ACTION_CONTRACT,
    )
    if not args.json:
        print(f"source: {args.source_repo}")
        print(f"target: {args.target_repo}")
        print(f"shape-compatible: {report['compatible_by_shape_only']}")
        print(f"semantically interchangeable: {report['semantically_interchangeable']}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
