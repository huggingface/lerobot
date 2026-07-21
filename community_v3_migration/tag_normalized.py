"""Tag the migrated (v3.0) datasets whose SO-arm joints are still in normalized units.

Walks every `{user}/{dataset}` sub-dataset in the destination repo, re-classifies it from its
v3.0 metadata (meta/info.json + meta/stats.json), and adds the `normalized` card tag to any whose
joint state/action are in normalized units (-100..100 / 0..100) rather than physical degrees. These
are the datasets run_migration.py left un-converted (uncalibrated -> APPROXIMATE). Idempotent:
already-tagged datasets are skipped. Dry-run by default; pass --yes to actually push the edited card.

  python tag_normalized.py --dst-repo lerobot/community_dataset_v3            # dry-run
  python tag_normalized.py --dst-repo lerobot/community_dataset_v3 --yes
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from huggingface_hub import DatasetCard, HfApi, hf_hub_download

from classify import (
    encoding_from_bounds,
    is_end_effector,
    is_mislabeled_so,
    is_so_robot_type,
    load_info,
    so_joint_count,
)
from run_migration import NORMALIZED_TAG

DST_REPO = "lerobot/community_dataset_v3"


def is_normalized(root: Path) -> bool:
    """True when the SO-arm joints of a v3.0 sub-dataset at ``root`` are still normalized."""
    info = load_info(root)
    rt = info.get("robot_type", "") or ""
    if is_end_effector(info) or not is_so_robot_type(rt) or is_mislabeled_so(info):
        return False
    stats = json.loads((root / "meta" / "stats.json").read_text())
    key = next((k for k in ("action", "observation.state") if k in stats), None)
    if key is None:
        return False
    lo = np.asarray(stats[key]["min"], dtype=float)
    hi = np.asarray(stats[key]["max"], dtype=float)
    n = so_joint_count(info, key) or len(hi)
    return encoding_from_bounds(lo[:n], hi[:n], rt)["encoding"] == "normalized"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst-repo", default=DST_REPO, metavar="ORG/NAME",
                    help="Destination HF dataset repo whose sub-datasets are inspected and tagged.")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Inspect only the first N sub-datasets (alphabetical). Useful for a smoke test.")
    ap.add_argument("--yes", action="store_true",
                    help="Actually push the tag. Without it, only prints what would be tagged (dry-run).")
    args = ap.parse_args()

    api = HfApi()
    subs = sorted(p[: -len("/meta/info.json")]
                  for p in api.list_repo_files(args.dst_repo, repo_type="dataset")
                  if p.endswith("/meta/info.json"))
    if args.limit:
        subs = subs[: args.limit]
    print(f"{len(subs)} sub-datasets in {args.dst_repo}", file=sys.stderr)

    tagged, already, failed = [], [], []
    for i, sub in enumerate(subs):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for f in ("meta/info.json", "meta/stats.json"):
                    hf_hub_download(args.dst_repo, f"{sub}/{f}", repo_type="dataset", local_dir=tmp)
                if not is_normalized(Path(tmp) / sub):
                    continue
                readme = hf_hub_download(args.dst_repo, f"{sub}/README.md", repo_type="dataset", local_dir=tmp)
                card = DatasetCard.load(readme)
                card_tags = list(card.data.tags or [])
                if NORMALIZED_TAG in card_tags:
                    already.append(sub)
                    continue
                if not args.yes:
                    tagged.append(sub)
                    continue
                card.data.tags = card_tags + [NORMALIZED_TAG]
                card.save(readme)
                for attempt in range(1, 4):  # transient Hub ReadTimeouts are common; retry w/ backoff
                    try:
                        api.upload_file(path_or_fileobj=readme, path_in_repo=f"{sub}/README.md",
                                        repo_id=args.dst_repo, repo_type="dataset",
                                        commit_message=f"Tag {sub} '{NORMALIZED_TAG}'")
                        tagged.append(sub)
                        break
                    except Exception as e:
                        if attempt == 3:
                            failed.append(sub)
                            print(f"FAILED {sub}: {e}", file=sys.stderr)
                        else:
                            time.sleep(2 ** attempt)
        except Exception as e:
            failed.append(sub)
            print(f"ERROR {sub}: {e}", file=sys.stderr)
        print(f"[{i + 1}/{len(subs)}] {sub}", file=sys.stderr)

    verb = "tagged" if args.yes else "would tag"
    for sub in tagged:
        print(f"{verb}: {sub}")
    print(f"\n{len(tagged)} {verb} '{NORMALIZED_TAG}', {len(already)} already tagged, "
          f"{len(failed)} failed.", file=sys.stderr)
    if not args.yes and tagged:
        print("dry-run: nothing pushed. re-run with --yes to apply.", file=sys.stderr)


if __name__ == "__main__":
    main()
