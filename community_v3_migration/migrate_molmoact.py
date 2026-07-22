"""Migrate the SO-100/101 datasets referenced by ``allenai/MolmoAct2-SO100_101-Dataset``.

That repo does NOT store the datasets themselves; it lists them. Each
``language_annotations/{user}/{dataset}/...`` folder name is the HF repo id of a *standalone*
LeRobotDataset. This script derives those repo ids, drops any already present in
``lerobot/community_dataset_v3`` (already migrated) and in the destination (resume), then runs
the exact same per-dataset pipeline as ``run_migration.py`` on each remaining standalone repo
(download whole repo -> SO-arm joint fix -> v2.1->v3.0 convert -> card -> upload -> cleanup).

  python migrate_molmoact.py --dst-repo lerobot/community_dataset_v3 --work-dir ./molmo_work
Flags mirror run_migration.py: --only-classify, --no-push, --folder-name USER/DATASET [...],
--limit N, --reference-repo (the "already migrated" set to skip against).
"""
import argparse
import csv
import shutil
import sys
import traceback
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import classify  # noqa: E402
from run_migration import already_done, list_datasets, migrate_one  # noqa: E402

LIST_REPO = "allenai/MolmoAct2-SO100_101-Dataset"
REFERENCE_REPO = "lerobot/community_dataset_v3"  # the "already migrated" set to skip against
ANNOTATIONS_PREFIX = "language_annotations/"


def list_molmoact_datasets(api: HfApi, repo: str = LIST_REPO) -> list[str]:
    """Standalone dataset repo ids (``{user}/{dataset}``) derived from the folder names under
    ``language_annotations/`` in the MolmoAct listing repo."""
    files = api.list_repo_files(repo, repo_type="dataset")
    return sorted({"/".join(f.split("/")[1:3]) for f in files
                   if f.startswith(ANNOTATIONS_PREFIX) and len(f.split("/")) >= 3})


def pending_datasets(api: HfApi, subs: list[str], dst_repo: str | None,
                     reference_repo: str, no_upload: bool, only_classify: bool) -> list[str]:
    """Drop ids already in the reference repo (already migrated) and, unless classify/no-push,
    ids already in the destination repo (resume)."""
    skip = set(list_datasets(api, reference_repo))
    if not only_classify and not no_upload and dst_repo and dst_repo != reference_repo:
        skip |= {p[: -len("/meta/info.json")] for p in api.list_repo_files(dst_repo, repo_type="dataset")
                 if p.endswith("/meta/info.json")}
    return [s for s in subs if s not in skip]


def main():
    ap = argparse.ArgumentParser(
        description="Migrate the standalone SO-100/101 datasets listed by "
                    f"{LIST_REPO} to LeRobotDataset v3.0 (degrees), skipping any already present "
                    f"in --reference-repo. One dataset at a time (download -> fix -> convert -> "
                    "upload -> cleanup); resumable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dst-repo", default=REFERENCE_REPO, metavar="ORG/NAME",
                    help="Destination HF dataset repo to push the converted v3.0 datasets to "
                         "(created if missing).")
    ap.add_argument("--reference-repo", default=REFERENCE_REPO, metavar="ORG/NAME",
                    help="Repo whose datasets are considered already migrated and skipped.")
    ap.add_argument("--work-dir", default="./molmo_work", metavar="DIR",
                    help="Local scratch directory (one dataset lives here at a time on a push run).")
    ap.add_argument("--manifest", default="manifest_molmoact.csv", metavar="CSV",
                    help="CSV log appended to as datasets are processed. Reused across resumed runs.")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Process only the first N pending datasets (alphabetical). Ignored with "
                         "--folder-name.")
    ap.add_argument("--folder-name", nargs="+", default=None, metavar="USER/DATASET",
                    help="One or more specific standalone repo ids to process (must appear in the "
                         f"{LIST_REPO} listing).")
    ap.add_argument("--only-classify", action="store_true",
                    help="Detect robot type + joint encoding and write the manifest only; no "
                         "download of data, convert, or push.")
    ap.add_argument("--no-push", action="store_true",
                    help="Fix + convert locally but do NOT upload; output kept under --work-dir.")
    args = ap.parse_args()
    no_upload = args.no_push

    api = HfApi()
    all_ids = list_molmoact_datasets(api)
    if args.folder_name:
        wanted = {n.strip("/") for n in args.folder_name}
        subs = [s for s in all_ids if s in wanted]
        missing = wanted - set(subs)
        if missing:
            print(f"warning: not in {LIST_REPO} listing: {', '.join(sorted(missing))}", file=sys.stderr)
    else:
        subs = pending_datasets(api, all_ids, args.dst_repo, args.reference_repo, no_upload, args.only_classify)
        if args.limit:
            subs = subs[: args.limit]
    print(f"{len(subs)} dataset(s) to process (of {len(all_ids)} listed)", file=sys.stderr)
    if not subs:
        return

    if not args.only_classify and not no_upload:
        api.create_repo(args.dst_repo, repo_type="dataset", exist_ok=True)
    dst_files = set() if (args.only_classify or no_upload) else set(
        api.list_repo_files(args.dst_repo, repo_type="dataset"))

    first = not Path(args.manifest).exists()
    with open(args.manifest, "a", newline="") as mf:
        w = None
        for i, sub in enumerate(subs):
            try:
                if args.only_classify:
                    from huggingface_hub import snapshot_download
                    local = Path(args.work_dir) / sub
                    snapshot_download(repo_id=sub, repo_type="dataset", local_dir=str(local),
                                      allow_patterns=["meta/*"])
                    row = {"root": sub, **classify(local)}
                    shutil.rmtree(Path(args.work_dir) / sub.split("/")[0], ignore_errors=True)
                elif not no_upload and already_done(api, args.dst_repo, sub, dst_files):
                    row = {"root": sub, "action": "skipped: already present in destination repo"}
                else:
                    row = migrate_one(api, args.dst_repo, sub, args.work_dir, no_upload, standalone=True)
            except Exception as e:
                row = {"root": sub, "action": f"ERROR: {e}"}
                traceback.print_exc()
            if w is None:
                w = csv.DictWriter(mf, fieldnames=sorted(
                    {"root", "robot_type", "is_so", "encoding", "action_dim",
                     "maxabs", "ambiguous", "action", "codebase_version", "note"}))
                if first:
                    w.writeheader()
            w.writerow({k: row.get(k) for k in w.fieldnames})
            mf.flush()
            print(f"[{i+1}/{len(subs)}] {sub}: {row.get('action')}", file=sys.stderr)


if __name__ == "__main__":
    main()
