"""Extract one sub-dataset from a LeRobotDataset monorepo into a standalone HF dataset repo.

Downloads only ``{src-repo}/{folder}/...`` (scoped listing, no whole-repo walk) and re-uploads
its contents at the ROOT of a NEW dataset repo, so the result is a self-contained LeRobotDataset.

  python extract_dataset.py --folder wannrrr/etnai --dst-repo CarolinePascal/etnai
"""
import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_migration import download_subfolder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--folder", required=True, metavar="USER/DATASET",
                    help="Sub-dataset path within the source monorepo, e.g. 'wannrrr/etnai'.")
    ap.add_argument("--dst-repo", required=True, metavar="ORG/NAME",
                    help="New standalone destination dataset repo (must differ from the source).")
    ap.add_argument("--src-repo", default="lerobot/community_dataset_v3", metavar="ORG/NAME",
                    help="Source monorepo to pull the sub-dataset from.")
    ap.add_argument("--work-dir", default="./extract_work", help="Local scratch directory.")
    ap.add_argument("--private", action="store_true", help="Create the destination repo as private.")
    args = ap.parse_args()

    if args.dst_repo in (args.src_repo, args.folder):
        ap.error("--dst-repo must be a new repo name, distinct from the source repo/folder.")

    local = Path(args.work_dir) / args.folder
    if local.parent.exists():
        shutil.rmtree(local.parent, ignore_errors=True)

    print(f"downloading {args.src_repo}/{args.folder} ...", file=sys.stderr)
    download_subfolder(args.folder, args.work_dir, repo=args.src_repo)
    if not (local / "meta" / "info.json").exists():
        ap.error(f"'{args.folder}' is not a LeRobotDataset (no meta/info.json) in {args.src_repo}")

    api = HfApi()
    api.create_repo(args.dst_repo, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"uploading -> {args.dst_repo} ...", file=sys.stderr)
    api.upload_folder(repo_id=args.dst_repo, repo_type="dataset", folder_path=str(local),
                      commit_message=f"Standalone copy of {args.folder} from {args.src_repo}")
    shutil.rmtree(Path(args.work_dir) / args.folder.split("/")[0], ignore_errors=True)
    print(f"done: https://huggingface.co/datasets/{args.dst_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
