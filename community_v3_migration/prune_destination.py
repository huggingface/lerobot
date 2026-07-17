"""Prune datasets from the migrated destination repo based on the run manifest(s).

Selects, from the manifest CSV(s) written by run_migration.py / slurm_migrate.py, the datasets to
remove and deletes their folders from the destination repo. Two independent filters:

  --errored        rows whose migration action starts with "ERROR:" (failed to convert; usually
                   absent from the repo, but any partial upload left behind is cleaned up).
  --mislabeled-so  rows whose robot_type claims SO but the dataset isn't a real 6-DOF SO arm
                   (action_dim not a multiple of 6, or a classification note saying so).

Only folders actually present in the destination repo are touched. Dry-run by default; pass --yes
to perform the deletions.

  python prune_destination.py --manifest /fsx/$USER/cdv3_manifests/manifest_*.csv \
      --dst-repo lerobot/community_dataset_v3 --errored --mislabeled-so        # dry-run
  python prune_destination.py --manifest manifest_*.csv --errored --mislabeled-so --yes
"""
import argparse
import sys

import pandas as pd
from huggingface_hub import HfApi

from classify import is_so_robot_type

DST_REPO = "lerobot/community_dataset_v3"


def _is_errored(row) -> bool:
    return str(row.get("action") or "").strip().upper().startswith("ERROR")


def _is_mislabeled_so(row) -> bool:
    if "claims SO but" in str(row.get("note") or ""):
        return True
    if not is_so_robot_type(str(row.get("robot_type") or "")):
        return False
    try:
        return int(float(row["action_dim"])) % 6 != 0
    except (TypeError, ValueError, KeyError):
        return False


def select(df: pd.DataFrame, present: set[str], errored: bool, mislabeled: bool) -> dict[str, str]:
    """Map each dataset root that is present in the repo AND matches an enabled filter to a reason.
    'errored' takes precedence over 'mislabeled-so' when a row matches both."""
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        root = str(row.get("root") or "").strip()
        if not root or root not in present or root in out:
            continue
        if errored and _is_errored(row):
            out[root] = "errored"
        elif mislabeled and _is_mislabeled_so(row):
            out[root] = "mislabeled-so"
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, metavar="CSV",
                    help="One or more manifest CSVs from the migration run (per-rank files ok).")
    ap.add_argument("--dst-repo", default=DST_REPO, metavar="ORG/NAME",
                    help="Destination HF dataset repo to prune.")
    ap.add_argument("--errored", action="store_true",
                    help="Delete datasets that errored during migration.")
    ap.add_argument("--mislabeled-so", action="store_true",
                    help="Delete datasets labelled SO but not a real 6-DOF SO arm.")
    ap.add_argument("--yes", action="store_true",
                    help="Actually delete. Without it, only prints what would be deleted (dry-run).")
    ap.add_argument("--list-missing", action="store_true",
                    help="Report datasets in the manifest that are NOT present in the destination "
                         "repo (grouped by their migration action), then exit without deleting.")
    ap.add_argument("--report", action="store_true",
                    help="List all three categories (errored, mislabeled-so, missing) without "
                         "deleting anything, then exit. '*' marks datasets absent from the repo.")
    args = ap.parse_args()
    if not (args.errored or args.mislabeled_so or args.list_missing or args.report):
        ap.error("enable at least one of: --errored, --mislabeled-so, --list-missing, --report")

    df = pd.concat([pd.read_csv(p) for p in args.manifest], ignore_index=True)
    if "root" not in df.columns:
        ap.error("manifest has no 'root' column; is this a run_migration.py manifest?")
    # A dataset can appear multiple times across per-rank / resumed manifests (e.g. it timed out
    # once then succeeded on retry). Collapse to one row per dataset, preferring a SUCCESS over an
    # errored attempt so a later successful migration isn't shadowed by a stale ERROR row.
    df = (df.assign(_err=df.apply(_is_errored, axis=1))
            .sort_values("_err", kind="stable")
            .drop_duplicates(subset="root", keep="first")
            .drop(columns="_err"))

    api = HfApi()
    present = {p[: -len("/meta/info.json")]
               for p in api.list_repo_files(args.dst_repo, repo_type="dataset")
               if p.endswith("/meta/info.json")}

    if args.report or args.list_missing:
        def _dump(title, sub):
            print(f"== {title} ({len(sub)}) ==")
            for root in sorted(sub):
                print(f"  {' ' if root in present else '*'} {root}")
            print()
        missing = set(df.loc[~df["root"].isin(present), "root"])
        if args.report:
            _dump("errored", set(df.loc[df.apply(_is_errored, axis=1), "root"]))
            _dump("mislabeled-so", set(df.loc[df.apply(_is_mislabeled_so, axis=1), "root"]))
        _dump("missing from repo", missing)
        print(f"{df['root'].nunique()} unique manifest rows, {len(present)} datasets in "
              f"{args.dst_repo}. '*' = absent from repo.", file=sys.stderr)
        return

    to_delete = select(df, present, args.errored, args.mislabeled_so)
    for root, reason in sorted(to_delete.items()):
        print(f"{reason:14s} {root}")
    print(f"\n{len(to_delete)} dataset(s) present in {args.dst_repo} match "
          f"({df['root'].nunique()} rows in manifest, {len(present)} datasets in repo).", file=sys.stderr)

    if not args.yes:
        print("dry-run: nothing deleted. re-run with --yes to delete.", file=sys.stderr)
        return
    for root, reason in sorted(to_delete.items()):
        api.delete_folder(path_in_repo=root, repo_id=args.dst_repo, repo_type="dataset",
                          commit_message=f"Prune {root} ({reason})")
        print(f"deleted {root} ({reason})", file=sys.stderr)


if __name__ == "__main__":
    main()
