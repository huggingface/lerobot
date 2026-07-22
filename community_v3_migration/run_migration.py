"""End-to-end migration of the community_dataset_v3 monorepo to v3.0 + SO-arm degrees.

For each `{user}/{dataset}` sub-dataset: stream-download it, fix SO-arm joint values
(if applicable), run the stock v2.1->v3.0 structural converter locally, upload the v3.0
result under the same path into a NEW repo, then delete the local copy. Resumable.

  uv run python run_migration.py --dst-repo HuggingFaceVLA/community_dataset_v3_degrees \
      --work-dir /big/disk/cdv3_work --manifest manifest.csv
Flags: --only-classify (just write manifest), --no-push (fix+convert locally, keep output,
       no upload), --folder-name A [B ...] (target specific dataset folders), --limit N.
Uncalibrated `normalized` datasets keep their normalized joint units (flagged APPROXIMATE on
the card); paste fitted CANON ranges in so_arm_frame.py to convert them to degrees instead.
"""
import argparse, csv, json, shutil, sys, traceback
from pathlib import Path
from huggingface_hub import HfApi

import so_arm_frame
from classify import classify, is_end_effector, load_info
from fix_dataset import (
    data_video_episode_mismatch,
    fix_dataset_in_place,
    reconcile_episode_count,
    reindex_episodes,
)

SRC_REPO = "HuggingFaceVLA/community_dataset_v3"
NORMALIZED_TAG = "normalized"  # card tag for datasets whose SO joints stay in normalized units


def download_subfolder(sub: str, work_dir: str, patterns: list[str] | None = None, repo: str = SRC_REPO) -> None:
    """Download only ``{repo}/{sub}/...`` into ``work_dir``.

    ``snapshot_download`` walks the entire repo tree (``list_repo_tree(recursive=True)``
    with no path scope) before applying ``allow_patterns``. On this 791-dataset monorepo
    that whole-repo enumeration is pathologically slow and looks like a hang. Listing the
    scoped ``path_in_repo=sub`` subtree and fetching its files directly avoids it.
    """
    from fnmatch import fnmatch

    from huggingface_hub import hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    for entry in api.list_repo_tree(repo, path_in_repo=sub, repo_type="dataset", recursive=True):
        if not isinstance(entry, RepoFile):
            continue
        if patterns and not any(fnmatch(entry.path, pat) for pat in patterns):
            continue
        hf_hub_download(repo, filename=entry.path, repo_type="dataset", local_dir=work_dir)


def list_datasets(api: HfApi, repo: str) -> list[str]:
    files = api.list_repo_files(repo, repo_type="dataset")
    roots = {p[: -len("/meta/info.json")] for p in files if p.endswith("/meta/info.json")}
    return sorted(roots)


def resolve_folders(api: HfApi, repo: str, names: list[str]) -> list[str]:
    """Expand each --folder-name into concrete dataset roots (folders that contain
    meta/info.json). A name may be a full dataset path (returned as-is) or a namespace/prefix
    like 'Beegbrain' (expanded to every dataset beneath it). Unknown names pass through so
    they surface as a clear per-item error instead of a confusing FileNotFoundError."""
    out: list[str] = []
    for name in names:
        name = name.strip("/")
        try:
            paths = [e.path for e in api.list_repo_tree(
                repo, path_in_repo=name, recursive=True, repo_type="dataset")]
        except Exception:
            out.append(name)  # let it fail loudly downstream
            continue
        roots = sorted({p[: -len("/meta/info.json")] for p in paths if p.endswith("/meta/info.json")})
        out.extend(roots or [name])
    seen: set[str] = set()
    return [r for r in out if not (r in seen or seen.add(r))]


def already_done(api: HfApi, dst: str, sub: str, dst_files: set[str]) -> bool:
    return f"{sub}/meta/info.json" in dst_files  # present in target => skip (resume)


def _write_dataset_card(local: Path, sub: str, result: dict, standalone: bool = False) -> None:
    """Regenerate the sub-dataset's card the way LeRobot does (create_lerobot_dataset_card
    from meta/info.json), then append a migration section documenting provenance and the
    joint-encoding fix. When ``standalone`` is set, ``sub`` is itself the source dataset's HF
    repo id (rather than a folder inside the ``SRC_REPO`` monorepo)."""
    enc = result.get("encoding")
    converted_degrees = bool(result.get("converted"))
    approx = enc == "normalized" and not so_arm_frame.CANON_IS_CALIBRATED

    enc_labels = {
        "degrees_old": "legacy degrees (old community frame, pre-#777 convention)",
        "degrees_new": "degrees (recorded with `use_degrees=True`)",
        "normalized": "normalized units (joints -100..100, gripper 0..100)",
        "radians": "radians",
        "unknown": "undetermined",
    }
    joint_actions = {
        "degrees_old": "per-joint offsets and axis directions corrected to the post-#777 frame (values stay in degrees)",
        "degrees_new": "already in the post-#777 degrees frame; values unchanged",
        "normalized": ("un-normalized to physical degrees using calibrated joint ranges"
                       if converted_degrees else
                       "left in normalized units (-100..100 joints, 0..100 gripper); NOT converted to degrees"),
        "radians": "left unchanged (already in radians)",
        "unknown": "left unchanged (encoding could not be determined)",
    }

    lines = [
        "## Migration to LeRobotDataset v3.0",
        "",
        "Migrated to LeRobotDataset **v3.0**"
        + (" with SO-100/101 joint state/action mapped to the post-#777 physical frame (in degrees)."
           if converted_degrees else "."),
        "",
        (f"- Source: [`{sub}`](https://huggingface.co/datasets/{sub})" if standalone else
         f"- Source: [`{SRC_REPO}`](https://huggingface.co/datasets/{SRC_REPO}/tree/main/{sub}) (`{sub}`)"),
        "- Codebase version: v2.1 -> v3.0",
    ]
    if result.get("is_so"):
        lines += [
            f"- Original joint encoding: {enc_labels.get(enc, enc)}",
            f"- Joint values: {joint_actions.get(enc, 'left unchanged')}",
            f"- Robot type: `{result.get('robot_type')}`",
            f"- Action dimension: {result.get('action_dim')}",
        ]
    else:
        lines += ["- Joint values: not applicable (not an SO-100/101 dataset)"]
    if approx:
        lines += ["", "> **Note:** per-robot calibration was unavailable, so joint state/action were "
                  "left in their original *normalized* units (-100..100 joints, 0..100 gripper) rather "
                  "than converted to physical degrees. Treat these joint values as APPROXIMATE."]
    if result.get("ambiguous"):
        lines += ["", "> **Note:** joint-encoding detection was flagged ambiguous; conversion used the "
                  "best-guess encoding above and may warrant manual review."]

    section = "\n".join(lines) + "\n"

    readme = local / "README.md"
    try:
        try:
            from lerobot.datasets.utils import create_lerobot_dataset_card
        except ImportError:
            from lerobot.common.datasets.utils import create_lerobot_dataset_card

        class _Info(dict):            # satisfies both the dict and .to_dict() card variants
            def to_dict(self):
                return dict(self)

        rt = result.get("robot_type") or None
        tags = [rt] if rt else []
        if enc == "normalized" and not converted_degrees:
            tags.append(NORMALIZED_TAG)
        card = create_lerobot_dataset_card(
            tags=tags or None,
            dataset_info=_Info(load_info(local)),
            license="apache-2.0",
            repo_id=sub,
        )
        card.text = card.text.rstrip() + "\n\n" + section
        card.save(str(readme))
    except Exception:
        # LeRobot card generator unavailable at runtime: keep the standalone migration note.
        if readme.exists():
            readme.write_text(readme.read_text().rstrip() + "\n\n" + section)
        else:
            readme.write_text(f"# {sub}\n\n" + section)


def migrate_one(api, dst_repo, sub, work_dir, no_upload, src_repo: str = SRC_REPO,
                standalone: bool = False) -> dict:
    local = Path(work_dir) / sub
    if local.parent.exists():
        shutil.rmtree(local.parent, ignore_errors=True)  # clean any partial
    if standalone:
        # ``sub`` is a self-contained HF dataset repo (not a monorepo folder): pull it whole.
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=sub, repo_type="dataset", local_dir=str(local))
    else:
        download_subfolder(sub, work_dir, repo=src_repo)

    info = load_info(local)
    if info.get("codebase_version") != "v2.1":
        return {"root": sub, "action": f"skipped: source codebase is {info.get('codebase_version')} (expected v2.1)"}
    if is_end_effector(info):
        return {"root": sub, "robot_type": info.get("robot_type"),
                "action": "skipped: end-effector (task-space) dataset, out of scope"}
    mismatch = data_video_episode_mismatch(local)
    if mismatch:
        return {"root": sub, "robot_type": info.get("robot_type"),
                "action": f"skipped: {mismatch}"}

    result = fix_dataset_in_place(local)          # SO-arm value fix (or structural_only)

    reconciled = reconcile_episode_count(local)   # align stale meta counts to data+video files
    if reconciled:
        result["action"] = f"{result['action']}; {reconciled}"

    reindexed = reindex_episodes(local)           # compact non-contiguous episode indices to 0..N-1
    if reindexed:
        result["action"] = f"{result['action']}; {reindexed}"

    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
    convert_dataset(repo_id=sub, root=str(local), push_to_hub=False)  # v2.1 -> v3.0, in place

    _write_dataset_card(local, sub, result, standalone=standalone)  # document conversion in the card

    base = {k: result.get(k) for k in
            ("robot_type", "is_so", "encoding", "action_dim", "maxabs", "ambiguous", "action")}
    if no_upload:
        # keep the converted output on disk for inspection; do NOT delete or push
        base["action"] = f"{base['action']}; not pushed (kept locally at {local})"
        return {"root": sub, **base}
    api.upload_folder(repo_id=dst_repo, repo_type="dataset", folder_path=str(local),
                      path_in_repo=sub, commit_message=f"Add {sub} (v3.0, {result['action']})")
    shutil.rmtree(Path(work_dir) / sub.split("/")[0], ignore_errors=True)  # drop after successful push
    return {"root": sub, **base}


def main():
    ap = argparse.ArgumentParser(
        description="Migrate the HuggingFaceVLA/community_dataset_v3 monorepo to LeRobotDataset "
                    "v3.0, converting SO-100/101 joint state/action to physical degrees along "
                    "the way. Processes one sub-dataset at a time (download -> fix -> convert -> "
                    "upload -> cleanup) and is resumable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dst-repo", default=None, metavar="ORG/NAME",
                    help="Destination HF dataset repo to push the converted v3.0 datasets to "
                         "(created if missing). Required unless --no-push or --only-classify.")
    ap.add_argument("--work-dir", default="./cdv3_work", metavar="DIR",
                    help="Local scratch directory used to download, convert, and (unless pushing) "
                         "retain each sub-dataset. Only one dataset lives here at a time on a push run.")
    ap.add_argument("--manifest", default="manifest.csv", metavar="CSV",
                    help="CSV log appended to as datasets are processed (robot_type, detected "
                         "encoding, action taken, errors). Reused across resumed runs.")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Process only the first N sub-datasets (alphabetical). Ignored when "
                         "--folder-name is given. Useful for a quick end-to-end smoke test.")
    ap.add_argument("--folder-name", nargs="+", default=None, metavar="USER/DATASET",
                    help=f"One or more folders WITHIN the {SRC_REPO} monorepo to process. Either a "
                         "full dataset path ('Beegbrain/draw_pixel_art') or a whole namespace "
                         "('Beegbrain'), which expands to every dataset under it. Skips the full "
                         "791-dataset listing.")
    ap.add_argument("--only-classify", action="store_true",
                    help="Detect each dataset's robot type and joint encoding and write the "
                         "manifest, without downloading data, converting, or pushing. Run this "
                         "first to review scope (especially rows flagged ambiguous=True).")
    ap.add_argument("--no-push", action="store_true",
                    help="Fix + convert locally but do NOT upload; the converted v3.0 output is "
                         "kept under --work-dir for inspection instead of being deleted.")
    args = ap.parse_args()
    no_upload = args.no_push
    if not no_upload and not args.only_classify and not args.dst_repo:
        ap.error("--dst-repo is required unless --no-push or --only-classify is set.")

    api = HfApi()
    if args.folder_name:
        subs = resolve_folders(api, SRC_REPO, args.folder_name)
        print(f"targeting {len(subs)} sub-dataset(s): {', '.join(subs)}", file=sys.stderr)
    else:
        subs = list_datasets(api, SRC_REPO)
        if args.limit:
            subs = subs[: args.limit]
        print(f"{len(subs)} sub-datasets found", file=sys.stderr)

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
                    # classify without full download: fetch just the meta/ of this sub
                    download_subfolder(sub, args.work_dir, patterns=[f"{sub}/meta/*"])
                    row = {"root": sub, **classify(Path(args.work_dir) / sub)}
                    shutil.rmtree(Path(args.work_dir) / sub.split("/")[0], ignore_errors=True)
                elif not no_upload and already_done(api, args.dst_repo, sub, dst_files):
                    row = {"root": sub, "action": "skipped: already present in destination repo"}
                else:
                    row = migrate_one(api, args.dst_repo, sub, args.work_dir, no_upload)
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
