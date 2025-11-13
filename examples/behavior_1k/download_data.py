import shutil

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--local-dir", type=str, required=True)
    parser.add_argument("--force-download", action="store_true")

    args = parser.parse_args()

    if args.force_download:
        shutil.rmtree(args.local_dir, ignore_errors=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        force_download=args.force_download,
        max_workers=args.max_workers,
        local_dir=args.local_dir,
        ignore_patterns=["annotations/*"],  # NOTE(fracapuano): Dropping textual annotations right now
    )
