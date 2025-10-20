from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def main() -> None:
    base_dir = Path("/home/yihao/.cache/huggingface/lerobot/yihao-brain-bot")
    repo_ids = [
        "yihao-brain-bot/xlerobot-get-water",
        "yihao-brain-bot/xlerobot-get-hershey",
        "yihao-brain-bot/xlerobot-get-kitkat",
        "yihao-brain-bot/xlerobot-get-musketeers",
        "yihao-brain-bot/xlerobot-get-altereco",
    ]

    roots = [base_dir / repo_id.split("/", maxsplit=1)[1] for repo_id in repo_ids]

    missing = [path for path in roots if not path.exists()]
    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The following dataset directories were not found:\n"
            f"  - {formatted}\n"
            "Please verify the paths before running the aggregation."
        )

    aggr_repo_id = "yihao-brain-bot/xlerobot-get"
    aggr_root = base_dir / "xlerobot-get"

    if aggr_root.exists():
        raise FileExistsError(
            f"Destination directory already exists: {aggr_root}\n"
            "Move or delete it before running the aggregation to avoid overwriting data."
        )

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=aggr_repo_id,
        roots=roots,
        aggr_root=aggr_root,
    )

    print(f"Aggregation complete. New dataset stored at: {aggr_root}")


if __name__ == "__main__":
    main()
