import argparse
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub._download_raw import AVAILABLE_RAW_REPO_IDS
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub


def get_push_repo_id_from_raw(raw_repo_id: str, push_repo: str) -> str:
    dataset_id_raw = raw_repo_id.split("/")[1]
    dataset_id = dataset_id_raw.removesuffix("_raw")
    return f"{push_repo}/{dataset_id}"


def encode_datasets(
    raw_dir: Path,
    raw_repo_ids: list[str],
    push_repo: str,
    vcodec: str,
    pix_fmt: str,
    g: int,
    crf: int,
    local_dir: Path | None = None,
    tests_data_dir: Path | None = None,
    raw_format: str | None = None,
    dry_run: bool = False,
) -> None:
    if len(raw_repo_ids) == 1 and raw_repo_ids[0].lower() == "lerobot-raw":
        raw_repo_ids_format = AVAILABLE_RAW_REPO_IDS
    else:
        if raw_format is None:
            raise ValueError(raw_format)
        raw_repo_ids_format = {id_: raw_format for id_ in raw_repo_ids}

    for raw_repo_id, repo_raw_format in raw_repo_ids_format.items():
        check_repo_id(raw_repo_id)
        dataset_repo_id_push = get_push_repo_id_from_raw(raw_repo_id, push_repo)
        dataset_raw_dir = raw_dir / raw_repo_id
        dataset_dir = local_dir / dataset_repo_id_push if local_dir is not None else None
        encoding = {
            "vcodec": vcodec,
            "pix_fmt": pix_fmt,
            "g": g,
            "crf": crf,
        }

        if not (dataset_raw_dir).is_dir():
            raise NotADirectoryError(dataset_raw_dir)

        if not dry_run:
            push_dataset_to_hub(
                dataset_raw_dir,
                raw_format=repo_raw_format,
                repo_id=dataset_repo_id_push,
                local_dir=dataset_dir,
                encoding=encoding,
                tests_data_dir=tests_data_dir,
            )
        else:
            print(
                f"DRY RUN: {dataset_raw_dir}  -->  {dataset_dir}  -->  {dataset_repo_id_push}@{CODEBASE_VERSION}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data"),
        help="Directory where raw datasets are located.",
    )
    parser.add_argument(
        "--raw-repo-ids",
        type=str,
        nargs="*",
        default=["lerobot-raw"],
        help="""Raw dataset repo ids. if 'lerobot-raw', the keys from `AVAILABLE_RAW_REPO_IDS` will be
            used and raw datasets will be fetched from the 'lerobot-raw/' repo and pushed with their
            associated format. It is assumed that each dataset is located at `raw_dir / raw_repo_id` """,
    )
    parser.add_argument(
        "--raw-format",
        type=str,
        default=None,
        help="""Raw format to use for the raw repo-ids. Must be specified if --raw-repo-ids is not
            'lerobot-raw'""",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--push-repo",
        type=str,
        default="lerobot",
        help="Repo to upload datasets to",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        help="Codec to use for encoding videos",
    )
    parser.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        help="Pixel formats (chroma subsampling) to be used for encoding",
    )
    parser.add_argument(
        "--g",
        type=int,
        default=2,
        help="Group of pictures sizes to be used for encoding.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=30,
        help="Constant rate factors to be used for encoding.",
    )
    parser.add_argument(
        "--tests-data-dir",
        type=Path,
        default=None,
        help=(
            "When provided, save tests artifacts into the given directory "
            "(e.g. `--tests-data-dir tests/data` will save to tests/data/{--repo-id})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="If not set to 0, this script won't download or upload anything.",
    )
    args = parser.parse_args()
    encode_datasets(**vars(args))


if __name__ == "__main__":
    main()
