import argparse
from pathlib import Path

from lerobot import available_datasets
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub

SOURCE_FORMAT = {
    "aloha": "aloha_hdf5",
    "pusht": "pusht_zarr",
    "xarm": "xarm_pkl",
    "umi": "umi_zarr",
}


def convert_datasets(
    local_dir: Path,
    raw_repo_ids: list[str],
    push_repo: str,
    vcodec: str,
    pix_fmt: str,
    g: int,
    crf: int,
    dry_run: bool = False,
) -> None:
    if len(raw_repo_ids) == 1 and raw_repo_ids[0].lower() == "all":
        raw_repo_ids = [f"cadene/{id_.split('/')[1]}_raw" for id_ in available_datasets]

    for dataset_repo_id_raw in raw_repo_ids:
        dataset_id_raw = dataset_repo_id_raw.split("/")[1]
        dataset_id = dataset_id_raw[:-4]
        dataset_repo_id_push = f"{push_repo}/{dataset_id}"
        raw_format = SOURCE_FORMAT[dataset_id.split("_")[0]]
        local_dataset_dir = local_dir / dataset_repo_id_raw
        encoding = {
            "vcodec": vcodec,
            "pix_fmt": pix_fmt,
            "g": g,
            "crf": crf,
        }

        if not dry_run:
            if not (local_dataset_dir).is_dir():
                download_raw(local_dataset_dir, dataset_repo_id_raw)

            push_dataset_to_hub(
                local_dataset_dir, raw_format=raw_format, repo_id=dataset_repo_id_push, encoding=encoding
            )
        else:
            print("DRY RUN:")
            print(f"    - downloading {dataset_repo_id_raw} into {local_dataset_dir}")
            print(f"    - encoding it with {encoding=}")
            print(f"    - pushing it to {dataset_repo_id_push}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data"),
        help="Directory where raw datasets will be downloaded and encoded before being uploaded.",
    )
    parser.add_argument(
        "--raw-repo-ids",
        type=str,
        nargs="*",
        default=["all"],
        help="Dataset repo ids. if 'all', the list from `available_datasets` will be "
        "used and raw datasets will be fetched from the 'cadene/' repo.",
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
        "--dry-run",
        type=int,
        default=0,
        help="If not set to 0, this script won't download or upload anything.",
    )
    args = parser.parse_args()
    convert_datasets(**vars(args))


if __name__ == "__main__":
    main()
