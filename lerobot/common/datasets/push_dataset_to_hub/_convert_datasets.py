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
    raw_dir: Path, vcodec: str, pix_fmt: str, g: int, crf: int, dry_run: bool = False
) -> None:
    for repo_id in available_datasets:
        name = repo_id.split("/")[1]
        name_raw = f"{name}_raw"
        repo_id_raw = f"cadene/{name_raw}"
        raw_format = SOURCE_FORMAT[name.split("_")[0]]

        if not dry_run:
            download_raw(raw_dir, repo_id_raw)
            push_dataset_to_hub(raw_dir / name_raw, raw_format=raw_format, repo_id=repo_id)
        else:
            print("DRY RUN:")
            print(f"    - downloading {repo_id_raw} into {raw_dir / name_raw}")
            print(f"    - encoding it with {vcodec=}, {pix_fmt=}, {g=}, {crf=}")
            print(f"    - pushing it to {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory where raw datasets will be downloaded and encoded before being uploaded.",
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
