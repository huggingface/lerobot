"""
This file contains all obsolete download scripts. They are centralized here to not have to load
useless dependencies when using datasets.
"""

import io
import logging
import shutil
from pathlib import Path

import tqdm

ALOHA_RAW_URLS_DIR = "lerobot/common/datasets/push_dataset_to_hub/_aloha_raw_urls"


def download_raw(raw_dir, dataset_id):
    if "pusht" in dataset_id:
        download_pusht(raw_dir)
    elif "xarm" in dataset_id:
        download_xarm(raw_dir)
    elif "aloha" in dataset_id:
        download_aloha(raw_dir, dataset_id)
    elif "umi" in dataset_id:
        download_umi(raw_dir)
    else:
        raise ValueError(dataset_id)


def download_and_extract_zip(url: str, destination_folder: Path) -> bool:
    import zipfile

    import requests

    print(f"downloading from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)

        zip_file = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                zip_file.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()

        zip_file.seek(0)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(destination_folder)


def download_pusht(raw_dir: str):
    pusht_url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract_zip(pusht_url, raw_dir)
    # file is created inside a useful "pusht" directory, so we move it out and delete the dir
    zarr_path = raw_dir / "pusht_cchi_v7_replay.zarr"
    shutil.move(raw_dir / "pusht" / "pusht_cchi_v7_replay.zarr", zarr_path)
    shutil.rmtree(raw_dir / "pusht")


def download_xarm(raw_dir: Path):
    """Download all xarm datasets at once"""
    import zipfile

    import gdown

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # from https://github.com/fyhMer/fowm/blob/main/scripts/download_datasets.py
    url = "https://drive.google.com/uc?id=1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya"
    zip_path = raw_dir / "data.zip"
    gdown.download(url, str(zip_path), quiet=False)
    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zip_f:
        for pkl_path in zip_f.namelist():
            if pkl_path.startswith("data/xarm") and pkl_path.endswith(".pkl"):
                zip_f.extract(member=pkl_path)
                # move to corresponding raw directory
                extract_dir = pkl_path.replace("/buffer.pkl", "")
                raw_pkl_path = raw_dir / "buffer.pkl"
                shutil.move(pkl_path, raw_pkl_path)
                shutil.rmtree(extract_dir)
    zip_path.unlink()


def download_aloha(raw_dir: Path, dataset_id: str):
    import gdown

    subset_id = dataset_id.replace("aloha_", "")
    urls_path = Path(ALOHA_RAW_URLS_DIR) / f"{subset_id}.txt"
    assert urls_path.exists(), f"{subset_id}.txt not found in '{ALOHA_RAW_URLS_DIR}' directory."

    with open(urls_path) as f:
        # strip lines and ignore empty lines
        urls = [url.strip() for url in f if url.strip()]

    # sanity check
    for url in urls:
        assert (
            "drive.google.com/drive/folders" in url or "drive.google.com/file" in url
        ), f"Wrong url provided '{url}' in file '{urls_path}'."

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Start downloading from google drive for {dataset_id}")
    for url in urls:
        if "drive.google.com/drive/folders" in url:
            # when a folder url is given, download up to 50 files from the folder
            gdown.download_folder(url, output=str(raw_dir), remaining_ok=True)

        elif "drive.google.com/file" in url:
            # because of the 50 files limit per folder, we download the remaining files (file by file)
            gdown.download(url, output=str(raw_dir), fuzzy=True)

    logging.info(f"End downloading from google drive for {dataset_id}")


def download_umi(raw_dir: Path):
    url_cup_in_the_wild = "https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip"
    zarr_path = raw_dir / "cup_in_the_wild.zarr"

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract_zip(url_cup_in_the_wild, zarr_path)


if __name__ == "__main__":
    data_dir = Path("data")
    dataset_ids = [
        "pusht",
        "xarm_lift_medium",
        "xarm_lift_medium_replay",
        "xarm_push_medium",
        "xarm_push_medium_replay",
        "aloha_mobile_cabinet",
        "aloha_mobile_chair",
        "aloha_mobile_elevator",
        "aloha_mobile_shrimp",
        "aloha_mobile_wash_pan",
        "aloha_mobile_wipe_wine",
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
        "aloha_static_battery",
        "aloha_static_candy",
        "aloha_static_coffee",
        "aloha_static_coffee_new",
        "aloha_static_cups_open",
        "aloha_static_fork_pick_up",
        "aloha_static_pingpong_test",
        "aloha_static_pro_pencil",
        "aloha_static_screw_driver",
        "aloha_static_tape",
        "aloha_static_thread_velcro",
        "aloha_static_towel",
        "aloha_static_vinh_cup",
        "aloha_static_vinh_cup_left",
        "aloha_static_ziploc_slide",
        "umi_cup_in_the_wild",
    ]
    for dataset_id in dataset_ids:
        raw_dir = data_dir / f"{dataset_id}_raw"
        download_raw(raw_dir, dataset_id)
