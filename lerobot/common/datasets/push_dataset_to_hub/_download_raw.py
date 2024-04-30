"""
This file contains all obsolete download scripts. They are centralized here to not have to load
useless dependencies when using datasets.
"""

import io
import logging
import shutil
from pathlib import Path

import tqdm


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
    # TODO(rcadene): remove gdown and use hugging face download instead
    import gdown

    logging.warning(
        "Aloha download is broken and requires a custom version of gdown which is not limited on number of files"
    )

    folder_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/drive/folders/1RgyD0JgTX30H4IM5XZn8I3zSV_mr8pyF",
        "aloha_sim_insertion_scripted": "https://drive.google.com/drive/folders/1TsojQQSXtHEoGnqgJ3gmpPQR2DPLtS2N",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/drive/folders/1sc-E4QYW7A0o23m1u2VWNGVq5smAsfCo",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj",
    }

    ep48_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/file/d/18Cudl6nikDtgRolea7je8iF_gGKzynOP/view?usp=drive_link",
        "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/1wfMSZ24oOh5KR_0aaP3Cnu_c4ZCveduB/view?usp=drive_link",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/18smMymtr8tIxaNUQ61gW6dG50pt3MvGq/view?usp=drive_link",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1pnGIOd-E4-rhz2P3VxpknMKRZCoKt6eI/view?usp=drive_link",
    }

    ep49_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/file/d/1C1kZYyROzs-PrLc0SkDgUgMi4-L3lauE/view?usp=drive_link",
        "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/17EuCUWS6uCCr6yyNzpXdcdE-_TTNCKtf/view?usp=drive_link",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/1Nk7l53d9sJoGDBKAOnNrExX5nLacATc6/view?usp=drive_link",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1GKReZHrXU73NMiC5zKCq_UtqPVtYq8eo/view?usp=drive_link",
    }
    num_episodes = {  # noqa: F841 # we keep this for reference
        "aloha_sim_insertion_human": 50,
        "aloha_sim_insertion_scripted": 50,
        "aloha_sim_transfer_cube_human": 50,
        "aloha_sim_transfer_cube_scripted": 50,
    }

    episode_len = {  # noqa: F841 # we keep this for reference
        "aloha_sim_insertion_human": 500,
        "aloha_sim_insertion_scripted": 400,
        "aloha_sim_transfer_cube_human": 400,
        "aloha_sim_transfer_cube_scripted": 400,
    }

    cameras = {  # noqa: F841 # we keep this for reference
        "aloha_sim_insertion_human": ["top"],
        "aloha_sim_insertion_scripted": ["top"],
        "aloha_sim_transfer_cube_human": ["top"],
        "aloha_sim_transfer_cube_scripted": ["top"],
    }

    assert dataset_id in folder_urls
    assert dataset_id in ep48_urls
    assert dataset_id in ep49_urls

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(folder_urls[dataset_id], output=str(raw_dir))

    # because of the 50 files limit per directory, two files episode 48 and 49 were missing
    gdown.download(ep48_urls[dataset_id], output=str(raw_dir / "episode_48.hdf5"), fuzzy=True)
    gdown.download(ep49_urls[dataset_id], output=str(raw_dir / "episode_49.hdf5"), fuzzy=True)


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
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
        "umi_cup_in_the_wild",
    ]
    for dataset_id in dataset_ids:
        raw_dir = data_dir / f"{dataset_id}_raw"
        download_raw(raw_dir, dataset_id)
