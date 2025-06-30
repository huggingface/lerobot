

import json
import logging
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download
import tarfile

import tqdm
from examples.port_datasets.agibot_hdf5.port_agibot import port_agibot
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging
from huggingface_hub import HfApi, HfFileSystem

RAW_REPO_ID = "agibot-world/AgiBotWorld-Alpha"


def download(raw_dir, allow_patterns=None, ignore_patterns=None):
    snapshot_download(
        RAW_REPO_ID,
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

def download_proprio_stats(raw_dir):
    proprio_stats_dir = raw_dir / "proprio_stats"

    if proprio_stats_dir.exists():
        logging.info("Skipping download proprio stats")
        return

    download(raw_dir, allow_patterns="proprio_stats/*.tar")

    for path in proprio_stats_dir.glob("*.tar"):
        logging.info(f"Untar-ing {path}...")
        with tarfile.open(path, 'r') as tar:
            tar.extractall(path=proprio_stats_dir)
    
        logging.info(f"Deleting {path}...")
        path.unlink()


def download_parameters(raw_dir):
    params_dir = raw_dir / "parameters"

    if params_dir.exists():
        logging.info("Skipping download parameters")
        return

    download(raw_dir, allow_patterns="parameters/*.tar")

    for path in params_dir.glob("*.tar"):
        logging.info(f"Untar-ing {path}...")
        with tarfile.open(path, 'r') as tar:
            tar.extractall(path=params_dir)
        
        logging.info(f"Deleting {path}...")
        path.unlink()
    

def get_observations_files(raw_dir, raw_repo_id):
    files_json_path = raw_dir / "observations_files.json"
    sizes_json_path = raw_dir / "observations_sizes.json"
    if files_json_path.exists() and sizes_json_path.exists():
        with open(files_json_path) as f:
            files = json.load(f)
        with open(sizes_json_path) as f:
            sizes = json.load(f)
        return files, sizes

    api = HfApi()
    files = api.list_repo_files(repo_id=raw_repo_id, repo_type="dataset")
    files = [file for file in files if "observations/" in file]

    fs = HfFileSystem()
    sizes = []
    for file in tqdm.tqdm(files, desc="Downloading file sizes"):
        file_info = fs.info(f"datasets/{raw_repo_id}/{file}")
        size = file_info["size"] / 1000**3
        sizes.append(size)

    # Sort ASC to start with smaller size files
    sizes, files = zip(*sorted(zip(sizes, files)))

    with open(files_json_path, "w") as f:
        json.dump(files, f)
    with open(sizes_json_path, "w") as f:
        json.dump(sizes, f)
    return files, sizes

def display_observations_sizes(files, sizes):
    size_per_task = {}
    for i, (file, size) in enumerate(zip(files, sizes)):
        logging.info(f"{i}/{len(files)}: {file} {size:.2f}GB")

        task = int(file.split('/')[1])

        if task not in size_per_task:
            size_per_task[task] = 0

        size_per_task[task] += size

    for task, size in size_per_task.items():
        logging.info(f"{task} {size:.2f}GB")

    total_size = sum(list(size_per_task.values()))
    logging.info(f"Total size: {total_size:.2f}GB")


def download_meta_data(raw_dir):
    # Download task data
    download(raw_dir, allow_patterns="task_info/task_*.json")

    # Download all camera parameters ~170 GB
    download_parameters(raw_dir)

    # Download all proprio stats ~26 GB
    download_proprio_stats(raw_dir)

def no_depth(tarinfo, path):
    """ Utility to not untar depth data"""
    if "depth" in tarinfo.name:
        return None
    return tarinfo

def main():
    init_logging()

    repo_id = "cadene/agibot_alpha_v30"
    raw_dir = Path("/fsx/remi_cadene/data/AgiBotWorld-Alpha")
    download_meta_data(raw_dir)
    # Get list of tar files containing observation data (containing several episodes each)
    obs_files, obs_sizes = get_observations_files(raw_dir, RAW_REPO_ID)
    display_observations_sizes(obs_files, obs_sizes)

    shard_indices = range(len(obs_files))
    num_shards = len(obs_files)

    # TOOD: remove
    obs_files = obs_files[:2]
    shard_indices = [0,1]

    # Iterate on each subset of episodes
    for shard_index, obs_file in zip(shard_indices, obs_files):

        shard_repo_id = f"{repo_id}_world_{num_shards}_rank_{shard_index}"
        dataset_dir = HF_LEROBOT_HOME / shard_repo_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # Download subset
        download(raw_dir, allow_patterns=obs_file)

        tar_path = raw_dir / obs_file
        with tarfile.open(tar_path, 'r') as tar:
            extracted_files = tar.getnames()

        task_index = int(tar_path.parent.name)
        episode_names = [int(p) for p in extracted_files if '/' not in p]

        # Untar if needed
        if not all([(tar_path.parent / f"{ep_name}").exists() for ep_name in episode_names]):
            logging.info(f"Untar-ing {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=tar_path.parent, filter=no_depth)

        port_agibot(raw_dir, shard_repo_id, task_index, episode_names, push_to_hub=False)

        for ep_name in episode_names:
            shutil.rmtree(tar_path.parent / f"{ep_name}")

        tar_path.unlink()

        # dataset = LeRobotDataset(shard_repo_id, root=dataset_dir)
        # lol=1



if __name__ == "__main__":
    main()