"""Functions to create an empty dataset, and populate it with frames."""
# TODO(rcadene, aliberts): to adapt as class methods of next version of LeRobotDataset

import concurrent
import json
import logging
import multiprocessing
import shutil
from pathlib import Path

import torch
import tqdm
from PIL import Image

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.utils.utils import log_say
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)

########################################################################################
# Asynchrounous saving of images on disk
########################################################################################


def safe_stop_image_writer(func):
    # TODO(aliberts): Allow to pass custom exceptions
    # (e.g. ThreadServiceExit, KeyboardInterrupt, SystemExit, UnpluggedError, DynamixelCommError)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            image_writer = kwargs.get("dataset", {}).get("image_writer")
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                stop_image_writer(image_writer, timeout=20)
            raise e

    return wrapper


def save_image(img_tensor, key, frame_index, episode_index, videos_dir: str):
    img = Image.fromarray(img_tensor.numpy())
    path = Path(videos_dir) / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def loop_to_save_images_in_threads(image_queue, num_threads):
    if num_threads < 1:
        raise NotImplementedError(f"Only `num_threads>=1` is supported for now, but {num_threads=} given.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        while True:
            # Blocks until a frame is available
            frame_data = image_queue.get()

            # As usually done, exit loop when receiving None to stop the worker
            if frame_data is None:
                break

            image, key, frame_index, episode_index, videos_dir = frame_data
            futures.append(executor.submit(save_image, image, key, frame_index, episode_index, videos_dir))

        # Before exiting function, wait for all threads to complete
        with tqdm.tqdm(total=len(futures), desc="Writing images") as progress_bar:
            concurrent.futures.wait(futures)
            progress_bar.update(len(futures))


def start_image_writer_processes(image_queue, num_processes, num_threads_per_process):
    if num_processes < 1:
        raise ValueError(f"Only `num_processes>=1` is supported, but {num_processes=} given.")

    if num_threads_per_process < 1:
        raise NotImplementedError(
            "Only `num_threads_per_process>=1` is supported for now, but {num_threads_per_process=} given."
        )

    processes = []
    for _ in range(num_processes):
        process = multiprocessing.Process(
            target=loop_to_save_images_in_threads,
            args=(image_queue, num_threads_per_process),
        )
        process.start()
        processes.append(process)
    return processes


def stop_processes(processes, queue, timeout):
    # Send None to each process to signal them to stop
    for _ in processes:
        queue.put(None)

    # Wait maximum 20 seconds for all processes to terminate
    for process in processes:
        process.join(timeout=timeout)

    # If not terminated after 20 seconds, force termination
    if process.is_alive():
        process.terminate()

    # Close the queue, no more items can be put in the queue
    queue.close()

    # Ensure all background queue threads have finished
    queue.join_thread()


def start_image_writer(num_processes, num_threads):
    """This function abstract away the initialisation of processes or/and threads to
    save images on disk asynchrounously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it returns a dictionary containing a threads pool of size `num_threads`.
    When `num_processes>0`, it returns a dictionary containing a processes pool of size `num_processes`,
    where each subprocess starts their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """
    image_writer = {}

    if num_processes == 0:
        futures = []
        threads_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        image_writer["threads_pool"], image_writer["futures"] = threads_pool, futures
    else:
        # TODO(rcadene): When using num_processes>1, `multiprocessing.Manager().Queue()`
        # might be better than `multiprocessing.Queue()`. Source: https://www.geeksforgeeks.org/python-multiprocessing-queue-vs-multiprocessing-manager-queue
        image_queue = multiprocessing.Queue()
        processes_pool = start_image_writer_processes(
            image_queue, num_processes=num_processes, num_threads_per_process=num_threads
        )
        image_writer["processes_pool"], image_writer["image_queue"] = processes_pool, image_queue

    return image_writer


def async_save_image(image_writer, image, key, frame_index, episode_index, videos_dir):
    """This function abstract away the saving of an image on disk asynchrounously. It uses a dictionary
    called image writer which contains either a pool of processes or a pool of threads.
    """
    if "threads_pool" in image_writer:
        threads_pool, futures = image_writer["threads_pool"], image_writer["futures"]
        futures.append(threads_pool.submit(save_image, image, key, frame_index, episode_index, videos_dir))
    else:
        image_queue = image_writer["image_queue"]
        image_queue.put((image, key, frame_index, episode_index, videos_dir))


def stop_image_writer(image_writer, timeout):
    if "threads_pool" in image_writer:
        futures = image_writer["futures"]
        # Before exiting function, wait for all threads to complete
        with tqdm.tqdm(total=len(futures), desc="Writing images") as progress_bar:
            concurrent.futures.wait(futures, timeout=timeout)
            progress_bar.update(len(futures))
    else:
        processes_pool, image_queue = image_writer["processes_pool"], image_writer["image_queue"]
        stop_processes(processes_pool, image_queue, timeout=timeout)


########################################################################################
# Functions to initialize, resume and populate a dataset
########################################################################################


def init_dataset(
    repo_id,
    root,
    force_override,
    fps,
    video,
    write_images,
    num_image_writer_processes,
    num_image_writer_threads,
):
    local_dir = Path(root) / repo_id
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    episodes_dir = local_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    rec_info_path = episodes_dir / "data_recording_info.json"
    if rec_info_path.exists():
        with open(rec_info_path) as f:
            rec_info = json.load(f)
        num_episodes = rec_info["last_episode_index"] + 1
    else:
        num_episodes = 0

    dataset = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "videos_dir": videos_dir,
        "episodes_dir": episodes_dir,
        "fps": fps,
        "video": video,
        "rec_info_path": rec_info_path,
        "num_episodes": num_episodes,
    }

    if write_images:
        # Initialize processes or/and threads dedicated to save images on disk asynchronously,
        # which is critical to control a robot and record data at a high frame rate.
        image_writer = start_image_writer(
            num_processes=num_image_writer_processes,
            num_threads=num_image_writer_threads,
        )
        dataset["image_writer"] = image_writer

    return dataset


def add_frame(dataset, observation, action):
    if "current_episode" not in dataset:
        # initialize episode dictionary
        ep_dict = {}
        for key in observation:
            if key not in ep_dict:
                ep_dict[key] = []
        for key in action:
            if key not in ep_dict:
                ep_dict[key] = []

        ep_dict["episode_index"] = []
        ep_dict["frame_index"] = []
        ep_dict["timestamp"] = []
        ep_dict["next.done"] = []

        dataset["current_episode"] = ep_dict
        dataset["current_frame_index"] = 0

    ep_dict = dataset["current_episode"]
    episode_index = dataset["num_episodes"]
    frame_index = dataset["current_frame_index"]
    videos_dir = dataset["videos_dir"]
    video = dataset["video"]
    fps = dataset["fps"]

    ep_dict["episode_index"].append(episode_index)
    ep_dict["frame_index"].append(frame_index)
    ep_dict["timestamp"].append(frame_index / fps)
    ep_dict["next.done"].append(False)

    img_keys = [key for key in observation if "image" in key]
    non_img_keys = [key for key in observation if "image" not in key]

    # Save all observed modalities except images
    for key in non_img_keys:
        ep_dict[key].append(observation[key])

    # Save actions
    for key in action:
        ep_dict[key].append(action[key])

    if "image_writer" not in dataset:
        dataset["current_frame_index"] += 1
        return

    # Save images
    image_writer = dataset["image_writer"]
    for key in img_keys:
        imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
        async_save_image(
            image_writer,
            image=observation[key],
            key=key,
            frame_index=frame_index,
            episode_index=episode_index,
            videos_dir=str(videos_dir),
        )

        if video:
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            frame_info = {"path": f"videos/{fname}", "timestamp": frame_index / fps}
        else:
            frame_info = str(imgs_dir / f"frame_{frame_index:06d}.png")

        ep_dict[key].append(frame_info)

    dataset["current_frame_index"] += 1


def delete_current_episode(dataset):
    del dataset["current_episode"]
    del dataset["current_frame_index"]

    # delete temporary images
    episode_index = dataset["num_episodes"]
    videos_dir = dataset["videos_dir"]
    for tmp_imgs_dir in videos_dir.glob(f"*_episode_{episode_index:06d}"):
        shutil.rmtree(tmp_imgs_dir)


def save_current_episode(dataset):
    episode_index = dataset["num_episodes"]
    ep_dict = dataset["current_episode"]
    episodes_dir = dataset["episodes_dir"]
    rec_info_path = dataset["rec_info_path"]

    ep_dict["next.done"][-1] = True

    for key in ep_dict:
        if "observation" in key and "image" not in key:
            ep_dict[key] = torch.stack(ep_dict[key])

    ep_dict["action"] = torch.stack(ep_dict["action"])
    ep_dict["episode_index"] = torch.tensor(ep_dict["episode_index"])
    ep_dict["frame_index"] = torch.tensor(ep_dict["frame_index"])
    ep_dict["timestamp"] = torch.tensor(ep_dict["timestamp"])
    ep_dict["next.done"] = torch.tensor(ep_dict["next.done"])

    ep_path = episodes_dir / f"episode_{episode_index}.pth"
    torch.save(ep_dict, ep_path)

    rec_info = {
        "last_episode_index": episode_index,
    }
    with open(rec_info_path, "w") as f:
        json.dump(rec_info, f)

    # force re-initialization of episode dictionnary during add_frame
    del dataset["current_episode"]

    dataset["num_episodes"] += 1


def encode_videos(dataset, image_keys, play_sounds):
    log_say("Encoding videos", play_sounds)

    num_episodes = dataset["num_episodes"]
    videos_dir = dataset["videos_dir"]
    local_dir = dataset["local_dir"]
    fps = dataset["fps"]

    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            # key = f"observation.images.{name}"
            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = local_dir / "videos" / fname
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
            shutil.rmtree(tmp_imgs_dir)


def from_dataset_to_lerobot_dataset(dataset, play_sounds):
    log_say("Consolidate episodes", play_sounds)

    num_episodes = dataset["num_episodes"]
    episodes_dir = dataset["episodes_dir"]
    videos_dir = dataset["videos_dir"]
    video = dataset["video"]
    fps = dataset["fps"]
    repo_id = dataset["repo_id"]

    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    if video:
        image_keys = [key for key in data_dict if "image" in key]
        encode_videos(dataset, image_keys, play_sounds)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    return lerobot_dataset


def save_lerobot_dataset_on_disk(lerobot_dataset):
    hf_dataset = lerobot_dataset.hf_dataset
    info = lerobot_dataset.info
    stats = lerobot_dataset.stats
    episode_data_index = lerobot_dataset.episode_data_index
    local_dir = lerobot_dataset.videos_dir.parent
    meta_data_dir = local_dir / "meta_data"

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    save_meta_data(info, stats, episode_data_index, meta_data_dir)


def push_lerobot_dataset_to_hub(lerobot_dataset, tags):
    hf_dataset = lerobot_dataset.hf_dataset
    local_dir = lerobot_dataset.videos_dir.parent
    videos_dir = lerobot_dataset.videos_dir
    repo_id = lerobot_dataset.repo_id
    video = lerobot_dataset.video
    meta_data_dir = local_dir / "meta_data"

    if not (local_dir / "train").exists():
        raise ValueError(
            "You need to run `save_lerobot_dataset_on_disk(lerobot_dataset)` before pushing to the hub."
        )

    hf_dataset.push_to_hub(repo_id, revision="main")
    push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
    push_dataset_card_to_hub(repo_id, revision="main", tags=tags)
    if video:
        push_videos_to_hub(repo_id, videos_dir, revision="main")
    create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)


def create_lerobot_dataset(dataset, run_compute_stats, push_to_hub, tags, play_sounds):
    if "image_writer" in dataset:
        logging.info("Waiting for image writer to terminate...")
        image_writer = dataset["image_writer"]
        stop_image_writer(image_writer, timeout=20)

    lerobot_dataset = from_dataset_to_lerobot_dataset(dataset, play_sounds)

    if run_compute_stats:
        log_say("Computing dataset statistics", play_sounds)
        lerobot_dataset.stats = compute_stats(lerobot_dataset)
    else:
        logging.info("Skipping computation of the dataset statistics")
        lerobot_dataset.stats = {}

    save_lerobot_dataset_on_disk(lerobot_dataset)

    if push_to_hub:
        push_lerobot_dataset_to_hub(lerobot_dataset, tags)

    return lerobot_dataset
