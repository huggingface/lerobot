#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import torch
import tqdm
from PIL import Image

DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


def safe_stop_image_writer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset", None)
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise e

    return wrapper


class ImageWriter:
    """This class abstract away the initialisation of processes or/and threads to
    save images on disk asynchrounously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it creates a threads pool of size `num_threads`.
    When `num_processes>0`, it creates processes pool of size `num_processes`, where each subprocess starts
    their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """

    def __init__(self, write_dir: Path, num_processes: int = 0, num_threads: int = 1):
        self.dir = write_dir
        self.image_path = DEFAULT_IMAGE_PATH
        self.num_processes = num_processes
        self.num_threads = self.num_threads_per_process = num_threads

        if self.num_processes <= 0:
            self.type = "threads"
            self.threads = ThreadPoolExecutor(max_workers=self.num_threads)
            self.futures = []
        else:
            self.type = "processes"
            self.num_threads_per_process = self.num_threads
            self.image_queue = multiprocessing.Queue()
            self.processes: list[multiprocessing.Process] = []
            for _ in range(num_processes):
                process = multiprocessing.Process(target=self._loop_to_save_images_in_threads)
                process.start()
                self.processes.append(process)

    def _loop_to_save_images_in_threads(self) -> None:
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            while True:
                frame_data = self.image_queue.get()
                if frame_data is None:
                    break

                image, file_path = frame_data
                futures.append(executor.submit(self._save_image, image, file_path))

            with tqdm.tqdm(total=len(futures), desc="Writing images") as progress_bar:
                wait(futures)
                progress_bar.update(len(futures))

    def async_save_image(self, image: torch.Tensor, file_path: Path) -> None:
        """Save an image asynchronously using threads or processes."""
        if self.type == "threads":
            self.futures.append(self.threads.submit(self._save_image, image, file_path))
        else:
            self.image_queue.put((image, file_path))

    def _save_image(self, image: torch.Tensor, file_path: Path) -> None:
        img = Image.fromarray(image.numpy())
        img.save(str(file_path), quality=100)

    def get_image_file_path(
        self, episode_index: int, image_key: str, frame_index: int, return_str: bool = True
    ) -> str | Path:
        fpath = self.image_path.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return str(self.dir / fpath) if return_str else self.dir / fpath

    def get_episode_dir(self, episode_index: int, image_key: str, return_str: bool = True) -> str | Path:
        dir_path = self.get_image_file_path(
            episode_index=episode_index, image_key=image_key, frame_index=0, return_str=False
        ).parent
        return str(dir_path) if return_str else dir_path

    def stop(self, timeout=20) -> None:
        """Stop the image writer, waiting for all processes or threads to finish."""
        if self.type == "threads":
            with tqdm.tqdm(total=len(self.futures), desc="Writing images") as progress_bar:
                wait(self.futures, timeout=timeout)
                progress_bar.update(len(self.futures))
        else:
            self._stop_processes(self.processes, self.image_queue, timeout)

    def _stop_processes(self, timeout) -> None:
        for _ in self.processes:
            self.image_queue.put(None)

        for process in self.processes:
            process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()

        self.image_queue.close()
        self.image_queue.join_thread()
