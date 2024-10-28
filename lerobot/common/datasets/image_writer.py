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
import queue
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_IMAGE_PATH = "{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


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


def write_image(image_array: np.ndarray, fpath: Path):
    try:
        image = Image.fromarray(image_array)
        image.save(fpath)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


def worker_thread_process(queue: queue.Queue):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath = item
        write_image(image_array, fpath)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_process, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class ImageWriter:
    """
    This class abstract away the initialisation of processes or/and threads to
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
        self.write_dir = write_dir
        self.write_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = DEFAULT_IMAGE_PATH

        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []

        if self.num_processes == 0:
            # Use threading
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                t = threading.Thread(target=worker_thread_process, args=(self.queue,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # Use multiprocessing
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

    def get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = self.image_path.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.write_dir / fpath

    def get_episode_dir(self, episode_index: int, image_key: str) -> Path:
        return self.get_image_file_path(
            episode_index=episode_index, image_key=image_key, frame_index=0
        ).parent

    def save_image(self, image_array: torch.Tensor | np.ndarray, fpath: Path):
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.numpy()
        self.queue.put((image_array, fpath))

    def wait_until_done(self):
        self.queue.join()

    def stop(self):
        if self.num_processes == 0:
            # For threading
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            # For multiprocessing
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            self.queue.close()
            self.queue.join_thread()
            for p in self.processes:
                p.join()
