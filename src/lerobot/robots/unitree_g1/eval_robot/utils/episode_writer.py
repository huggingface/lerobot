import os
import cv2
import json
import datetime
import numpy as np
import time

from queue import Queue, Empty
from threading import Thread
import logging_mp

logger_mp = logging_mp.get_logger(__name__)


class EpisodeWriter:
    def __init__(self, task_dir, frequency=30, image_size=[640, 480]):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size

        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if "episode_" in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split("_")[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info("==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc()
        self.result = None
        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")

    def data_info(self, version="1.0.0", date=None, author=None):
        self.info = {
            "version": "1.0.0" if version is None else version,
            "date": datetime.date.today().strftime("%Y-%m-%d") if date is None else date,
            "author": "unitree" if author is None else author,
            "image": {"width": self.image_size[0], "height": self.image_size[1], "fps": self.frequency},
            "depth": {"width": self.image_size[0], "height": self.image_size[1], "fps": self.frequency},
            "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},  # PCM_S16
            "joint_names": {
                "left_arm": [
                    "kLeftShoulderPitch",
                    "kLeftShoulderRoll",
                    "kLeftShoulderYaw",
                    "kLeftElbow",
                    "kLeftWristRoll",
                    "kLeftWristPitch",
                    "kLeftWristyaw",
                ],
                "left_ee": [],
                "right_arm": [],
                "right_ee": [],
                "body": [],
            },
            "tactile_names": {
                "left_ee": [],
                "right_ee": [],
            },
            "sim_state": "",
        }

    def text_desc(self):
        self.text = {
            "goal": "Place the wooden blocks into the yellow frame, stacking them from bottom to top in the order: red, yellow, green.",
            "desc": "Using the gripper, first place the red wooden block into the yellow frame. Next, stack the yellow wooden block on top of the red one, and finally place the green wooden block on top of the yellow block.",
            "steps": "",
        }

    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info(
                "==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed."
            )
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1

        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, "colors")
        self.depth_dir = os.path.join(self.episode_dir, "depths")
        self.audio_dir = os.path.join(self.episode_dir, "audios")
        self.json_path = os.path.join(self.episode_dir, "data.json")
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created

    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            "idx": self.item_id,
            "colors": colors,
            "depths": depths,
            "states": states,
            "actions": actions,
            "tactiles": tactiles,
            "audios": audios,
            "sim_state": sim_state,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass

            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data["idx"]
        colors = item_data.get("colors", {})
        depths = item_data.get("depths", {})
        audios = item_data.get("audios", {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f"{str(idx).zfill(6)}_{color_key}.jpg"
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info("Failed to save color image.")
                item_data["colors"][color_key] = os.path.join("colors", color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f"{str(idx).zfill(6)}_{depth_key}.jpg"
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info("Failed to save depth image.")
                item_data["depths"][depth_key] = os.path.join("depths", depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f"audio_{str(idx).zfill(6)}_{mic}.npy"
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data["audios"][mic] = os.path.join("audios", audio_name)

        # Update episode data
        self.episode_data.append(item_data)

    def save_episode(self, result):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        self.result = result
        logger_mp.info("==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.data["info"] = self.info
        self.data["text"] = self.text
        self.data["data"] = self.episode_data
        self.data["result"] = self.result

        with open(self.json_path, "w", encoding="utf-8") as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.need_save = False  # Reset the save flag
        self.is_available = True  # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path} with result: {self.result}")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode(self.result)
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
