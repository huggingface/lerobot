import argparse
import logging
import os
import shutil
import traceback
from pathlib import Path

import cv2
import h5py
import torch

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


class AlohaHD5Extractor:
    TAGS = ["aloha", "robotics", "hdf5"]
    aloha_stationary = "aloha-stationary"
    aloha_mobile = "aloha-mobile"

    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        list of str
            A list of keys corresponding to RGB cameras in the dataset.
        """

        rgb_cameras = [key for key in hdf5_data["/observations/images"] if "depth" not in key]
        return rgb_cameras

    @staticmethod
    def check_format(episode_list: list[str] | list[Path], image_compressed: bool = True):
        """
        Check the format of the given list of HDF5 files.
        Parameters
        ----------
        episode_list : list of str or list of Path
            List of paths to the HDF5 files to be checked.
        image_compressed : bool, optional
            Flag indicating whether the images are compressed (default is True).
        Raises
        ------
        ValueError
            If the episode_list is empty.
            If any HDF5 file is missing required keys '/action' or '/observations/qpos'.
            If the '/action' or '/observations/qpos' keys do not have 2 dimensions.
            If the number of frames in '/action' and '/observations/qpos' keys do not match.
            If the number of frames in '/observations/images/{camera}' does not match the number of frames in '/action' and '/observations/qpos'.
            If the dimensions of images do not match the expected dimensions based on the image_compressed flag.
            If uncompressed images do not have the expected (h, w, c) format.
        """

        if not episode_list:
            raise ValueError(
                "No hdf5 files found in the raw directory. Make sure they are named 'episode_*.hdf5'"
            )
        for episode_path in episode_list:
            with h5py.File(episode_path, "r") as data:
                if not all(key in data for key in ["/action", "/observations/qpos"]):
                    raise ValueError(
                        "Missing required keys in the hdf5 file. Make sure the keys '/action' and '/observations/qpos' are present."
                    )

                if not data["/action"].ndim == data["/observations/qpos"].ndim == 2:
                    raise ValueError(
                        "The '/action' and '/observations/qpos' keys should have both 2 dimensions."
                    )

                if (num_frames := data["/action"].shape[0]) != data["/observations/qpos"].shape[0]:
                    raise ValueError(
                        "The '/action' and '/observations/qpos' keys should have the same number of frames."
                    )

                for camera in AlohaHD5Extractor.get_cameras(data):
                    if num_frames != data[f"/observations/images/{camera}"].shape[0]:
                        raise ValueError(
                            f"The number of frames in '/observations/images/{camera}' should be the same as in '/action' and '/observations/qpos' keys."
                        )

                    expected_dims = 2 if image_compressed else 4
                    if data[f"/observations/images/{camera}"].ndim != expected_dims:
                        raise ValueError(
                            f"Expect {expected_dims} dimensions for {'compressed' if image_compressed else 'uncompressed'} images but {data[f'/observations/images/{camera}'].ndim} provided."
                        )
                    if not image_compressed:
                        b, h, w, c = data[f"/observations/images/{camera}"].shape
                        if not c < h and c < w:
                            raise ValueError(f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided.")

    @staticmethod
    def extract_episode_frames(
        episode_path: str | Path, features: dict[str, dict], image_compressed: bool
    ) -> list[dict[str, torch.Tensor]]:
        """
        Extract frames from an episode stored in an HDF5 file.
        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        features : dict of str to dict
            Dictionary where keys are feature identifiers and values are dictionaries with feature details.
        image_compressed : bool
            Flag indicating whether the images are stored in a compressed format.
        Returns
        -------
        list of dict of str to torch.Tensor
            List of frames, where each frame is a dictionary mapping feature identifiers to tensors.
        """

        frames = []
        with h5py.File(episode_path, "r") as file:
            for frame_idx in range(file["/action"].shape[0]):
                frame = {}
                for feature_id in features:
                    feature_name_hd5 = (
                        feature_id.replace(".", "/")
                        .replace("observation", "observations")
                        .replace("state", "qpos")
                    )
                    if "images" in feature_id.split("."):
                        image = (
                            (file[feature_name_hd5][frame_idx])
                            if not image_compressed
                            else cv2.imdecode(file[feature_name_hd5][frame_idx], 1)
                        )
                        frame[feature_id] = torch.from_numpy(image.transpose(2, 0, 1))
                    else:
                        frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])
                frames.append(frame)
        return frames

    @staticmethod
    def define_features(
        hdf5_file_path: Path, image_compressed: bool = True, encode_as_video: bool = True
    ) -> dict[str, dict]:
        """
        Define features from an HDF5 file.
        Parameters
        ----------
        hdf5_file_path : Path
            The path to the HDF5 file.
        image_compressed : bool, optional
            Whether the images are compressed, by default True.
        encode_as_video : bool, optional
            Whether to encode images as video or as images, by default True.
        Returns
        -------
        dict[str, dict]
            A dictionary where keys are topic names and values are dictionaries
            containing feature information such as dtype, shape, and names.
        """

        # Initialize lists to store topics and features
        topics: list[str] = []
        features: dict[str, dict] = {}

        # Open the HDF5 file
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            # Collect all dataset names in the HDF5 file
            hdf5_file.visititems(
                lambda name, obj: topics.append(name) if isinstance(obj, h5py.Dataset) else None
            )

            # Iterate over each topic to define its features
            for topic in topics:
                # If the topic is an image, define it as a video feature
                destination_topic = (
                    topic.replace("/", ".").replace("observations", "observation").replace("qpos", "state")
                )
                if "images" in topic.split("/"):
                    sample = hdf5_file[topic][0]
                    features[destination_topic] = {
                        "dtype": "video" if encode_as_video else "image",
                        "shape": cv2.imdecode(hdf5_file[topic][0], 1).transpose(2, 0, 1).shape
                        if image_compressed
                        else sample.shape,
                        "names": [
                            "channel",
                            "height",
                            "width",
                        ],
                    }
                # Skip compressed length topics
                elif "compress_len" in topic.split("/"):
                    continue
                # Otherwise, define it as a regular feature
                else:
                    features[destination_topic] = {
                        "dtype": str(hdf5_file[topic][0].dtype),
                        "shape": (topic_shape := hdf5_file[topic][0].shape),
                        "names": [f"{topic.split('/')[-1]}_{k}" for k in range(topic_shape[0])],
                    }
        # Return the defined features
        return features


class DatasetConverter:
    """
    A class to convert datasets to Lerobot format.
    Parameters
    ----------
    raw_path : Path or str
        The path to the raw dataset.
    dataset_repo_id : str
        The repository ID where the dataset will be stored.
    fps : int
        Frames per second for the dataset.
    robot_type : str, optional
        The type of robot, by default "".
    encode_as_videos : bool, optional
        Whether to encode images as videos, by default True.
    image_compressed : bool, optional
        Whether the images are compressed, by default True.
    image_writer_processes : int, optional
        Number of processes for writing images, by default 0.
    image_writer_threads : int, optional
        Number of threads for writing images, by default 0.
    Methods
    -------
    extract_episode(episode_path, task_description='')
        Extracts frames from a single episode and saves it with a description.
    extract_episodes(episode_description='')
        Extracts frames from all episodes and saves them with a description.
    push_dataset_to_hub(dataset_tags=None, private=False, push_videos=True, license="apache-2.0")
        Pushes the dataset to the Hugging Face Hub.
    init_lerobot_dataset()
        Initializes the Lerobot dataset.
    """

    def __init__(
        self,
        raw_path: Path | str,
        dataset_repo_id: str,
        fps: int,
        robot_type: str = "",
        encode_as_videos: bool = True,
        image_compressed: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.image_compressed = image_compressed
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.encode_as_videos = encode_as_videos

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{'-' * 10} Aloha HD5 -> Lerobot Converter {'-' * 10}")
        self.logger.info(f"Processing Aloha HD5 dataset from {self.raw_path}")
        self.logger.info(f"Dataset will be stored in {self.dataset_repo_id}")
        self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Robot type: {self.robot_type}")
        self.logger.info(f"Image compressed: {self.image_compressed}")
        self.logger.info(f"Encoding images as videos: {self.encode_as_videos}")
        self.logger.info(f"#writer processes: {self.image_writer_processes}")
        self.logger.info(f"#writer threads: {self.image_writer_threads}")

        self.episode_list = list(self.raw_path.glob("episode_*.hdf5"))
        AlohaHD5Extractor.check_format(self.episode_list, image_compressed=self.image_compressed)
        self.features = AlohaHD5Extractor.define_features(
            self.episode_list[0],
            image_compressed=self.image_compressed,
            encode_as_video=self.encode_as_videos,
        )

    def extract_episode(self, episode_path, task_description: str = ""):
        """
        Extracts frames from an episode and saves them to the dataset.

        Parameters
        ----------
        episode_path : str
            The path to the episode file.
        task_description : str, optional
            A description of the task associated with the episode (default is an empty string).

        Returns
        -------
        None
        """

        for frame in AlohaHD5Extractor.extract_episode_frames(
            episode_path, self.features, self.image_compressed
        ):
            self.dataset.add_frame(frame)
        self.logger.info(f"Saving Episode with Description: {task_description} ...")
        self.dataset.save_episode(task=task_description)

    def extract_episodes(self, episode_description: str = ""):
        """
        Extracts episodes from the episode list and processes them.
        Parameters
        ----------
        episode_description : str, optional
            A description of the task to be passed to the extract_episode method (default is '').
        Raises
        ------
        Exception
            If an error occurs during the processing of an episode, it will be caught and printed.
        Notes
        -----
        After processing all episodes, the dataset is consolidated.
        """

        for episode_path in self.episode_list:
            try:
                self.extract_episode(episode_path, task_description=episode_description)
            except Exception as e:
                print(f"Error processing episode {episode_path}", f"{e}")
                traceback.print_exc()
                continue
        self.dataset.consolidate()

    def push_dataset_to_hub(
        self,
        dataset_tags: list[str] | None = None,
        private: bool = False,
        push_videos: bool = True,
        license: str | None = "apache-2.0",
    ):
        """
        Pushes the dataset to the Hugging Face Hub.
        Parameters
        ----------
        dataset_tags : list of str, optional
            A list of tags to associate with the dataset on the Hub. Default is None.
        private : bool, optional
            If True, the dataset will be private. Default is False.
        push_videos : bool, optional
            If True, videos will be pushed along with the dataset. Default is True.
        license : str, optional
            The license under which the dataset is released. Default is "apache-2.0".
        Returns
        -------
        None
        """

        self.logger.info(f"Pushing dataset to Hugging Face Hub. ID: {self.dataset_repo_id} ...")
        self.dataset.push_to_hub(
            tags=dataset_tags,
            license=license,
            push_videos=push_videos,
            private=private,
        )

    def init_lerobot_dataset(self):
        """
        Initializes the LeRobot dataset.
        This method cleans the cache if the dataset already exists and then creates a new LeRobot dataset.
        Returns
        -------
        LeRobotDataset
            The initialized LeRobot dataset.
        """

        # Clean the cache if the dataset already exists
        if os.path.exists(LEROBOT_HOME / self.dataset_repo_id):
            shutil.rmtree(LEROBOT_HOME / self.dataset_repo_id)
        self.dataset = LeRobotDataset.create(
            repo_id=self.dataset_repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=self.image_writer_processes,
        )

        return self.dataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    elif value in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    """
    Convert Aloha HD5 dataset and push to Hugging Face hub.
    This script processes raw HDF5 files from the Aloha dataset, converts them into a specified format,
    and optionally uploads the dataset to the Hugging Face hub.
    Parameters
    ----------
    --raw-path : Path
        Directory containing the raw HDF5 files.
    --dataset-repo-id : str
        Repository ID where the dataset will be stored.
    --fps : int
        Frames per second for the dataset.
    --robot-type : str, optional
        Type of robot, either "aloha-stationary" or "aloha-mobile". Default is "aloha-stationary".
    --private : bool, optional
        Set to True to make the dataset private. Default is False.
    --push-videos : bool, optional
        Set to True to push videos to the hub. Default is True.
    --license : str, optional
        License for the dataset. Default is "apache-2.0".
    --image-compressed : bool, optional
        Set to True if the images are compressed. Default is True.
    --video-encoding : bool, optional
        Set to True to encode images as videos. Default is True.
    --nproc : int, optional
        Number of image writer processes. Default is 10.
    --nthreads : int, optional
        Number of image writer threads. Default is 5.
    """

    parser = argparse.ArgumentParser(description="Convert Aloha HD5 dataset and push to Hugging Face hub.")
    parser.add_argument(
        "--raw-path", type=Path, required=True, help="Directory containing the raw hdf5 files."
    )
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True, help="Repository ID where the dataset will be stored."
    )
    parser.add_argument("--fps", type=int, required=True, help="Frames per second for the dataset.")
    parser.add_argument(
        "--description", type=str, help="Description of the dataset.", default="Aloha recorded dataset."
    )

    parser.add_argument(
        "--robot-type",
        type=str,
        choices=["aloha-stationary", "aloha-mobile"],
        default="aloha-stationary",
        help="Type of robot.",
    )
    parser.add_argument(
        "--private", type=str2bool, default=False, help="Set to True to make the dataset private."
    )
    parser.add_argument("--push", type=str2bool, default=True, help="Set to True to push videos to the hub.")
    parser.add_argument("--license", type=str, default="apache-2.0", help="License for the dataset.")
    parser.add_argument(
        "--image-compressed", type=str2bool, default=True, help="Set to True if the images are compressed."
    )
    parser.add_argument(
        "--video-encoding", type=str2bool, default=True, help="Set to True to encode images as videos."
    )

    parser.add_argument("--nproc", type=int, default=10, help="Number of image writer processes.")
    parser.add_argument("--nthreads", type=int, default=5, help="Number of image writer threads.")

    args = parser.parse_args()
    print(
        args.video_encoding,
        "-------------------------------------------------------------------------------------------------------",
    )

    converter = DatasetConverter(
        raw_path=args.raw_path,
        dataset_repo_id=args.dataset_repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        image_compressed=args.image_compressed,
        encode_as_videos=args.video_encoding,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
    )
    converter.init_lerobot_dataset()
    converter.extract_episodes(episode_description=args.description)

    if args.push:
        converter.push_dataset_to_hub(
            dataset_tags=AlohaHD5Extractor.TAGS, private=args.private, push_videos=True, license=args.license
        )


if __name__ == "__main__":
    main()
