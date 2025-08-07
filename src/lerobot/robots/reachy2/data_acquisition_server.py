from pathlib import Path
import json
import os
import shutil

from concurrent import futures
from typing import Dict
import threading

import grpc
from logging import getLogger

import torch.multiprocessing as mp

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig
from lerobot.teleoperators.reachy2_fake_teleoperator import Reachy2FakeTeleoperator, Reachy2FakeTeleoperatorConfig
from lerobot.utils.utils import log_say
from lerobot.record import record_loop

from data_acquisition_api.data_acquisition_pb2 import (
    ActionAck,
    EpisodeRating,
    SessionParams,
    Dataset,
    DatasetList,
    DatasetPushState,
)
from data_acquisition_api.data_acquisition_pb2_grpc import add_DataAcquisitionServiceServicer_to_server
from google.protobuf.empty_pb2 import Empty


class DataAcquisitionServicer():
    def __init__(
        self,
    ):
        self._logger = getLogger(__name__)
        self.play_sound = True

        self.thread: threading.Thread = None
        self.events: Dict = {}
        self.events["exit_early"] = False

        self.setup_over = False

        self.dataset_list_path = Path("datasets.json")
        self.robot: Reachy2Robot = None
        self.task: str = None
        self.dataset: LeRobotDataset = None

        self.fps: int
        self.episode_duration: int
        self.break_time_duration: int

        self.episode_recording_in_progress: bool = False
        self.episode_saved: bool = False
        self.episode_recorded_in_session: bool = False

        self.run_compute_stats: bool = True

    def register_to_server(self, server):
        add_DataAcquisitionServiceServicer_to_server(self, server)

    def GetDatasetList(self, request: Empty, context: grpc.ServicerContext) -> DatasetList:
        dataset_list = self.read_datasets_from_json(self.dataset_list_path)
        return dataset_list

    def RemoveDataset(self, request: Dataset, context: grpc.ServicerContext) -> ActionAck:
        self.remove_dataset_by_name(request.dataset_name, self.dataset_list_path)
        return ActionAck(success_ack=True)

    def AddDataset(self, request: Dataset, context: grpc.ServicerContext) -> ActionAck:
        # Create a new dataset
        dataset = self.create_dataset(request.dataset_name, DatasetPushState.PUSHED, 0)
        # Add it to the JSON file
        self.add_dataset_to_json_file(dataset, self.dataset_list_path)
        return ActionAck(success_ack=True)

    def UpdateDataset(self, request: Dataset, context: grpc.ServicerContext) -> ActionAck:
        self.update_dataset(request, self.dataset_list_path)
        return ActionAck(success_ack=True)

    def ClearAllDatasets(self, request: Empty, context: grpc.ServicerContext) -> ActionAck:
        # Clear all datasets from the JSON file
        self.clear_all_datasets(self.dataset_list_path)
        return ActionAck(success_ack=True)

    def RemoveSession(self, request: SessionParams, context: grpc.ServicerContext) -> ActionAck:
        self.delete_folder(request.session_name)
        return ActionAck(success_ack=True)

    def ClearAllSessions(self, request: Empty, context: grpc.ServicerContext) -> ActionAck:
        pass

    # Session and Episode Management with LeRobot

    def StartSession(self, request: SessionParams, context: grpc.ServicerContext) -> ActionAck:
        self._logger.error(f"Starting session with params: {request}")
        try:
            self.setup_recording_session(request, context)
            self.setup_over = True
            ack = ActionAck(success_ack=True)
        except Exception as e:
            self._logger.error(f"Error starting session: {e}")
            ack = ActionAck(success_ack=False)
        return ack

    def StopSession(self, request: Empty, context: grpc.ServicerContext) -> ActionAck:
        self._logger.error("Stopping session")
        try:
            if self.thread and self.thread.is_alive():
                self.thread.join()  # Wait for the thread to finish
            log_say("Stop recording", play_sounds=self.play_sound, blocking=True)
            self.robot.disconnect()
            ack = ActionAck(success_ack=True)
            self._logger.error("Session stopped")
        except Exception as e:
            self._logger.error(f"Error stopping session: {e}")
            ack = ActionAck(success_ack=False)
        return ack

    def StartEpisode(self, request: Empty, context: grpc.ServicerContext) -> ActionAck:
        self._logger.error("Starting episode")
        try:
            if self.episode_recorded_in_session and not self.episode_saved:
                self._logger.error("Episode not saved. Clearing episode buffer.")
                self.dataset.clear_episode_buffer()
            if not self.setup_over:
                raise RuntimeError("Setup not completed. Please call StartSession first.")
            if self.episode_recording_in_progress:
                raise RuntimeError("Episode recording already in progress. Please stop it before starting a new one.")
            self.episode_saved = False
            self.thread = threading.Thread(target=self.record_episode)
            self.thread.daemon = True
            self.thread.start()
            self.episode_recorded_in_session = True
            ack = ActionAck(success_ack=True)
            self._logger.error("Episode started")
        except Exception as e:
            self._logger.error(f"Error starting episode: {e}")
            ack = ActionAck(success_ack=False)
        return ack

    def StopEpisode(self, request: Empty, context: grpc.ServicerContext) -> ActionAck:
        self._logger.error("Stopping episode")
        try:
            self.events["exit_early"] = True
            if self.thread and self.thread.is_alive():
                self.thread.join()  # Wait for the thread to finish
            self.events["exit_early"] = False
            ack = ActionAck(success_ack=True)
            self._logger.error("Episode stopped")
        except Exception as e:
            self._logger.error(f"Error stopping session: {e}")
            ack = ActionAck(success_ack=False)
        return ack

    def SaveEpisode(self, request: EpisodeRating, context: grpc.ServicerContext) -> ActionAck:
        self._logger.error("Saving episode")
        try:
            if self.episode_recording_in_progress:
                raise RuntimeError("Episode recording in progress. Please stop it before saving.")
            self.dataset.save_episode()
            self.episode_saved = True
            ack = ActionAck(success_ack=True)
            self._logger.error("Episode saved")
        except Exception as e:
            self._logger.error(f"Error saving episode: {e}")
            ack = ActionAck(success_ack=False)
        return ack

    def UploadSession(self, request: Empty, context: grpc.ServicerContext) -> ActionAck:
        try:
            if self.episode_recording_in_progress:
                raise RuntimeError("Episode recording in progress. Please stop it before uploading.")
            self.dataset.push_to_hub()
            ack = ActionAck(success_ack=True)
            self._logger.error("Session uploaded")
        except Exception as e:
            self._logger.error(f"Error uploading session: {e}")
            ack = ActionAck(success_ack=False)
        return ack

    def setup_recording_session(
            self,
            request: SessionParams,
            context: grpc.ServicerContext,
    ):
        # Create the robot and teleoperator configurations
        self.robot_config = Reachy2RobotConfig(
            ip_address=request.robot.ip_address,
            id=request.robot.robot_id,
            use_external_commands=True,
        )
        self.teleop_config = Reachy2FakeTeleoperatorConfig(
            ip_address=request.robot.ip_address,
        )

        # Initialize the robot and teleoperator
        self.robot = Reachy2Robot(self.robot_config)
        self.teleop = Reachy2FakeTeleoperator(self.teleop_config)

        self.fps = request.fps if request.HasField("fps") else 30
        self.task = request.task_description
        self.episode_duration = request.episode_duration
        self.break_time_duration = request.break_time_duration

        # Configure the dataset features
        action_features = hw_to_dataset_features(self.robot.action_features, "action")
        obs_features = hw_to_dataset_features(self.robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}

        print(f"Dataset name: {request.dataset_name}")

        if request.resume:
            self.dataset = LeRobotDataset(
                request.dataset_name,
                root="/home/demo/.cache/huggingface/lerobot/" + request.dataset_name,
            )
            if hasattr(self.robot, "cameras") and len(self.robot.cameras) > 0:
                self.dataset.start_image_writer()
        else:
            # Create the dataset
            self.dataset = LeRobotDataset.create(
                repo_id=request.dataset_name,
                fps=self.fps,
                features=dataset_features,
                robot_type=self.robot.name,
                use_videos=True,
                image_writer_threads=4,
            )

            current_dataset = self.create_dataset(request.dataset_name, DatasetPushState.LOCAL_ONLY, 0)
            self.add_dataset_to_json_file(current_dataset, self.dataset_list_path)

        # Connect the robot and teleoperator
        if not self.robot.is_connected:
            self.robot.connect()
        if not self.teleop.is_connected:
            self.teleop.connect()

    def record_episode(
        self,
        display_cameras: bool = False,
        play_sounds: bool = True,
    ):
        self.episode_recording_in_progress = True
        record_loop(
            robot=self.robot,
            events=self.events,
            fps=self.fps,
            teleop=self.teleop,
            dataset=self.dataset,
            control_time_s=self.episode_duration,
            single_task=self.task,
            display_data=True,
        )
        self.episode_recording_in_progress = False

    def create_dataset(self, dataset_name: str, pushed: DatasetPushState, nb_episodes: int) -> Dataset:
        dataset = Dataset()
        dataset.dataset_name = dataset_name
        dataset.pushed = pushed
        dataset.nb_episodes = nb_episodes
        return dataset

    def dataset_to_dict(self, dataset: Dataset) -> dict:
        """Convert Dataset proto to a dict manually."""
        return {
            "dataset_name": dataset.dataset_name,
            "pushed": DatasetPushState.Name(dataset.pushed),
            "nb_episodes": dataset.nb_episodes
        }

    def add_dataset_to_json_file(self, dataset: Dataset, json_filename: str):
        if os.path.exists(json_filename):
            with open(json_filename, "r") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print("Warning: JSON root is not a list, resetting.")
                        data = []
                except json.JSONDecodeError:
                    print("Warning: Invalid JSON file, resetting.")
                    data = []
        else:
            data = []

        # Add the new dataset
        data.append(self.dataset_to_dict(dataset))

        # Save back to file
        with open(json_filename, "w") as f:
            json.dump(data, f, indent=2)

    def update_dataset(self, dataset: Dataset, json_filename: str):
        """Update the 'pushed' field of a dataset in the JSON file."""
        if not os.path.exists(json_filename):
            print(f"File '{json_filename}' does not exist.")
            return

        with open(json_filename, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    print("Warning: JSON root is not a list. No update performed.")
                    return
            except json.JSONDecodeError:
                print("Warning: JSON file is invalid. No update performed.")
                return

        dataset_found = False
        for d in data:
            if d.get("dataset_name") == dataset.dataset_name:
                d["pushed"] = DatasetPushState.Name(dataset.pushed)  # Convert enum to string
                dataset_found = True
                break

        if not dataset_found:
            print(f"No dataset with name '{dataset.dataset_name}' found.")
            return

        with open(json_filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Dataset '{dataset.dataset_name}' updated successfully.")

    def dict_to_dataset(self, d: Dict) -> Dataset:
        """Convert a dictionary to a Dataset proto."""
        dataset = Dataset()
        dataset.dataset_name = d.get("dataset_name", "")

        pushed_str = d.get("pushed", "UNKNOWN")
        if isinstance(pushed_str, str):
            # Convert string to enum value
            dataset.pushed = DatasetPushState.Value(pushed_str)
        elif isinstance(pushed_str, int):
            # Already an integer value
            dataset.pushed = pushed_str
        else:
            dataset.pushed = DatasetPushState.UNKNOWN

        dataset.nb_episodes = d.get("nb_episodes", 0)
        return dataset

    def read_datasets_from_json(self, json_filename: str) -> DatasetList:
        dataset_list = DatasetList()

        if not os.path.exists(json_filename):
            print(f"File {json_filename} does not exist.")
            return dataset_list  # Empty list

        with open(json_filename, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    print("Warning: JSON root is not a list. Ignored.")
                    return dataset_list
            except json.JSONDecodeError:
                print("Warning: Invalid JSON file. Ignored.")
                return dataset_list

        for d in data:
            dataset = self.dict_to_dataset(d)
            dataset_list.datasets.append(dataset)

        return dataset_list

    def remove_dataset_by_name(self, dataset_name: str, json_filename: str):
        """Remove a dataset with a given name from the JSON file."""
        if not os.path.exists(json_filename):
            print(f"File {json_filename} does not exist.")
            return

        with open(json_filename, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    print("Warning: JSON root is not a list. No action taken.")
                    return
            except json.JSONDecodeError:
                print("Warning: Invalid JSON file. No action taken.")
                return

        original_length = len(data)
        # Filter out datasets with the matching name
        data = [d for d in data if d.get("dataset_name") != dataset_name]

        if len(data) == original_length:
            print(f"No dataset found with name '{dataset_name}'. No action taken.")
            return

        # Save the modified list back to the file
        with open(json_filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Dataset '{dataset_name}' removed successfully.")

    def clear_all_datasets(self, json_filename: str):
        """Clear all datasets from the JSON file."""
        if not os.path.exists(json_filename):
            print(f"File {json_filename} does not exist. Nothing to clear.")
            return

        # Overwrite the file with an empty list
        with open(json_filename, "w") as f:
            json.dump([], f, indent=2)
        print(f"All datasets have been cleared from '{json_filename}'.")

    def delete_folder(self, folder_path: str):
        """Delete an entire folder and its content."""
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        if not os.path.isdir(folder_path):
            print(f"Path '{folder_path}' is not a folder.")
            return

        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and all its contents have been deleted.")
        except Exception as e:
            print(f"Error while deleting folder '{folder_path}': {e}")


def main():
    mp.set_start_method('spawn', force=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    data_acquisition_servicer = DataAcquisitionServicer()
    data_acquisition_servicer.register_to_server(server)
    server.add_insecure_port('[::]:50062')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
