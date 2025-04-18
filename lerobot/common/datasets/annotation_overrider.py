import json
from pathlib import Path
from typing import Dict, Optional, Union

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset


class TaskAnnotationOverrider:
    """
    A class to override task annotations in LeRobotDataset or MultiLeRobotDataset
    without modifying the original dataset files.

    This allows users to use datasets shared by others but with customized task descriptions.
    """

    def __init__(self, annotation_file: Union[str, Path]):
        """
        Initialize the TaskAnnotationOverrider with an annotation file.

        Args:
            annotation_file (Union[str, Path]): Path to the JSON file containing task annotation overrides.
                The file should have the following structure:
                {
                    "repo_id1": {
                        "0": "new task description for task 0",
                        "1": "new task description for task 1"
                    },
                    "repo_id2": {
                        "0": "new task description for task 0"
                    }
                }
        """
        self.annotation_file = Path(annotation_file)
        self.load_annotations()

    def load_annotations(self) -> None:
        """Load task annotations from the specified JSON file."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file) as f:
            self.annotations = json.load(f)

    def apply_overrides(self, dataset: Union[LeRobotDataset, MultiLeRobotDataset]) -> None:
        """
        Apply task annotation overrides to a LeRobotDataset or MultiLeRobotDataset.

        Args:
            dataset (Union[LeRobotDataset, MultiLeRobotDataset]): The dataset to override task annotations for.
        """
        if isinstance(dataset, MultiLeRobotDataset):
            self._apply_overrides_multi(dataset)
        elif isinstance(dataset, LeRobotDataset):
            self._apply_overrides_single(dataset)
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def _apply_overrides_single(self, dataset: LeRobotDataset) -> None:
        """
        Apply task annotation overrides to a single LeRobotDataset.

        Args:
            dataset (LeRobotDataset): The dataset to override task annotations for.
        """
        repo_id = dataset.repo_id
        if repo_id not in self.annotations:
            print(f"No annotations found for repository: {repo_id}")
            return

        repo_annotations = self.annotations[repo_id]
        modified = False

        # Create mapping of old task descriptions to new ones
        task_description_mapping = {}

        # Update the tasks in the metadata
        for task_idx_str, new_description in repo_annotations.items():
            task_idx = int(task_idx_str)

            # Check if the task index exists in the dataset
            if task_idx >= dataset.meta.total_tasks:
                print(f"Warning: Task index {task_idx} not found in dataset {repo_id}")
                continue

            # Update the task description
            if task_idx in dataset.meta.tasks:
                original_description = dataset.meta.tasks[task_idx]
                dataset.meta.tasks[task_idx] = new_description
                task_description_mapping[original_description] = new_description
                modified = True
            else:
                print(f"Warning: Task index {task_idx} not found in dataset {repo_id}")

        # If any annotations were modified, update the dataset
        if modified:
            self._update_task_index_mapping(dataset)
            self._update_episode_tasks(dataset, task_description_mapping)
            self._update_item_access(dataset)

    def _apply_overrides_multi(self, multi_dataset: MultiLeRobotDataset) -> None:
        """
        Apply task annotation overrides to each LeRobotDataset in a MultiLeRobotDataset.

        Args:
            multi_dataset (MultiLeRobotDataset): The multi-dataset to override task annotations for.
        """
        for dataset in multi_dataset._datasets:
            self._apply_overrides_single(dataset)

    def _update_task_index_mapping(self, dataset: LeRobotDataset) -> None:
        """
        Update the task_to_task_index mapping to reflect the new task descriptions.

        Args:
            dataset (LeRobotDataset): The dataset to update the mapping for.
        """
        # Rebuild the task_to_task_index mapping
        dataset.meta.task_to_task_index = {
            task_desc: task_idx for task_idx, task_desc in dataset.meta.tasks.items()
        }

    def _update_episode_tasks(self, dataset: LeRobotDataset, task_mapping: Dict[str, str]) -> None:
        """
        Update the task descriptions in the episodes metadata.

        Args:
            dataset (LeRobotDataset): The dataset to update.
            task_mapping (Dict[str, str]): Mapping from original to new task descriptions.
        """
        for ep_idx, episode in dataset.meta.episodes.items():
            if "tasks" in episode:
                # Replace task descriptions in the episode's tasks list
                updated_tasks = []
                for task in episode["tasks"]:
                    if task in task_mapping:
                        updated_tasks.append(task_mapping[task])
                    else:
                        updated_tasks.append(task)

                if updated_tasks != episode["tasks"]:
                    episode["tasks"] = updated_tasks

    def _update_item_access(self, dataset: LeRobotDataset) -> None:
        """
        Modify the __getitem__ method of the dataset to use the updated task annotations.

        Args:
            dataset (LeRobotDataset): The dataset to update.
        """
        # Since task lookup is done in __getitem__, no need to modify anything else here
        # The overridden tasks in dataset.meta.tasks will be used automatically
        pass

    def save_overridden_metadata(self, dataset: LeRobotDataset, output_path: Optional[Path] = None) -> None:
        """
        Save the overridden metadata to a new file without modifying the original dataset.

        This allows saving the modified tasks as a separate file that can be loaded later.

        Args:
            dataset (LeRobotDataset): The dataset with overridden task annotations.
            output_path (Optional[Path]): The path to save the overridden metadata.
                If None, will save to dataset.root/meta/tasks_overridden.json
        """
        if output_path is None:
            output_path = dataset.root / "meta" / "tasks_overridden.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the overridden tasks
        tasks_dict = {str(idx): desc for idx, desc in dataset.meta.tasks.items()}
        with open(output_path, "w") as f:
            json.dump(tasks_dict, f, indent=2)

        # Optionally, also save the updated episodes
        episodes_path = output_path.parent / "episodes_overridden.json"
        with open(episodes_path, "w") as f:
            json.dump(dataset.meta.episodes, f, indent=2)


# Usage example:
#
# from lerobot.common.datasets.task_annotation_overrider import TaskAnnotationOverrider
#
# # Load a dataset
# dataset = LeRobotDataset("jpata/so100_pick_place_tangerine")
#
# # Create and apply overrides
# overrider = TaskAnnotationOverrider("task_annotations.json")
# overrider.apply_overrides(dataset)
#
# # Now dataset has updated task descriptions
# # The original dataset files are unchanged
#
# # When using the dataset, task descriptions will be updated
# sample = dataset[0]
# print(sample["task"])  # This will use the overridden task description
