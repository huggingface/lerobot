import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    create_subtask_index_array,
    create_subtasks_dataframe,
    save_subtasks,
)

if TYPE_CHECKING:
    from lerobot.data_processing.data_annotations.vlm_annotations import BaseVLM


# Skill Annotation Data Structures
class Skill:
    """Represents a single atomic skill/subtask in a demonstration."""

    def __init__(self, name: str, start: float, end: float):
        self.name = name
        self.start = start  # Start timestamp in seconds
        self.end = end  # End timestamp in seconds

    def to_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(name=data["name"], start=data["start"], end=data["end"])

    def __repr__(self) -> str:
        return f"Skill(name='{self.name}', start={self.start:.2f}, end={self.end:.2f})"


class EpisodeSkills:
    """Container for all skills in an episode."""

    def __init__(self, episode_index: int, description: str, skills: list[Skill]):
        self.episode_index = episode_index
        self.description = description
        self.skills = skills

    def to_dict(self) -> dict:
        return {
            "episode_index": self.episode_index,
            "description": self.description,
            "skills": [s.to_dict() for s in self.skills],
        }


# Video Extraction Utilities


class VideoExtractor:
    """Utilities for extracting and processing video segments from LeRobot datasets."""

    def __init__(self) -> None:
        pass

    def extract_episode_video(
        self,
        video_path: Path,
        start_timestamp: float,
        end_timestamp: float,
        target_fps: int = 1,
    ) -> Path:
        """
        Extract a specific episode segment from a concatenated video file.

        Args:
            video_path: Path to the source video file
            start_timestamp: Start time in seconds
            end_timestamp: End time in seconds
            target_fps: Target frames per second for output

        Returns:
            Path to the extracted temporary video file
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        duration = end_timestamp - start_timestamp

        print(f"Extracting: {start_timestamp:.1f}s - {end_timestamp:.1f}s ({duration:.1f}s)")

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-ss",
            str(start_timestamp),
            "-t",
            str(duration),
            "-r",
            str(target_fps),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-an",
            "-y",
            str(tmp_path),
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e}") from e
        except FileNotFoundError as e:
            raise RuntimeError("FFmpeg not found. Please install ffmpeg.") from e

        if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError("Video extraction produced invalid file")

        return tmp_path

    def add_timer_overlay(self, video_path: Path) -> Path:
        """
        Add a visible timer overlay to each frame (elapsed time in seconds) in one corner.
        Used so the VLM can read the timestamp from the image instead of relying on file metadata.
        Draws a black box with white text at top-right. Writes to a new temporary file and returns its path.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_file:
            out_path = Path(out_file.name)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(1.2, min(h, w) / 350.0)
        thickness = max(2, int(font_scale))

        padding = 15
        margin = 30

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / fps
            text = f"{t_sec:.2f} s"

            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Top-right placement
            x_text = w - tw - margin - padding
            y_text = margin + th + padding

            # Rectangle coordinates (black box behind text)
            x1 = x_text - padding
            y1 = y_text - th - padding
            x2 = x_text + tw + padding
            y2 = y_text + baseline + padding

            # Draw black filled rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # Draw white text
            cv2.putText(
                frame,
                text,
                (x_text, y_text),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        if not out_path.exists() or out_path.stat().st_size < 1024:
            if out_path.exists():
                out_path.unlink()
            raise RuntimeError("Timer overlay produced invalid file")
        return out_path

    def get_video_duration(self, video_path: Path) -> float:
        """Get duration of a video file in seconds."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count / fps


# Skill Annotation Pipeline
class SkillAnnotator:
    """
    Main class for annotating LeRobot datasets with skill labels.

    This class orchestrates the full annotation pipeline:
    1. Load dataset
    2. Extract video segments for each episode
    3. Run VLM-based skill segmentation
    4. Update dataset task metadata
    """

    def __init__(
        self,
        vlm: "BaseVLM",
        video_extractor: VideoExtractor | None = None,
        batch_size: int = 8,
        add_timer_overlay: bool = True,
    ):
        self.vlm = vlm
        self.video_extractor = video_extractor or VideoExtractor()
        self.batch_size = batch_size
        self.add_timer_overlay = add_timer_overlay

    def annotate_dataset(
        self,
        dataset: LeRobotDataset,
        video_key: str,
        episodes: list[int] | None = None,
        skip_existing: bool = False,
        subtask_labels: list[str] | None = None,
    ) -> dict[int, EpisodeSkills]:
        """
        Annotate all episodes in a dataset with skill labels using batched processing.

        Args:
            dataset: LeRobot dataset to annotate
            video_key: Key for video observations (e.g., "observation.images.base")
            episodes: Specific episode indices to annotate (None = all)
            skip_existing: Skip episodes that already have skill annotations
            subtask_labels: If provided, model must choose only from these labels (closed vocabulary)

        Returns:
            Dictionary mapping episode index to EpisodeSkills
        """
        episode_indices = episodes or list(range(dataset.meta.total_episodes))
        annotations: dict[int, EpisodeSkills] = {}
        failed_episodes: dict[int, str] = {}  # Track failed episodes with error messages

        # Get coarse task description if available
        coarse_goal = self._get_coarse_goal(dataset)

        # Filter out episodes that already have annotations if skip_existing is True
        if skip_existing:
            existing_annotations = load_skill_annotations(dataset.root)
            if existing_annotations and "episodes" in existing_annotations:
                # Only skip episodes that exist AND have non-empty skills
                existing_episode_indices = set()
                for idx_str, episode_data in existing_annotations["episodes"].items():
                    idx = int(idx_str)
                    # Check if skills list exists and is not empty
                    if "skills" in episode_data and episode_data["skills"]:
                        existing_episode_indices.add(idx)

                original_count = len(episode_indices)
                episode_indices = [ep for ep in episode_indices if ep not in existing_episode_indices]
                skipped_count = original_count - len(episode_indices)
                if skipped_count > 0:
                    print(f"Skipping {skipped_count} episodes with existing non-empty annotations")

        if not episode_indices:
            print("No episodes to annotate (all already annotated)")
            return annotations

        print(f"Annotating {len(episode_indices)} episodes in batches of {self.batch_size}...")

        # Process episodes in batches
        for batch_start in range(0, len(episode_indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(episode_indices))
            batch_episodes = episode_indices[batch_start:batch_end]

            print(
                f"Processing batch {batch_start // self.batch_size + 1}/{(len(episode_indices) + self.batch_size - 1) // self.batch_size} (episodes {batch_episodes[0]} to {batch_episodes[-1]})..."
            )

            try:
                batch_annotations = self._annotate_episodes_batch(
                    dataset, batch_episodes, video_key, coarse_goal, subtask_labels
                )

                for ep_idx in batch_episodes:
                    if ep_idx in batch_annotations and batch_annotations[ep_idx]:
                        skills = batch_annotations[ep_idx]
                        annotations[ep_idx] = EpisodeSkills(
                            episode_index=ep_idx,
                            description=coarse_goal,
                            skills=skills,
                        )
                        print(f" Episode {ep_idx}: {len(skills)} skills identified")
                    else:
                        failed_episodes[ep_idx] = "Empty or missing skills from batch processing"
                        print(f"⚠ Episode {ep_idx}: No skills extracted, will retry")
            except Exception as e:
                print(f"✗ Batch failed: {e}. Falling back to single-episode processing...")
                # Fallback: process episodes one by one
                for ep_idx in batch_episodes:
                    try:
                        skills = self._annotate_episode(
                            dataset, ep_idx, video_key, coarse_goal, subtask_labels
                        )
                        if skills:
                            annotations[ep_idx] = EpisodeSkills(
                                episode_index=ep_idx,
                                description=coarse_goal,
                                skills=skills,
                            )
                            print(f" Episode {ep_idx}: {len(skills)} skills identified")
                        else:
                            failed_episodes[ep_idx] = "Empty skills list from single-episode processing"
                            print(f"⚠ Episode {ep_idx}: No skills extracted, will retry")
                    except Exception as ep_error:
                        failed_episodes[ep_idx] = str(ep_error)
                        print(f"⚠ Episode {ep_idx} failed: {ep_error}, will retry")

        # Retry failed episodes one more time
        if failed_episodes:
            print(f"\nRetrying {len(failed_episodes)} failed episodes...")
            retry_count = 0
            for ep_idx, error_msg in list(failed_episodes.items()):
                print(f"Retry attempt for episode {ep_idx} (previous error: {error_msg})")
                try:
                    skills = self._annotate_episode(dataset, ep_idx, video_key, coarse_goal, subtask_labels)
                    if skills:
                        annotations[ep_idx] = EpisodeSkills(
                            episode_index=ep_idx,
                            description=coarse_goal,
                            skills=skills,
                        )
                        print(f" Episode {ep_idx} (retry): {len(skills)} skills identified")
                        del failed_episodes[ep_idx]
                        retry_count += 1
                    else:
                        print(f"✗ Episode {ep_idx} (retry): Still no skills extracted")
                except Exception as retry_error:
                    failed_episodes[ep_idx] = str(retry_error)
                    print(f"✗ Episode {ep_idx} (retry) failed: {retry_error}")

            if retry_count > 0:
                print(f"Successfully recovered {retry_count} episodes on retry")

            if failed_episodes:
                print(f"\n⚠ Warning: {len(failed_episodes)} episodes still failed after retry:")
                for ep_idx, error_msg in failed_episodes.items():
                    print(f"  Episode {ep_idx}: {error_msg}")

        return annotations

    def _get_coarse_goal(self, dataset: LeRobotDataset) -> str:
        """Extract or generate the coarse task description."""
        # Try to get from existing task metadata
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            # Get the first task description
            first_task = dataset.meta.tasks.index[0]
            if first_task:
                return str(first_task)

        return "Perform the demonstrated manipulation task."

    def _annotate_episodes_batch(
        self,
        dataset: LeRobotDataset,
        episode_indices: list[int],
        video_key: str,
        coarse_goal: str,
        subtask_labels: list[str] | None = None,
    ) -> dict[int, list[Skill]]:
        """Annotate multiple episodes with skill labels in a batch."""
        # Extract all videos for this batch
        extracted_paths = []
        timer_paths = []
        paths_for_vlm = []
        durations = []
        valid_episode_indices = []

        for ep_idx in episode_indices:
            try:
                # Get video path and timestamps
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)

                if not video_path.exists():
                    print(f"Warning: Video not found for episode {ep_idx}")
                    continue

                # Get episode timestamps from metadata
                ep = dataset.meta.episodes[ep_idx]
                start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
                end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
                duration = end_ts - start_ts

                # Extract episode segment to temporary file
                extracted_path = self.video_extractor.extract_episode_video(
                    video_path, start_ts, end_ts, target_fps=dataset.meta.fps
                )

                if self.add_timer_overlay:
                    video_for_vlm = self.video_extractor.add_timer_overlay(extracted_path)
                    extracted_paths.append(extracted_path)
                    timer_paths.append(video_for_vlm)
                else:
                    video_for_vlm = extracted_path
                    extracted_paths.append(extracted_path)
                    timer_paths.append(None)

                paths_for_vlm.append(video_for_vlm)
                durations.append(duration)
                valid_episode_indices.append(ep_idx)

            except Exception as e:
                print(f"Warning: Failed to extract video for episode {ep_idx}: {e}")
                continue

        if not paths_for_vlm:
            return {}

        try:
            # Run VLM skill segmentation in batch
            all_skills = self.vlm.segment_skills_batch(paths_for_vlm, durations, coarse_goal, subtask_labels)

            # Map results back to episode indices
            results = {}
            for ep_idx, skills in zip(valid_episode_indices, all_skills, strict=True):
                results[ep_idx] = skills

            return results

        finally:
            # Clean up all temporary files (extracted and timer-overlay)
            for path in extracted_paths:
                if path.exists():
                    path.unlink()
            for path in timer_paths:
                if path is not None and path.exists():
                    path.unlink()

    def _annotate_episode(
        self,
        dataset: LeRobotDataset,
        episode_index: int,
        video_key: str,
        coarse_goal: str,
        subtask_labels: list[str] | None = None,
    ) -> list[Skill]:
        """Annotate a single episode with skill labels."""
        # Get video path and timestamps for this episode
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, video_key)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get episode timestamps from metadata
        ep = dataset.meta.episodes[episode_index]
        start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
        end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
        duration = end_ts - start_ts

        # Extract episode segment to temporary file
        extracted_path = self.video_extractor.extract_episode_video(
            video_path, start_ts, end_ts, target_fps=1
        )
        if self.add_timer_overlay:
            video_for_vlm = self.video_extractor.add_timer_overlay(extracted_path)
        else:
            video_for_vlm = extracted_path

        try:
            # Run VLM skill segmentation
            skills = self.vlm.segment_skills(video_for_vlm, duration, coarse_goal, subtask_labels)
            return skills
        finally:
            # Clean up temporary files (extracted and optionally timer-overlay)
            if extracted_path.exists():
                extracted_path.unlink()
            if self.add_timer_overlay and video_for_vlm != extracted_path and video_for_vlm.exists():
                video_for_vlm.unlink()


# Metadata Writer - Updates per-frame task_index based on skills


def get_skill_for_timestamp(skills: list[Skill], timestamp: float) -> Skill | None:
    """
    Find which skill covers a given timestamp.

    Args:
        skills: List of skills with start/end times
        timestamp: Frame timestamp in seconds

    Returns:
        The Skill that covers this timestamp, or None if not found
    """
    for skill in skills:
        if skill.start <= timestamp < skill.end:
            return skill
        # Handle the last frame (end boundary)
        if timestamp >= skill.end and skill == skills[-1]:
            return skill
    return skills[-1] if skills else None  # Fallback to last skill


def save_skill_annotations(
    dataset: LeRobotDataset,
    annotations: dict[int, EpisodeSkills],
    output_dir: Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """
    Save skill annotations to the dataset by:
    1. Creating a subtasks.parquet file with unique subtasks
    2. Adding a subtask_index feature to the dataset
    3. Saving raw skill annotations as JSON for reference

    This function does NOT modify tasks.parquet - it keeps the original tasks intact
    and creates a separate subtask hierarchy.

    Args:
        dataset: The LeRobot dataset to annotate
        annotations: Dictionary of episode skills
        output_dir: Optional directory to save the modified dataset
        repo_id: Optional repository ID for the new dataset

    Returns:
        New dataset with subtask_index feature added
    """
    if not annotations:
        print("No annotations to save")
        return dataset

    # Step 1: Create subtasks DataFrame
    print("Creating subtasks DataFrame...")
    subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe(annotations)

    # Step 2: Create subtask_index array for all frames
    print("Creating subtask_index array...")
    subtask_indices = create_subtask_index_array(dataset, annotations, skill_to_subtask_idx)

    # Step 3: Save subtasks.parquet to the original dataset root
    save_subtasks(subtasks_df, dataset.root)

    # Step 4: Save the raw skill annotations as JSON for reference
    skills_path = dataset.root / "meta" / "skills.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing skills data if it exists and is not empty
    existing_skills_data = None
    if skills_path.exists():
        try:
            with open(skills_path) as f:
                existing_skills_data = json.load(f)
                if existing_skills_data and len(existing_skills_data.get("episodes", {})) > 0:
                    print(
                        f"Found existing skills.json with {len(existing_skills_data.get('episodes', {}))} episodes, merging..."
                    )
        except (OSError, json.JSONDecodeError):
            print("Warning: Could not load existing skills.json, will create new file")
            existing_skills_data = None

    # Prepare new annotations
    new_episodes = {str(ep_idx): ann.to_dict() for ep_idx, ann in annotations.items()}

    # Merge with existing data if available
    if existing_skills_data:
        # Preserve existing episodes that are not being updated
        merged_episodes = existing_skills_data.get("episodes", {}).copy()
        merged_episodes.update(new_episodes)

        # Merge skill_to_subtask_index mappings
        merged_skill_to_subtask = existing_skills_data.get("skill_to_subtask_index", {}).copy()
        merged_skill_to_subtask.update(skill_to_subtask_idx)

        # Use existing coarse_description if available, otherwise use new one
        coarse_desc = existing_skills_data.get(
            "coarse_description", annotations[next(iter(annotations))].description
        )

        skills_data = {
            "coarse_description": coarse_desc,
            "skill_to_subtask_index": merged_skill_to_subtask,
            "episodes": merged_episodes,
        }
        print(
            f"Updated {len(new_episodes)} episode(s), total episodes in skills.json: {len(merged_episodes)}"
        )
    else:
        # No existing data, create new
        skills_data = {
            "coarse_description": annotations[next(iter(annotations))].description,
            "skill_to_subtask_index": skill_to_subtask_idx,
            "episodes": new_episodes,
        }

    with open(skills_path, "w") as f:
        json.dump(skills_data, f, indent=2)

    print(f" Saved skill annotations to {skills_path}")

    # Step 5: Add subtask_index feature to dataset using add_features
    print("Adding subtask_index feature to dataset...")

    # Determine output directory and repo_id
    output_dir = dataset.root.parent / f"{dataset.root.name}" if output_dir is None else Path(output_dir)

    if repo_id is None:
        repo_id = f"{dataset.repo_id}"

    # Add feature using dataset_tools
    feature_info = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    new_dataset = add_features(
        dataset=dataset,
        features={
            "subtask_index": (subtask_indices, feature_info),
        },
        output_dir=output_dir,
        repo_id=repo_id,
    )

    # Copy subtasks.parquet to new output directory
    import shutil

    shutil.copy(dataset.root / "meta" / "subtasks.parquet", output_dir / "meta" / "subtasks.parquet")
    shutil.copy(dataset.root / "meta" / "skills.json", output_dir / "meta" / "skills.json")

    print(" Successfully added subtask_index feature!")
    print(f"  New dataset saved to: {new_dataset.root}")
    print(f"  Total subtasks: {len(subtasks_df)}")

    return new_dataset


def load_skill_annotations(dataset_root: Path) -> dict | None:
    """Load existing skill annotations from a dataset."""
    skills_path = dataset_root / "meta" / "skills.json"
    if skills_path.exists():
        with open(skills_path) as f:
            return json.load(f)
    return None
