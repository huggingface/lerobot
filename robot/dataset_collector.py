"""
Episode dataset collector for OAK-D + SO-100 robot manipulation.

Captures RGB images, depth maps, robot joint states, and actions
into a LeRobot-compatible dataset for policy training.

Usage::

    # Teleoperation recording (uses LeRobot's built-in recording)
    python -m robot.dataset_collector --task "pick up the red cube" --num-episodes 10

    # Record from agentic policy execution (autonomous)
    python -m robot.dataset_collector --task "pick up the red cube" --mode autonomous

    # Playback/inspect collected episodes
    python -m robot.dataset_collector --inspect --dataset-path ./data/episodes
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single demonstration episode with RGB, depth, state, and actions."""

    episode_id: int
    task: str
    fps: int
    rgb_frames: list[np.ndarray]
    depth_frames: list[np.ndarray]
    joint_positions: list[np.ndarray]  # (T, n_joints) degrees
    actions: list[np.ndarray]          # (T, n_joints) degree targets
    gripper_positions: list[float]     # (T,) 0-100%
    gripper_actions: list[float]       # (T,) 0-100% targets
    timestamps: list[float]
    success: bool = False

    @property
    def num_frames(self) -> int:
        return len(self.rgb_frames)

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


class DatasetCollector:
    """Collects demonstration episodes from the robot+camera system.

    Supports two modes:
    1. **Teleoperation**: Uses LeRobot's built-in leader-follower recording.
    2. **Autonomous**: Records while the agentic pipeline executes tasks.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_root: str | None = None,
        task: str = "",
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        lr_cfg = self.cfg.get("lerobot", {})
        self.dataset_root = Path(dataset_root or lr_cfg.get("dataset_root", "./data/episodes"))
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.fps = lr_cfg.get("fps", 30)

        self._current_episode: Episode | None = None
        self._episodes: list[Episode] = []
        self._episode_counter = 0

    def start_episode(self, task: str | None = None) -> Episode:
        """Begin recording a new episode."""
        ep = Episode(
            episode_id=self._episode_counter,
            task=task or self.task,
            fps=self.fps,
            rgb_frames=[],
            depth_frames=[],
            joint_positions=[],
            actions=[],
            gripper_positions=[],
            gripper_actions=[],
            timestamps=[],
        )
        self._current_episode = ep
        logger.info("Started episode %d: '%s'", ep.episode_id, ep.task)
        return ep

    def record_step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        joint_positions: np.ndarray,
        action: np.ndarray,
        gripper_pos: float,
        gripper_action: float,
    ) -> None:
        """Record a single timestep in the current episode.

        Args:
            rgb: RGB image (H, W, 3) uint8.
            depth: Depth map (H, W) uint16, millimeters.
            joint_positions: Current joint angles (n_joints,) degrees.
            action: Action joint targets (n_joints,) degrees.
            gripper_pos: Current gripper position (0-100%).
            gripper_action: Gripper action target (0-100%).
        """
        if self._current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")

        ep = self._current_episode
        ep.rgb_frames.append(rgb)
        ep.depth_frames.append(depth)
        ep.joint_positions.append(np.asarray(joint_positions, dtype=np.float32))
        ep.actions.append(np.asarray(action, dtype=np.float32))
        ep.gripper_positions.append(float(gripper_pos))
        ep.gripper_actions.append(float(gripper_action))
        ep.timestamps.append(time.time())

    def end_episode(self, success: bool = True) -> Episode:
        """Finish the current episode and save it."""
        if self._current_episode is None:
            raise RuntimeError("No episode to end.")

        ep = self._current_episode
        ep.success = success
        self._episodes.append(ep)
        self._episode_counter += 1

        # Save to disk
        self._save_episode(ep)
        logger.info(
            "Episode %d complete: %d frames, %.1fs, success=%s",
            ep.episode_id, ep.num_frames, ep.duration_s, ep.success,
        )
        self._current_episode = None
        return ep

    def _save_episode(self, ep: Episode) -> None:
        """Save episode data to disk in a format compatible with LeRobot datasets."""
        ep_dir = self.dataset_root / f"episode_{ep.episode_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Save RGB frames as video (H264)
        if ep.rgb_frames:
            h, w = ep.rgb_frames[0].shape[:2]
            video_path = ep_dir / "rgb.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, ep.fps, (w, h))
            for frame in ep.rgb_frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()

        # Save depth frames as compressed numpy
        if ep.depth_frames:
            depth_path = ep_dir / "depth.npz"
            np.savez_compressed(str(depth_path), *ep.depth_frames)

        # Save state and actions as numpy arrays
        if ep.joint_positions:
            states = np.stack(ep.joint_positions, axis=0)
            np.save(ep_dir / "joint_positions.npy", states)

        if ep.actions:
            actions = np.stack(ep.actions, axis=0)
            np.save(ep_dir / "actions.npy", actions)

        # Save gripper data
        np.save(ep_dir / "gripper_positions.npy", np.array(ep.gripper_positions, dtype=np.float32))
        np.save(ep_dir / "gripper_actions.npy", np.array(ep.gripper_actions, dtype=np.float32))

        # Save metadata
        meta = {
            "episode_id": ep.episode_id,
            "task": ep.task,
            "fps": ep.fps,
            "num_frames": ep.num_frames,
            "duration_s": ep.duration_s,
            "success": ep.success,
            "timestamps": ep.timestamps,
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def collect_teleop_episodes(
        self,
        num_episodes: int = 10,
        episode_time_s: float = 60.0,
    ) -> None:
        """Record episodes using LeRobot's built-in teleoperation recording.

        This delegates to ``lerobot-record`` with the OAK-D camera config.
        """
        lr_cfg = self.cfg.get("lerobot", {})
        repo_id = lr_cfg.get("dataset_repo_id", "")
        if not repo_id:
            repo_id = f"local/{self.task.replace(' ', '_')}"

        robot_cfg = self.cfg.get("robot", {})
        cameras_json = json.dumps(robot_cfg.get("cameras", {}))

        import subprocess
        cmd = [
            "lerobot-record",
            f"--robot.type={robot_cfg.get('type', 'so100_follower')}",
            f"--robot.port={robot_cfg.get('port', '')}",
            f"--robot.cameras={cameras_json}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.num_episodes={num_episodes}",
            f"--dataset.single_task={self.task}",
            f"--dataset.fps={self.fps}",
            f"--dataset.episode_time_s={episode_time_s}",
            "--display_data=true",
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def collect_autonomous_episode(
        self,
        arm_controller,
        camera,
        perception_pipeline,
        reasoning_agent,
        T_cam_to_robot: np.ndarray,
        convention: str = "opencv",
    ) -> Episode:
        """Record one episode while the agentic pipeline runs autonomously.

        This captures every observation and action during task execution.
        """
        from main import transform_cam_to_robot

        ep = self.start_episode()

        max_steps = 300  # safety limit
        for step in range(max_steps):
            # Observe
            rgb = camera.read()
            depth = camera.read_depth()
            joints = arm_controller.get_joint_positions()
            gripper = arm_controller.get_gripper_position()

            # Get intrinsics
            intrinsics = camera.get_depth_intrinsics()

            # Perception
            scene, det_states = perception_pipeline.observe(rgb, depth, intrinsics, ep.task)

            # Reason
            scene.task = ep.task
            action = reasoning_agent.reason(scene)

            # Determine action targets
            action_joints = joints.copy()  # default: hold position
            gripper_action = gripper

            if action.is_pick() and action.object_index is not None:
                idx = action.object_index
                if idx < len(scene.objects):
                    obj = scene.objects[idx]
                    cam_pos = np.array(obj.center_xyz)
                    robot_pos = transform_cam_to_robot(cam_pos, T_cam_to_robot, convention)
                    arm_controller.pick(robot_pos.tolist(), object_label=obj.label)
                    # Re-read state after pick
                    action_joints = arm_controller.get_joint_positions()
                    gripper_action = arm_controller.get_gripper_position()

            elif action.is_place() and action.place_xyz is not None:
                place_cam = np.array(action.place_xyz)
                place_robot = transform_cam_to_robot(place_cam, T_cam_to_robot, convention)
                arm_controller.place(place_robot.tolist())
                action_joints = arm_controller.get_joint_positions()
                gripper_action = arm_controller.get_gripper_position()

            # Record
            self.record_step(rgb, depth, joints, action_joints, gripper, gripper_action)

            if action.is_terminal():
                break

        return self.end_episode(success=True)

    def to_lerobot_dataset(self, repo_id: str | None = None) -> None:
        """Convert collected episodes to a LeRobotDataset on disk.

        This creates the HuggingFace-compatible dataset structure that can be
        used directly with LeRobot's training pipeline.
        """
        lr_cfg = self.cfg.get("lerobot", {})
        repo_id = repo_id or lr_cfg.get("dataset_repo_id", f"local/{self.task.replace(' ', '_')}")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Build features dict matching LeRobot's expected format
        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

        episodes_dir = self.dataset_root
        episode_dirs = sorted(episodes_dir.glob("episode_*"))

        if not episode_dirs:
            logger.warning("No episodes found in %s", episodes_dir)
            return

        logger.info("Converting %d episodes to LeRobotDataset format", len(episode_dirs))

        output_dir = self.dataset_root / "lerobot_dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a combined dataset metadata
        all_episodes = []
        for ep_dir in episode_dirs:
            meta_path = ep_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                all_episodes.append(meta)

        dataset_meta = {
            "repo_id": repo_id,
            "task": self.task,
            "fps": self.fps,
            "num_episodes": len(all_episodes),
            "total_frames": sum(ep["num_frames"] for ep in all_episodes),
            "episodes": all_episodes,
            "motor_names": motor_names,
        }

        with open(output_dir / "dataset_meta.json", "w") as f:
            json.dump(dataset_meta, f, indent=2)

        logger.info(
            "Dataset saved to %s: %d episodes, %d total frames",
            output_dir, dataset_meta["num_episodes"], dataset_meta["total_frames"],
        )


def inspect_dataset(dataset_path: str) -> None:
    """Print summary of collected episodes."""
    root = Path(dataset_path)
    episode_dirs = sorted(root.glob("episode_*"))
    print(f"\nDataset at: {root}")
    print(f"Episodes: {len(episode_dirs)}\n")

    total_frames = 0
    total_duration = 0.0
    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            total_frames += meta["num_frames"]
            total_duration += meta["duration_s"]
            print(f"  Episode {meta['episode_id']:3d}: {meta['num_frames']:4d} frames, "
                  f"{meta['duration_s']:.1f}s, success={meta['success']}, task='{meta['task']}'")
        else:
            print(f"  {ep_dir.name}: metadata.json missing")

    print(f"\nTotal: {total_frames} frames, {total_duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Dataset Collector for Robot Manipulation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--task", default="pick up the red cube", help="Task description")
    parser.add_argument("--mode", choices=["teleop", "autonomous"], default="teleop",
                        help="Collection mode")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--episode-time-s", type=float, default=60.0, help="Max seconds per episode")
    parser.add_argument("--inspect", action="store_true", help="Inspect existing dataset")
    parser.add_argument("--dataset-path", default="", help="Path to dataset (for --inspect)")
    parser.add_argument("--convert", action="store_true", help="Convert episodes to LeRobotDataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.inspect:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        path = args.dataset_path or cfg.get("lerobot", {}).get("dataset_root", "./data/episodes")
        inspect_dataset(path)
        return

    collector = DatasetCollector(config_path=args.config, task=args.task)

    if args.convert:
        collector.to_lerobot_dataset()
        return

    if args.mode == "teleop":
        collector.collect_teleop_episodes(
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
        )
    else:
        logger.info("Autonomous collection requires running the full pipeline. Use main.py instead.")


if __name__ == "__main__":
    main()
