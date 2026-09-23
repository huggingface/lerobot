# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Bridge-style random command substitution, with reviewed ReBot grounding provenance."""

import bisect
import json
from pathlib import Path

import numpy as np

from lerobot.utils.steering import render_steering_command

from .language_task import RecipeTaskDataset

STYLES = {"subtask", "motion", "point", "trace", "combination"}


class SteeringCommands:
    """Index half-open frame intervals; every alternative labels the same demonstrated actions."""

    def __init__(self, manifest: dict):
        if manifest.get("version") != 1 or not manifest.get("source", {}).get("revision"):
            raise ValueError("Steering manifest needs version=1 and a pinned source revision")
        self.source = manifest["source"]
        self.episodes: dict[int, list[dict]] = {}
        self.starts: dict[int, list[int]] = {}
        for span in manifest["segments"]:
            episode, start, end = (span[k] for k in ("episode_index", "start_frame", "end_frame"))
            if any(type(n) is not int for n in (episode, start, end)) or episode < 0 or not 0 <= start < end:
                raise ValueError("Invalid episode/frame interval")
            if span.get("review", {}).get("verdict") != "accepted" or not span["review"].get("reviewer"):
                raise ValueError("Every training segment must have an accepted, attributed review")
            if not span.get("commands"):
                raise ValueError("Steering segment has no commands")
            for command in span["commands"]:
                if (
                    command.get("style") not in STYLES
                    or not isinstance(command.get("text"), str)
                    or not command["text"].strip()
                ):
                    raise ValueError("Invalid steering command")
                if not command.get("evidence"):
                    raise ValueError("Each steering command needs grounding provenance")
                render_steering_command(command)
            self.episodes.setdefault(episode, []).append(span)
        for episode, spans in self.episodes.items():
            spans.sort(key=lambda s: s["start_frame"])
            if any(a["end_frame"] > b["start_frame"] for a, b in zip(spans, spans[1:], strict=False)):
                raise ValueError("Overlapping steering intervals are ambiguous")
            self.starts[episode] = [s["start_frame"] for s in spans]

    def at(self, episode: int, frame: int) -> list[dict]:
        spans = self.episodes.get(episode, [])
        index = bisect.bisect_right(self.starts.get(episode, []), frame) - 1
        if index < 0 or frame >= spans[index]["end_frame"]:
            raise ValueError(f"Missing reviewed steering commands for episode {episode}, frame {frame}")
        return spans[index]["commands"]

    def coverage(self, episode_lengths: dict[int, int]) -> dict:
        """Check every requested frame before training, including wholly absent episodes."""
        gaps = []
        covered = 0
        for episode, length in episode_lengths.items():
            cursor = 0
            for span in self.episodes.get(episode, []):
                start, end = span["start_frame"], span["end_frame"]
                if end > length:
                    raise ValueError(f"Steering interval exceeds episode {episode} length {length}")
                if start > cursor:
                    gaps.append({"episode_index": episode, "start_frame": cursor, "end_frame": start})
                covered += end - start
                cursor = end
            if cursor < length:
                gaps.append({"episode_index": episode, "start_frame": cursor, "end_frame": length})
        return {
            "complete": not gaps,
            "covered_frames": covered,
            "total_frames": sum(episode_lengths.values()),
            "gaps": gaps,
        }

    def sample(self, sample: dict, task_probability: float, *, deterministic: bool = False) -> dict:
        commands = self.at(int(sample["episode_index"]), int(sample["frame_index"]))
        # Training uses the worker-seeded RNG on every visit, like Bridge. Evaluation is frame-stable.
        rng = np.random.default_rng(int(sample["index"])) if deterministic else np.random
        if rng.random() < task_probability:
            return sample
        index = int(rng.integers(len(commands))) if deterministic else int(rng.randint(len(commands)))
        command = commands[index]
        return {**sample, "task": render_steering_command(command)}


class SteeringCommandDataset(RecipeTaskDataset):
    """Reuse LeRobot decoding and recipe task fallback; change only the conditioning instruction."""

    def __init__(
        self, *args, steering_manifest: str, task_probability: float = 0.2, deterministic=False, **kwargs
    ):
        self.steering = SteeringCommands(json.loads(Path(steering_manifest).read_text()))
        self.task_probability = task_probability
        self.deterministic = deterministic
        if not 0 <= task_probability <= 1:
            raise ValueError("task_probability must be in [0, 1]")
        super().__init__(*args, **kwargs)
        if (
            self.repo_id != self.steering.source["repo_id"]
            or kwargs.get("revision") != self.steering.source["revision"]
        ):
            raise ValueError("Steering manifest does not match the pinned dataset")
        episodes = self.episodes if self.episodes is not None else range(self.meta.total_episodes)
        report = self.steering.coverage({ep: int(self.meta.episodes[ep]["length"]) for ep in episodes})
        if not report["complete"]:
            raise ValueError(
                f"Missing reviewed steering coverage: {len(report['gaps'])} gaps; first: {report['gaps'][0]}"
            )
        for spans in self.steering.episodes.values():
            for span in spans:
                for command in span["commands"]:
                    if command.get("points"):
                        feature = self.meta.features.get(command["camera"], {})
                        width, height = command["image_size"]
                        if tuple(feature.get("shape", ())) != (height, width, 3):
                            raise ValueError(
                                "Steering coordinates do not match the dataset camera dimensions"
                            )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return self.steering.sample(
            super().__getitem__(idx), self.task_probability, deterministic=self.deterministic
        )
