#!/usr/bin/env python3

"""Manual emotion annotation tool for LeRobot datasets.

Usage example:
    python useful_scripts/annotate_emotions.py \
        --repo-id HSP-IIT/HRI3 \
        --output /tmp/hri3_emotions.npz

Controls:
    - 0/1/2/3: set emotion for current frame
    - Left/Right: move 1 frame (fills labels in between)
    - PageUp/PageDown: move 10 frames (fills labels in between)
    - Home/End: jump to first/last frame (fills labels in between)
    - s: save annotations
    - q or Esc: save and quit
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


EMOTION_MAP = {
    0: "neutral",
    1: "happy",
    2: "alert",
    3: "shy",
}


class EmotionAnnotator:
    def __init__(
        self,
        repo_id: str,
        root: str | None,
        camera_key: str | None,
        output_path: Path,
        jump_size: int,
        cache_size: int,
    ):
        self.dataset = LeRobotDataset(repo_id, root=root)
        self.output_path = output_path
        self.jump_size = max(1, jump_size)

        if not self.dataset.meta.camera_keys:
            raise ValueError("Dataset has no camera keys. Cannot open frame annotation UI.")

        self.camera_key = camera_key or self.dataset.meta.camera_keys[0]
        if self.camera_key not in self.dataset.meta.camera_keys:
            raise ValueError(
                f"Camera key '{self.camera_key}' not found. Available: {self.dataset.meta.camera_keys}"
            )

        self.num_frames = self.dataset.num_frames
        self.current_index = 0
        self.default_label = 0
        self.cache_size = max(1, int(cache_size))
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        self.labels = np.full(self.num_frames, -1, dtype=np.int64)
        self._load_if_exists()
        self.labeled_count = int(np.sum(self.labels >= 0))

        if self.labels[0] < 0:
            user_input = input("Initial label for first frame [default 0]: ").strip()
            if user_input:
                self.default_label = self._parse_label(user_input)
            self.labels[0] = self.default_label
            self.labeled_count += 1

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        first_image = self._get_image(self.current_index)
        if first_image.ndim == 2 or (first_image.ndim == 3 and first_image.shape[-1] == 1):
            self._image_artist = self.ax.imshow(np.squeeze(first_image), cmap="gray")
        else:
            self._image_artist = self.ax.imshow(first_image)
        self.ax.axis("off")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._controls_text = None
        self._redraw()

    @staticmethod
    def _parse_label(value: str) -> int:
        label = int(value)
        if label not in EMOTION_MAP:
            raise ValueError("Label must be one of: 0, 1, 2, 3")
        return label

    def _load_if_exists(self) -> None:
        if not self.output_path.exists():
            return
        data = np.load(self.output_path)
        loaded = data["labels"]
        if loaded.shape[0] != self.num_frames:
            raise ValueError(
                f"Annotation length mismatch: file has {loaded.shape[0]} labels, dataset has {self.num_frames} frames"
            )
        self.labels = loaded.astype(np.int64)
        print(f"Loaded existing annotations from {self.output_path}")

    def _to_image(self, frame: dict) -> np.ndarray:
        image = frame[self.camera_key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)

        return image

    def _get_image(self, index: int) -> np.ndarray:
        cached = self._frame_cache.get(index)
        if cached is not None:
            self._frame_cache.move_to_end(index)
            return cached

        frame = self.dataset[index]
        image = self._to_image(frame)
        self._frame_cache[index] = image
        if len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)
        return image

    def _fill_between(self, old_index: int, new_index: int) -> None:
        if old_index == new_index:
            return
        label = self.labels[old_index]
        if label < 0:
            label = self.default_label
        lo = min(old_index, new_index)
        hi = max(old_index, new_index)
        unlabeled_in_range = int(np.sum(self.labels[lo : hi + 1] < 0))
        self.labels[lo : hi + 1] = label
        self.labeled_count += unlabeled_in_range

    def _move(self, delta: int) -> None:
        old = self.current_index
        new = int(np.clip(old + delta, 0, self.num_frames - 1))
        self._fill_between(old, new)
        self.current_index = new

    def _set_label(self, label: int) -> None:
        if self.labels[self.current_index] < 0:
            self.labeled_count += 1
        self.labels[self.current_index] = label

    def _status(self) -> str:
        current_label = self.labels[self.current_index]
        name = EMOTION_MAP.get(int(current_label), "unset") if current_label >= 0 else "unset"
        return (
            f"frame {self.current_index + 1}/{self.num_frames} | "
            f"label={current_label} ({name}) | labeled={self.labeled_count}/{self.num_frames}"
        )

    def _redraw(self) -> None:
        image = self._get_image(self.current_index)
        self._image_artist.set_data(np.squeeze(image) if (image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1)) else image)
        self.ax.set_title(self._status())

        controls = (
            "0=neutral 1=happy 2=alert 3=shy | "
            "←/→ move 1 | PgUp/PgDn move 10 | Home/End jump | s save | q/esc quit"
        )
        if self._controls_text is None:
            self._controls_text = self.fig.text(0.5, 0.01, controls, ha="center", va="bottom", fontsize=9)
        else:
            self._controls_text.set_text(controls)
        self.fig.canvas.draw_idle()

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, labels=self.labels)

        csv_path = self.output_path.with_suffix(".csv")
        ep_idx = np.asarray(self.dataset.hf_dataset["episode_index"], dtype=np.int64)
        frame_idx = np.asarray(self.dataset.hf_dataset["frame_index"], dtype=np.int64)
        df = pd.DataFrame(
            {
                "global_index": np.arange(self.num_frames, dtype=np.int64),
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "emotions": self.labels,
            }
        )
        df.to_csv(csv_path, index=False)
        print(f"Saved annotations: {self.output_path}")
        print(f"Saved CSV: {csv_path}")

    def export_dataset(self, repo_id: str, output_dir: str | None) -> None:
        export_labels = self.labels.copy()
        if np.any(export_labels < 0):
            print("Warning: found unlabeled frames. Filling with 0 (neutral) for dataset export.")
            export_labels[export_labels < 0] = 0

        emotions = export_labels.astype(np.float32).reshape(-1, 1)
        feature_info = {
            "dtype": "float32",
            "shape": [1],
            "names": ["emotions"],
        }

        _ = add_features(
            dataset=self.dataset,
            features={"emotions": (emotions, feature_info)},
            output_dir=output_dir,
            repo_id=repo_id,
        )
        print(f"Exported dataset with 'emotions' feature as repo_id={repo_id}")

    def _on_key(self, event) -> None:
        key = event.key

        try:
            if key in {"0", "1", "2", "3"}:
                self._set_label(int(key))
            elif key == "right":
                self._move(+1)
            elif key == "left":
                self._move(-1)
            elif key == "pageup":
                self._move(+self.jump_size)
            elif key == "pagedown":
                self._move(-self.jump_size)
            elif key == "home":
                self._fill_between(self.current_index, 0)
                self.current_index = 0
            elif key == "end":
                self._fill_between(self.current_index, self.num_frames - 1)
                self.current_index = self.num_frames - 1
            elif key == "s":
                self.save()
            elif key in {"q", "escape"}:
                self.save()
                plt.close(self.fig)
                return
            else:
                return
        except Exception as exc:
            print(f"Input error: {exc}")

        self._redraw()

    def run(self) -> None:
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual emotion annotation tool for LeRobot datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID, e.g. HSP-IIT/HRI3")
    parser.add_argument("--root", type=str, default=None, help="Optional local LeRobot datasets root")
    parser.add_argument("--camera-key", type=str, default=None, help="Camera key to visualize")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/emotions.npz"),
        help="Output NPZ annotations file",
    )
    parser.add_argument(
        "--jump-size",
        type=int,
        default=10,
        help="Frame jump size for PageUp/PageDown",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=64,
        help="LRU cache size (number of decoded frames to keep in memory)",
    )
    parser.add_argument(
        "--export-repo-id",
        type=str,
        default=None,
        help="If set, export a new dataset with an added 'emotions' column",
    )
    parser.add_argument(
        "--export-output-dir",
        type=str,
        default=None,
        help="Optional output dir for exported dataset",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    annotator = EmotionAnnotator(
        repo_id=args.repo_id,
        root=args.root,
        camera_key=args.camera_key,
        output_path=args.output,
        jump_size=args.jump_size,
        cache_size=args.cache_size,
    )
    annotator.run()

    if args.export_repo_id:
        annotator.export_dataset(repo_id=args.export_repo_id, output_dir=args.export_output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
