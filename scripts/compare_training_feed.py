#!/usr/bin/env python

"""Compare a recorded training camera frame with the current live camera feed.

This is a local helper for checking whether lighting, framing, and object placement
match the data used to train a policy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def find_policy_configs(outputs_dir: Path) -> list[Path]:
    return sorted(
        outputs_dir.glob("*/checkpoints/*/pretrained_model/train_config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def choose_policy(config_paths: list[Path]) -> Path:
    if not config_paths:
        raise FileNotFoundError("No policy train_config.json files found under outputs/train.")

    print("\nAvailable trained policies:\n")
    for i, path in enumerate(config_paths):
        rel = path.parent
        try:
            cfg = json.loads(path.read_text())
            dataset_id = cfg.get("dataset", {}).get("repo_id", "?")
        except Exception:
            dataset_id = "?"
        print(f"[{i}] {rel}\n    dataset: {dataset_id}")

    while True:
        raw = input("\nSelect policy number: ").strip()
        try:
            idx = int(raw)
            return config_paths[idx]
        except (ValueError, IndexError):
            print(f"Enter a number from 0 to {len(config_paths) - 1}.")


def image_key_from_config(config: dict) -> str:
    input_features = config.get("policy", {}).get("input_features", {})
    visual_keys = [
        key
        for key, value in input_features.items()
        if isinstance(value, dict) and value.get("type") == "VISUAL"
    ]
    if not visual_keys:
        raise ValueError("Could not find a VISUAL input feature in the policy config.")
    if len(visual_keys) > 1:
        print("Multiple camera inputs found; using the first one:", visual_keys[0])
    return visual_keys[0]


def repo_root_from_repo_id(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / Path(repo_id)


def to_uint8_rgb(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if image.dtype != np.uint8:
        max_value = float(np.nanmax(image)) if image.size else 1.0
        if max_value <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image[..., :3]


def luminance_mean(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def add_label(rgb: np.ndarray, label: str, y: int = 28) -> None:
    cv2.putText(rgb, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(rgb, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)


def make_panel(train_rgb: np.ndarray, live_bgr: np.ndarray, sample_idx: int) -> np.ndarray:
    live_rgb = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2RGB)
    live_rgb = cv2.resize(live_rgb, (train_rgb.shape[1], train_rgb.shape[0]), interpolation=cv2.INTER_AREA)

    train_mean = luminance_mean(train_rgb)
    live_mean = luminance_mean(live_rgb)
    delta = live_mean - train_mean

    train_view = train_rgb.copy()
    live_view = live_rgb.copy()
    add_label(train_view, f"Training frame {sample_idx} | light {train_mean:.1f}")
    add_label(live_view, f"Live camera | light {live_mean:.1f} | delta {delta:+.1f}")

    panel = np.concatenate([train_view, live_view], axis=1)
    footer = np.zeros((56, panel.shape[1], 3), dtype=np.uint8)
    footer_text = "Keys: space pause/play | n next | p previous | r random | q quit | Aim for similar light delta near 0"
    cv2.putText(footer, footer_text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
    return cv2.cvtColor(np.concatenate([panel, footer], axis=0), cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/train"))
    parser.add_argument("--policy", type=Path, help="Path to pretrained_model or train_config.json.")
    parser.add_argument("--dataset-repo-id", help="Override dataset repo id from policy config.")
    parser.add_argument("--dataset-root", type=Path, help="Override local dataset root path.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--play-fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.policy:
        policy_path = args.policy
        config_path = policy_path / "train_config.json" if policy_path.is_dir() else policy_path
    else:
        config_path = choose_policy(find_policy_configs(args.outputs_dir))

    config = json.loads(config_path.read_text())
    image_key = image_key_from_config(config)
    repo_id = args.dataset_repo_id or config["dataset"]["repo_id"]
    dataset_root = args.dataset_root or config.get("dataset", {}).get("root")

    if dataset_root is None:
        candidate = repo_root_from_repo_id(repo_id)
        dataset_root = candidate if candidate.exists() else None

    print(f"\nPolicy config: {config_path}")
    print(f"Dataset: {repo_id}")
    print(f"Dataset root: {dataset_root or '(default LeRobot cache)'}")
    print(f"Camera key: {image_key}")
    print(f"Live camera index: {args.camera_index}\n")

    dataset = LeRobotDataset(repo_id, root=dataset_root)
    if image_key not in dataset[0]:
        raise KeyError(f"Image key '{image_key}' not found in dataset sample. Keys: {sorted(dataset[0])}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    sample_idx = max(0, min(args.sample_index, len(dataset) - 1))
    playing = True
    frame_delay_ms = max(1, int(1000 / args.play_fps))
    cv2.namedWindow("Training vs Live Camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, live_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from live camera.")

            sample = dataset[sample_idx]
            train_rgb = to_uint8_rgb(sample[image_key])
            panel = make_panel(train_rgb, live_bgr, sample_idx)
            cv2.imshow("Training vs Live Camera", panel)

            key = cv2.waitKey(frame_delay_ms if playing else 30) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                playing = not playing
            if key == ord("n"):
                sample_idx = min(sample_idx + 1, len(dataset) - 1)
                playing = False
            elif key == ord("p"):
                sample_idx = max(sample_idx - 1, 0)
                playing = False
            elif key == ord("r"):
                sample_idx = random.randrange(len(dataset))
                playing = False
            elif playing:
                sample_idx = (sample_idx + 1) % len(dataset)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
