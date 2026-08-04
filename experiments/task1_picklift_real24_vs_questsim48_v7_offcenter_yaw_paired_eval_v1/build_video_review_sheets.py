from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def red_centroid(frame: np.ndarray) -> tuple[float, int]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, np.array([0, 80, 45]), np.array([12, 255, 255]))
    high = cv2.inRange(hsv, np.array([170, 80, 45]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(low | high, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return float("nan"), 0
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(centroids[component][1]), int(stats[component, cv2.CC_STAT_AREA])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.operator_manifest.read_text(encoding="utf-8"))
    metrics = []
    strips = []
    timelines = []
    for row in manifest["scored_trials"]:
        capture = cv2.VideoCapture(row["video_path"])
        frames: list[np.ndarray] = []
        ys: list[float] = []
        areas: list[int] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            y, area = red_centroid(frame)
            frames.append(frame)
            ys.append(y)
            areas.append(area)
        capture.release()
        valid = [
            index
            for index, (y, area) in enumerate(zip(ys, areas, strict=True))
            if np.isfinite(y) and area >= 80
        ]
        if not valid:
            raise RuntimeError(f"No red-cube candidate found in {row['trial_id']}.")
        baseline_candidates = [index for index in valid if index < min(40, len(frames))]
        baseline = int(np.median(baseline_candidates)) if baseline_candidates else valid[0]
        highest = min(valid, key=lambda index: ys[index])
        picks = [0, max(0, highest - 10), highest, min(len(frames) - 1, highest + 10)]
        tiles = []
        for frame_index in picks:
            tile = cv2.resize(frames[frame_index], (240, 180))
            text = (
                f"{row['trial_id']} {row['model_id']} "
                f"f{frame_index} y={ys[frame_index]:.1f}"
            )
            cv2.putText(
                tile,
                text,
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.37,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        strips.append(np.hstack(tiles))
        timeline_tiles = []
        for sample_index in range(60):
            frame_index = min(len(frames) - 1, round(sample_index * len(frames) / 60))
            tile = cv2.resize(frames[frame_index], (96, 72))
            cv2.putText(
                tile,
                f"{sample_index / 2:.1f}s",
                (2, 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.24,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            timeline_tiles.append(tile)
        timeline = np.vstack(
            [
                np.hstack(timeline_tiles[row_index * 10 : (row_index + 1) * 10])
                for row_index in range(6)
            ]
        )
        header = np.zeros((24, timeline.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"{row['trial_id']} {row['model_id']}",
            (4, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        timelines.append(np.vstack([header, timeline]))
        metrics.append(
            {
                "trial_id": row["trial_id"],
                "model_id": row["model_id"],
                "frames": len(frames),
                "baseline_frame": baseline,
                "baseline_y": ys[baseline],
                "highest_frame": highest,
                "highest_y": ys[highest],
                "upward_pixel_delta": ys[baseline] - ys[highest],
                "highest_area": areas[highest],
            }
        )
    for sheet_index in range(4):
        chunk = strips[sheet_index * 6 : (sheet_index + 1) * 6]
        cv2.imwrite(
            str(args.output / f"sheet_{sheet_index + 1}.jpg"),
            np.vstack(chunk),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        timeline_chunk = timelines[sheet_index * 6 : (sheet_index + 1) * 6]
        cv2.imwrite(
            str(args.output / f"timeline_{sheet_index + 1}.jpg"),
            np.vstack(timeline_chunk),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
