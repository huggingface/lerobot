from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import cv2
import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


AUDIT_ID = "task1_picklift_evalv2_systematic_offset_audit_v1"
EXPECTED_OWNING_PARENT_COMMIT = "de2fce222a8e7e94670efea5b2eae3ff0ac3dfb7"
RESEARCH_COMMIT = "f4d288dcf6c5f7402560fa2cb0a5e5676cd375a6"
RESEARCH_RESULT_MANIFEST_SHA256 = (
    "1b726459c4664919644a246d4955946cbdcd83fda660330fef004b26ed2b367b"
)
RESEARCH_DECISION_LOG_SHA256 = (
    "1dbfaa761a70cbd4cf367e73bfaece66533b3e573dee60a361b3a15e2f0de457"
)
FROZEN_DATASET_TREE_SHA256 = (
    "251cbdc079b304425ccdfbd7a08f15d34858ea0dd8c19345544b8da9f3adb9f2"
)
FROZEN_MODEL_SHA256 = (
    "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
)

EVAL_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evaluation/"
    "task1_picklift_real24_offcenter_yaw_eval_v2_difficulty_pilot_v2_grid15mm"
)
REVIEW_ROOT = EVAL_ROOT / "canonical_video_review_v1"
REVIEW_TRIALS = REVIEW_ROOT / "trials.jsonl"
EXPECTED_REVIEW_TRIALS_SHA256 = (
    "218062238c4c8dd218567cd316b33e742244edcf5025a94201b198d1c4baaf92"
)
DATASET_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728"
)
DATA_PARQUET = DATASET_ROOT / "data/chunk-000/file-000.parquet"
DATA_VIDEO = (
    DATASET_ROOT
    / "videos/observation.images.front/chunk-000/file-000.mp4"
)
GEOMETRY_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/analysis/task1_picklift_camera_geometry_audit_v1"
)
REAL_CENTROIDS = GEOMETRY_ROOT / "real_cube_centroids_mean.json"
GEOMETRY_METRICS = GEOMETRY_ROOT / "review_v6_production/geometry_metrics.json"
READY_FRAME_MANIFEST = GEOMETRY_ROOT / "real_ready_frames/manifest.json"
URDF_PATH = Path(
    "/home/ubuntu24/.cache/huggingface/lerobot/robot-urdfs/so101/"
    "so101_new_calib.urdf"
)
OUTPUT_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/analysis/"
    "task1_picklift_evalv2_systematic_offset_audit_v1"
)

EXPECTED_INPUT_HASHES = {
    "review_trials_jsonl": EXPECTED_REVIEW_TRIALS_SHA256,
    "real_cube_centroids_mean": (
        "2a84e671767d351e404dffbf97c0a92877b81302391985f7500eb6240985729e"
    ),
    "geometry_metrics": (
        "46c8a9a5f3f2373d1738abfe27ae39a121c72e88256b8c62a02507cd82e8e3f7"
    ),
    "real_ready_frames_manifest": (
        "4ba7e0fdcc2ef8d678bf3ad7d842bd227b7ad5ee1e5a80e5d75f1361fa4957bb"
    ),
    "dataset_info": (
        "fdc3d07728ee634963e2082ccae4943eb6fc1028a8703b4fc140d3b0d4bbda26"
    ),
    "dataset_data_parquet": (
        "960fed916b6a28c3f5569827896669630eac28b026e175b1a5eb5cc52c041709"
    ),
    "urdf": "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c",
}

CELL_CENTERS_TASK_M = {
    f"r{row}c{column}": (0.225 + 0.05 * (row - 1), -0.075 + 0.05 * (column - 1))
    for row in range(1, 4)
    for column in range(1, 5)
}
JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
EARLY_FRAME_INDICES = tuple(range(5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros(maximum.shape, dtype=np.float64)
    nonzero = delta > 0
    red = (maximum == rgb[:, :, 0]) & nonzero
    green = (maximum == rgb[:, :, 1]) & nonzero
    blue = (maximum == rgb[:, :, 2]) & nonzero
    hue[red] = ((rgb[:, :, 1][red] - rgb[:, :, 2][red]) / delta[red]) % 6.0
    hue[green] = (
        (rgb[:, :, 2][green] - rgb[:, :, 0][green]) / delta[green] + 2.0
    )
    hue[blue] = (
        (rgb[:, :, 0][blue] - rgb[:, :, 1][blue]) / delta[blue] + 4.0
    )
    hue *= 60.0
    saturation = np.zeros(maximum.shape, dtype=np.float64)
    np.divide(delta, maximum, out=saturation, where=maximum > 0)
    mask = (
        ((hue < 28.0) | (hue > 345.0))
        & (saturation > 0.45)
        & (maximum > 89)
    )
    yy, xx = np.indices(mask.shape)
    return mask & (xx > 50) & (xx < 390) & (yy > 210) & (yy < 430)


def red_component(
    rgb: np.ndarray, expected_px: Iterable[float] | None = None
) -> dict:
    mask = (_red_mask(rgb).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 25]
    if not contours:
        raise RuntimeError("No red component found in canonical table ROI.")
    expected = (
        np.asarray(list(expected_px), dtype=np.float64)
        if expected_px is not None
        else None
    )
    if expected is None:
        contour = max(contours, key=cv2.contourArea)
        selection_distance = None
        selection_rule = "largest component"
    else:
        def contour_distance(contour: np.ndarray) -> float:
            moments = cv2.moments(contour)
            centroid = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
            return float(np.linalg.norm(centroid - expected))

        contour = min(contours, key=contour_distance)
        selection_distance = contour_distance(contour)
        if selection_distance > 60.0:
            raise RuntimeError(
                f"No plausible red cube component within 60 px of expected position; "
                f"nearest={selection_distance:.2f}px"
            )
        selection_rule = "nearest component to outcome-independent nominal projection"
    points = contour.reshape(-1, 2)
    component_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
    yy, xx = np.nonzero(component_mask)
    left, top, width, height = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    orientation = math.degrees(math.atan2(float(edge[1]), float(edge[0]))) % 90.0
    major = float(max(rect[1]))
    minor = float(min(rect[1]))
    return {
        "centroid_px": [float(xx.mean()), float(yy.mean())],
        "bbox_xyxy": [left, top, left + width - 1, top + height - 1],
        "bbox_size_px": [width, height],
        "bbox_geometric_mean_px": math.sqrt(width * height),
        "segmented_area_px": int(len(xx)),
        "component_selection": {
            "rule": selection_rule,
            "expected_px": expected.astype(float).tolist() if expected is not None else None,
            "distance_to_expected_px": selection_distance,
            "minimum_contour_area_px": 25,
            "maximum_expected_distance_px": 60 if expected is not None else None,
        },
        "minimum_area_rectangle": {
            "orientation_modulo_90_degrees": orientation,
            "major_axis_px": major,
            "minor_axis_px": minor,
            "aspect_ratio": major / minor if minor else None,
            "corners_px": box.astype(float).tolist(),
            "interpretation": (
                "image-space shape evidence only; the near-square cube, perspective, "
                "manual placement, and segmentation make this unsuitable as yaw measurement truth"
            ),
        },
    }


def fit_task_to_pixel_homography(real_centroids: dict[str, list[float]]) -> tuple[np.ndarray, dict]:
    task = np.asarray([CELL_CENTERS_TASK_M[cell] for cell in sorted(real_centroids)], dtype=np.float64)
    pixels = np.asarray([real_centroids[cell] for cell in sorted(real_centroids)], dtype=np.float64)
    homography, _ = cv2.findHomography(task, pixels, method=0)
    projected = project_points(homography, task)
    residuals = np.linalg.norm(projected - pixels, axis=1)
    return homography, {
        "anchor_count": len(task),
        "median_anchor_residual_px": float(np.median(residuals)),
        "maximum_anchor_residual_px": float(np.max(residuals)),
        "anchor_residuals_px": {
            cell: float(value) for cell, value in zip(sorted(real_centroids), residuals)
        },
    }


def project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def image_to_task(homography: np.ndarray, point: list[float]) -> list[float]:
    inverse = np.linalg.inv(homography)
    return project_points(inverse, np.asarray([point]))[0].astype(float).tolist()


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    rx = np.array(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    )
    ry = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    return rz @ ry @ rx


def _origin_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _axis_transform(axis: list[float], degrees: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


@dataclass
class UrdfForwardKinematics:
    joints: dict[str, tuple[np.ndarray, list[float]]]
    gripper_frame_origin: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "UrdfForwardKinematics":
        root = ET.parse(path).getroot()
        joints = {}
        fixed = None
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            origin = joint.find("origin")
            xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
            origin_tf = _origin_transform(xyz, rpy)
            axis_node = joint.find("axis")
            axis = (
                [float(value) for value in axis_node.attrib["xyz"].split()]
                if axis_node is not None
                else [0.0, 0.0, 0.0]
            )
            if name in JOINT_ORDER:
                joints[name] = (origin_tf, axis)
            if name == "gripper_frame_joint":
                fixed = origin_tf
        if set(joints) != set(JOINT_ORDER) or fixed is None:
            raise RuntimeError("URDF does not contain the expected SO101 chain.")
        return cls(joints=joints, gripper_frame_origin=fixed)

    def position(self, state: Iterable[float]) -> np.ndarray:
        transform = np.eye(4)
        values = list(state)
        for index, name in enumerate(JOINT_ORDER):
            origin, axis = self.joints[name]
            transform = transform @ origin @ _axis_transform(axis, values[index])
        transform = transform @ self.gripper_frame_origin
        return transform[:3, 3].copy()


def first_closure(rows: list[dict], fk: UrdfForwardKinematics) -> dict:
    states = np.asarray([row["observation_state"] for row in rows], dtype=np.float64)
    gripper = states[:, 5]
    minimum_tick = int(np.argmin(gripper))
    peak_tick = int(np.argmax(gripper[: minimum_tick + 1]))
    peak = float(gripper[peak_tick])
    minimum = float(gripper[minimum_tick])
    threshold = (peak + minimum) / 2.0
    candidates = np.where(gripper[peak_tick : minimum_tick + 1] <= threshold)[0]
    close_tick = peak_tick + int(candidates[0]) if len(candidates) else minimum_tick
    window_end = min(len(rows), close_tick + 11)
    positions = np.asarray([fk.position(state[:5]) for state in states])
    low_tick = peak_tick + int(np.argmin(positions[peak_tick:window_end, 2]))
    return {
        "open_peak_tick": peak_tick,
        "open_peak_gripper": peak,
        "closure_minimum_tick": minimum_tick,
        "closure_minimum_gripper": minimum,
        "first_closure_tick": close_tick,
        "first_closure_threshold": threshold,
        "low_approach_tick": low_tick,
        "low_approach_gripper_frame_xyz_m": positions[low_tick].astype(float).tolist(),
        "first_closure_gripper_frame_xyz_m": positions[close_tick].astype(float).tolist(),
        "proxy_definition": (
            "first midpoint crossing from the pre-closure open peak toward the episode "
            "minimum; low approach is the minimum URDF gripper-frame Z from that peak "
            "through ten ticks after first closure"
        ),
    }


def decode_selected_frames(video_path: Path, selected: set[int]) -> tuple[dict[int, np.ndarray], int]:
    frames = {}
    index = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if index in selected:
                frames[index] = frame.to_ndarray(format="rgb24")
            index += 1
    missing = selected - set(frames)
    if missing:
        raise RuntimeError(f"Missing selected frames in {video_path}: {sorted(missing)}")
    return frames, index


def decode_frame(video_path: Path, index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for current, frame in enumerate(container.decode(stream)):
            if current == index:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Cannot decode frame {index}: {video_path}")


def euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)))


def convex_hull_contains(points: np.ndarray, query: Iterable[float]) -> bool:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return cv2.pointPolygonTest(hull, tuple(map(float, query)), False) >= 0


def linear_diagnostics(rows: list[dict], field: str) -> dict:
    y = np.asarray([row[field] for row in rows], dtype=np.float64)
    x_position = np.asarray(
        [
            [
                1.0,
                row["visible_cube_task_xy_m"][0],
                row["visible_cube_task_xy_m"][1],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    x_yaw = np.column_stack(
        [x_position, [1.0 if row["nominal_yaw_degrees_modulo_90"] == 45 else 0.0 for row in rows]]
    )
    beta_position = np.linalg.lstsq(x_position, y, rcond=None)[0]
    beta_yaw = np.linalg.lstsq(x_yaw, y, rcond=None)[0]
    sse_position = float(np.sum((y - x_position @ beta_position) ** 2))
    sse_yaw = float(np.sum((y - x_yaw @ beta_yaw) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "position_only_coefficients": beta_position.astype(float).tolist(),
        "position_plus_yaw_coefficients": beta_yaw.astype(float).tolist(),
        "position_only_sse": sse_position,
        "position_plus_yaw_sse": sse_yaw,
        "position_only_r_squared": 1.0 - sse_position / sst if sst > 0 else None,
        "position_plus_yaw_r_squared": 1.0 - sse_yaw / sst if sst > 0 else None,
        "yaw_added_sse_reduction_fraction": (
            (sse_position - sse_yaw) / sse_position if sse_position > 0 else None
        ),
        "sample_size": len(rows),
        "interpretation": "descriptive fit only; n=12 is insufficient for a causal yaw claim",
    }


def draw_cross(draw: ImageDraw.ImageDraw, point: Iterable[float], color: str, size: int = 7) -> None:
    x, y = map(float, point)
    draw.line((x - size, y, x + size, y), fill=color, width=2)
    draw.line((x, y - size, x, y + size), fill=color, width=2)


def placement_contact_sheet(rows: list[dict], path: Path) -> None:
    sheet = Image.new("RGB", (1280, 1080), "black")
    for index, row in enumerate(rows):
        image = Image.open(row["pre_action_frame"]["path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_cross(draw, row["expected_cube_centroid_px"], "cyan")
        draw_cross(draw, row["visible_cube_centroid_px"], "lime")
        draw.line(
            (*row["expected_cube_centroid_px"], *row["visible_cube_centroid_px"]),
            fill="yellow",
            width=2,
        )
        draw.rectangle((0, 0, 640, 35), fill="black")
        dx, dy = row["visible_minus_nominal_task_xy_m"]
        draw.text(
            (5, 5),
            (
                f"{row['order']:02d} {row['cell_id']} yaw{row['nominal_yaw_degrees_modulo_90']} "
                f"visible-nominal=({dx * 100:+.1f},{dy * 100:+.1f})cm"
            ),
            fill="white",
        )
        image.thumbnail((320, 240))
        x = (index % 4) * 320
        y = (index // 4) * 360
        sheet.paste(image, (x, y))
    sheet.save(path)


def approach_contact_sheet(rows: list[dict], path: Path) -> None:
    sheet = Image.new("RGB", (1280, 1080), "black")
    for index, row in enumerate(rows):
        image = Image.open(row["approach_frame"]["path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw_cross(draw, row["approach_visible_cube_centroid_px"], "lime")
        draw_cross(draw, row["demo_calibrated_gripper_projection_px"], "magenta")
        draw.line(
            (
                *row["approach_visible_cube_centroid_px"],
                *row["demo_calibrated_gripper_projection_px"],
            ),
            fill="yellow",
            width=2,
        )
        draw.rectangle((0, 0, 640, 35), fill="black")
        rx, ry = row["demo_calibrated_policy_residual_xy_m"]
        draw.text(
            (5, 5),
            (
                f"{row['order']:02d} {row['cell_id']} residual=({rx * 100:+.1f},"
                f"{ry * 100:+.1f})cm review={'S' if row['review_success'] else 'F'}"
            ),
            fill="white",
        )
        image.thumbnail((320, 240))
        x = (index % 4) * 320
        y = (index // 4) * 360
        sheet.paste(image, (x, y))
    sheet.save(path)


def coverage_plot(training_rows: list[dict], trial_rows: list[dict], path: Path) -> None:
    image = Image.new("RGB", (640, 480), "#202020")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 210, 390, 430), outline="#777777", width=1)
    for row in training_rows:
        x, y = row["median_early_centroid_px"]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#4da6ff")
    for row in trial_rows:
        x, y = row["visible_cube_centroid_px"]
        color = "#00ff66" if row["review_success"] else "#ff5c5c"
        draw.line((x - 5, y - 5, x + 5, y + 5), fill=color, width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill=color, width=2)
        draw.text((x + 5, y - 8), str(row["order"]), fill=color)
    draw.text((8, 8), "blue dots: Real24 episode early medians", fill="#4da6ff")
    draw.text((8, 26), "red/green X: Eval-v2 failed/success pre-action centroids", fill="white")
    image.save(path)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite immutable audit root: {OUTPUT_ROOT}")
    checks = {
        "review_trials_jsonl": (REVIEW_TRIALS, EXPECTED_INPUT_HASHES["review_trials_jsonl"]),
        "real_cube_centroids_mean": (
            REAL_CENTROIDS,
            EXPECTED_INPUT_HASHES["real_cube_centroids_mean"],
        ),
        "geometry_metrics": (GEOMETRY_METRICS, EXPECTED_INPUT_HASHES["geometry_metrics"]),
        "real_ready_frames_manifest": (
            READY_FRAME_MANIFEST,
            EXPECTED_INPUT_HASHES["real_ready_frames_manifest"],
        ),
        "dataset_info": (
            DATASET_ROOT / "meta/info.json",
            EXPECTED_INPUT_HASHES["dataset_info"],
        ),
        "dataset_data_parquet": (
            DATA_PARQUET,
            EXPECTED_INPUT_HASHES["dataset_data_parquet"],
        ),
        "urdf": (URDF_PATH, EXPECTED_INPUT_HASHES["urdf"]),
    }
    for name, (path, expected) in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name} hash mismatch: {actual}")

    review_rows = read_jsonl(REVIEW_TRIALS)
    if len(review_rows) != 12 or sum(row["review_success"] for row in review_rows) != 1:
        raise RuntimeError("Frozen reviewed result identity mismatch.")
    if any("__replacement1" not in row["artifact_stem"] for row in review_rows if row["order"] == 8):
        raise RuntimeError("Scored t08 is not the frozen linked replacement.")

    geometry_metrics = read_json(GEOMETRY_METRICS)
    accepted_fit_note = geometry_metrics["v6_candidate"]["camera"]["alignment"][
        "lens_distortion_decision"
    ]
    real_centroids = read_json(REAL_CENTROIDS)
    homography, homography_fit = fit_task_to_pixel_homography(real_centroids)
    inverse_homography = np.linalg.inv(homography)
    fk = UrdfForwardKinematics.load(URDF_PATH)

    dataframe = pd.read_parquet(DATA_PARQUET)
    if len(dataframe) != 3790 or dataframe["episode_index"].nunique() != 24:
        raise RuntimeError("Frozen Real24 table identity mismatch.")
    dataset_sequences = {}
    selected_global_indices = set()
    dataset_closures = {}
    for episode_index, group in dataframe.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        rows = [
            {"observation_state": np.asarray(value, dtype=float).tolist()}
            for value in group["observation.state"]
        ]
        closure = first_closure(rows, fk)
        dataset_closures[int(episode_index)] = closure
        globals_for_early = []
        for frame_index in EARLY_FRAME_INDICES:
            match = group[group["frame_index"] == frame_index]
            if len(match) != 1:
                raise RuntimeError(f"Episode {episode_index} missing early frame {frame_index}.")
            global_index = int(match.iloc[0]["index"])
            selected_global_indices.add(global_index)
            globals_for_early.append(global_index)
        selected_global_indices.add(
            int(group.iloc[closure["low_approach_tick"]]["index"])
        )
        dataset_sequences[int(episode_index)] = {
            "group": group,
            "early_global_indices": globals_for_early,
            "approach_global_index": int(group.iloc[closure["low_approach_tick"]]["index"]),
        }
    selected_frames, decoded_dataset_frames = decode_selected_frames(
        DATA_VIDEO, selected_global_indices
    )
    if decoded_dataset_frames != 3790:
        raise RuntimeError("Frozen Real24 video frame count mismatch.")

    training_rows = []
    demo_fk_points = []
    demo_cube_points = []
    training_early_points = []
    for episode_index in range(24):
        provenance_path = (
            DATASET_ROOT
            / f"provenance/episodes/episode_{episode_index:06d}.json"
        )
        provenance = read_json(provenance_path)
        expected_training_px = project_points(
            homography,
            np.asarray(
                [[provenance["spawn_x_cm"] / 100.0, provenance["spawn_y_cm"] / 100.0]]
            ),
        )[0]
        early = [
            red_component(selected_frames[index], expected_training_px)
            for index in dataset_sequences[episode_index]["early_global_indices"]
        ]
        early_centroids = np.asarray([row["centroid_px"] for row in early])
        median_centroid = np.median(early_centroids, axis=0)
        cube_task = project_points(
            inverse_homography, np.asarray([median_centroid])
        )[0]
        closure = dataset_closures[episode_index]
        approach_xyz = np.asarray(closure["low_approach_gripper_frame_xyz_m"])
        demo_fk_points.append(approach_xyz[:2])
        demo_cube_points.append(cube_task)
        training_early_points.extend(early_centroids.tolist())
        training_rows.append(
            {
                "episode_index": episode_index,
                "cell_id": provenance["spawn_region"],
                "nominal_spawn_x_m": provenance["spawn_x_cm"] / 100.0,
                "nominal_spawn_y_m": provenance["spawn_y_cm"] / 100.0,
                "early_frame_indices": list(EARLY_FRAME_INDICES),
                "early_centroids_px": early_centroids.astype(float).tolist(),
                "median_early_centroid_px": median_centroid.astype(float).tolist(),
                "median_early_cube_task_xy_m_approx": cube_task.astype(float).tolist(),
                "maximum_early_centroid_deviation_px": float(
                    np.max(np.linalg.norm(early_centroids - median_centroid, axis=1))
                ),
                "closure_proxy": closure,
                "low_approach_gripper_frame_xy_m": approach_xyz[:2].astype(float).tolist(),
                "provenance": {
                    "path": str(provenance_path),
                    "sha256": sha256_file(provenance_path),
                    "result": provenance["result"],
                    "success": provenance["success"],
                    "yaw_distribution_claim": provenance["yaw_distribution_claim"],
                },
            }
        )
    demo_fk_points_array = np.asarray(demo_fk_points)
    demo_cube_points_array = np.asarray(demo_cube_points)
    fk_design = np.column_stack(
        [demo_fk_points_array, np.ones(len(demo_fk_points_array))]
    )
    fk_to_task_affine = np.linalg.lstsq(
        fk_design, demo_cube_points_array, rcond=None
    )[0]
    demo_calibrated_predictions = fk_design @ fk_to_task_affine
    demo_calibration_residuals = demo_calibrated_predictions - demo_cube_points_array
    demo_calibration_rmse_m = float(
        np.sqrt(np.mean(np.sum(demo_calibration_residuals**2, axis=1)))
    )
    for training_row, prediction, residual in zip(
        training_rows, demo_calibrated_predictions, demo_calibration_residuals
    ):
        training_row["demo_calibrated_grasp_xy_m"] = prediction.astype(float).tolist()
        training_row["demo_calibration_residual_xy_m"] = residual.astype(float).tolist()
    training_early_array = np.asarray(training_early_points)
    training_episode_medians = np.asarray(
        [row["median_early_centroid_px"] for row in training_rows]
    )
    cell_training_points = defaultdict(list)
    for row in training_rows:
        cell_training_points[row["cell_id"]].extend(row["early_centroids_px"])
    cell_training_means = {
        cell: np.mean(points, axis=0) for cell, points in cell_training_points.items()
    }
    training_support_radius_px = float(
        max(
            np.linalg.norm(np.asarray(point) - cell_training_means[cell])
            for cell, points in cell_training_points.items()
            for point in points
        )
    )

    frames_root = OUTPUT_ROOT / "frames"
    OUTPUT_ROOT.mkdir(parents=True)
    frames_root.mkdir()
    trial_rows = []
    for row in sorted(review_rows, key=lambda item: item["order"]):
        evidence = read_json(Path(row["evidence"]["path"]))
        trial = evidence["trial"]
        pre_action_path = Path(str(row["evidence"]["path"]).replace(".json", ".pre_action.png"))
        if not pre_action_path.exists():
            raise RuntimeError(f"Missing pre-action frame: {pre_action_path}")
        pre_rgb = np.asarray(Image.open(pre_action_path).convert("RGB"))
        nominal = np.asarray(
            [trial["nominal_x_forward_m"], trial["nominal_y_lateral_m"]],
            dtype=float,
        )
        expected_px = project_points(homography, nominal[None, :])[0]
        visible = red_component(pre_rgb, expected_px)
        visible_px = np.asarray(visible["centroid_px"])
        visible_task = project_points(inverse_homography, visible_px[None, :])[0]
        nominal_residual_task = visible_task - nominal
        nominal_residual_px = visible_px - expected_px

        step_rows = read_jsonl(Path(evidence["steps_jsonl"]["path"]))
        closure = first_closure(step_rows, fk)
        approach_tick = closure["low_approach_tick"]
        approach_rgb = decode_frame(Path(evidence["video"]["path"]), approach_tick)
        approach_component = red_component(approach_rgb, visible_px)
        approach_visible_px = np.asarray(approach_component["centroid_px"])
        approach_cube_task = project_points(
            inverse_homography, approach_visible_px[None, :]
        )[0]
        approach_xyz = np.asarray(closure["low_approach_gripper_frame_xyz_m"])
        calibrated_grasp_xy = (
            np.asarray([approach_xyz[0], approach_xyz[1], 1.0])
            @ fk_to_task_affine
        )
        calibrated_residual = calibrated_grasp_xy - approach_cube_task
        fk_ground_px = project_points(
            homography, calibrated_grasp_xy[None, :]
        )[0]

        approach_frame_path = frames_root / f"{row['artifact_stem']}.approach.png"
        Image.fromarray(approach_rgb).save(approach_frame_path)
        same_cell_mean = cell_training_means[trial["cell_id"]]
        nearest_early = float(
            np.min(np.linalg.norm(training_early_array - visible_px, axis=1))
        )
        nearest_episode_median = float(
            np.min(np.linalg.norm(training_episode_medians - visible_px, axis=1))
        )
        same_cell_distance = euclidean(visible_px, same_cell_mean)
        trial_rows.append(
            {
                "order": row["order"],
                "trial_id": row["trial_id"],
                "artifact_stem": row["artifact_stem"],
                "cell_id": row["cell_id"],
                "quadrant": row["quadrant"],
                "nominal_x_forward_m": trial["nominal_x_forward_m"],
                "nominal_y_lateral_m": trial["nominal_y_lateral_m"],
                "nominal_yaw_degrees_modulo_90": trial[
                    "nominal_yaw_degrees_modulo_90"
                ],
                "review_success": row["review_success"],
                "review_failure_category": row["review_failure_category"],
                "pre_action_frame": {
                    "path": str(pre_action_path),
                    "sha256": sha256_file(pre_action_path),
                },
                "visible_cube_centroid_px": visible_px.astype(float).tolist(),
                "visible_cube_bbox_xyxy": visible["bbox_xyxy"],
                "visible_cube_bbox_size_px": visible["bbox_size_px"],
                "visible_cube_segmented_area_px": visible["segmented_area_px"],
                "visible_cube_shape_yaw_evidence": visible["minimum_area_rectangle"],
                "expected_cube_centroid_px": expected_px.astype(float).tolist(),
                "visible_minus_expected_centroid_px": nominal_residual_px.astype(float).tolist(),
                "expected_visible_centroid_residual_norm_px": float(
                    np.linalg.norm(nominal_residual_px)
                ),
                "visible_cube_task_xy_m": visible_task.astype(float).tolist(),
                "visible_minus_nominal_task_xy_m": nominal_residual_task.astype(float).tolist(),
                "visible_nominal_task_residual_norm_m": float(
                    np.linalg.norm(nominal_residual_task)
                ),
                "manual_nominal_is_measurement_truth": False,
                "training_coverage": {
                    "nearest_real24_early_frame_distance_px": nearest_early,
                    "nearest_real24_episode_median_distance_px": nearest_episode_median,
                    "same_cell_early_mean_distance_px": same_cell_distance,
                    "outside_sampled_early_support_radius": (
                        nearest_early > training_support_radius_px
                    ),
                    "inside_global_real24_early_convex_hull": convex_hull_contains(
                        training_early_array, visible_px
                    ),
                },
                "closure_proxy": closure,
                "approach_frame": {
                    "path": str(approach_frame_path),
                    "sha256": sha256_file(approach_frame_path),
                    "source_video_path": evidence["video"]["path"],
                    "source_video_sha256": evidence["video"]["sha256"],
                    "policy_tick": approach_tick,
                },
                "approach_visible_cube_centroid_px": approach_visible_px.astype(float).tolist(),
                "approach_visible_cube_task_xy_m": approach_cube_task.astype(float).tolist(),
                "fk_gripper_frame_xyz_m": approach_xyz.astype(float).tolist(),
                "demo_calibrated_gripper_projection_px": fk_ground_px.astype(float).tolist(),
                "demo_calibrated_grasp_xy_m": calibrated_grasp_xy.astype(float).tolist(),
                "fk_to_task_affine_matrix": fk_to_task_affine.astype(float).tolist(),
                "successful_demo_calibration_rmse_m": demo_calibration_rmse_m,
                "demo_calibrated_policy_residual_xy_m": calibrated_residual.astype(float).tolist(),
                "demo_calibrated_policy_residual_norm_m": float(
                    np.linalg.norm(calibrated_residual)
                ),
                "upstream_action_modified_events": evidence[
                    "upstream_action_modified_events"
                ],
                "operator_notes": row["operator_notes"],
                "review_notes": row["review_notes"],
            }
        )

    residuals = np.asarray(
        [row["demo_calibrated_policy_residual_xy_m"] for row in trial_rows]
    )
    placement_residuals_m = np.asarray(
        [row["visible_nominal_task_residual_norm_m"] for row in trial_rows]
    )
    placement_residuals_px = np.asarray(
        [row["expected_visible_centroid_residual_norm_px"] for row in trial_rows]
    )
    nearest_early = np.asarray(
        [row["training_coverage"]["nearest_real24_early_frame_distance_px"] for row in trial_rows]
    )
    yaw_groups = {}
    for yaw in (0, 45):
        selected = [row for row in trial_rows if row["nominal_yaw_degrees_modulo_90"] == yaw]
        yaw_groups[str(yaw)] = {
            "trials": len(selected),
            "successes": sum(row["review_success"] for row in selected),
            "median_policy_residual_norm_m": float(
                np.median([row["demo_calibrated_policy_residual_norm_m"] for row in selected])
            ),
            "median_placement_residual_norm_m": float(
                np.median([row["visible_nominal_task_residual_norm_m"] for row in selected])
            ),
            "median_image_shape_orientation_modulo_90_degrees": float(
                np.median(
                    [
                        row["visible_cube_shape_yaw_evidence"][
                            "orientation_modulo_90_degrees"
                        ]
                        for row in selected
                    ]
                )
            ),
        }

    eval_visible_task = np.asarray([row["visible_cube_task_xy_m"] for row in trial_rows])
    eval_calibrated_grasp_points = np.asarray(
        [row["demo_calibrated_grasp_xy_m"] for row in trial_rows]
    )
    reachability = {
        "task_contract_bounds_m": {
            "x": [0.20, 0.35],
            "y": [-0.10, 0.10],
        },
        "nominal_eval_points_inside_task_contract_bounds": sum(
            0.20 <= row["nominal_x_forward_m"] <= 0.35
            and -0.10 <= row["nominal_y_lateral_m"] <= 0.10
            for row in trial_rows
        ),
        "visible_eval_points_inside_real24_cube_convex_hull": sum(
            convex_hull_contains(
                np.asarray(
                    [training_row["median_early_cube_task_xy_m_approx"] for training_row in training_rows]
                ),
                point,
            )
            for point in eval_visible_task
        ),
        "eval_calibrated_approach_points_inside_successful_demo_grasp_hull": sum(
            convex_hull_contains(demo_cube_points_array, point)
            for point in eval_calibrated_grasp_points
        ),
        "successful_demo_grasp_xy_range_m": {
            "minimum": np.min(demo_cube_points_array, axis=0).astype(float).tolist(),
            "maximum": np.max(demo_cube_points_array, axis=0).astype(float).tolist(),
        },
        "eval_calibrated_approach_xy_range_m": {
            "minimum": np.min(eval_calibrated_grasp_points, axis=0).astype(float).tolist(),
            "maximum": np.max(eval_calibrated_grasp_points, axis=0).astype(float).tolist(),
        },
        "interpretation": (
            "offline proxy only: nominal points are within the frozen task bounds and "
            "the videos show near-object approaches, but URDF FK and observed demonstration "
            "coverage cannot prove physical reachability without hardware"
        ),
    }

    confirmed_setup_rows = [
        row["trial_id"]
        for row in trial_rows
        if row["visible_nominal_task_residual_norm_m"] > 0.01
    ]
    confirmed_training_gap_rows = [
        row["trial_id"]
        for row in trial_rows
        if row["training_coverage"]["outside_sampled_early_support_radius"]
    ]
    median_residual = np.median(residuals, axis=0)
    same_x_sign = int(np.sum(np.sign(residuals[:, 0]) == np.sign(median_residual[0])))
    same_y_sign = int(np.sum(np.sign(residuals[:, 1]) == np.sign(median_residual[1])))
    x_diagnostics = linear_diagnostics(
        [
            {**row, "residual_x": row["demo_calibrated_policy_residual_xy_m"][0]}
            for row in trial_rows
        ],
        "residual_x",
    )
    y_diagnostics = linear_diagnostics(
        [
            {**row, "residual_y": row["demo_calibrated_policy_residual_xy_m"][1]}
            for row in trial_rows
        ],
        "residual_y",
    )
    consistent_direction = max(same_x_sign, same_y_sign) >= 10
    position_trend = max(
        x_diagnostics["position_only_r_squared"],
        y_diagnostics["position_only_r_squared"],
    ) >= 0.50
    residual_above_demo_fit = float(
        np.median(np.linalg.norm(residuals, axis=1))
    ) > 2.0 * demo_calibration_rmse_m
    systematic_offset_confirmed = (
        residual_above_demo_fit and (consistent_direction or position_trend)
    )
    summary = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "frozen_result": {
            "reviewed_successes": 1,
            "trials": 12,
            "missed_grasp_failures": 11,
            "scored_t08": "replacement1",
        },
        "homography": {
            "task_to_real_canonical_matrix": homography.astype(float).tolist(),
            "real_canonical_to_task_matrix": inverse_homography.astype(float).tolist(),
            "fit": homography_fit,
            "accepted_geometry_evidence_note": accepted_fit_note,
            "manual_nominal_is_measurement_truth": False,
        },
        "placement_and_grid_mapping": {
            "median_expected_visible_residual_px": float(np.median(placement_residuals_px)),
            "maximum_expected_visible_residual_px": float(np.max(placement_residuals_px)),
            "median_approx_task_residual_m": float(np.median(placement_residuals_m)),
            "maximum_approx_task_residual_m": float(np.max(placement_residuals_m)),
            "trials_over_1cm_approx": len(confirmed_setup_rows),
            "trials_over_1cm_ids": confirmed_setup_rows,
            "classification": (
                "confirmed placement/setup discrepancy for listed trials"
                if confirmed_setup_rows
                else "no confirmed placement/setup error above 1 cm"
            ),
            "boundary": (
                "The inverse-homography values are approximate task-plane coordinates. "
                "Manual nominal placement is an instruction, not measured physical truth."
            ),
        },
        "real24_image_position_coverage": {
            "training_episodes": 24,
            "early_frames_per_episode": len(EARLY_FRAME_INDICES),
            "sampled_early_support_radius_px": training_support_radius_px,
            "eval_nearest_early_distance_px": {
                "median": float(np.median(nearest_early)),
                "maximum": float(np.max(nearest_early)),
            },
            "outside_sampled_early_support_trials": len(confirmed_training_gap_rows),
            "outside_sampled_early_support_trial_ids": confirmed_training_gap_rows,
            "inside_global_training_convex_hull_trials": sum(
                row["training_coverage"]["inside_global_real24_early_convex_hull"]
                for row in trial_rows
            ),
            "classification": (
                "confirmed sparse image-space training-position coverage gap"
                if confirmed_training_gap_rows
                else "no confirmed image-space gap under the sampled-support definition"
            ),
            "boundary": "Real24 exact metric cube poses and yaw are unknown.",
        },
        "policy_offset": {
            "fk_to_task_affine_matrix": fk_to_task_affine.astype(float).tolist(),
            "successful_demo_calibration_rmse_m": demo_calibration_rmse_m,
            "median_demo_calibrated_residual_xy_m": median_residual.astype(float).tolist(),
            "median_demo_calibrated_residual_norm_m": float(
                np.median(np.linalg.norm(residuals, axis=1))
            ),
            "maximum_demo_calibrated_residual_norm_m": float(
                np.max(np.linalg.norm(residuals, axis=1))
            ),
            "same_sign_counts": {"x": same_x_sign, "y": same_y_sign},
            "systematic_rule": (
                "median residual norm exceeds two times the successful-demo calibration RMSE "
                "and either at least 10/12 share one axis sign or a position-only descriptive "
                "fit has R-squared at least 0.50"
            ),
            "consistent_direction": consistent_direction,
            "position_trend": position_trend,
            "residual_above_demo_fit": residual_above_demo_fit,
            "classification": (
                "confirmed systematic position-dependent policy approach offset"
                if systematic_offset_confirmed
                else "unresolved; no single offset direction satisfies the frozen descriptive rule"
            ),
            "dominant_observed_trend": (
                "X residual contracts toward the middle of the sampled X range: "
                "near-row poses tend to overshoot X while far-row poses tend to fall short. "
                f"The descriptive X-residual coefficient on visible cube X is "
                f"{x_diagnostics['position_only_coefficients'][1]:+.3f} m/m."
            ),
            "x_diagnostics": x_diagnostics,
            "y_diagnostics": y_diagnostics,
            "boundary": (
                "URDF gripper-frame FK is calibrated against the 24 successful demonstrations; "
                "it is a reproducible approach proxy, not a direct fingertip contact measurement."
            ),
        },
        "yaw_diagnostic": {
            "groups": yaw_groups,
            "classification": (
                "possible but not separated: yaw45 had 0/6 successes versus yaw0 1/6, "
                "while adding yaw explained no more than "
                f"{100 * max(x_diagnostics['yaw_added_sse_reduction_fraction'], y_diagnostics['yaw_added_sse_reduction_fraction']):.1f}% "
                "of position-adjusted residual SSE"
            ),
            "boundary": (
                "Only 6 yaw0 and 6 yaw45 trials are available; visible position residual and "
                "manual yaw uncertainty prevent a causal conclusion."
            ),
        },
        "reachability_proxy": reachability,
        "cause_buckets": {
            "confirmed_placement_setup_error": (
                f"{len(confirmed_setup_rows)}/12 trials exceed 1 cm approximate nominal-visible residual"
                if confirmed_setup_rows
                else "not confirmed above the 1 cm audit threshold"
            ),
            "confirmed_training_coverage_gap": (
                f"{len(confirmed_training_gap_rows)}/12 eval positions lie outside sampled Real24 early-frame support"
            ),
            "confirmed_systematic_policy_offset": (
                "yes; position-dependent X contraction rather than one constant direction"
                if systematic_offset_confirmed
                else "no"
            ),
            "possible_yaw_effect": (
                "remains possible but is not separated from position; diagnostic only"
            ),
            "unresolved": [
                "exact physical cube pose and yaw for manual Eval-v2 placements",
                "direct fingertip contact point and force",
                "physical reachability margin outside observed trajectories",
                "causal separation of position, yaw, and policy feedback after a miss",
            ],
        },
    }

    # Select the next protocol recommendation from the frozen diagnostic evidence.
    placement_problem = len(confirmed_setup_rows) >= 4
    coverage_problem = len(confirmed_training_gap_rows) >= 8
    if placement_problem:
        recommendation = {
            "decision": "shrink_and_realign_evalv2_before_new_data",
            "next_step": (
                "Do not enter new aligned Sim collection yet. First version a placement "
                "protocol with directly auditable image targets; use yaw0 only and reduce "
                "offset magnitude until nominal-visible residual is controlled."
            ),
            "avoid": "Do not choose the next points from individual failure cells.",
        }
    elif coverage_problem:
        recommendation = {
            "decision": "retain_15mm_offsets_and_isolate_position_with_yaw0_before_new_aligned_sim_collection",
            "next_step": (
                "Keep the same predeclared balanced 15 mm off-center magnitude because the "
                "visible placements matched the nominal projection; use yaw0 only in the next "
                "fixed diagnostic to isolate position. If that confirms the same coverage-linked "
                "X contraction, then define aligned Sim collection over the full predeclared "
                "off-center distribution."
            ),
            "avoid": "Do not mine individual failed cells or mix a yaw intervention into the same gate.",
        }
    else:
        recommendation = {
            "decision": "retain_offsets_but_isolate_yaw0",
            "next_step": (
                "The placement and sampled coverage audits do not justify shrinking offsets; "
                "repeat only a frozen yaw0 diagnostic before considering aligned Sim collection."
            ),
            "avoid": "Do not select points based on observed failures.",
        }
    summary["recommended_next_gate"] = recommendation

    trial_jsonl = OUTPUT_ROOT / "trials.jsonl"
    training_jsonl = OUTPUT_ROOT / "real24_training_coverage.jsonl"
    write_jsonl(trial_jsonl, trial_rows)
    write_jsonl(training_jsonl, training_rows)
    trial_csv = OUTPUT_ROOT / "trials.csv"
    with trial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order",
                "trial_id",
                "cell_id",
                "nominal_x_forward_m",
                "nominal_y_lateral_m",
                "nominal_yaw_degrees_modulo_90",
                "review_success",
                "visible_cube_centroid_x_px",
                "visible_cube_centroid_y_px",
                "expected_cube_centroid_x_px",
                "expected_cube_centroid_y_px",
                "placement_residual_px",
                "placement_residual_m_approx",
                "nearest_real24_early_px",
                "policy_residual_x_m",
                "policy_residual_y_m",
                "policy_residual_norm_m",
            ],
        )
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(
                {
                    "order": row["order"],
                    "trial_id": row["trial_id"],
                    "cell_id": row["cell_id"],
                    "nominal_x_forward_m": row["nominal_x_forward_m"],
                    "nominal_y_lateral_m": row["nominal_y_lateral_m"],
                    "nominal_yaw_degrees_modulo_90": row[
                        "nominal_yaw_degrees_modulo_90"
                    ],
                    "review_success": row["review_success"],
                    "visible_cube_centroid_x_px": row["visible_cube_centroid_px"][0],
                    "visible_cube_centroid_y_px": row["visible_cube_centroid_px"][1],
                    "expected_cube_centroid_x_px": row["expected_cube_centroid_px"][0],
                    "expected_cube_centroid_y_px": row["expected_cube_centroid_px"][1],
                    "placement_residual_px": row[
                        "expected_visible_centroid_residual_norm_px"
                    ],
                    "placement_residual_m_approx": row[
                        "visible_nominal_task_residual_norm_m"
                    ],
                    "nearest_real24_early_px": row["training_coverage"][
                        "nearest_real24_early_frame_distance_px"
                    ],
                    "policy_residual_x_m": row[
                        "demo_calibrated_policy_residual_xy_m"
                    ][0],
                    "policy_residual_y_m": row[
                        "demo_calibrated_policy_residual_xy_m"
                    ][1],
                    "policy_residual_norm_m": row[
                        "demo_calibrated_policy_residual_norm_m"
                    ],
                }
            )
    summary_path = OUTPUT_ROOT / "summary.json"
    write_json(summary_path, summary)
    placement_sheet = OUTPUT_ROOT / "placement_expected_visible_overlay.png"
    approach_sheet = OUTPUT_ROOT / "approach_policy_offset_overlay.png"
    coverage_image = OUTPUT_ROOT / "real24_eval_image_position_coverage.png"
    placement_contact_sheet(trial_rows, placement_sheet)
    approach_contact_sheet(trial_rows, approach_sheet)
    coverage_plot(training_rows, trial_rows, coverage_image)

    manifest = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": "offline_read_only_inputs",
        "prohibitions_observed": [
            "no serial",
            "no hardware camera",
            "no robot or torque",
            "no 12V",
            "no Quest or Remote service",
            "no policy or MuJoCo rollout",
            "no training or fine-tuning",
            "no Dataset writes",
        ],
        "research_control": {
            "commit": RESEARCH_COMMIT,
            "result_manifest_sha256": RESEARCH_RESULT_MANIFEST_SHA256,
            "decision_log_sha256": RESEARCH_DECISION_LOG_SHA256,
        },
        "owning_parent_commit": EXPECTED_OWNING_PARENT_COMMIT,
        "fixed_model_sha256": FROZEN_MODEL_SHA256,
        "frozen_dataset_tree_sha256": FROZEN_DATASET_TREE_SHA256,
        "input_hashes": {
            name: {"path": str(path), "sha256": expected}
            for name, (path, expected) in checks.items()
        },
        "segmentation": {
            "contract": (
                "same fixed Real HSV mask and table ROI as the accepted camera geometry audit; "
                "when multiple red objects exist, select the component nearest the "
                "outcome-independent nominal projection (maximum 60 px)"
            ),
            "source_reference": (
                "IAmRobotTrainerResearch/artifacts/picklift-camera-geometry-audit-v1/"
                "analyze_canonical_frames.py"
            ),
            "source_reference_sha256": (
                "4a795498937fd2e454b925c5f05b5dbc1dbd744024d124bf6a09407981366a27"
            ),
        },
        "homography": summary["homography"],
        "fk_proxy": {
            "urdf": str(URDF_PATH),
            "urdf_sha256": EXPECTED_INPUT_HASHES["urdf"],
            "target_frame": "gripper_frame_link",
            "calibration": (
                "least-squares affine mapping from raw URDF gripper-frame XY to "
                "image-homography cube XY using all 24 successful Real24 demonstrations"
            ),
            "affine_matrix": fk_to_task_affine.astype(float).tolist(),
            "successful_demo_fit_rmse_m": demo_calibration_rmse_m,
        },
        "outputs": {},
    }
    output_files = [
        trial_jsonl,
        trial_csv,
        training_jsonl,
        summary_path,
        placement_sheet,
        approach_sheet,
        coverage_image,
        *sorted(frames_root.glob("*.png")),
    ]
    for path in output_files:
        manifest["outputs"][path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    manifest_path = OUTPUT_ROOT / "audit_manifest.json"
    write_json(manifest_path, manifest)
    hashes_path = OUTPUT_ROOT / "hashes.sha256"
    hash_targets = [*output_files, manifest_path]
    hashes_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in hash_targets),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
