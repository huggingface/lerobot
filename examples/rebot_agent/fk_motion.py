# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Ground arm-motion labels in measured joints using a calibrated ReBot URDF.

Called by feature extraction after selecting a short motion-consistent interval.
No robot control or IK is performed. Camera points still require visual tracking
or independently validated camera calibration; FK poses are not pixel coordinates.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def measured_positions(states, state_names: list[str], config: dict, *, kinematics=None) -> np.ndarray:
    """Apply the supplied joint mapping and compute per-arm tool positions with main's FK."""
    if config["arm"] not in {"left", "right"} or config["units"] not in {"degrees", "radians"}:
        raise ValueError("Specify the arm and measured revolute-joint units")
    if not config.get("calibration") or not config.get("frame"):
        raise ValueError("FK requires calibration provenance and a named coordinate frame")
    urdf = Path(config["urdf"])
    joint_names, state_keys = config["joint_names"], config["state_keys"]
    if (
        not joint_names
        or len(joint_names) != len(state_keys)
        or len(set(joint_names)) != len(joint_names)
        or len(set(state_keys)) != len(state_keys)
        or len(set(state_names)) != len(state_names)
    ):
        raise ValueError("Each modeled revolute joint needs exactly one state key")
    states = np.asarray(states, dtype=float)
    if (
        states.ndim != 2
        or states.shape[1] != len(state_names)
        or len(states) < 2
        or not np.isfinite(states).all()
    ):
        raise ValueError("Need at least two finite measured states")
    joints = states[:, [state_names.index(key) for key in state_keys]]
    if config["units"] == "radians":
        joints = np.rad2deg(joints)
    signs = np.asarray(config["signs"], dtype=float)
    offsets = np.asarray(config["offset_degrees"], dtype=float)
    if (
        signs.shape != (len(joint_names),)
        or offsets.shape != signs.shape
        or not np.isin(signs, [-1, 1]).all()
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("Joint calibration needs one sign and finite offset per joint")
    joints = joints * signs + offsets
    if kinematics is None:
        from lerobot.model.kinematics import RobotKinematics

        kinematics = RobotKinematics(str(urdf), config["tool_frame"], joint_names)
    positions = np.asarray([kinematics.forward_kinematics(q)[:3, 3] for q in joints])
    if not np.isfinite(positions).all():
        raise ValueError("FK produced invalid poses")
    return positions


def extract_motion(states, state_names: list[str], config: dict, *, kinematics=None) -> list[dict]:
    """Return measured Cartesian direction labels in the explicitly named base frame."""
    if config.get("calibration_status") != "verified":
        raise ValueError(
            "FK motion labels require verified calibration; use measured_positions for diagnostics"
        )
    positions = measured_positions(states, state_names, config, kinematics=kinematics)
    urdf = Path(config["urdf"])
    delta = positions[-1] - positions[0]
    threshold = float(config["deadband_m"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Use a measured positive displacement deadband")
    result = []
    for axis, displacement in enumerate(delta):
        if abs(displacement) < threshold:
            continue
        # Reject net-direction labels for trajectories that substantially reverse direction.
        travel = np.abs(np.diff(positions[:, axis])).sum()
        if travel > 1.5 * abs(displacement):
            raise ValueError("Split this reversing trajectory before assigning an atomic motion")
        direction = config["axis_directions"][axis]["positive" if displacement > 0 else "negative"]
        result.append(
            {
                "text": f"move the {config['arm']} gripper {direction}",
                "evidence": {
                    "method": "measured-joint FK",
                    "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                    "calibration": config["calibration"],
                    "calibration_status": config["calibration_status"],
                    "frame": config["frame"],
                    "tool_frame": config["tool_frame"],
                    "joint_mapping": {
                        key: config[key]
                        for key in ("units", "state_keys", "joint_names", "signs", "offset_degrees")
                    },
                    "axis_directions": config["axis_directions"],
                    "displacement_m": delta.tolist(),
                },
            }
        )
    return result


def segment_motion(
    timestamps,
    positions: dict[str, np.ndarray],
    grippers: dict[str, np.ndarray],
    *,
    boundaries: list[int],
    translation_speed_m_s: float,
    gripper_speed_deg_s: float,
    median_window: int = 5,
    minimum_duration_s: float = 0.15,
    translation_reversal_m: float = 0.01,
    gripper_reversal_deg: float = 10.0,
) -> list[dict]:
    """Partition every frame into offline candidate intervals; never emit language labels.

    A sample i owns transition i -> i+1. The final sample has no measured outgoing
    transition and is kept as an unresolved singleton. Filtering uses future frames
    inside each semantic interval, so this function must not be used at rollout time.
    """
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Need at least two finite timestamps")
    dt = np.diff(times)
    if (dt <= 0).any() or (dt > 1.5 * np.median(dt)).any():
        raise ValueError("Timestamps must increase without unobserved temporal gaps")
    n = len(times)
    if not positions or set(positions) != set(grippers) or not set(positions) <= {"left", "right"}:
        raise ValueError("Positions and gripper angles must identify the same ReBot arms")
    if (
        not isinstance(median_window, int)
        or median_window < 1
        or median_window % 2 != 1
        or not np.isfinite(minimum_duration_s)
        or minimum_duration_s < 0
        or not np.isfinite(
            [translation_speed_m_s, gripper_speed_deg_s, translation_reversal_m, gripper_reversal_deg]
        ).all()
        or min(translation_speed_m_s, gripper_speed_deg_s, translation_reversal_m, gripper_reversal_deg) <= 0
    ):
        raise ValueError("Use positive speed thresholds, an odd median window, and a nonnegative duration")
    if any(not isinstance(b, int) or b < 0 or b > n for b in boundaries):
        raise ValueError("Semantic boundaries must be episode-local frame indices")
    columns, names, thresholds = [], [], []
    for arm in sorted(positions):
        xyz, grip = np.asarray(positions[arm], dtype=float), np.asarray(grippers[arm], dtype=float)
        if (
            xyz.shape != (n, 3)
            or grip.shape != (n,)
            or not np.isfinite(xyz).all()
            or not np.isfinite(grip).all()
        ):
            raise ValueError("Every arm needs finite Nx3 positions and N measured gripper angles")
        columns.extend([xyz, grip[:, None]])
        names.extend([f"{arm}.{axis}" for axis in ("x", "y", "z", "gripper")])
        thresholds.extend([translation_speed_m_s] * 3 + [gripper_speed_deg_s])
    values = np.concatenate(columns, axis=1)
    thresholds = np.asarray(thresholds)
    reversal_thresholds = np.tile([translation_reversal_m] * 3 + [gripper_reversal_deg], len(positions))
    cuts = sorted({0, n, *boundaries})
    result = []
    for start, end in zip(cuts[:-1], cuts[1:], strict=True):
        # Include the closing observation, but never smooth a transition across the
        # semantic boundary. Gripper angles stay in measured degrees, not open/closed.
        stop = min(end, n - 1)
        rates = np.diff(values[start : stop + 1], axis=0) / dt[start:stop, None]
        if not len(rates):
            continue
        pad = median_window // 2
        padded = np.pad(rates, ((pad, pad), (0, 0)), mode="edge")
        filtered = np.median(np.lib.stride_tricks.sliding_window_view(padded, median_window, axis=0), axis=-1)
        signs = np.where(np.abs(filtered) >= thresholds, np.sign(filtered), 0).astype(int)
        raw_signs = np.where(np.abs(rates) >= thresholds, np.sign(rates), 0).astype(int)
        # Debounce each channel independently so asynchronous arm/axis changes do
        # not prevent a sustained change in another channel from being retained.
        for channel in range(signs.shape[1]):
            changes = np.flatnonzero(np.diff(signs[:, channel])) + 1
            runs = [0, *changes.tolist(), len(signs)]
            previous = 0
            for a, b in zip(runs[:-1], runs[1:], strict=True):
                duration = times[start + b] - times[start + a]
                delta = abs(values[start + b, channel] - values[start + a, channel])
                if duration < minimum_duration_s and delta < reversal_thresholds[channel]:
                    signs[a:b, channel] = previous
                else:
                    previous = signs[a, channel]
        changes = set((np.flatnonzero(np.any(np.diff(signs, axis=0), axis=1)) + 1).tolist())
        # Preserve raw excursions above the displacement deadband even if brief:
        # velocity denoising must not erase a real reversal or rapid gripper pulse.
        for channel, threshold in enumerate(reversal_thresholds):
            raw = values[start : stop + 1, channel]
            direction, extreme = 0, 0
            for i in range(1, len(raw)):
                delta = raw[i] - raw[extreme]
                if direction == 0:
                    if delta == 0:
                        extreme = i
                    elif abs(delta) >= threshold:
                        changes.add(extreme)
                        direction, extreme = int(np.sign(delta)), i
                elif direction * delta > 0:
                    extreme = i
                elif abs(delta) >= threshold:
                    changes.add(extreme)
                    direction, extreme = -direction, i
            if direction:
                changes.add(extreme)
        runs = sorted({0, len(signs), *changes})
        for a, b in zip(runs[:-1], runs[1:], strict=True):
            first, last = start + a, start + b
            displacement = values[last] - values[first]
            travel = np.abs(np.diff(values[first : last + 1], axis=0)).sum(axis=0)
            flags = []
            if times[last] - times[first] < minimum_duration_s:
                flags.append("short_interval")
            if np.any(raw_signs[a:b] != signs[a]):
                flags.append("raw_and_filtered_velocity_disagree")
            active = signs[a] != 0
            if np.any(active & ((displacement * signs[a] <= 0) | (travel > 1.5 * np.abs(displacement)))):
                flags.append("raw_motion_disagrees_with_filtered_direction")
            result.append(
                {
                    "start_frame": first,
                    "end_frame": last,
                    "start_time": float(times[first]),
                    "end_time": float(times[last]),
                    "channel_signs": dict(zip(names, signs[a].tolist(), strict=True)),
                    "raw_velocity_signs": {
                        name: np.unique(raw_signs[a:b, c]).tolist() for c, name in enumerate(names)
                    },
                    "measured_delta": dict(zip(names, displacement.tolist(), strict=True)),
                    "review_flags": flags,
                    "review": "pending",
                }
            )
    result.append(
        {
            "start_frame": n - 1,
            "end_frame": n,
            "start_time": float(times[-1]),
            "end_time": None,
            "channel_signs": None,
            "raw_velocity_signs": None,
            "measured_delta": None,
            "review_flags": ["no_outgoing_observation"],
            "review": "pending",
        }
    )
    return result


def main():
    from lerobot.annotations.steerable_pipeline.reader import iter_episodes, reconstruct_subtask_spans

    parser = argparse.ArgumentParser(description="Extract offline FK/gripper interval candidates, not labels")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = json.loads((args.dataset_root / "source.json").read_text())
    if not source.get("repo_id") or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("Dataset source.json must contain a repo ID and immutable revision")
    config["segmentation"] = {
        "median_window": 5,
        "minimum_duration_s": 0.15,
        "translation_reversal_m": 0.01,
        "gripper_reversal_deg": 10.0,
        **config["segmentation"],
    }
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    arms = config["arms"]
    if set(arms) != {"left", "right"} or any(
        a["arm"] != arm or a.get("calibration_status") not in {"verified", "unverified"}
        for arm, a in arms.items()
    ):
        raise ValueError("Provide both arm mappings and their explicit calibration status")
    records = list(iter_episodes(args.dataset_root, only_episodes=tuple(args.episodes)))
    if sorted(r.episode_index for r in records) != sorted(set(args.episodes)):
        raise ValueError("Requested complete episodes are missing or split across shards")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "fk_motion_source.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "source": source,
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "urdf_sha256": {
            arm: hashlib.sha256(Path(a["urdf"]).read_bytes()).hexdigest() for arm, a in arms.items()
        },
        "accepted_training_labels": False,
        "episodes": [],
    }
    for record in records:
        times = np.asarray(record.frame_timestamps)
        if not np.array_equal(record.frame_indices, np.arange(len(times))):
            raise ValueError("Require complete, contiguous episode-local frame indices")
        frame_data = record.frames_df()
        states = np.stack(frame_data["observation.state"])
        positions = {arm: measured_positions(states, names, a) for arm, a in arms.items()}
        grippers = {}
        for arm, a in arms.items():
            values = states[:, names.index(a["gripper_state_key"])].astype(float)
            grippers[arm] = np.rad2deg(values) if a["units"] == "radians" else values
        spans = reconstruct_subtask_spans(
            frame_data.iloc[0].get("language_persistent", []),
            episode_end_t=float(times[-1] + np.median(np.diff(times))),
        )
        boundaries = [int(np.searchsorted(times, s["start"])) for s in spans]
        intervals = segment_motion(
            times, positions, grippers, boundaries=boundaries, **config["segmentation"]
        )
        for interval in intervals:
            matches = [s for s in spans if s["start"] <= times[interval["start_frame"]] < s["end"]]
            interval["subtask"] = matches[0]["text"] if len(matches) == 1 else None
            if len(matches) != 1:
                interval["review_flags"].append("missing_or_ambiguous_subtask")
        arrays = args.output / f"episode_{record.episode_index:06d}.npz"
        np.savez_compressed(
            arrays,
            timestamps=times,
            **{f"{arm}_xyz": v for arm, v in positions.items()},
            **{f"{arm}_gripper_degrees": v for arm, v in grippers.items()},
        )
        report["episodes"].append(
            {
                "episode_index": record.episode_index,
                "frames": len(times),
                "source_parquet_sha256": hashlib.sha256(record.data_path.read_bytes()).hexdigest(),
                "arrays": arrays.name,
                "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                "intervals": intervals,
            }
        )
        (args.output / "intervals.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"episode": record.episode_index, "frames": len(times), "intervals": len(intervals)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
