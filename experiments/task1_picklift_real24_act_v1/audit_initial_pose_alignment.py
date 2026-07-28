"""Audit frozen Real-24 episode starts against gate12 Nexus reset states.

This script reads Parquet and JSON evidence only. It imports no camera, serial,
robot, torque, gateway, Quest, MuJoCo, or policy execution module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EXPECTED_DATASET_TREE_SHA256 = (
    "251cbdc079b304425ccdfbd7a08f15d34858ea0dd8c19345544b8da9f3adb9f2"
)
EXPECTED_DATA_FILE_SHA256 = (
    "960fed916b6a28c3f5569827896669630eac28b026e175b1a5eb5cc52c041709"
)
EXPECTED_NEXUS_RESET_EVIDENCE_SHA256 = (
    "7ce34c0e083f0237f83de839edd84d7507b747aa87db772e2618197c4a295fd4"
)
EXPECTED_EPISODES = 24
EXPECTED_GATE_RESETS = 12
EARLY_FRAME_COUNT = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_tree_sha256(dataset_root: Path) -> tuple[str, int]:
    lines = []
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(dataset_root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(), len(files)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def joint_statistics(values: np.ndarray) -> list[dict[str, float]]:
    if values.ndim != 2 or values.shape[1] != len(JOINT_ORDER):
        raise RuntimeError("Joint matrix must have shape [N,6].")
    q1 = np.quantile(values, 0.25, axis=0)
    median = np.median(values, axis=0)
    q3 = np.quantile(values, 0.75, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    result = []
    for index, joint in enumerate(JOINT_ORDER):
        result.append(
            {
                "joint": joint,
                "min": float(values[:, index].min()),
                "q1": float(q1[index]),
                "median": float(median[index]),
                "q3": float(q3[index]),
                "max": float(values[:, index].max()),
                "range": float(np.ptp(values[:, index])),
                "std_population": float(values[:, index].std(ddof=0)),
                "iqr": float(q3[index] - q1[index]),
                "mad": float(mad[index]),
            }
        )
    return result


def choose_representatives(
    episode_indices: np.ndarray,
    states: np.ndarray,
) -> dict[str, Any]:
    coordinate_median = np.median(states, axis=0)
    distance_to_median = np.linalg.norm(states - coordinate_median, axis=1)
    closest_index = int(np.argmin(distance_to_median))
    pairwise = np.linalg.norm(
        states[:, np.newaxis, :] - states[np.newaxis, :, :],
        axis=2,
    )
    medoid_index = int(np.argmin(pairwise.sum(axis=1)))

    def record(index: int) -> dict[str, Any]:
        return {
            "episode_index": int(episode_indices[index]),
            "frame_index": 0,
            "state": states[index].tolist(),
            "distance_to_coordinatewise_median_l2": float(
                distance_to_median[index]
            ),
            "sum_pairwise_l2_distance": float(pairwise[index].sum()),
        }

    return {
        "coordinatewise_median_not_a_command": coordinate_median.tolist(),
        "selected_ready_pose_rule": (
            "observed frame0 state minimizing raw-unit L2 distance to the "
            "coordinatewise frame0 median"
        ),
        "selected_ready_pose": record(closest_index),
        "raw_l2_medoid": record(medoid_index),
        "selected_and_medoid_same_episode": bool(
            closest_index == medoid_index
        ),
    }


def compare_distributions(
    real_states: np.ndarray,
    nexus_states: np.ndarray,
    representative_state: np.ndarray,
) -> tuple[list[dict[str, Any]], list[str]]:
    real_stats = joint_statistics(real_states)
    nexus_stats = joint_statistics(nexus_states)
    comparisons = []
    ood_joints = []
    for index, joint in enumerate(JOINT_ORDER):
        real = real_stats[index]
        nexus = nexus_stats[index]
        real_min = real["min"]
        real_max = real["max"]
        outside = (nexus_states[:, index] < real_min) | (
            nexus_states[:, index] > real_max
        )
        if nexus["median"] < real_min:
            nearest_observed_bound_delta = nexus["median"] - real_min
        elif nexus["median"] > real_max:
            nearest_observed_bound_delta = nexus["median"] - real_max
        else:
            nearest_observed_bound_delta = 0.0
        lower_tukey = real["q1"] - 1.5 * real["iqr"]
        upper_tukey = real["q3"] + 1.5 * real["iqr"]
        strict_ood = bool(outside.all())
        distribution_shift = bool(
            nexus["median"] < lower_tukey
            or nexus["median"] > upper_tukey
        )
        if strict_ood or distribution_shift:
            ood_joints.append(joint)
        comparisons.append(
            {
                "joint": joint,
                "real_frame0": real,
                "nexus_reset": nexus,
                "nexus_minus_real_median": float(
                    nexus["median"] - real["median"]
                ),
                "nexus_median_minus_representative_ready_pose": float(
                    nexus["median"] - representative_state[index]
                ),
                "nexus_median_distance_in_real_population_std": (
                    float(
                        abs(nexus["median"] - real["median"])
                        / real["std_population"]
                    )
                    if real["std_population"] > 0
                    else None
                ),
                "nexus_sample_outside_real_observed_range_count": int(
                    outside.sum()
                ),
                "nexus_sample_outside_real_observed_range_fraction": float(
                    outside.mean()
                ),
                "nexus_median_nearest_observed_bound_signed_delta": float(
                    nearest_observed_bound_delta
                ),
                "real_tukey_inner_fence": [lower_tukey, upper_tukey],
                "strict_ood_all_12_outside_real_min_max": strict_ood,
                "distribution_shift_nexus_median_outside_real_tukey_fence": (
                    distribution_shift
                ),
            }
        )
    return comparisons, ood_joints


def load_real_episode_starts(
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    data_file = dataset_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(
        data_file,
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "observation.state",
        ],
    )
    payload = table.to_pydict()
    episodes = np.asarray(payload["episode_index"], dtype=np.int64)
    frames = np.asarray(payload["frame_index"], dtype=np.int64)
    timestamps = np.asarray(payload["timestamp"], dtype=np.float64)
    states = np.asarray(payload["observation.state"], dtype=np.float64)
    if states.shape != (len(episodes), len(JOINT_ORDER)):
        raise RuntimeError("Parquet observation.state is not [frames,6].")
    records = []
    frame0_states = []
    episode_indices = []
    for episode_index in sorted(set(episodes.tolist())):
        mask = episodes == episode_index
        order = np.argsort(frames[mask])
        episode_frames = frames[mask][order]
        episode_timestamps = timestamps[mask][order]
        episode_states = states[mask][order]
        if episode_frames[0] != 0 or len(episode_states) < EARLY_FRAME_COUNT:
            raise RuntimeError(
                f"Episode {episode_index} lacks frame0 or five early frames."
            )
        if not np.array_equal(
            episode_frames[:EARLY_FRAME_COUNT],
            np.arange(EARLY_FRAME_COUNT),
        ):
            raise RuntimeError(
                f"Episode {episode_index} early frame indices are not 0..4."
            )
        provenance_path = (
            dataset_root
            / "provenance/episodes"
            / f"episode_{episode_index:06d}.json"
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        early = episode_states[:EARLY_FRAME_COUNT]
        record = {
            "episode_index": int(episode_index),
            "spawn_region": provenance["spawn_region"],
            "spawn_id": provenance["spawn_id"],
            "frame0_timestamp": float(episode_timestamps[0]),
            "frame0_state": early[0].tolist(),
            "first_five_frame_states": early.tolist(),
            "first_five_max_abs_delta_from_frame0": np.max(
                np.abs(early - early[0]), axis=0
            ).tolist(),
            "first_five_range": np.ptp(early, axis=0).tolist(),
        }
        records.append(record)
        frame0_states.append(early[0])
        episode_indices.append(episode_index)
    if len(records) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Expected 24 episodes, found {len(records)}."
        )
    return (
        records,
        np.asarray(episode_indices, dtype=np.int64),
        np.asarray(frame0_states, dtype=np.float64),
    )


def load_nexus_states(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    encoded = path.read_bytes()
    if encoded.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = encoded.decode("utf-16")
    else:
        text = encoded.decode("utf-8-sig")
    records = [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_GATE_RESETS:
        raise RuntimeError("Expected exactly 12 Nexus reset records.")
    if {record["seed"] for record in records} != set(range(9000, 9012)):
        raise RuntimeError("Nexus gate reset seeds are not frozen 9000..9011.")
    states = np.asarray([record["state"] for record in records], dtype=np.float64)
    if states.shape != (EXPECTED_GATE_RESETS, len(JOINT_ORDER)):
        raise RuntimeError("Nexus reset states are not [12,6].")
    if not np.isfinite(states).all():
        raise RuntimeError("Nexus reset evidence contains non-finite state.")
    return records, states


def build_ready_pose_contract(
    selected: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    state = selected["state"]
    state_sha = hashlib.sha256(
        json.dumps(state, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "profile_id": "task1_real24_ready_pose_reset_v1",
        "schema_version": 1,
        "joint_order": list(JOINT_ORDER),
        "state_dataset_units": state,
        "state_dtype": "float32_on_adapter_boundary",
        "state_shape": [6],
        "state_sha256_canonical_json": state_sha,
        "source": {
            "dataset_tree_sha256": EXPECTED_DATASET_TREE_SHA256,
            "freeze_id": "task1_picklift_formal24_s03_freeze_v1",
            "subset_id": "task1_picklift_formal24_s03_accepted_v1",
            "episode_index": selected["episode_index"],
            "frame_index": 0,
            "spawn_region": provenance["spawn_region"],
            "selection_rule": (
                "observed frame0 closest in raw-unit L2 to coordinatewise "
                "median across all 24 accepted real episodes"
            ),
        },
        "remote_interface_requirement": {
            "proposed_type": (
                "PolicyReadyPose(profile_id, joint_order, state_dataset_units, "
                "source_dataset_tree_sha256, source_episode_index, "
                "source_frame_index, state_sha256)"
            ),
            "proposed_signature": (
                "adapter.reset(trial, *, ready_pose: PolicyReadyPose | None = None)"
            ),
            "sequence": [
                "execute the existing trial reset, object spawn, and Nexus settle",
                "after settle and before policy tick0, convert the six dataset-unit joints to simulator qpos",
                "overwrite robot qpos only, zero corresponding robot qvel, and call mj_forward without advancing simulation time",
                "preserve the already-frozen object pose/seed and do not re-run object placement",
                "reset env_step, policy_tick, success counters, timeout state, and policy action queue to zero/empty",
                "render the policy tick0 RGB and return the post-override state",
            ],
            "required_validation": [
                "exact joint order and finite float32[6]",
                "state lies in Remote/Nexus action-coordinate contract",
                "post-override observed state matches requested ready pose within an explicit small numeric tolerance",
                "episode manifest records requested pose, observed tick0 pose, per-joint delta, profile id, and source hashes",
            ],
            "nexus_robot_init_noise": (
                "do not retain it in the policy tick0 robot state; existing "
                "Nexus reset noise may run but is deterministically overwritten"
            ),
            "unchanged_contracts": [
                "object spawn and seed",
                "camera and canonical RGB",
                "official success predicate",
                "20-to-50 Hz hold schedule",
                "30 second / 600 policy tick / 1500 environment step limit",
                "frozen ACT checkpoint and processors",
                "output calibration clamp and max_relative_target=5.0",
            ],
            "excluded_from_rollout": (
                "all reset, settle, and direct ready-pose placement work"
            ),
        },
    }


def run(dataset_root: Path, nexus_reset_jsonl: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite evidence: {output_dir}")
    data_file = dataset_root / "data/chunk-000/file-000.parquet"
    if sha256_file(data_file) != EXPECTED_DATA_FILE_SHA256:
        raise RuntimeError("Frozen Real-24 data Parquet hash mismatch.")
    tree_sha, file_count = dataset_tree_sha256(dataset_root)
    if tree_sha != EXPECTED_DATASET_TREE_SHA256:
        raise RuntimeError("Frozen Real-24 dataset tree hash mismatch.")
    if sha256_file(nexus_reset_jsonl) != EXPECTED_NEXUS_RESET_EVIDENCE_SHA256:
        raise RuntimeError("Frozen Nexus reset-state evidence hash mismatch.")

    real_records, episode_indices, real_states = load_real_episode_starts(
        dataset_root
    )
    nexus_records, nexus_states = load_nexus_states(nexus_reset_jsonl)
    representatives = choose_representatives(episode_indices, real_states)
    selected = representatives["selected_ready_pose"]
    selected_real_record = next(
        record
        for record in real_records
        if record["episode_index"] == selected["episode_index"]
    )
    selected_provenance = json.loads(
        (
            dataset_root
            / "provenance/episodes"
            / f"episode_{selected['episode_index']:06d}.json"
        ).read_text(encoding="utf-8")
    )
    comparisons, ood_joints = compare_distributions(
        real_states,
        nexus_states,
        np.asarray(selected["state"], dtype=np.float64),
    )
    early_delta = np.asarray(
        [
            record["first_five_max_abs_delta_from_frame0"]
            for record in real_records
        ],
        dtype=np.float64,
    )
    nexus_to_selected_l2 = np.linalg.norm(
        nexus_states - np.asarray(selected["state"], dtype=np.float64),
        axis=1,
    )
    nexus_to_real_pairwise_l2 = np.linalg.norm(
        nexus_states[:, np.newaxis, :] - real_states[np.newaxis, :, :],
        axis=2,
    )
    nexus_nearest_real_l2 = nexus_to_real_pairwise_l2.min(axis=1)

    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "real_episode_initial_states.jsonl", real_records)
    nexus_comparison_records = []
    real_stats = joint_statistics(real_states)
    for record in nexus_records:
        state = np.asarray(record["state"], dtype=np.float64)
        per_joint = []
        for index, joint in enumerate(JOINT_ORDER):
            stats = real_stats[index]
            value = float(state[index])
            per_joint.append(
                {
                    "joint": joint,
                    "value": value,
                    "real_min": stats["min"],
                    "real_median": stats["median"],
                    "real_max": stats["max"],
                    "minus_real_median": value - stats["median"],
                    "outside_real_observed_range": bool(
                        value < stats["min"] or value > stats["max"]
                    ),
                }
            )
        nexus_comparison_records.append(
            {
                **record,
                "relative_to_real_frame0": per_joint,
            }
        )
    write_jsonl(
        output_dir / "nexus_gate12_reset_comparisons.jsonl",
        nexus_comparison_records,
    )

    ready_pose_contract = build_ready_pose_contract(
        selected,
        selected_provenance,
    )
    summary = {
        "schema_version": 1,
        "audit_id": "task1_real24_vs_nexus_gate12_initial_pose_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_diagnostic_not_a_real_robot_or_paper_result",
        "scope": {
            "hardware_accessed": False,
            "simulation_rollout_executed": False,
            "frozen120_executed": False,
            "remote_modified": False,
            "dataset_modified": False,
            "checkpoint_loaded": False,
        },
        "inputs": {
            "dataset_root": str(dataset_root),
            "dataset_repo_id": (
                "local/task1_picklift_formal24_s03_20260728"
            ),
            "dataset_freeze_id": "task1_picklift_formal24_s03_freeze_v1",
            "dataset_subset_id": "task1_picklift_formal24_s03_accepted_v1",
            "dataset_tree_sha256": tree_sha,
            "dataset_file_count": file_count,
            "data_parquet_sha256": EXPECTED_DATA_FILE_SHA256,
            "episodes": len(real_records),
            "frames": 3790,
            "fps": 20,
            "nexus_reset_evidence": str(nexus_reset_jsonl),
            "nexus_reset_evidence_sha256": (
                EXPECTED_NEXUS_RESET_EVIDENCE_SHA256
            ),
            "nexus_resets": len(nexus_records),
            "nexus_seeds": [record["seed"] for record in nexus_records],
        },
        "method": {
            "primary_real_state": "frame_index=0 observation.state",
            "early_stability_window": "frame_index 0..4 inclusive",
            "episode_dispersion": (
                "population std, observed range, IQR, and MAD over 24 frame0 states"
            ),
            "strict_ood": (
                "all 12 Nexus reset values are outside the observed Real-24 "
                "frame0 min/max for that joint"
            ),
            "distribution_shift": (
                "Nexus reset median is outside the Real-24 frame0 Tukey "
                "inner fence [Q1-1.5*IQR,Q3+1.5*IQR]"
            ),
        },
        "real_frame0_joint_statistics": real_stats,
        "real_first_five_frame_stability": [
            {
                "joint": joint,
                "episodes_with_nonzero_delta_from_frame0": int(
                    (early_delta[:, index] > 1.0e-9).sum()
                ),
                "median_of_episode_max_abs_delta_from_frame0": float(
                    np.median(early_delta[:, index])
                ),
                "max_of_episode_max_abs_delta_from_frame0": float(
                    early_delta[:, index].max()
                ),
            }
            for index, joint in enumerate(JOINT_ORDER)
        ],
        "representative_ready_pose": {
            **representatives,
            "selected_spawn_region": selected_provenance["spawn_region"],
            "selected_spawn_id": selected_provenance["spawn_id"],
            "selected_first_five_frame_states": selected_real_record[
                "first_five_frame_states"
            ],
            "selected_first_five_max_abs_delta_from_frame0": (
                selected_real_record[
                    "first_five_max_abs_delta_from_frame0"
                ]
            ),
        },
        "nexus_vs_real_by_joint": comparisons,
        "nexus_reset_pose_distance_raw_l2": {
            "to_selected_real_ready_pose": {
                "min": float(nexus_to_selected_l2.min()),
                "median": float(np.median(nexus_to_selected_l2)),
                "max": float(nexus_to_selected_l2.max()),
            },
            "to_nearest_of_24_real_frame0_states": {
                "min": float(nexus_nearest_real_l2.min()),
                "median": float(np.median(nexus_nearest_real_l2)),
                "max": float(nexus_nearest_real_l2.max()),
            },
            "units_note": (
                "raw L2 across the six shared degree/0-100 dataset coordinates"
            ),
        },
        "ood_or_shifted_joints": ood_joints,
        "interpretation": {
            "initial_pose_mismatch_present": bool(ood_joints),
            "zero_of_twelve_may_be_affected": bool(ood_joints),
            "assessment": (
                "initial pose mismatch is a plausible material contributor "
                "and should be controlled before interpreting task success"
                if ood_joints
                else "no material initial pose mismatch detected"
            ),
            "causal_limit": (
                "This offline audit cannot attribute the 0/12 result to the "
                "initial pose alone; a controlled gate12 rerun after Remote "
                "implements the frozen ready-pose contract is required."
            ),
            "paper_effect_claim": False,
        },
        "recommendation": {
            "decision": (
                "implement_remote_ready_pose_contract_then_rerun_gate12_"
                "before_considering_frozen120"
                if ood_joints
                else "no_ready_pose_change_required"
            ),
            "ready_pose_contract": ready_pose_contract,
        },
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_json(
        output_dir / "ready_pose_contract_proposal.json",
        ready_pose_contract,
    )
    names = (
        "real_episode_initial_states.jsonl",
        "nexus_gate12_reset_comparisons.jsonl",
        "audit_summary.json",
        "ready_pose_contract_proposal.json",
    )
    hashes = {name: sha256_file(output_dir / name) for name in names}
    (output_dir / "hashes.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "hashes": hashes}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Real-24 versus Nexus gate12 initial joint states."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--nexus-reset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        dataset_root=args.dataset_root.resolve(),
        nexus_reset_jsonl=args.nexus_reset_jsonl.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
