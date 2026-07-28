"""Independently verify immutable Task1 aligned frozen120 evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_MODEL_SHA256 = (
    "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
)
EXPECTED_PLAN_SHA256 = (
    "c26bf3caa788708cf8ccb332e2f84771d271b05795634a9b1022ed97d06b8c86"
)
EXPECTED_REMOTE_COMMIT = "1dfac5c108e830a55b114b1a75bd00bdb5d877b7"
EXPECTED_RESET_PROFILE = "picklift_real24_aligned_reset_v2"
EXPECTED_READY_PROFILE = "task1_real24_ready_pose_reset_v1"
EXPECTED_READY_STATE_SHA256 = (
    "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
)
EXPECTED_READY_TOLERANCE = 1.0e-5
EXPECTED_EPISODES = 120
EXPECTED_TICKS = 72_000
EXPECTED_ENV_STEPS = 180_000
LEGAL_TERMINATION_REASONS = {
    "environment_terminated",
    "environment_truncated",
    "max_steps_reached",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_vector(value: Any, length: int = 6) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def bool_vector(value: Any, length: int = 6) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, bool) for item in value)
    )


def expected_episode_identity(index: int) -> tuple[str, int, int]:
    repeat_index = index // 12
    cell_index = index % 12
    row = cell_index // 4 + 1
    column = cell_index % 4 + 1
    return f"r{row}c{column}", 1000 + repeat_index, repeat_index


def verify_primary_hashes(run_dir: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for line in (run_dir / "hashes.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split(None, 1)
        recorded[name.strip()] = digest
    expected_names = {
        "episodes.jsonl",
        "ticks.jsonl",
        "run_manifest.json",
        "summary.json",
    }
    require(set(recorded) == expected_names, "primary hash file set drifted")
    for name, digest in recorded.items():
        require(
            sha256_file(run_dir / name) == digest,
            f"primary evidence hash mismatch: {name}",
        )
    return recorded


def verify_episodes(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    with (run_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
        episodes = [json.loads(line) for line in handle]
    require(len(episodes) == EXPECTED_EPISODES, "episode count is not 120")
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        cell, seed, repeat_index = expected_episode_identity(index)
        require(episode["phase_episode_index"] == index, "phase index drifted")
        require(episode["repeat_index"] == repeat_index, "repeat index drifted")
        require(episode["cell"] == cell, "cell order drifted")
        require(episode["seed"] == seed, "seed order drifted")
        require(episode["phase_id"] == "frozen120", "phase id drifted")
        require(episode["model_sha256"] == EXPECTED_MODEL_SHA256, "model drifted")
        require(
            episode["remote_deployed_commit"] == EXPECTED_REMOTE_COMMIT,
            "Remote commit drifted",
        )
        require(episode["interface_valid"] is True, "interface-invalid episode")
        require(episode["interface_error"] is None, "episode interface error")
        require(
            episode["ready_pose_tick0_valid"] is True,
            "ready tick0 invalid",
        )
        require(
            episode["object_spawn_plan_valid"] is True,
            "object spawn invalid",
        )
        require(episode["env_steps"] == 1500, "episode env steps not 1500")
        require(episode["raw_action_count"] == 600, "raw count not 600")
        require(episode["sent_action_count"] == 600, "sent count not 600")
        require(
            episode["valid_observation_count"] == 600,
            "observation count not 600",
        )
        require(
            episode["termination_reason"] in LEGAL_TERMINATION_REASONS,
            "illegal termination reason",
        )
        require(
            bool(episode["terminated"])
            or bool(episode["truncated"])
            or bool(episode["timeout"]),
            "episode lacks legal terminal flag",
        )
        require(
            episode["reset_profile_id"] == EXPECTED_RESET_PROFILE,
            "reset profile drifted",
        )
        require(
            episode["ready_pose_profile_id"] == EXPECTED_READY_PROFILE,
            "ready profile drifted",
        )
        ready = episode["ready_pose_validation"]
        require(ready is not None, "ready validation missing")
        require(
            float(ready["absolute_tolerance"]) == EXPECTED_READY_TOLERANCE,
            "ready tolerance drifted",
        )
        require(
            float(ready["maximum_absolute_tick0_delta"])
            <= EXPECTED_READY_TOLERANCE,
            "ready delta exceeds tolerance",
        )
        for field in (
            "requested_tick0_state",
            "observed_tick0_state",
            "per_joint_tick0_delta",
        ):
            require(finite_vector(ready[field]), f"invalid ready vector: {field}")
        require(
            episode["trial_manifest"]["spawn"]
            == episode["object_spawn_validation"]["planned_spawn"],
            "planned spawn evidence drifted",
        )
        observed_spawn = episode["object_spawn"]
        planned_spawn = episode["trial_manifest"]["spawn"]
        require(
            all(
                observed_spawn.get(field) == planned_value
                for field, planned_value in planned_spawn.items()
            ),
            "observed spawn differs from planned fields",
        )
        require(
            isinstance(observed_spawn.get("actual_initial_pose"), dict)
            and observed_spawn.get("placement_method")
            == "post_reset_freejoint_then_nexus_settle",
            "observed spawn placement evidence is missing",
        )
        identity = (cell, seed)
        require(identity not in by_identity, "duplicate cell/seed episode")
        by_identity[identity] = episode
    require(len(by_identity) == 120, "cell/seed identity count drifted")
    return episodes, by_identity


def verify_ticks(
    run_dir: Path,
    episodes_by_identity: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    tick_counts: Counter[tuple[str, int]] = Counter()
    held_steps: Counter[tuple[str, int]] = Counter()
    calibration_action_counts: Counter[tuple[str, int]] = Counter()
    relative_action_counts: Counter[tuple[str, int]] = Counter()
    calibration_joint_counts: Counter[tuple[str, int]] = Counter()
    relative_joint_counts: Counter[tuple[str, int]] = Counter()
    environment_action_counts: Counter[tuple[str, int]] = Counter()
    sim_projection_counts: Counter[tuple[str, int]] = Counter()
    max_projection_delta: defaultdict[tuple[str, int], float] = defaultdict(float)
    expected_identity_index = 0
    expected_tick_in_episode = 0
    total_ticks = 0
    with (run_dir / "ticks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            tick = json.loads(line)
            cell, seed, _ = expected_episode_identity(expected_identity_index)
            identity = (cell, seed)
            require(
                (tick["cell"], tick["seed"]) == identity,
                "tick cell/seed order drifted",
            )
            require(tick["phase_id"] == "frozen120", "tick phase drifted")
            observation = tick["observation"]
            policy = tick["policy"]
            remote = tick["remote"]
            require(
                observation["policy_tick"] == expected_tick_in_episode,
                "observation policy tick drifted",
            )
            require(
                remote["policy_tick"] == expected_tick_in_episode,
                "remote policy tick drifted",
            )
            require(
                finite_vector(observation["observation_state"]),
                "non-finite observation state",
            )
            require(
                isinstance(observation["front_rgb_sha256"], str)
                and len(observation["front_rgb_sha256"]) == 64,
                "front RGB hash invalid",
            )
            for field in (
                "raw_action",
                "calibration_clipped_action",
                "sent_action",
                "safety_reference_state",
                "sim_state_projection_delta",
            ):
                require(finite_vector(policy[field]), f"invalid policy {field}")
            for field in (
                "calibration_clip_mask",
                "relative_clip_mask",
                "sim_state_projection_mask",
            ):
                require(bool_vector(policy[field]), f"invalid policy {field}")
            require(policy["raw_action_finite"] is True, "raw action not finite")
            require(policy["sent_action_finite"] is True, "sent action not finite")
            require(
                finite_vector(remote["sent_action"]),
                "invalid remote sent action",
            )
            require(
                finite_vector(remote["environment_action"]),
                "invalid environment action",
            )
            require(
                bool_vector(remote["environment_clipped_mask"]),
                "invalid environment clip mask",
            )
            require(
                policy["sent_action"] == remote["sent_action"],
                "policy/Remote sent action mismatch",
            )
            held = int(remote["held_env_steps"])
            require(held in (2, 3), "20-to-50 Hz hold is not 2 or 3")
            tick_counts[identity] += 1
            held_steps[identity] += held
            calibration_action_counts[identity] += int(
                any(policy["calibration_clip_mask"])
            )
            relative_action_counts[identity] += int(
                any(policy["relative_clip_mask"])
            )
            calibration_joint_counts[identity] += sum(
                policy["calibration_clip_mask"]
            )
            relative_joint_counts[identity] += sum(policy["relative_clip_mask"])
            environment_action_counts[identity] += int(
                any(remote["environment_clipped_mask"])
            )
            sim_projection_counts[identity] += int(
                policy["sim_state_projected"]
            )
            max_projection_delta[identity] = max(
                max_projection_delta[identity],
                max(abs(float(value)) for value in policy[
                    "sim_state_projection_delta"
                ]),
            )
            total_ticks += 1
            expected_tick_in_episode += 1
            if expected_tick_in_episode == 600:
                expected_identity_index += 1
                expected_tick_in_episode = 0
    require(total_ticks == EXPECTED_TICKS, "tick count is not 72000")
    require(expected_identity_index == 120, "tick episodes incomplete")
    require(expected_tick_in_episode == 0, "partial final tick episode")
    for identity, episode in episodes_by_identity.items():
        require(tick_counts[identity] == 600, "per-episode tick count drifted")
        require(held_steps[identity] == 1500, "per-episode hold sum drifted")
        comparisons = (
            ("calibration_clipped_action_count", calibration_action_counts),
            ("relative_clipped_action_count", relative_action_counts),
            ("calibration_clipped_joint_value_count", calibration_joint_counts),
            ("relative_clipped_joint_value_count", relative_joint_counts),
            ("environment_clipped_action_count", environment_action_counts),
            ("sim_state_projected_tick_count", sim_projection_counts),
        )
        for field, counter in comparisons:
            require(
                int(episode[field]) == counter[identity],
                f"episode/tick aggregate mismatch: {field}",
            )
        require(
            math.isclose(
                float(episode["maximum_absolute_sim_state_projection_delta"]),
                max_projection_delta[identity],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "episode/tick maximum sim projection mismatch",
        )
    return {
        "ticks": total_ticks,
        "environment_steps_from_hold_schedule": sum(held_steps.values()),
        "finite_observation_states": total_ticks,
        "finite_raw_actions": total_ticks,
        "finite_sent_actions": total_ticks,
        "finite_environment_actions": total_ticks,
    }


def verify_summary(
    run_dir: Path,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    require(summary["frozen120_interface_pass"] is True, "summary gate failed")
    require(summary["model_sha256"] == EXPECTED_MODEL_SHA256, "summary model drift")
    require(summary["plan_sha256"] == EXPECTED_PLAN_SHA256, "summary plan drift")
    require(
        summary["remote_deployed_commit"] == EXPECTED_REMOTE_COMMIT,
        "summary Remote commit drift",
    )
    successes = sum(bool(episode["success"]) for episode in episodes)
    require(summary["overall"]["episodes"] == 120, "summary episode drift")
    require(
        summary["overall"]["interface_valid_episodes"] == 120,
        "summary interface drift",
    )
    require(
        summary["overall"]["ready_pose_tick0_valid_episodes"] == 120,
        "summary ready drift",
    )
    require(
        summary["overall"]["object_spawn_plan_valid_episodes"] == 120,
        "summary spawn drift",
    )
    require(summary["overall"]["env_steps"] == EXPECTED_ENV_STEPS, "summary steps")
    require(summary["overall"]["raw_action_count"] == EXPECTED_TICKS, "summary ticks")
    require(summary["overall"]["successes"] == successes, "summary success drift")
    for row in range(1, 4):
        for column in range(1, 5):
            cell = f"r{row}c{column}"
            group = [
                episode for episode in episodes if episode["cell"] == cell
            ]
            cell_summary = summary["by_cell"][cell]
            require(cell_summary["episodes"] == 10, "by-cell episode drift")
            require(
                cell_summary["successes"]
                == sum(bool(episode["success"]) for episode in group),
                "by-cell success drift",
            )
    return summary


def verify(run_dir: Path) -> dict[str, Any]:
    primary_hashes = verify_primary_hashes(run_dir)
    manifest = load_json(run_dir / "run_manifest.json")
    require(manifest["status"] == "frozen120_complete", "run not complete")
    require(manifest["phase_id"] == "frozen120", "manifest phase drift")
    require(manifest["episode_count"] == 120, "manifest episode drift")
    require(manifest["plan_sha256"] == EXPECTED_PLAN_SHA256, "manifest plan drift")
    require(manifest["model_sha256"] == EXPECTED_MODEL_SHA256, "manifest model drift")
    require(
        manifest["remote_deployed_commit"] == EXPECTED_REMOTE_COMMIT,
        "manifest Remote commit drift",
    )
    require(manifest["hardware_accessed"] is False, "hardware flag is not false")
    require(
        manifest["gateway_or_quest_started"] is False,
        "gateway/Quest flag is not false",
    )
    require(
        manifest["lerobot_dataset_written"] is False,
        "Dataset write flag is not false",
    )
    require(
        manifest["ready_pose"]["state_sha256"]
        == EXPECTED_READY_STATE_SHA256,
        "manifest ready state SHA drift",
    )
    episodes, episodes_by_identity = verify_episodes(run_dir)
    tick_validation = verify_ticks(run_dir, episodes_by_identity)
    summary = verify_summary(run_dir, episodes)
    return {
        "schema_version": 1,
        "status": "pass",
        "research_status": (
            "diagnostic_only_not_real_robot_evaluation_not_a_paper_result"
        ),
        "run_id": run_dir.name,
        "source_commit": manifest["source_commit"],
        "primary_hashes": primary_hashes,
        "episodes": len(episodes),
        "ticks": tick_validation["ticks"],
        "environment_steps": tick_validation[
            "environment_steps_from_hold_schedule"
        ],
        "interface_valid_episodes": summary["overall"][
            "interface_valid_episodes"
        ],
        "ready_pose_tick0_valid_episodes": summary["overall"][
            "ready_pose_tick0_valid_episodes"
        ],
        "object_spawn_plan_valid_episodes": summary["overall"][
            "object_spawn_plan_valid_episodes"
        ],
        "successes": summary["overall"]["successes"],
        "by_cell_successes": {
            cell: cell_summary["successes"]
            for cell, cell_summary in summary["by_cell"].items()
        },
        "finite_checks": tick_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve())
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"independent_validation_sha256={digest}")


if __name__ == "__main__":
    main()
