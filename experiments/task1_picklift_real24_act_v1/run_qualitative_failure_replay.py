"""Replay one predeclared frozen120 failure with canonical ACT-input video.

This runner is hardware-free and evaluation-neutral. It selects exactly the
frozen r2c2 / seed 1008 trial, sends the actual canonical 640x480 RGB input
seen by ACT to ffmpeg at 20 FPS, and does not modify the frozen120 evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from run_remote_sim_frozen120 import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_PLAN_SHA256,
    EXPECTED_READY_POSE_PROFILE_ID,
    EXPECTED_READY_POSE_STATE_SHA256,
    EXPECTED_READY_POSE_TOLERANCE,
    EXPECTED_REMOTE_DEPLOYED_COMMIT,
    EXPECTED_RESET_PROFILE_ID,
    append_jsonl,
    git_head,
    load_contract,
    sha256_file,
    validate_deployment,
    validate_object_spawn,
    validate_policy_inputs,
    validate_ready_pose_evidence,
    write_json,
)


EXPERIMENT_DIR = Path(
    "/home/ubuntu24/Teleop/lerobot/experiments/"
    "task1_picklift_real24_act_v1"
)
REMOTE_ADAPTER_DIR = Path(
    "/home/ubuntu24/SO101QuestRemote/robot-host"
)
TARGET_CELL = "r2c2"
TARGET_SEED = 1008
TARGET_PHASE_EPISODE_INDEX = 101
TARGET_REPEAT_INDEX = 8
EXPECTED_POLICY_TICKS = 600
EXPECTED_ENV_STEPS = 1500
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 20
VIDEO_DURATION_SECONDS = 30.0
VIDEO_NAME = "r2c2_seed1008_canonical_act_input_20fps.mp4"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def select_target_trial(trials: tuple[Any, ...]) -> Any:
    matches = [
        trial
        for trial in trials
        if trial.cell_name == TARGET_CELL and trial.seed == TARGET_SEED
    ]
    if len(matches) != 1:
        raise RuntimeError("Frozen target trial is not unique.")
    trial = matches[0]
    if (
        trial.phase_episode_index != TARGET_PHASE_EPISODE_INDEX
        or trial.repeat_index != TARGET_REPEAT_INDEX
        or trial.phase_id != "frozen120"
    ):
        raise RuntimeError("Frozen target trial identity drifted.")
    return trial


def ffmpeg_command(video_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        str(video_path),
    ]


def probe_video(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,"
            "avg_frame_rate,nb_read_frames:format=duration"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def validate_video_probe(probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise RuntimeError("Replay MP4 must contain exactly one video stream.")
    stream = streams[0]
    video_format = probe.get("format")
    if not isinstance(video_format, dict):
        raise RuntimeError("Replay MP4 format evidence is missing.")
    checks = {
        "codec_name": stream.get("codec_name"),
        "width": int(stream.get("width")),
        "height": int(stream.get("height")),
        "pix_fmt": stream.get("pix_fmt"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "frames": int(stream.get("nb_read_frames")),
        "duration_seconds": float(video_format.get("duration")),
    }
    if checks["codec_name"] != "h264":
        raise RuntimeError("Replay MP4 codec is not H.264.")
    if (checks["width"], checks["height"]) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        raise RuntimeError("Replay MP4 dimensions drifted.")
    if checks["pix_fmt"] != "yuv420p":
        raise RuntimeError("Replay MP4 pixel format is not yuv420p.")
    if (
        checks["r_frame_rate"] != "20/1"
        or checks["avg_frame_rate"] != "20/1"
    ):
        raise RuntimeError("Replay MP4 frame rate is not exactly 20 FPS.")
    if checks["frames"] != EXPECTED_POLICY_TICKS:
        raise RuntimeError("Replay MP4 does not contain exactly 600 frames.")
    if not np.isclose(
        checks["duration_seconds"],
        VIDEO_DURATION_SECONDS,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("Replay MP4 duration is not exactly 30 seconds.")
    return checks


def validate_contract_only() -> dict[str, Any]:
    validation = validate_deployment()
    _, contract, trials = load_contract()
    trial = select_target_trial(trials)
    if contract["clock"]["maximum_policy_ticks"] != EXPECTED_POLICY_TICKS:
        raise RuntimeError("Policy tick contract drifted.")
    if contract["clock"]["maximum_env_steps"] != EXPECTED_ENV_STEPS:
        raise RuntimeError("Environment step contract drifted.")
    return {
        "status": "pass",
        "trial": trial.manifest(),
        "model_sha256": validation["model_sha256"],
        "plan_sha256": validation["plan_sha256"],
        "remote_deployed_commit": validation["remote_deployed_commit"],
        "reset_profile": validation["reset_profile"],
        "ready_pose": validation["ready_pose"],
        "ready_pose_tolerance": validation["ready_pose_tolerance"],
        "clock": validation["clock"],
        "video": {
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "fps": VIDEO_FPS,
            "frames": EXPECTED_POLICY_TICKS,
            "duration_seconds": VIDEO_DURATION_SECONDS,
            "overlay": False,
        },
    }


def run_replay(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(
            f"Replay evidence already exists; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    validation = validate_deployment()
    _, contract, trials = load_contract()
    trial = select_target_trial(trials)

    if str(REMOTE_ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(REMOTE_ADAPTER_DIR))
    from nexus_picklift_policy_adapter import (
        create_nexus_picklift_policy_adapter,
    )
    from picklift_ready_pose import REAL24_READY_POSE
    from sim_policy_inference import Task1ActSimInference

    source_commit = git_head(EXPERIMENT_DIR.parents[1])
    video_path = output_dir / VIDEO_NAME
    ticks_path = output_dir / "trajectory_ticks.jsonl"
    manifest_path = output_dir / "replay_manifest.json"
    result_path = output_dir / "replay_result.json"
    run_manifest = {
        "schema_version": 1,
        "replay_id": output_dir.name,
        "status": "running",
        "research_status": (
            "qualitative_hardware_free_replay_not_an_evaluation_result"
        ),
        "started_at_utc": started_at.isoformat(),
        "source_commit": source_commit,
        "target_trial": trial.manifest(),
        "checkpoint_path": contract["upstream_act_binding"]["checkpoint_path"],
        "model_sha256": EXPECTED_MODEL_SHA256,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "remote_deployed_commit": EXPECTED_REMOTE_DEPLOYED_COMMIT,
        "reset_profile_id": EXPECTED_RESET_PROFILE_ID,
        "ready_pose_profile_id": EXPECTED_READY_POSE_PROFILE_ID,
        "ready_pose_state_sha256": EXPECTED_READY_POSE_STATE_SHA256,
        "ready_pose_tolerance": EXPECTED_READY_POSE_TOLERANCE,
        "clock": contract["clock"],
        "video_contract": {
            "content": "actual_canonical_rgb_input_to_act_each_policy_tick",
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "fps": VIDEO_FPS,
            "frames": EXPECTED_POLICY_TICKS,
            "duration_seconds": VIDEO_DURATION_SECONDS,
            "overlay": False,
            "audio": False,
            "encoder_command": ffmpeg_command(video_path),
        },
        "sim_state_projection_tolerance_dataset_units": 0.05,
        "hardware_accessed": False,
        "serial_accessed": False,
        "hardware_camera_accessed": False,
        "gateway_or_quest_started": False,
        "robot_or_torque_enabled": False,
        "twelve_volt_enabled": False,
        "lerobot_dataset_written": False,
        "training_or_finetuning_performed": False,
        "checkpoint_changed": False,
    }
    write_json(manifest_path, run_manifest)

    policy = Task1ActSimInference()
    if policy.model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Loaded policy model SHA drifted.")
    adapter = create_nexus_picklift_policy_adapter()
    ffmpeg = subprocess.Popen(
        ffmpeg_command(video_path),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ffmpeg.stdin is None or ffmpeg.stderr is None:
        raise RuntimeError("Failed to open ffmpeg pipes.")

    raw_action_count = 0
    sent_action_count = 0
    valid_observation_count = 0
    calibration_clipped_action_count = 0
    calibration_clipped_joint_value_count = 0
    relative_clipped_action_count = 0
    relative_clipped_joint_value_count = 0
    environment_clipped_action_count = 0
    sim_state_projected_tick_count = 0
    maximum_absolute_sim_state_projection_delta = 0.0
    frame_stream_sha256 = hashlib.sha256()
    ready_pose_validation: dict[str, Any] | None = None
    spawn_validation: dict[str, Any] | None = None
    final_manifest: dict[str, Any] | None = None
    replay_error: str | None = None
    try:
        policy.reset_episode()
        observation = adapter.reset(trial, ready_pose=REAL24_READY_POSE)
        reset_manifest = adapter.episode_manifest()
        reset_inputs = observation.policy_inputs()
        validate_policy_inputs(reset_inputs)
        ready_pose_validation = validate_ready_pose_evidence(
            reset_manifest["ready_pose_reset"],
            reset_inputs["observation.state"],
            validation["ready_pose"],
        )
        spawn_validation = validate_object_spawn(
            trial.manifest(),
            reset_manifest["task_manifest"],
        )
        with ticks_path.open("x", encoding="utf-8", buffering=1) as ticks_handle:
            while adapter.phase == "active":
                inputs = observation.policy_inputs()
                validate_policy_inputs(inputs)
                state = inputs["observation.state"]
                frame = inputs["observation.images.front"]
                frame_bytes = frame.tobytes(order="C")
                frame_stream_sha256.update(frame_bytes)
                ffmpeg.stdin.write(frame_bytes)
                valid_observation_count += 1

                policy_step = policy.infer(state, frame)
                raw_action_count += 1
                calibration_clipped_action_count += int(
                    bool(policy_step.calibration_clip_mask.any())
                )
                calibration_clipped_joint_value_count += int(
                    policy_step.calibration_clip_mask.sum()
                )
                relative_clipped_action_count += int(
                    bool(policy_step.relative_clip_mask.any())
                )
                relative_clipped_joint_value_count += int(
                    policy_step.relative_clip_mask.sum()
                )
                if bool(policy_step.sim_state_projection_mask.any()):
                    sim_state_projected_tick_count += 1
                    maximum_absolute_sim_state_projection_delta = max(
                        maximum_absolute_sim_state_projection_delta,
                        float(
                            np.max(
                                np.abs(policy_step.sim_state_projection_delta)
                            )
                        ),
                    )
                tick_result = adapter.apply_action(policy_step.sent_action)
                sent_action_count += 1
                environment_clipped_action_count += int(
                    any(tick_result.environment_clipped_mask)
                )
                append_jsonl(
                    ticks_handle,
                    {
                        "phase_id": trial.phase_id,
                        "phase_episode_index": trial.phase_episode_index,
                        "cell": trial.cell_name,
                        "seed": trial.seed,
                        "canonical_rgb_sha256": hashlib.sha256(
                            frame_bytes
                        ).hexdigest(),
                        "observation": observation.evidence(),
                        "policy": policy_step.to_jsonable(),
                        "remote": tick_result.evidence(),
                    },
                )
                if adapter.phase == "active":
                    observation = adapter.observe()
        final_manifest = adapter.episode_manifest()
    except Exception as exc:
        replay_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass
        ffmpeg_returncode = ffmpeg.wait()
        ffmpeg_stderr = ffmpeg.stderr.read().decode(
            "utf-8", errors="replace"
        )
        adapter.close()

    if replay_error is not None:
        raise RuntimeError(f"Replay failed: {replay_error}")
    if ffmpeg_returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with {ffmpeg_returncode}: {ffmpeg_stderr}"
        )
    if final_manifest is None:
        raise RuntimeError("Final Remote episode manifest is missing.")

    runtime = final_manifest["runtime_summary"]
    video_probe = validate_video_probe(probe_video(video_path))
    if (
        raw_action_count != EXPECTED_POLICY_TICKS
        or sent_action_count != EXPECTED_POLICY_TICKS
        or valid_observation_count != EXPECTED_POLICY_TICKS
    ):
        raise RuntimeError("Replay did not execute exactly 600 policy ticks.")
    if int(runtime["env_steps"]) != EXPECTED_ENV_STEPS:
        raise RuntimeError("Replay did not execute exactly 1500 env steps.")

    result = {
        "schema_version": 1,
        "replay_id": output_dir.name,
        "status": (
            "predeclared_failure_reproduced"
            if runtime["success"] is False
            else "unexpected_success_determinism_drift"
        ),
        "research_status": (
            "qualitative_hardware_free_replay_not_an_evaluation_result"
        ),
        "target_trial": trial.manifest(),
        "runtime_summary": runtime,
        "ready_pose_validation": ready_pose_validation,
        "object_spawn_validation": spawn_validation,
        "action_accounting": {
            "valid_observation_count": valid_observation_count,
            "raw_action_count": raw_action_count,
            "calibration_clipped_action_count": (
                calibration_clipped_action_count
            ),
            "calibration_clipped_joint_value_count": (
                calibration_clipped_joint_value_count
            ),
            "relative_clipped_action_count": (
                relative_clipped_action_count
            ),
            "relative_clipped_joint_value_count": (
                relative_clipped_joint_value_count
            ),
            "sent_action_count": sent_action_count,
            "environment_clipped_action_count": (
                environment_clipped_action_count
            ),
            "sim_state_projected_tick_count": (
                sim_state_projected_tick_count
            ),
            "maximum_absolute_sim_state_projection_delta": (
                maximum_absolute_sim_state_projection_delta
            ),
        },
        "frame_stream": {
            "description": (
                "concatenated raw uint8 HWC RGB bytes actually passed to ACT"
            ),
            "frames": valid_observation_count,
            "sha256": frame_stream_sha256.hexdigest(),
        },
        "video": {
            "path": str(video_path),
            "sha256": sha256_file(video_path),
            "probe": video_probe,
            "overlay": False,
        },
        "trajectory_ticks": {
            "path": str(ticks_path),
            "sha256": sha256_file(ticks_path),
            "lines": raw_action_count,
        },
        "runtime_seconds": time.monotonic() - started_monotonic,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_path, result)
    run_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": result["completed_at_utc"],
            "runtime_seconds": result["runtime_seconds"],
            "official_success": runtime["success"],
            "video_sha256": result["video"]["sha256"],
            "trajectory_ticks_sha256": result["trajectory_ticks"]["sha256"],
        }
    )
    write_json(manifest_path, run_manifest)
    primary_hashes = {
        VIDEO_NAME: sha256_file(video_path),
        "trajectory_ticks.jsonl": sha256_file(ticks_path),
        "replay_manifest.json": sha256_file(manifest_path),
        "replay_result.json": sha256_file(result_path),
    }
    write_json(output_dir / "primary_hashes.json", primary_hashes)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "result": result,
                "primary_hashes": primary_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if runtime["success"] is not False:
        raise SystemExit(2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-contract-only", action="store_true")
    args = parser.parse_args()
    if args.validate_contract_only == (args.output_dir is not None):
        parser.error(
            "choose exactly one of --validate-contract-only or --output-dir"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.validate_contract_only:
        print(json.dumps(validate_contract_only(), indent=2, sort_keys=True))
        return
    run_replay(args.output_dir.resolve())


if __name__ == "__main__":
    main()
