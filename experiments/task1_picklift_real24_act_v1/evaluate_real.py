from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import stat
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deployment_safety import (
    JOINT_ORDER,
    sha256_file,
    verify_frozen_calibration,
)

from lerobot.utils.robot_utils import precise_sleep

EXPECTED_MODEL_SHA256 = (
    "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
)
EXPECTED_PLAN_SHA256 = (
    "a067615e37ae8b64b57d663e85c1b49be61ef881daa58c3d9159506caf9be048"
)
EXPECTED_PROFILE_SHA256 = (
    "01e50c86adc1a03f2bb1675469502e6969e2cc6a4a51dc8db0b75e6049b5d4c5"
)
EXPECTED_EVALUATION_ID = (
    "task1_picklift_real24_act100k_ready_pose_official_send_eval12_v1"
)
EXPECTED_FOLLOWER_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00"
)
EXPECTED_FOLLOWER_SERIAL = "5C82110904"
EXPECTED_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-icSpring_icspring_camera_202404160005-video-index0"
)
CAMERA_PROFILE_ID = "icspring_front_crop_1280x960_to_640x480_v1"
CONTROL_FPS = 20
ACT_CHUNK_SIZE = 67
ACT_ACTION_STEPS = 67
READY_POSE_PROFILE_ID = "task1_real24_ready_pose_reset_v1"
READY_POSE_STATE_SHA256 = (
    "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
)
READY_POSE = np.asarray(
    [
        7.4285712242126465,
        -98.32967376708984,
        45.010990142822266,
        92.21977996826172,
        1.8461538553237915,
        19.765840530395508,
    ],
    dtype=np.float32,
)
READY_MOVE_PROFILE_ID = "task1_real24_ready_pose_direct_official_send_v1"
READY_MOVE_CONTROL_FPS = 20
READY_MOVE_TOLERANCE = 1.0
READY_MOVE_TIMEOUT_SECONDS = 60.0
POLICY_TIMING_PROFILE_ID = "lerobot_per_tick_pacing_no_catchup_v1"
ACTION_PATH_PROFILE_ID = "lerobot_so101_official_send_no_custom_clamp_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One explicitly armed Task1 PickLift real-robot evaluation trial "
            "using the frozen ready pose and official SO101 action path."
        )
    )
    parser.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--operator-confirmed-ready", action="store_true")
    parser.add_argument("--spawn-region", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument(
        "--camera-device",
        default=EXPECTED_CAMERA_DEVICE,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/ubuntu24/Teleop/artifacts/training/"
            "task1_picklift_real24_act_v1/full_100k/checkpoints/"
            "100000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/"
            "so_follower/so101_follower_main.json"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "experiments/task1_picklift_real24_act_v1/"
            "evaluation_plan_ready_pose_official_send_v1.json"
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "experiments/task1_picklift_real24_act_v1/"
            "real_evaluation_profile_ready_pose_official_send_v1.json"
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(
            "/home/ubuntu24/Teleop/artifacts/evaluation/"
            "task1_picklift_real24_act100k_ready_pose_official_send_eval12_v1"
        ),
    )
    parser.add_argument("--maximum-trial-seconds", type=float, default=30.0)
    return parser.parse_args()


def finite_joint_vector(values: Any, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (6,):
        raise RuntimeError(f"{label} must have shape (6,), got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise RuntimeError(f"{label} contains NaN or infinity.")
    return vector


def ready_pose_state_sha256(values: np.ndarray = READY_POSE) -> str:
    payload = json.dumps(
        [float(value) for value in finite_joint_vector(values, "ready pose")],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_contract(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if plan["evaluation_id"] != EXPECTED_EVALUATION_ID:
        raise RuntimeError("Unexpected evaluation identity.")
    if sha256_file(args.plan) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Evaluation plan hash differs from the frozen plan.")
    if sha256_file(args.profile) != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Evaluation profile hash differs from the frozen profile.")
    profile_ref = plan["evaluation_profile"]
    if profile_ref["profile_id"] != profile["profile_id"]:
        raise RuntimeError("Evaluation plan/profile identity mismatch.")
    if profile_ref["sha256"] != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Evaluation plan/profile hash reference mismatch.")
    if profile["ready_pose"]["state_sha256"] != READY_POSE_STATE_SHA256:
        raise RuntimeError("Ready-pose state SHA differs from the frozen contract.")
    if ready_pose_state_sha256() != READY_POSE_STATE_SHA256:
        raise RuntimeError("Embedded ready-pose vector hash mismatch.")
    setup = plan["setup"]
    if setup["max_relative_target"] is not None:
        raise RuntimeError("Formal deployment requires max_relative_target=None.")
    if setup["custom_absolute_action_clamp"] is not False:
        raise RuntimeError("Formal deployment must not apply a custom absolute clamp.")
    if setup["control_fps"] != CONTROL_FPS:
        raise RuntimeError("Control FPS differs from the frozen plan.")
    if setup["act_chunk_size"] != ACT_CHUNK_SIZE:
        raise RuntimeError("ACT chunk size differs from the frozen checkpoint.")
    if setup["act_n_action_steps"] != ACT_ACTION_STEPS:
        raise RuntimeError("ACT action-step count differs from the frozen checkpoint.")
    if args.maximum_trial_seconds != setup["maximum_trial_seconds"]:
        raise RuntimeError("Trial duration differs from the frozen plan.")
    matches = [
        trial
        for trial in plan["trials"]
        if trial["spawn_region"] == args.spawn_region
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "spawn region must identify exactly one frozen trial: "
            f"{args.spawn_region}"
        )
    return plan, profile, matches[0]


def preflight(args: argparse.Namespace) -> dict:
    plan, profile, trial = load_frozen_contract(args)
    model_hash = sha256_file(args.checkpoint / "model.safetensors")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Selected model hash is not the frozen 100k checkpoint.")
    calibration_hash = verify_frozen_calibration(args.calibration)
    if args.follower_port != EXPECTED_FOLLOWER_PORT:
        raise RuntimeError("Follower port is not the frozen by-id identity.")
    follower_path = Path(args.follower_port)
    if not follower_path.is_symlink():
        raise RuntimeError("Follower by-id symlink is unavailable.")
    follower_target = follower_path.resolve(strict=True)
    if not stat.S_ISCHR(follower_target.stat().st_mode):
        raise RuntimeError("Follower by-id target is not a character device.")
    if str(args.camera_device) != EXPECTED_CAMERA_DEVICE:
        raise RuntimeError("Camera device is not the frozen by-id identity.")
    camera_path = Path(args.camera_device)
    if not camera_path.is_symlink():
        raise RuntimeError("Head-camera by-id symlink is unavailable.")
    camera_target = camera_path.resolve(strict=True)
    if not stat.S_ISCHR(camera_target.stat().st_mode):
        raise RuntimeError("Head-camera by-id target is not a character device.")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is unavailable for immutable trial video.")
    return {
        "status": "hardware_ready_not_connected",
        "evaluation_id": plan["evaluation_id"],
        "evaluation_role": plan["comparison_role"],
        "trial": trial,
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "evaluation_profile_id": profile["profile_id"],
        "evaluation_profile_sha256": EXPECTED_PROFILE_SHA256,
        "checkpoint": str(args.checkpoint),
        "model_sha256": model_hash,
        "calibration": str(args.calibration),
        "calibration_sha256": calibration_hash,
        "follower_port": args.follower_port,
        "follower_serial": EXPECTED_FOLLOWER_SERIAL,
        "follower_resolved_device": str(follower_target),
        "camera_device": str(args.camera_device),
        "camera_resolved_device": str(camera_target),
        "camera_profile_id": CAMERA_PROFILE_ID,
        "control_fps": CONTROL_FPS,
        "act_chunk_size": ACT_CHUNK_SIZE,
        "act_n_action_steps": ACT_ACTION_STEPS,
        "max_relative_target": None,
        "custom_absolute_action_clamp": False,
        "action_path_profile_id": ACTION_PATH_PROFILE_ID,
        "policy_timing_profile_id": POLICY_TIMING_PROFILE_ID,
        "ready_pose": profile["ready_pose"],
        "ffmpeg": ffmpeg_path,
    }


def set_verified_torque(robot, enabled: bool) -> None:
    value = 1 if enabled else 0
    robot.bus.sync_write("Torque_Enable", value, normalize=False, num_retry=2)
    actual = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    if any(int(state) != value for state in actual.values()):
        raise RuntimeError(
            f"Follower torque verification failed: expected {value}, got {actual}"
        )


def video_command(video_path: Path) -> list[str]:
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
        "640x480",
        "-framerate",
        str(CONTROL_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
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


def action_dict(values: np.ndarray) -> dict[str, float]:
    requested = finite_joint_vector(values, "requested action")
    return {
        f"{joint}.pos": float(requested[index])
        for index, joint in enumerate(JOINT_ORDER)
    }


def sent_action_vector(sent: dict[str, float]) -> np.ndarray:
    return finite_joint_vector(
        [sent[f"{joint}.pos"] for joint in JOINT_ORDER],
        "upstream returned sent action",
    )


def read_present_position(robot) -> np.ndarray:
    observed = robot.bus.sync_read("Present_Position")
    return finite_joint_vector(
        [observed[joint] for joint in JOINT_ORDER],
        "Follower observed state",
    )


def move_to_frozen_ready_pose(
    robot,
    trajectory_handle,
    *,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> dict:
    started = now_fn()
    period = 1.0 / READY_MOVE_CONTROL_FPS
    steps = 0
    while True:
        tick_started = now_fn()
        observed = read_present_position(robot)
        delta = READY_POSE.astype(np.float64) - observed.astype(np.float64)
        maximum_error = float(np.max(np.abs(delta)))
        if maximum_error <= READY_MOVE_TOLERANCE:
            return {
                "status": "ready_pose_observed",
                "profile_id": READY_MOVE_PROFILE_ID,
                "requested_state": READY_POSE.tolist(),
                "observed_state": observed.tolist(),
                "delta_requested_minus_observed": delta.tolist(),
                "maximum_absolute_error": maximum_error,
                "steps": steps,
                "elapsed_seconds": now_fn() - started,
            }
        if now_fn() - started >= READY_MOVE_TIMEOUT_SECONDS:
            raise RuntimeError(
                "Frozen ready pose was not reached within the existing "
                f"{READY_MOVE_TIMEOUT_SECONDS}-second movement timeout."
            )
        requested = READY_POSE.copy()
        sent = sent_action_vector(robot.send_action(action_dict(requested)))
        upstream_modified_mask = ~np.isclose(
            sent,
            requested,
            rtol=0.0,
            atol=1.0e-6,
        )
        compute_seconds = now_fn() - tick_started
        scheduled_sleep_seconds = max(0.0, period - compute_seconds)
        record = {
            "movement_step": steps,
            "observed_state": observed.tolist(),
            "requested_action": requested.tolist(),
            "sent_action": sent.tolist(),
            "upstream_action_modified_mask": upstream_modified_mask.tolist(),
            "delta_requested_minus_observed": delta.tolist(),
            "maximum_absolute_error_before_step": maximum_error,
            "loop_compute_seconds": compute_seconds,
            "scheduled_sleep_seconds": scheduled_sleep_seconds,
        }
        trajectory_handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        steps += 1
        sleep_fn(scheduled_sleep_seconds)


def run_paced_ticks(
    duration_seconds: float,
    tick_fn: Callable[[int, float, float], dict],
    record_fn: Callable[[dict], None],
    *,
    period: float,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> list[dict]:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite.")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite.")
    records: list[dict] = []
    loop_started = now_fn()
    step = 0
    while now_fn() - loop_started < duration_seconds:
        tick_started = now_fn()
        record = tick_fn(step, tick_started, loop_started)
        compute_seconds = now_fn() - tick_started
        scheduled_sleep_seconds = max(0.0, period - compute_seconds)
        record.update(
            {
                "step": step,
                "tick_started_elapsed_seconds": tick_started - loop_started,
                "loop_compute_seconds": compute_seconds,
                "scheduled_sleep_seconds": scheduled_sleep_seconds,
            }
        )
        record_fn(record)
        records.append(record)
        sleep_fn(scheduled_sleep_seconds)
        record["tick_completed_elapsed_seconds"] = now_fn() - loop_started
        step += 1
    return records


def reset_policy_after_ready_pose(model, ready_pose_result: dict) -> str:
    if ready_pose_result.get("status") != "ready_pose_observed":
        raise RuntimeError("Policy queue reset requires an observed frozen ready pose.")
    model.reset()
    return datetime.now(UTC).isoformat()


def run_hardware_trial(args: argparse.Namespace, preflight_record: dict) -> None:
    from examples.picklift_v3.camera_profile import (
        camera_profile,
        canonicalize_front,
    )
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    if not args.operator_confirmed_ready:
        raise RuntimeError("--execute-hardware requires --operator-confirmed-ready.")

    profile = camera_profile(CAMERA_PROFILE_ID)
    camera = OpenCVCameraConfig(
        index_or_path=args.camera_device,
        width=profile["source"]["width"],
        height=profile["source"]["height"],
        fps=profile["source"]["fps"],
        fourcc=profile["source"]["fourcc"],
    )
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id="so101_follower_main",
            cameras={"front": camera},
            use_degrees=True,
            max_relative_target=None,
        )
    )

    model = ACTPolicy.from_pretrained(args.checkpoint)
    model.to("cuda")
    model.eval()
    if model.config.chunk_size != ACT_CHUNK_SIZE:
        raise RuntimeError("Loaded checkpoint ACT chunk size changed.")
    if model.config.n_action_steps != ACT_ACTION_STEPS:
        raise RuntimeError("Loaded checkpoint ACT n_action_steps changed.")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_dir / f"{args.spawn_region}.json"
    video_path = args.evidence_dir / f"{args.spawn_region}.mp4"
    steps_path = args.evidence_dir / f"{args.spawn_region}.steps.jsonl"
    ready_path = args.evidence_dir / f"{args.spawn_region}.ready.jsonl"
    return_path = args.evidence_dir / f"{args.spawn_region}.return.jsonl"
    for path, label in (
        (evidence_path, "Evidence"),
        (video_path, "Video"),
        (steps_path, "Steps"),
        (ready_path, "Ready-pose trajectory"),
        (return_path, "Return trajectory"),
    ):
        if path.exists():
            raise RuntimeError(f"{label} already exists; refusing to overwrite {path}.")

    started_at = datetime.now(UTC).isoformat()
    termination = "maximum_duration"
    torque_may_be_enabled = False
    torque_disable_verified = False
    connected_state_before_ready: np.ndarray | None = None
    ready_result: dict | None = None
    return_result: dict | None = None
    policy_reset_at_utc: str | None = None
    run_error: str | None = None
    shutdown_errors: list[str] = []
    video_process: subprocess.Popen[bytes] | None = None
    video_stderr = ""
    steps: list[dict] = []
    steps_handle = steps_path.open("x", encoding="utf-8", buffering=1)
    ready_handle = ready_path.open("x", encoding="utf-8", buffering=1)
    return_handle = return_path.open("x", encoding="utf-8", buffering=1)

    try:
        robot.bus.connect(handshake=True)
        initial_torque = robot.bus.sync_read(
            "Torque_Enable",
            normalize=False,
            num_retry=2,
        )
        if any(int(value) != 0 for value in initial_torque.values()):
            set_verified_torque(robot, False)
            raise RuntimeError(
                "Follower torque was already enabled before the trial; "
                "disabled torque and refused execution."
            )
        for connected_camera in robot.cameras.values():
            connected_camera.connect()

        follower_raw = robot.bus.sync_read("Present_Position", normalize=False)
        robot.bus.sync_write("Goal_Position", follower_raw, normalize=False)
        connected_state_before_ready = read_present_position(robot)
        torque_may_be_enabled = True
        set_verified_torque(robot, True)

        ready_result = move_to_frozen_ready_pose(robot, ready_handle)

        video_process = subprocess.Popen(
            video_command(video_path),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if video_process.stdin is None or video_process.stderr is None:
            raise RuntimeError("Failed to open ffmpeg video pipes.")
        policy_reset_at_utc = reset_policy_after_ready_pose(model, ready_result)

        def policy_tick(
            step: int,
            tick_started: float,
            loop_started: float,
        ) -> dict:
            observation = robot.get_observation()
            state = finite_joint_vector(
                [observation[f"{joint}.pos"] for joint in JOINT_ORDER],
                "policy observation state",
            )
            front = canonicalize_front(
                np.asarray(observation["front"]),
                CAMERA_PROFILE_ID,
            )
            frame_bytes = front.tobytes(order="C")
            assert video_process is not None
            assert video_process.stdin is not None
            video_process.stdin.write(frame_bytes)
            policy_input = prepare_observation_for_inference(
                {
                    "observation.state": state,
                    "observation.images.front": front,
                },
                device=torch.device("cuda"),
                task="Task 1 PickLift v1",
                robot_type="so101_follower",
            )
            with torch.inference_mode():
                processed = preprocessor(policy_input)
                raw_action_tensor = postprocessor(model.select_action(processed))
            raw_action = finite_joint_vector(
                raw_action_tensor.detach().cpu().numpy().reshape(-1),
                "policy raw action",
            )
            requested_action = raw_action.copy()
            sent_action = sent_action_vector(
                robot.send_action(action_dict(requested_action))
            )
            upstream_modified_mask = ~np.isclose(
                sent_action,
                requested_action,
                rtol=0.0,
                atol=1.0e-6,
            )
            record = {
                "observation_state": state.tolist(),
                "canonical_rgb_sha256": hashlib.sha256(frame_bytes).hexdigest(),
                "raw_action": raw_action.tolist(),
                "requested_action": requested_action.tolist(),
                "custom_action_transform": "none",
                "sent_action": sent_action.tolist(),
                "upstream_action_modified_mask": (
                    upstream_modified_mask.tolist()
                ),
            }
            if step == 0:
                tick0_delta = (
                    READY_POSE.astype(np.float64) - state.astype(np.float64)
                )
                record["tick0_ready_pose"] = {
                    "requested_state": READY_POSE.tolist(),
                    "observed_state": state.tolist(),
                    "delta_requested_minus_observed": tick0_delta.tolist(),
                }
            return record

        def write_step(record: dict) -> None:
            steps_handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )

        steps = run_paced_ticks(
            args.maximum_trial_seconds,
            policy_tick,
            write_step,
            period=1.0 / CONTROL_FPS,
        )
        return_result = move_to_frozen_ready_pose(robot, return_handle)

    except KeyboardInterrupt:
        termination = "operator_abort"
        run_error = "KeyboardInterrupt: operator emergency abort"
    except Exception as exc:
        termination = "hardware_or_runtime_error"
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        steps_handle.close()
        ready_handle.close()
        return_handle.close()
        if robot.bus.is_connected:
            try:
                if torque_may_be_enabled:
                    try:
                        set_verified_torque(robot, False)
                        torque_disable_verified = True
                    except Exception as exc:
                        shutdown_errors.append(
                            f"torque_disable={type(exc).__name__}: {exc}"
                        )
            finally:
                try:
                    robot.bus.disconnect(disable_torque=False)
                except Exception as exc:
                    shutdown_errors.append(
                        f"bus_disconnect={type(exc).__name__}: {exc}"
                    )
        for connected_camera in robot.cameras.values():
            if connected_camera.is_connected:
                try:
                    connected_camera.disconnect()
                except Exception as exc:
                    shutdown_errors.append(
                        f"camera_disconnect={type(exc).__name__}: {exc}"
                    )
        if video_process is not None:
            if video_process.stdin is not None:
                with contextlib.suppress(Exception):
                    video_process.stdin.close()
            video_returncode = video_process.wait()
            if video_process.stderr is not None:
                video_stderr = video_process.stderr.read().decode(
                    "utf-8",
                    errors="replace",
                )
            if video_returncode != 0:
                shutdown_errors.append(
                    f"ffmpeg_returncode={video_returncode}: {video_stderr}"
                )

    ended_at = datetime.now(UTC).isoformat()
    video_exists = video_path.exists()
    tick0 = steps[0].get("tick0_ready_pose") if steps else None
    evidence = {
        **preflight_record,
        "status": (
            "completed_pending_operator_annotation"
            if run_error is None and not shutdown_errors
            else "aborted_with_error"
        ),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "termination": termination,
        "run_error": run_error,
        "shutdown_errors": shutdown_errors,
        "torque_disable_verified": torque_disable_verified,
        "operator_annotation_status": "pending",
        "connected_state_before_ready": (
            connected_state_before_ready.tolist()
            if connected_state_before_ready is not None
            else None
        ),
        "ready_pose_alignment": {
            "before_policy_window": True,
            "profile_id": READY_MOVE_PROFILE_ID,
            "source_profile_id": READY_POSE_PROFILE_ID,
            "state_sha256": READY_POSE_STATE_SHA256,
            "requested_state": READY_POSE.tolist(),
            "result": ready_result,
            "trajectory": {
                "path": str(ready_path),
                "sha256": sha256_file(ready_path),
                "lines": len(
                    ready_path.read_text(encoding="utf-8").splitlines()
                ),
            },
        },
        "policy_start": {
            "policy_reset_after_ready_pose": True,
            "policy_reset_at_utc": policy_reset_at_utc,
            "tick0_observation_after_reset": tick0,
        },
        "steps_jsonl": {
            "path": str(steps_path),
            "sha256": sha256_file(steps_path),
            "lines": len(steps),
        },
        "upstream_action_modified_events": sum(
            any(step["upstream_action_modified_mask"]) for step in steps
        ),
        "automatic_return": {
            "outside_evaluation_window": True,
            "profile_id": READY_MOVE_PROFILE_ID,
            "target_is_same_frozen_ready_pose": True,
            "control_fps": READY_MOVE_CONTROL_FPS,
            "arrival_tolerance": READY_MOVE_TOLERANCE,
            "timeout_seconds": READY_MOVE_TIMEOUT_SECONDS,
            "result": return_result,
            "trajectory": {
                "path": str(return_path),
                "sha256": sha256_file(return_path),
                "lines": len(
                    return_path.read_text(encoding="utf-8").splitlines()
                ),
            },
        },
        "video": {
            "path": str(video_path),
            "exists": video_exists,
            "sha256": sha256_file(video_path) if video_exists else None,
            "frames": len(steps),
            "encoded_fps": CONTROL_FPS,
            "source": "canonical_rgb_act_input",
            "overlay": False,
        },
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved immutable trial evidence to {evidence_path}")
    if run_error is not None or shutdown_errors:
        raise RuntimeError(
            f"Trial aborted safely: run_error={run_error}, "
            f"shutdown_errors={shutdown_errors}"
        )


def main() -> None:
    args = parse_args()
    record = preflight(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not args.execute_hardware:
        print("DRY RUN ONLY: no camera, serial port, torque, or robot action was accessed.")
        return
    run_hardware_trial(args, record)


if __name__ == "__main__":
    main()
