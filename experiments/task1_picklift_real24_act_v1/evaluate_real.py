from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from deployment_safety import (
    JOINT_ORDER,
    clamp_action_fail_closed,
    sha256_file,
    verify_frozen_calibration,
)

EXPECTED_MODEL_SHA256 = "ba233bd25aad5bb63a8a863a8fe1f8ebfdb804547a5add33520c8e1e93f890fb"
CAMERA_PROFILE_ID = "icspring_front_crop_1280x960_to_640x480_v1"
CONTROL_FPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One explicitly armed Task1 PickLift real-robot evaluation trial."
    )
    parser.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--spawn-region", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument(
        "--camera-device",
        default="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_act_v1/"
            "full_100k/checkpoints/100000/pretrained_model"
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
        default=Path("experiments/task1_picklift_real24_act_v1/evaluation_plan.json"),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(
            "/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_act_v1_eval12"
        ),
    )
    parser.add_argument("--maximum-trial-seconds", type=float, default=30.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    return parser.parse_args()


def load_frozen_trial(args: argparse.Namespace) -> dict:
    plan = json.loads(args.plan.read_text())
    matches = [trial for trial in plan["trials"] if trial["spawn_region"] == args.spawn_region]
    if len(matches) != 1:
        raise RuntimeError(f"spawn region must identify exactly one frozen trial: {args.spawn_region}")
    if args.maximum_trial_seconds != plan["setup"]["maximum_trial_seconds"]:
        raise RuntimeError("maximum trial duration differs from the frozen evaluation plan.")
    return matches[0]


def preflight(args: argparse.Namespace) -> dict:
    trial = load_frozen_trial(args)
    model_hash = sha256_file(args.checkpoint / "model.safetensors")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Selected model hash is not the frozen 100k checkpoint.")
    calibration_hash = verify_frozen_calibration(args.calibration)
    return {
        "status": "hardware_ready_not_connected",
        "trial": trial,
        "checkpoint": str(args.checkpoint),
        "model_sha256": model_hash,
        "calibration": str(args.calibration),
        "calibration_sha256": calibration_hash,
        "camera_profile_id": CAMERA_PROFILE_ID,
        "control_fps": CONTROL_FPS,
    }


def set_verified_torque(robot, enabled: bool) -> None:
    value = 1 if enabled else 0
    robot.bus.sync_write("Torque_Enable", value, normalize=False, num_retry=2)
    actual = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    if any(int(state) != value for state in actual.values()):
        raise RuntimeError(f"Follower torque verification failed: expected {value}, got {actual}")


def annotate_result() -> tuple[str, str | None]:
    result = input("Trial result [success/failure/aborted]: ").strip().lower()
    if result not in {"success", "failure", "aborted"}:
        raise RuntimeError("Trial result must be success, failure, or aborted.")
    if result != "failure":
        return result, None
    category = input(
        "Failure category [perception_failure/missed_grasp/unstable_grasp/"
        "wrong_trajectory/collision/premature_release/timeout/out_of_workspace/unknown]: "
    ).strip()
    allowed = {
        "perception_failure",
        "missed_grasp",
        "unstable_grasp",
        "wrong_trajectory",
        "collision",
        "premature_release",
        "timeout",
        "out_of_workspace",
        "unknown",
    }
    if category not in allowed:
        raise RuntimeError("Failure category is outside the frozen evaluation contract.")
    return result, category


def run_hardware_trial(args: argparse.Namespace, preflight_record: dict) -> None:
    from examples.picklift_v3.camera_profile import camera_profile, canonicalize_front
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

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
            max_relative_target=args.max_relative_target,
        )
    )

    model = ACTPolicy.from_pretrained(args.checkpoint)
    model.to("cuda")
    model.eval()
    model.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    input(
        f"Place the cube at {args.spawn_region} "
        f"({preflight_record['trial']['spawn_x_cm']} cm, "
        f"{preflight_record['trial']['spawn_y_cm']} cm), place the arm in the approved "
        "evaluation start pose, verify the emergency stop path, then press ENTER to connect. "
    )

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_dir / f"{args.spawn_region}.json"
    if evidence_path.exists():
        raise RuntimeError(f"Evidence already exists; refusing to overwrite {evidence_path}.")

    started_at = datetime.now(UTC).isoformat()
    steps: list[dict] = []
    stop_event = threading.Event()
    termination = "maximum_duration"
    torque_may_be_enabled = False

    def wait_for_operator_end() -> None:
        input("Policy active. Press ENTER to end the trial early. Ctrl+C is emergency abort.\n")
        stop_event.set()

    try:
        robot.bus.connect(handshake=True)
        for connected_camera in robot.cameras.values():
            connected_camera.connect()

        follower_raw = robot.bus.sync_read("Present_Position", normalize=False)
        robot.bus.sync_write("Goal_Position", follower_raw, normalize=False)
        initial_state_dict = robot.bus.sync_read("Present_Position")
        initial_state = np.asarray([initial_state_dict[joint] for joint in JOINT_ORDER], dtype=np.float32)
        clamp_action_fail_closed(initial_state)

        torque_may_be_enabled = True
        set_verified_torque(robot, True)
        threading.Thread(target=wait_for_operator_end, daemon=True).start()

        period = 1.0 / CONTROL_FPS
        next_tick = time.perf_counter()
        deadline = next_tick + args.maximum_trial_seconds
        while time.perf_counter() < deadline and not stop_event.is_set():
            observation = robot.get_observation()
            state = np.asarray(
                [observation[f"{joint}.pos"] for joint in JOINT_ORDER],
                dtype=np.float32,
            )
            front = canonicalize_front(np.asarray(observation["front"]), CAMERA_PROFILE_ID)
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
            raw_action = raw_action_tensor.detach().cpu().numpy().reshape(-1)
            clipped_action, clip_mask = clamp_action_fail_closed(raw_action)
            requested = {
                f"{joint}.pos": float(clipped_action[index])
                for index, joint in enumerate(JOINT_ORDER)
            }
            sent_dict = robot.send_action(requested)
            sent_action = [float(sent_dict[f"{joint}.pos"]) for joint in JOINT_ORDER]
            steps.append(
                {
                    "step": len(steps),
                    "raw_action": raw_action.tolist(),
                    "calibration_clipped_action": clipped_action.tolist(),
                    "calibration_clip_mask": clip_mask.tolist(),
                    "sent_action": sent_action,
                }
            )
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

        if stop_event.is_set():
            termination = "operator_end"
    except KeyboardInterrupt:
        termination = "operator_abort"
    finally:
        if robot.bus.is_connected:
            try:
                if torque_may_be_enabled:
                    set_verified_torque(robot, False)
            finally:
                robot.bus.disconnect(disable_torque=False)
        for connected_camera in robot.cameras.values():
            if connected_camera.is_connected:
                connected_camera.disconnect()

    result, failure_category = annotate_result()
    evidence = {
        **preflight_record,
        "status": "completed",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "termination": termination,
        "result": result,
        "failure_category": failure_category,
        "initial_state": initial_state.tolist(),
        "steps": steps,
        "calibration_clip_events": sum(any(step["calibration_clip_mask"]) for step in steps),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"Saved immutable trial evidence to {evidence_path}")


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
