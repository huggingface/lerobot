from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_PLAN = EXPERIMENT_DIR / "evaluation_plan_ready_tolerance3_v2.json"
EXPECTED_PLAN_SHA256 = (
    "1eb82cc573a661f85d7c21ced5651a9cc64c8809873a3299d13e86d59fd19ada"
)
EXPECTED_RESEARCH_PLAN_SHA256 = (
    "c62fd6505368fe7417048530d8538799026aaa05f85236d5c957d02135305ce0"
)
EXPECTED_ENGINE_SHA256 = (
    "380b8c1c13f0f38a59e129b78d845a1cbd8916411af1f61a56b9267e83205f96"
)
EXPECTED_PROFILE_SHA256 = (
    "60025f6478a63bcf9b301a75cd27124b19f1cf4f6b142f8576e6b10abf4f95a5"
)
EXPECTED_READY_MOVE_TOLERANCE = 3.0
EXPECTED_EVALUATION_ID = (
    "task1_picklift_real24_vs_mixed_paired_real24_eval24_v1"
)
EXPECTED_FOLLOWER_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00"
)
EXPECTED_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-icSpring_icspring_camera_202404160005-video-index0"
)
DEFAULT_CALIBRATION = Path(
    "/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/"
    "so_follower/so101_follower_main.json"
)
READY_POSE_STATE_SHA256 = (
    "ecb871efad5692e192ac0f690bc0e959fef371bbb8338a31b23ca697741e3b56"
)
MODEL_IDS = ("real24_only", "real24_questsim24")
EXPECTED_SCHEDULE = (
    ("r1c1", "real24_only"),
    ("r1c1", "real24_questsim24"),
    ("r1c2", "real24_questsim24"),
    ("r1c2", "real24_only"),
    ("r1c3", "real24_only"),
    ("r1c3", "real24_questsim24"),
    ("r1c4", "real24_questsim24"),
    ("r1c4", "real24_only"),
    ("r2c1", "real24_only"),
    ("r2c1", "real24_questsim24"),
    ("r2c2", "real24_questsim24"),
    ("r2c2", "real24_only"),
    ("r2c3", "real24_only"),
    ("r2c3", "real24_questsim24"),
    ("r2c4", "real24_questsim24"),
    ("r2c4", "real24_only"),
    ("r3c1", "real24_only"),
    ("r3c1", "real24_questsim24"),
    ("r3c2", "real24_questsim24"),
    ("r3c2", "real24_only"),
    ("r3c3", "real24_only"),
    ("r3c3", "real24_questsim24"),
    ("r3c4", "real24_questsim24"),
    ("r3c4", "real24_only"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_joint_vector(values: Any, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (6,):
        raise RuntimeError(f"{label} must have shape (6,), got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise RuntimeError(f"{label} contains NaN or infinity.")
    return vector


def ready_pose_state_sha256(values: Any) -> str:
    vector = finite_joint_vector(values, "ready pose")
    payload = json.dumps(
        [float(value) for value in vector],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_frozen_plan(path: Path = DEFAULT_PLAN) -> dict:
    if sha256_file(path) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Paired evaluation plan hash differs from the frozen plan.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["evaluation_id"] != EXPECTED_EVALUATION_ID:
        raise RuntimeError("Unexpected paired evaluation identity.")
    if plan["research_contract"]["sha256"] != EXPECTED_RESEARCH_PLAN_SHA256:
        raise RuntimeError("Research-contract hash reference changed.")
    if plan["execution_engine"]["source_sha256"] != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Official-send engine hash reference changed.")
    if plan["evaluation_profile"]["sha256"] != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Evaluation-profile hash reference changed.")
    if tuple(plan["models"]) != MODEL_IDS:
        raise RuntimeError("Frozen model identities or ordering changed.")
    setup = plan["setup"]
    if setup["max_relative_target"] is not None:
        raise RuntimeError("Paired evaluation requires max_relative_target=None.")
    if setup["custom_absolute_action_clamp"] is not False:
        raise RuntimeError("Runner-side absolute action clamp must remain disabled.")
    if setup["custom_relative_step_limit_degrees"] is not None:
        raise RuntimeError("Runner-side relative step limiter must remain disabled.")
    if setup["control_fps"] != 20 or setup["maximum_trial_seconds"] != 30:
        raise RuntimeError("Frozen policy timing changed.")
    if setup["act_chunk_size"] != 67 or setup["act_n_action_steps"] != 67:
        raise RuntimeError("Frozen ACT action queue configuration changed.")
    if setup["ready_pose_before_every_trial"] is not True:
        raise RuntimeError("Ready pose is required before every trial.")
    if setup["ready_pose_after_every_trial"] is not True:
        raise RuntimeError("Ready pose is required after every trial.")
    if setup["policy_reset_after_ready_pose"] is not True:
        raise RuntimeError("Policy reset must happen after ready pose.")
    if (
        setup["ready_pose_arrival_tolerance_degrees"]
        != EXPECTED_READY_MOVE_TOLERANCE
    ):
        raise RuntimeError("Ready-pose arrival tolerance differs from revision v2.")
    if ready_pose_state_sha256(setup["ready_pose_state"]) != READY_POSE_STATE_SHA256:
        raise RuntimeError("Frozen ready-pose vector hash mismatch.")
    success = plan["success_contract"]
    if success["must_remain_held_until_timeout"] is not False:
        raise RuntimeError("Success must not require a hold through timeout.")
    if success["changes_policy_action_window"] is not False:
        raise RuntimeError("A success label must not shorten the action window.")
    trials = plan["trials"]
    if len(trials) != 24:
        raise RuntimeError("Paired plan must contain exactly 24 scored trials.")
    actual_schedule = tuple(
        (trial["cell_id"], trial["model_id"]) for trial in trials
    )
    if actual_schedule != EXPECTED_SCHEDULE:
        raise RuntimeError("Paired trial order differs from the research contract.")
    if [trial["order"] for trial in trials] != list(range(1, 25)):
        raise RuntimeError("Paired trial order indices must be contiguous 1..24.")
    if [trial["trial_id"] for trial in trials] != [
        f"t{index:02d}" for index in range(1, 25)
    ]:
        raise RuntimeError("Paired trial ids must be contiguous t01..t24.")
    if len({trial["spawn_region"] for trial in trials}) != 24:
        raise RuntimeError("Each trial needs a unique immutable artifact stem.")
    for cell in sorted({trial["cell_id"] for trial in trials}):
        models = [
            trial["model_id"] for trial in trials if trial["cell_id"] == cell
        ]
        if sorted(models) != sorted(MODEL_IDS):
            raise RuntimeError(f"Cell {cell} does not contain both models once.")
    return plan


def verify_static_files(plan: dict) -> dict:
    engine_path = resolve_repo_path(plan["execution_engine"]["path"])
    profile_path = resolve_repo_path(plan["evaluation_profile"]["path"])
    if sha256_file(engine_path) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Current official-send engine differs from commit 34cc7ac.")
    if sha256_file(profile_path) != EXPECTED_PROFILE_SHA256:
        raise RuntimeError("Current evaluation profile differs from the frozen profile.")
    models: dict[str, dict] = {}
    for model_id, model in plan["models"].items():
        checkpoint = Path(model["checkpoint"])
        weights = checkpoint / "model.safetensors"
        config_path = checkpoint / "config.json"
        actual_hash = sha256_file(weights)
        if actual_hash != model["model_sha256"]:
            raise RuntimeError(f"{model_id} checkpoint hash mismatch.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("type") != "act":
            raise RuntimeError(f"{model_id} is not an ACT checkpoint.")
        if config.get("chunk_size") != 67 or config.get("n_action_steps") != 67:
            raise RuntimeError(f"{model_id} ACT queue configuration changed.")
        models[model_id] = {
            "checkpoint": str(checkpoint),
            "model_sha256": actual_hash,
            "config_sha256": sha256_file(config_path),
            "chunk_size": config["chunk_size"],
            "n_action_steps": config["n_action_steps"],
        }
    return {
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "engine_path": str(engine_path),
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "profile_path": str(profile_path),
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "models": models,
    }


class FakeBus:
    def __init__(self, ready_pose: np.ndarray) -> None:
        self.state = ready_pose.copy()
        self.torque_enabled = False
        self.sent: list[np.ndarray] = []

    def move_to_ready(self, requested: np.ndarray) -> np.ndarray:
        self.state = finite_joint_vector(requested, "fake ready request").copy()
        return self.state.copy()

    def send(self, requested: np.ndarray) -> np.ndarray:
        action = finite_joint_vector(requested, "fake policy action").copy()
        self.sent.append(action)
        self.state = action.copy()
        return action


class FakeCamera:
    def capture(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeRobot:
    def __init__(self, ready_pose: np.ndarray) -> None:
        self.bus = FakeBus(ready_pose)
        self.camera = FakeCamera()

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bus.state.copy(), self.camera.capture()


class FakePolicy:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def action(self, state: np.ndarray) -> np.ndarray:
        return finite_joint_vector(state, "fake observation").copy()


def run_fake_protocol(plan: dict) -> dict:
    ready_pose = finite_joint_vector(
        plan["setup"]["ready_pose_state"],
        "ready pose",
    )
    robot = FakeRobot(ready_pose)
    policies = {model_id: FakePolicy(model_id) for model_id in MODEL_IDS}
    records = []
    for trial in plan["trials"]:
        robot.bus.torque_enabled = True
        observed_ready = robot.bus.move_to_ready(ready_pose)
        policy = policies[trial["model_id"]]
        policy.reset()
        tick0_state, canonical_rgb = robot.observation()
        raw_requested = policy.action(tick0_state)
        official_sent = robot.bus.send(raw_requested)
        observed_return = robot.bus.move_to_ready(ready_pose)
        robot.bus.torque_enabled = False
        records.append(
            {
                "order": trial["order"],
                "trial_id": trial["trial_id"],
                "cell_id": trial["cell_id"],
                "model_id": trial["model_id"],
                "ready_before_policy_matches": bool(
                    np.array_equal(observed_ready, ready_pose)
                ),
                "policy_reset_before_tick0": True,
                "tick0_state": tick0_state.tolist(),
                "canonical_rgb_shape": list(canonical_rgb.shape),
                "raw_requested_action": raw_requested.tolist(),
                "official_sent_action": official_sent.tolist(),
                "ready_after_trial_matches": bool(
                    np.array_equal(observed_return, ready_pose)
                ),
                "torque_disabled": not robot.bus.torque_enabled,
            }
        )
    return {
        "fake_hardware_only": True,
        "real_device_accessed": False,
        "trials_exercised": len(records),
        "models_in_frozen_order": [row["model_id"] for row in records],
        "policy_reset_calls": {
            model_id: policy.reset_calls for model_id, policy in policies.items()
        },
        "all_ready_before_policy": all(
            row["ready_before_policy_matches"] for row in records
        ),
        "all_ready_after_trial": all(
            row["ready_after_trial_matches"] for row in records
        ),
        "all_canonical_rgb_640x480": all(
            row["canonical_rgb_shape"] == [480, 640, 3] for row in records
        ),
        "all_official_sent_equals_requested": all(
            row["official_sent_action"] == row["raw_requested_action"]
            for row in records
        ),
        "all_torque_disabled": all(row["torque_disabled"] for row in records),
        "success_contract_probe": {
            "valid_success_seen_inside_window": True,
            "held_at_window_end": False,
            "scored_success": (
                plan["success_contract"]["must_remain_held_until_timeout"] is False
            ),
            "policy_window_unchanged": (
                plan["success_contract"]["changes_policy_action_window"] is False
            ),
        },
        "records": records,
    }


def run_paced_ticks(
    duration_seconds: float,
    tick_fn: Callable[[int, float, float], dict],
    record_fn: Callable[[dict], None],
    *,
    period: float,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict]:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite.")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite.")
    records = []
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


def software_dry_run(plan: dict) -> dict:
    static = verify_static_files(plan)
    fake = run_fake_protocol(plan)
    if fake["trials_exercised"] != 24:
        raise RuntimeError("Fake protocol did not exercise all 24 trials.")
    if fake["policy_reset_calls"] != {
        "real24_only": 12,
        "real24_questsim24": 12,
    }:
        raise RuntimeError("Each model must be reset once for each of its 12 trials.")
    checks = (
        fake["all_ready_before_policy"],
        fake["all_ready_after_trial"],
        fake["all_canonical_rgb_640x480"],
        fake["all_official_sent_equals_requested"],
        fake["all_torque_disabled"],
        fake["success_contract_probe"]["scored_success"],
        fake["success_contract_probe"]["policy_window_unchanged"],
    )
    if not all(checks):
        raise RuntimeError("Fake paired protocol verification failed.")
    return {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "software_dry_run_passed_hardware_not_accessed",
        "hardware_access": {
            "serial": False,
            "camera": False,
            "robot": False,
            "torque": False,
            "rollout": False,
        },
        "static_verification": static,
        "fake_protocol": fake,
        "next_gate": "Stop before the user turns on Follower 12 V.",
    }


def write_software_evidence(plan: dict, dry_run: dict) -> dict:
    evidence_root = Path(plan["evidence_root"])
    software_root = evidence_root / "software_preparation_v1"
    software_root.mkdir(parents=True, exist_ok=True)
    plan_copy = software_root / "evaluation_plan.json"
    dry_run_path = software_root / "dry_run.json"
    manifest_path = software_root / "manifest.json"
    hashes_path = software_root / "hashes.sha256"
    for path in (plan_copy, dry_run_path, manifest_path, hashes_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite frozen evidence: {path}")
    shutil.copyfile(DEFAULT_PLAN, plan_copy)
    dry_run_path.write_text(
        json.dumps(dry_run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "status": dry_run["status"],
        "evidence_root": str(evidence_root),
        "research_contract": plan["research_contract"],
        "plan": {
            "repo_path": str(DEFAULT_PLAN),
            "evidence_copy": str(plan_copy),
            "sha256": sha256_file(plan_copy),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "static_verification": dry_run["static_verification"],
        "dry_run": {
            "path": str(dry_run_path),
            "sha256": sha256_file(dry_run_path),
        },
        "hardware_access": dry_run["hardware_access"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hash_rows = [
        (sha256_file(plan_copy), plan_copy),
        (sha256_file(dry_run_path), dry_run_path),
        (sha256_file(manifest_path), manifest_path),
    ]
    hashes_path.write_text(
        "".join(f"{digest}  {path.name}\n" for digest, path in hash_rows),
        encoding="utf-8",
    )
    return {
        "evidence_root": str(evidence_root),
        "software_root": str(software_root),
        "plan_sha256": sha256_file(plan_copy),
        "dry_run_sha256": sha256_file(dry_run_path),
        "manifest_sha256": sha256_file(manifest_path),
        "hashes_sha256": sha256_file(hashes_path),
    }


def find_trial(plan: dict, trial_id: str) -> dict:
    matches = [trial for trial in plan["trials"] if trial["trial_id"] == trial_id]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown frozen trial id: {trial_id}")
    return matches[0]


def original_evidence_path(plan: dict, trial: dict) -> Path:
    return Path(plan["evidence_root"]) / "trials" / f"{trial['spawn_region']}.json"


def validate_execution_order(
    plan: dict,
    trial: dict,
    *,
    replacement: bool,
) -> tuple[str, str | None]:
    trials_root = Path(plan["evidence_root"]) / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    original_path = original_evidence_path(plan, trial)
    first_missing = next(
        (
            candidate
            for candidate in plan["trials"]
            if not original_evidence_path(plan, candidate).exists()
        ),
        None,
    )
    if not replacement:
        if first_missing is None:
            raise RuntimeError("All 24 frozen original trials already have evidence.")
        if first_missing["trial_id"] != trial["trial_id"]:
            raise RuntimeError(
                "Requested trial is not the next missing trial in frozen order: "
                f"expected {first_missing['trial_id']}."
            )
        return trial["spawn_region"], None
    if not original_path.exists():
        raise RuntimeError("Replacement requires preserved original evidence.")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    infrastructure_invalid = (
        original.get("status") == "aborted_with_error"
        or original.get("termination") == "hardware_or_runtime_error"
    )
    if not infrastructure_invalid:
        raise RuntimeError("Replacement is allowed only for infrastructure-invalid trials.")
    replacement_stem = f"{trial['spawn_region']}__replacement1"
    if (trials_root / f"{replacement_stem}.json").exists():
        raise RuntimeError("The one allowed linked replacement already exists.")
    return replacement_stem, trial["spawn_region"]


def load_official_engine(plan: dict):
    engine_path = resolve_repo_path(plan["execution_engine"]["path"])
    if sha256_file(engine_path) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("Official-send engine source hash changed.")
    if str(engine_path.parent) not in sys.path:
        sys.path.insert(0, str(engine_path.parent))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "task1_official_send_engine",
        engine_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the frozen official-send engine.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_paired_sidecar(
    plan: dict,
    trial: dict,
    artifact_stem: str,
    replacement_for: str | None,
) -> None:
    trials_root = Path(plan["evidence_root"]) / "trials"
    engine_evidence_path = trials_root / f"{artifact_stem}.json"
    if not engine_evidence_path.exists():
        return
    sidecar_path = trials_root / f"{artifact_stem}.paired.json"
    if sidecar_path.exists():
        raise RuntimeError(f"Refusing to overwrite paired sidecar: {sidecar_path}")
    evidence = json.loads(engine_evidence_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(evidence["started_at_utc"])
    ended = datetime.fromisoformat(evidence["ended_at_utc"])
    sidecar = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "trial": trial,
        "artifact_stem": artifact_stem,
        "replacement_for": replacement_for,
        "engine_evidence": {
            "path": str(engine_evidence_path),
            "sha256": sha256_file(engine_evidence_path),
        },
        "actual_policy_ticks": evidence["steps_jsonl"]["lines"],
        "wall_duration_seconds": (ended - started).total_seconds(),
        "operator_label": {"status": "pending"},
        "canonical_video_review_label": {"status": "pending"},
        "success_contract": plan["success_contract"],
        "return": evidence["automatic_return"],
        "torque_disable_verified": evidence["torque_disable_verified"],
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_hardware(args: argparse.Namespace, plan: dict) -> None:
    if not args.operator_confirmed_ready:
        raise RuntimeError("--execute-hardware requires --operator-confirmed-ready.")
    if args.trial_id is None:
        raise RuntimeError("--execute-hardware requires --trial-id.")
    verify_static_files(plan)
    trial = find_trial(plan, args.trial_id)
    artifact_stem, replacement_for = validate_execution_order(
        plan,
        trial,
        replacement=args.replacement,
    )
    engine = load_official_engine(plan)
    model = plan["models"][trial["model_id"]]
    engine.EXPECTED_MODEL_SHA256 = model["model_sha256"]
    engine.EXPECTED_PLAN_SHA256 = EXPECTED_PLAN_SHA256
    engine.EXPECTED_PROFILE_SHA256 = EXPECTED_PROFILE_SHA256
    engine.EXPECTED_EVALUATION_ID = EXPECTED_EVALUATION_ID
    profile_path = resolve_repo_path(plan["evaluation_profile"]["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    ready_movement = profile["ready_pose_movement"]
    if (
        ready_movement["arrival_tolerance_degrees"]
        != EXPECTED_READY_MOVE_TOLERANCE
    ):
        raise RuntimeError("Loaded profile has the wrong ready-pose tolerance.")
    engine.READY_MOVE_TOLERANCE = EXPECTED_READY_MOVE_TOLERANCE
    engine.READY_MOVE_PROFILE_ID = ready_movement["profile_id"]
    engine_args = argparse.Namespace(
        execute_hardware=True,
        operator_confirmed_ready=True,
        spawn_region=trial["spawn_region"],
        follower_port=args.follower_port,
        camera_device=args.camera_device,
        checkpoint=Path(model["checkpoint"]),
        calibration=args.calibration,
        plan=args.plan,
        profile=profile_path,
        evidence_dir=Path(plan["evidence_root"]) / "trials",
        maximum_trial_seconds=30.0,
    )
    preflight = engine.preflight(engine_args)
    preflight.update(
        {
            "paired_evaluation_id": plan["evaluation_id"],
            "paired_plan_sha256": EXPECTED_PLAN_SHA256,
            "paired_trial": trial,
            "replacement_for": replacement_for,
            "success_contract": plan["success_contract"],
            "operator_label": {"status": "pending"},
            "canonical_video_review_label": {"status": "pending"},
        }
    )
    engine_args.spawn_region = artifact_stem
    try:
        engine.run_hardware_trial(engine_args, preflight)
    finally:
        write_paired_sidecar(
            plan,
            trial,
            artifact_stem,
            replacement_for,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Task1 paired Real24-only versus mixed ACT evaluator. "
            "Software dry-run never inspects camera, serial, robot, or torque."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--software-dry-run", action="store_true")
    mode.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--freeze-software-evidence", action="store_true")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--trial-id")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--operator-confirmed-ready", action="store_true")
    parser.add_argument("--follower-port", default=EXPECTED_FOLLOWER_PORT)
    parser.add_argument("--camera-device", default=EXPECTED_CAMERA_DEVICE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_frozen_plan(args.plan)
    if args.software_dry_run:
        if args.trial_id or args.replacement or args.operator_confirmed_ready:
            raise RuntimeError("Hardware-only arguments are invalid in dry-run mode.")
        dry_run = software_dry_run(plan)
        if args.freeze_software_evidence:
            dry_run["frozen_evidence"] = write_software_evidence(plan, dry_run)
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        print(
            "DRY RUN ONLY: no serial, camera, robot, torque, 12 V, or rollout "
            "was accessed."
        )
        return
    if args.freeze_software_evidence:
        raise RuntimeError("--freeze-software-evidence is dry-run only.")
    execute_hardware(args, plan)


if __name__ == "__main__":
    main()
