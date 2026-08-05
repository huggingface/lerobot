from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_trial_evidence(evidence: dict) -> list[str]:
    errors: list[str] = []
    profile = evidence.get("success_early_stop_profile", {})
    enabled = profile.get("enabled") is True
    termination = evidence.get("termination")
    steps = evidence.get("steps_jsonl", {}).get("lines")
    frames = evidence.get("video", {}).get("frames")
    if steps != frames:
        errors.append("video_frames_must_equal_steps")
    if evidence.get("actual_policy_ticks") != steps:
        errors.append("actual_policy_ticks_must_equal_steps")
    if evidence.get("maximum_trial_seconds") != 30.0:
        errors.append("maximum_trial_seconds_must_be_30")
    if termination == "success_early_stop":
        if not enabled:
            errors.append("early_stop_termination_without_opt_in_profile")
        for key in (
            "success_signal_path",
            "success_signal_sha256",
            "success_signal_created_at_utc",
            "success_signal_observed_policy_step",
            "success_signal_observed_elapsed_seconds",
        ):
            if evidence.get(key) is None:
                errors.append(f"missing_{key}")
        observed_step = evidence.get("success_signal_observed_policy_step")
        if isinstance(observed_step, int) and steps != observed_step + 1:
            errors.append("confirming_step_must_be_last_recorded_step")
    elif any(
        evidence.get(key) is not None
        for key in (
            "success_signal_sha256",
            "success_signal_created_at_utc",
            "success_signal_observed_policy_step",
            "success_signal_observed_elapsed_seconds",
        )
    ):
        errors.append("success_signal_provenance_without_early_stop_termination")
    if evidence.get("torque_disable_verified") is not True:
        errors.append("torque_disable_not_verified")
    automatic_return = evidence.get("automatic_return", {})
    if automatic_return.get("outside_evaluation_window") is not True:
        errors.append("automatic_return_not_outside_window")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    errors = validate_trial_evidence(evidence)
    print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors}))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
