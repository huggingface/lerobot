from __future__ import annotations

from copy import deepcopy

from examples.picklift_v3.camera_profile import ALIGNED_FRONT_CAMERA_PROFILE_ID
from examples.picklift_v3.task_frame import (
    TASK_GRID_FRAME_V1_ID,
    TASK_GRID_FRAME_V2_ID,
)

ALIGNMENT_REFERENCE_V1_ID = "picklift_red_cube_alignment_v1"
ALIGNMENT_REFERENCE_V2_ID = "picklift_red_cube_alignment_v2"

_ALIGNMENT_REFERENCES = {
    ALIGNMENT_REFERENCE_V1_ID: {
        "reference_id": ALIGNMENT_REFERENCE_V1_ID,
        "task_frame_id": TASK_GRID_FRAME_V1_ID,
        "camera_profile_id": ALIGNED_FRONT_CAMERA_PROFILE_ID,
        "red_cube_center_m": {"x_forward": 0.15, "y_lateral": 0.0},
        "canonical_image_target": "legacy_frozen_alignment_evidence",
        "physical_confirmation_status": "confirmed_2026-07-28",
    },
    ALIGNMENT_REFERENCE_V2_ID: {
        "reference_id": ALIGNMENT_REFERENCE_V2_ID,
        "task_frame_id": TASK_GRID_FRAME_V2_ID,
        "camera_profile_id": ALIGNED_FRONT_CAMERA_PROFILE_ID,
        "red_cube_center_m": {"x_forward": 0.25, "y_lateral": 0.0},
        "canonical_image_target": "red_cube_center_at_canonical_640x480_image_center",
        "physical_confirmation_status": "pending_new_25cm_screenshot",
    },
}


def alignment_reference(reference_id: str) -> dict:
    try:
        return deepcopy(_ALIGNMENT_REFERENCES[reference_id])
    except KeyError as exc:
        raise ValueError(f"unsupported alignment_reference_id: {reference_id}") from exc


def validate_alignment_reference_config(cfg: dict) -> dict:
    reference = alignment_reference(str(cfg.get("alignment_reference_id", "")))
    if reference["task_frame_id"] != cfg.get("task_frame_id"):
        raise ValueError("alignment reference task_frame_id does not match configuration")
    if cfg.get("mode") == "real" and reference["camera_profile_id"] != cfg.get("camera_profile_id"):
        raise ValueError("alignment reference camera_profile_id does not match real configuration")
    return reference
