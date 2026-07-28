from __future__ import annotations

from copy import deepcopy

TASK_GRID_FRAME_V1_ID = "picklift_task_grid_v1"
TASK_GRID_FRAME_V2_ID = "picklift_task_grid_v2"
TASK_GRID_FRAME_ID = TASK_GRID_FRAME_V2_ID

_TASK_GRID_FRAME_V1 = {
    "frame_id": TASK_GRID_FRAME_V1_ID,
    "reference": "physical_mat_grid_aligned_with_mujoco_grid",
    "origin": {
        "description": (
            "SO-101 base rotation center projected onto the task plane, fixed at the mat bottom 30 cm mark"
        ),
        "coordinates_m": {"x": 0.0, "y": 0.0},
    },
    "centerline": {
        "definition": "the task-grid line through the origin along +X is Y=0",
    },
    "axes": {
        "x": {
            "name": "forward",
            "positive_direction": (
                "along the mat grid from the robot region into the workspace "
                "and toward the calibrated red-cube location"
            ),
        },
        "y": {
            "name": "lateral",
            "positive_direction": (
                "along the perpendicular task-grid direction designated +Y, "
                "identical to the MuJoCo grid +Y direction"
            ),
        },
    },
    "units": {
        "canonical": "meter",
        "operator_config": "centimeter",
    },
    "measurement_rule": (
        "measure only along the physical task-grid axes; never infer axes "
        "from camera image horizontal/vertical or table edges"
    ),
    "known_reference": {
        "description": "red cube center in the frozen alignment evidence",
        "coordinates_m": {"x": 0.15, "y": 0.0},
    },
}

_TASK_GRID_FRAME_V2 = deepcopy(_TASK_GRID_FRAME_V1)
_TASK_GRID_FRAME_V2.update(
    {
        "frame_id": TASK_GRID_FRAME_V2_ID,
        "geometry_parent_frame_id": TASK_GRID_FRAME_V1_ID,
        "alignment_reference_policy": "separate_versioned_object",
    }
)
_TASK_GRID_FRAME_V2.pop("known_reference")

_TASK_FRAMES = {
    TASK_GRID_FRAME_V1_ID: _TASK_GRID_FRAME_V1,
    TASK_GRID_FRAME_V2_ID: _TASK_GRID_FRAME_V2,
}


def task_frame(frame_id: str) -> dict:
    try:
        return deepcopy(_TASK_FRAMES[frame_id])
    except KeyError as exc:
        raise ValueError(f"unsupported task_frame_id: {frame_id}") from exc


def validate_task_frame_config(cfg: dict) -> dict:
    return task_frame(str(cfg.get("task_frame_id", "")))
