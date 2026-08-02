from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESEARCH_CONTRACT_COMMIT = "73908355df1add52cd04753216c13f8b1c0b400a"
RESEARCH_CONTRACT_PARENT = "361cdd6e423e4697d300057eb23ae9e4991bb6c2"
COLLECTION_PLAN_ID = "task1_picklift_real96_collection_v1"
COLLECTION_PLAN_SHA256 = "f8d9ab2de3e7f6915dacafc2b8f70cc523373154b2df6b3abc8536fcfb623ef7"
POSE_MANIFEST_ID = "task1_picklift_real96_poses_v1"
POSE_MANIFEST_SHA256 = "1845f6e51f7afecfb21caf8278141aab4371e3e8667f2d81a14dbb318478f7d8"
SUBSET_MANIFEST_ID = "task1_picklift_real48_real96_subsets_v1"
SUBSET_MANIFEST_SHA256 = "bdf7ad387cd4d8f619ead947d40c7642682859a3986d6e30df591f767b24ac5b"
SESSION_SEQUENCE_SHA256 = {
    1: "bda0389c211dc728f53071e8d77f8b332a50b9ec3f22e024d73c66b59fc2d5d1",
    2: "c81826685dc906e6bbf9d160e43fcb3986c146ecdc5ec020e73eb7c36ad05b98",
    3: "99ea1973aa5dccf00d7f4b241dc6c8a902d393b06d52aab60f3e161592385011",
}
SESSION_ITEMS_SHA256 = {
    1: "814dc518da9de7d631209a95022bf2d9e5bfc4401d53b2d93cc60fbc85d0b478",
    2: "3cb86c9c176828405cc1cc838a119b7f4bd848a7d28f612d1624844342da0c37",
    3: "baf03acb08aa4f26e1fa5eed63c02349ee59854391c4d1433640aa86e22ac483",
}

ROW_CENTERS_MM = (225, 275, 325)
COLUMN_CENTERS_MM = (-75, -25, 25, 75)
QUADRANTS = {
    "Q0": (-15, -15),
    "Q1": (-15, 15),
    "Q2": (15, -15),
    "Q3": (15, 15),
}
QUADRANT_NAMES = {
    "Q0": "x_minus_y_minus",
    "Q1": "x_minus_y_plus",
    "Q2": "x_plus_y_minus",
    "Q3": "x_plus_y_plus",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_item(
    *,
    row: int,
    column: int,
    session_index: int,
    subset_role: str,
    position_kind: str,
    yaw: int,
    quadrant: str | None,
    replicate_index: int,
) -> dict:
    cell_index = (row - 1) * 4 + column - 1
    x_mm = ROW_CENTERS_MM[row - 1]
    y_mm = COLUMN_CENTERS_MM[column - 1]
    if quadrant is not None:
        dx_mm, dy_mm = QUADRANTS[quadrant]
        x_mm += dx_mm
        y_mm += dy_mm
    position_token = "center" if quadrant is None else quadrant.lower()
    plan_item_id = (
        f"real96_s{session_index:02d}_r{row}c{column}_{subset_role}_{position_token}_"
        f"rep{replicate_index:02d}_yaw{yaw:02d}"
    )
    order_sha256 = _sha256(f"{COLLECTION_PLAN_ID}|session={session_index}|{plan_item_id}".encode())
    return {
        "plan_item_id": plan_item_id,
        "session_id": f"task1_real96_s{session_index:02d}",
        "session_index": session_index,
        "session_order": None,
        "global_order": None,
        "order_sha256": order_sha256,
        "cell": f"r{row}c{column}",
        "row": row,
        "column": column,
        "cell_index": cell_index,
        "subset_role": subset_role,
        "subset_memberships": ["Real48", "Real96"] if subset_role == "core" else ["Real96"],
        "real48_member": subset_role == "core",
        "position_kind": position_kind,
        "quadrant": quadrant,
        "quadrant_name": QUADRANT_NAMES[quadrant] if quadrant is not None else None,
        "replicate_index": replicate_index,
        "x_forward_m": x_mm / 1000,
        "y_lateral_m": y_mm / 1000,
        "yaw_degrees_modulo_90": yaw,
        "nominal_pose_key": f"r{row}c{column}|x={x_mm}|y={y_mm}|yaw={yaw}",
        "control_source": "leader_follower",
    }


def real96_items() -> list[dict]:
    by_session: dict[int, list[dict]] = {index: [] for index in range(1, 5)}
    for row in range(1, 4):
        for column in range(1, 5):
            index = (row - 1) * 4 + column - 1
            definitions = (
                (1, "core", "center", 0, None, 1),
                (1, "extension", "offset", 45, f"Q{(index + 3) % 4}", 1),
                (2, "core", "offset", 45, f"Q{(index + 2) % 4}", 1),
                (2, "extension", "center", 0, None, 2),
                (3, "core", "center", 45, None, 1),
                (3, "extension", "offset", 0, f"Q{(index + 1) % 4}", 1),
                (4, "core", "offset", 0, f"Q{index % 4}", 1),
                (4, "extension", "center", 45, None, 2),
            )
            for session, subset, kind, yaw, quadrant, replicate in definitions:
                by_session[session].append(
                    _make_item(
                        row=row,
                        column=column,
                        session_index=session,
                        subset_role=subset,
                        position_kind=kind,
                        yaw=yaw,
                        quadrant=quadrant,
                        replicate_index=replicate,
                    )
                )
    ordered: list[dict] = []
    for session_index in range(1, 5):
        session_items = sorted(by_session[session_index], key=lambda item: item["order_sha256"])
        for session_order, item in enumerate(session_items, 1):
            item["session_order"] = session_order
            item["global_order"] = len(ordered) + 1
            ordered.append(item)
    return ordered


def session_items(session_index: int) -> list[dict]:
    return [item for item in real96_items() if item["session_index"] == session_index]


def compact_session_bytes(session_index: int) -> bytes:
    return json.dumps(session_items(session_index), ensure_ascii=False, separators=(",", ":")).encode()


def session_sequence_bytes(session_index: int) -> bytes:
    return "".join(f"{item['plan_item_id']}\n" for item in session_items(session_index)).encode()


def validate_session_source(session_index: int) -> None:
    expected = SESSION_ITEMS_SHA256.get(session_index)
    if expected is None:
        raise ValueError(f"no independently transferred source hash for session {session_index}")
    actual = _sha256(compact_session_bytes(session_index))
    if actual != expected:
        raise RuntimeError(f"session {session_index} source hash mismatch: {actual} != {expected}")
    sequence_actual = _sha256(session_sequence_bytes(session_index))
    sequence_expected = SESSION_SEQUENCE_SHA256[session_index]
    if sequence_actual != sequence_expected:
        raise RuntimeError(
            f"session {session_index} sequence hash mismatch: {sequence_actual} != {sequence_expected}"
        )


def batch_spawns(session_index: int) -> list[dict]:
    validate_session_source(session_index)
    return [
        {
            **item,
            "spawn_id": item["plan_item_id"],
            "spawn_region": item["cell"],
            "spawn_x_cm": item["x_forward_m"] * 100,
            "spawn_y_cm": item["y_lateral_m"] * 100,
            "spawn_yaw_deg": item["yaw_degrees_modulo_90"],
        }
        for item in session_items(session_index)
    ]


def write_compact_session(path: Path, session_index: int) -> None:
    validate_session_source(session_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compact_session_bytes(session_index))
