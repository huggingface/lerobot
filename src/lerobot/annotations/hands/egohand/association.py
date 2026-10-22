# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Offline physical-instance association before handedness-conditioned pose.

Research-only frozen heuristic. No GT input. Screen order is used only for a
stable, separated bimanual anchor, never to sort arbitrary overlapping boxes.
Unknown observations stay missing. This is not a learned identity classifier.
"""

from dataclasses import asdict, dataclass
from functools import partial
from itertools import combinations, product
from unittest.mock import patch

import numpy as np


@dataclass(frozen=True)
class Policy:
    strong: float = 0.4  # Existing native detector threshold.
    weak: float = 0.15
    rescue_iou: float = 0.35
    duplicate_iou: float = 0.9
    containment: float = 0.9  # Same fixed duplicate predicate as v2.
    conflict_iou: float = 0.5
    independent_margin: float = 0.25
    max_gap_s: float = 0.5
    support_frames: int = 3
    max_center_distance: float = 1.0  # Normalized by larger box diagonal.


def iou(a, b):
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a), np.asarray(b)
    lo = np.maximum(a[:2] - a[2:4] / 2, b[:2] - b[2:4] / 2)
    hi = np.minimum(a[:2] + a[2:4] / 2, b[:2] + b[2:4] / 2)
    intersection = float(np.prod(np.maximum(hi - lo, 0)))
    return intersection / max(float(np.prod(a[2:4]) + np.prod(b[2:4])) - intersection, 1e-12)


def containment(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lo = np.maximum(a[:2] - a[2:4] / 2, b[:2] - b[2:4] / 2)
    hi = np.minimum(a[:2] + a[2:4] / 2, b[:2] + b[2:4] / 2)
    return float(np.prod(np.maximum(hi - lo, 0)) / max(min(np.prod(a[2:4]), np.prod(b[2:4])), 1e-12))


def proposals(record, policy):
    pool = []
    for c in sorted(record["L"] + record["R"], key=lambda c: -c[4]):
        if not np.isfinite(c).all() or min(c[2:4]) <= 0 or c[4] < policy.weak:
            continue
        # Collapse near-identical hypotheses across detector classes.
        if not any(iou(c, p) > policy.duplicate_iou for p in pool):
            pool.append(list(c))
    return pool[:5]


def stable(pool, t, c, direction, policy):
    """Birth/anchor support from multiple observations, not filled boxes."""
    for offset in range(1, policy.support_frames):
        index = t + offset * direction
        if not 0 <= index < len(pool):
            return False
        matches = [p for p in pool[index] if p[4] > policy.strong and iou(c, p) >= policy.rescue_iou]
        if not matches:
            return False
        c = max(matches, key=lambda p: iou(c, p))
    return True


def anchors(pool, policy):
    result = []
    for t, candidates in enumerate(pool):
        for a, b in combinations(candidates, 2):
            if min(a[4], b[4]) <= policy.strong or iou(a, b) > 0:
                continue
            if not any(stable(pool, t, a, d, policy) and stable(pool, t, b, d, policy) for d in (-1, 1)):
                continue
            left, right = sorted((a, b), key=lambda c: c[0])
            result.append((min(a[4], b[4]), t, left, right))
    return result


def assign_physical(records, *, fps=30.0, policy=Policy()):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Finite positive FPS required")
    pool = [proposals(record, policy) for record in records]
    boxes = np.zeros((len(records), 2, 3), np.float32)
    valid = np.zeros((len(records), 2), bool)
    stats = {
        "rescued": 0,
        "label_overridden": 0,
        "crossed": 0,
        "conflict_pairs_rejected": 0,
        "unknown_identity_frames": 0,
        "anchor_frame": -1,
    }
    candidates = anchors(pool, policy)
    if not candidates:
        stats["unknown_identity_frames"] = len(records)
        return {s: (boxes[:, h], valid[:, h]) for h, s in enumerate(("left", "right"))}, stats
    _, anchor, a, b = max(candidates, key=lambda row: (row[0], -row[1]))
    stats["anchor_frame"] = anchor
    max_gap = max(1, round(policy.max_gap_s * fps))

    def record(t, selected):
        for h, c in enumerate(selected):
            if c is not None:
                boxes[t, h] = [c[0], c[1], max(c[2:4]) * 1.4]
                valid[t, h] = True
                stats["rescued"] += int(c[4] <= policy.strong)
                stats["label_overridden"] += int(c[5] != h + 1)
        stats["crossed"] += int(all(c is not None for c in selected) and selected[0][0] > selected[1][0])

    record(anchor, (a, b))
    for direction in (-1, 1):
        state, last = [a, b], [anchor, anchor]
        for t in range(anchor + direction, len(pool) if direction == 1 else -1, direction):
            active = [abs(t - last[h]) <= max_gap for h in range(2)]
            costs = {}
            for h, c in product(range(2), range(len(pool[t]))):
                proposal = pool[t][c]
                overlap = iou(proposal, state[h]) if active[h] else 0.0
                distance = np.linalg.norm(np.asarray(proposal[:2]) - state[h][:2]) / max(
                    np.linalg.norm(proposal[2:4]), np.linalg.norm(state[h][2:4]), 1
                )
                matched = (
                    active[h]
                    and distance <= policy.max_center_distance
                    and (proposal[4] > policy.strong or overlap >= policy.rescue_iou)
                )
                # Association requires motion/overlap support; absent hypotheses
                # score zero, so weak unmatched proposals do not win by default.
                if matched:
                    score = proposal[4] - policy.strong + 1.5 * overlap - 0.5 * distance
                    costs[h, c] = (score, True)
                elif proposal[4] > policy.strong and stable(pool, t, proposal, direction, policy):
                    costs[h, c] = (proposal[4] - policy.strong, False)
            best = (0.0, (None, None))
            options = [None] + list(range(len(pool[t])))
            for left, right in product(options, repeat=2):
                chosen = (left, right)
                if left is not None and left == right:
                    continue
                if any((h, c) not in costs for h, c in enumerate(chosen) if c is not None):
                    continue
                if left is not None and right is not None:
                    x, y = pool[t][left], pool[t][right]
                    conflict = containment(x, y) >= policy.containment or iou(x, y) > policy.conflict_iou
                    # Genuine interacting tracks may overlap if both maintain
                    # distinct, recent, independently preferred associations.
                    independent = all(abs(t - last[h]) <= policy.support_frames for h in range(2)) and (
                        iou(x, state[0]) - iou(x, state[1]) > policy.independent_margin
                        and iou(y, state[1]) - iou(y, state[0]) > policy.independent_margin
                    )
                    if conflict and not independent:
                        stats["conflict_pairs_rejected"] += 1
                        continue
                births = [h for h, c in enumerate(chosen) if c is not None and not costs[h, c][1]]
                if births:
                    if left is None or right is None:
                        continue  # A lone unanchored track has unknown identity.
                    if iou(pool[t][left], pool[t][right]) > 0:
                        continue  # A birth cannot borrow the existing hand's box.
                    if len(births) == 2 and (any(active) or pool[t][left][0] >= pool[t][right][0]):
                        continue  # Reanchor only when both tracks were lost.
                score = sum(costs[h, c][0] for h, c in enumerate(chosen) if c is not None)
                if score > best[0]:
                    best = (score, chosen)
            selected = [pool[t][c] if c is not None else None for c in best[1]]
            record(t, selected)
            for h, c in enumerate(selected):
                if c is not None:
                    state[h], last[h] = c, t
    return {s: (boxes[:, h], valid[:, h]) for h, s in enumerate(("left", "right"))}, stats


def annotations_from_candidates(raw, matrix, width, fps):
    from gem.common.detectors.hand import HandDetection
    from gem.common.detectors.hand_boxes import smooth_hand_boxes
    from gem.common.detectors.hand_tracking import track_hand_detections

    detections = [[HandDetection(**d) for d in frame] for frame in raw]
    with patch("gem.common.detectors.hand_tracking._assign", partial(assign_physical, fps=fps)):
        tracks = track_hand_detections(detections, width, enforce_lr_order=False)
    tracks = smooth_hand_boxes(tracks, np.arange(len(raw)) / fps)
    annotations = {
        t: {
            "intrinsics": np.asarray(matrix).tolist(),
            "instances": [
                {
                    "track_id": side,
                    "side": side,
                    "bbox": tracks.boxes_xys[side][t].tolist(),
                    "bbox_observed": bool(tracks.raw_present[side][t]),
                }
                for side in ("left", "right")
                if tracks.emitted[side][t]
            ],
        }
        for t in range(len(raw))
    }
    return annotations, {
        "policy": "physical-anchor-bidirectional-v3",
        "parameters": asdict(Policy()),
        "stats": tracks.stats,
        "limitation": "Box/temporal evidence only; ambiguous real hand overlap may be rejected. Offline anchor assumes ego screen order when separated.",
    }
