import re
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.policies.safediff_vla import stageplan as sp
from lerobot.policies.safediff_vla.utils import IMAGE_MODALITY, TEXT_MODALITY, STATE_MODALITY


def _tokens(b=2, n_img=12, n_txt=5, d=32):
    n = n_img + n_txt + 1
    tok = torch.randn(b, n, d)
    mod = torch.cat([torch.full((n_img,), IMAGE_MODALITY), torch.full((n_txt,), TEXT_MODALITY), torch.tensor([STATE_MODALITY])]).repeat(b, 1)
    return tok, torch.ones(b, n, dtype=torch.bool), mod


def test_head_shapes_finite_and_unit_rot():
    head = sp.StagePlanHead(32, d_model=32, n_heads=4, n_layers=1)
    tok, pad, mod = _tokens()
    out = head(tok, pad, mod, torch.randn(2, 7))
    assert out["xyz"].shape == (2, 4, 3) and out["rot_sc"].shape == (2, 4, 3, 2) and out["release_logit"].shape == (2,)
    assert all(torch.isfinite(v).all() for v in out.values())
    assert torch.allclose(out["rot_sc"].norm(dim=-1), torch.ones(2, 4, 3), atol=1e-5)


def test_camera_ids_split_into_equal_blocks():
    head = sp.StagePlanHead(32, d_model=32, n_heads=4, n_layers=1)
    _, _, mod = _tokens(b=1, n_img=12)
    ids = head.camera_ids(mod == IMAGE_MODALITY)[0, :12]
    assert ids.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def _poses():
    e = np.array([-np.pi / 2, -np.pi / 2, 0.0])
    return np.array([[0.0, 0.5, 0.40, *e], [0.0, 0.62, 0.38, *e], [0.0, 0.50, 0.38, *e], [0.2, 0.3, 0.3, 0.0, 0.0, 0.0]])


def _run(release: bool, steps=400):
    c = sp.StageController(sp.StageConstants())
    c.set_plan(_poses(), release)
    ee, eul = np.array([0.0, 0.25, 0.47]), np.array([3.14, 0.02, -1.6])
    log = []
    for _ in range(steps):
        pose, grip = c.next_action(ee, eul)
        ee, eul = pose[:3].copy(), pose[3:].copy()  # perfect tracker
        log.append((c.phase, grip, ee.copy()))
    return log


def test_controller_stage_order_and_close_at_p1():
    log = _run(release=False)
    phases = [p for p, _, _ in log]
    first = {k: phases.index(k) for k in ("PREGRASP", "GRASP", "CLOSE", "POSTGRASP", "HOLD")}
    assert first["PREGRASP"] < first["GRASP"] < first["CLOSE"] < first["POSTGRASP"] < first["HOLD"]
    ic = next(i for i, (_, g, _) in enumerate(log) if g == sp.GRIPPER_CLOSED)
    assert np.linalg.norm(log[ic][2] - _poses()[1, :3]) < 1e-6  # close happens only once P1 is reached
    assert all(g == sp.GRIPPER_CLOSED for _, g, _ in log[ic:])  # release_valid=0 -> never opens
    assert np.linalg.norm(log[-1][2] - _poses()[2, :3]) < 1e-6


def test_controller_release_opens_at_p3_only():
    log = _run(release=True)
    io = next(i for i, (_, g, _) in enumerate(log) if i > 0 and g == sp.GRIPPER_OPEN and log[i - 1][1] == sp.GRIPPER_CLOSED)
    assert np.linalg.norm(log[io][2] - _poses()[3, :3]) < 1e-6
    assert log[-1][1] == sp.GRIPPER_OPEN


def test_segment_duration_from_global_speed_only():
    c = sp.StageController(sp.StageConstants(v_xyz=0.01, v_rot=0.05))
    seg = c._segment(np.zeros(3), np.zeros(3), np.array([0.1, 0, 0, 0, 0, 0]))
    assert len(seg) == 10 and np.allclose(seg[-1][0][:3], [0.1, 0, 0])


def test_policy_reads_only_allowed_inputs():
    """Static inference-information audit: the StagePlan inference code (docstrings/comments excluded) never references
    simulator/oracle/task-identity symbols."""
    import ast

    tree = ast.parse(Path(sp.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    for forbidden in ("get_grasped_keypoints", "is_grasped", "target_entity", "oracle", "physics", "VLABenchEnv", "select_book", "select_drink", "select_toy", "select_fruit", "select_poker", "select_mahjong", "task_name", "expert"):
        assert forbidden not in code, forbidden
