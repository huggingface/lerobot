import math

import numpy as np
import pytest
import torch

from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    _decode_rotation,
    to_absolute_poses,
    to_relative_poses,
)
from lerobot.utils.constants import ACTION, OBS_STATE

REF_KEY = "observation.ee_pose"
NAMES = {
    "axis_angle": ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"],
    "rot6d": ["x", "y", "z", *[f"r6d_{i}" for i in range(6)], "proximal", "distal"],
}


def _random_poses(shape, fmt):
    pos = torch.randn(*shape, 3)
    rotvec = torch.randn(*shape, 3)
    rotvec = rotvec / rotvec.norm(dim=-1, keepdim=True) * torch.rand(*shape, 1) * (math.pi - 0.1)
    if fmt == "axis_angle":
        rot = rotvec
    else:
        rot = _decode_rotation(rotvec, "axis_angle")[..., :2, :].reshape(*shape, 6)
    return torch.cat([pos, rot, torch.rand(*shape, 2)], dim=-1)


def _indices(fmt):
    return [0, 1, 2], list(range(3, 6 if fmt == "axis_angle" else 9))


@pytest.mark.parametrize("fmt", ["axis_angle", "rot6d"])
@pytest.mark.parametrize("shape", [(4, 10), (4,)])
def test_pose_roundtrip_keeps_non_pose_dims(fmt, shape):
    pos, rot = _indices(fmt)
    actions = _random_poses(shape, fmt)
    reference = _random_poses((4,), fmt)[:, pos + rot]

    relative = to_relative_poses(actions, reference, pos, rot, fmt)
    torch.testing.assert_close(relative[..., -2:], actions[..., -2:])
    torch.testing.assert_close(
        to_absolute_poses(relative, reference, pos, rot, fmt), actions, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("fmt", ["axis_angle", "rot6d"])
def test_without_reference_key_the_first_action_is_identity(fmt):
    pos, rot = _indices(fmt)
    step = RelativeActionsProcessorStep(
        enabled=True, mode="pose", rotation_format=fmt, action_names=NAMES[fmt]
    )
    actions = _random_poses((2, 5), fmt)
    out = step(create_transition(observation={}, action=actions))[TransitionKey.ACTION]

    identity = torch.zeros(len(rot)) if fmt == "axis_angle" else torch.tensor([1.0, 0, 0, 0, 1, 0])
    torch.testing.assert_close(out[:, 0, pos], torch.zeros(2, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[:, 0, rot], identity.expand(2, -1), atol=1e-5, rtol=0)

    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)
    passthrough = absolute(create_transition(action=out))[TransitionKey.ACTION]
    torch.testing.assert_close(passthrough, out)


@pytest.mark.parametrize("fmt", ["axis_angle", "rot6d"])
def test_reference_key_roundtrip_and_is_hidden_from_the_model(fmt):
    pos, rot = _indices(fmt)
    step = RelativeActionsProcessorStep(
        enabled=True, mode="pose", rotation_format=fmt, action_names=NAMES[fmt], reference_key=REF_KEY
    )
    actions = _random_poses((2, 5), fmt)
    reference = _random_poses((2,), fmt)[:, pos + rot]
    state = torch.rand(2, 2)
    out = step(create_transition(observation={REF_KEY: reference, OBS_STATE: state}, action=actions))

    assert REF_KEY not in out[TransitionKey.OBSERVATION]
    torch.testing.assert_close(out[TransitionKey.OBSERVATION][OBS_STATE], state)
    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)
    recovered = absolute(create_transition(action=out[TransitionKey.ACTION]))[TransitionKey.ACTION]
    torch.testing.assert_close(recovered, actions, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "nope"},
        {"mode": "pose", "rotation_format": "quat"},
        {"mode": "pose", "action_names": ["x", "y", "z", "rx", "ry", "rz"]},
        {"mode": "pose", "action_names": None},
    ],
)
def test_invalid_pose_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        step = RelativeActionsProcessorStep(enabled=True, **kwargs)
        step(create_transition(observation={}, action=torch.zeros(1, 2, 6)))


def test_pose_names_match_exactly():
    names = ["proximal", "x", "y", "z", "ax", "ay", "az"]
    step = RelativeActionsProcessorStep(enabled=True, mode="pose", action_names=names)
    assert step._pose_indices() == ([1, 2, 3], [4, 5, 6])


def test_relative_stats_match_the_step():
    datasets = pytest.importorskip("datasets")
    from lerobot.datasets.compute_stats import compute_relative_action_stats

    fmt, chunk = "axis_angle", 4
    pos, rot = _indices(fmt)
    actions = _random_poses((12,), fmt).numpy()
    episodes = np.repeat([0, 1], 6)
    hf = datasets.Dataset.from_dict({ACTION: actions.tolist(), "episode_index": episodes.tolist()})
    features = {ACTION: {"shape": (8,), "names": NAMES[fmt]}}
    stats = compute_relative_action_stats(hf, features, chunk, mode="pose")

    chunks = [
        to_relative_poses(
            torch.from_numpy(actions[s : s + chunk])[None],
            torch.from_numpy(actions[s, pos + rot])[None],
            pos,
            rot,
            fmt,
        )[0]
        for s in [0, 1, 2, 6, 7, 8]
    ]
    expected = torch.cat(chunks).numpy()
    np.testing.assert_allclose(stats["mean"], expected.mean(0), atol=1e-5)
    np.testing.assert_allclose(stats["max"], expected.max(0), atol=1e-5)


def test_recompute_stats_records_the_pose_settings(tmp_path, empty_lerobot_dataset_factory):
    pytest.importorskip("datasets")
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.utils import RELATIVE_ACTION_PATH
    from lerobot.utils.io_utils import load_json

    fmt = "rot6d"
    features = {
        ACTION: {"dtype": "float32", "shape": (11,), "names": NAMES[fmt]},
        REF_KEY: {"dtype": "float32", "shape": (9,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "ds", features=features, use_videos=False)
    for _ in range(2):
        for pose in _random_poses((8,), fmt).numpy():
            dataset.add_frame({ACTION: pose, REF_KEY: pose[:9], "task": "t"})
        dataset.save_episode()
    dataset.finalize()

    kwargs = {
        "relative_action_mode": "pose",
        "relative_pose_rotation_format": fmt,
        "relative_reference_key": REF_KEY,
    }
    recompute_stats(dataset, relative_action=True, chunk_size=4, **kwargs)
    step = RelativeActionsProcessorStep(mode="pose", rotation_format=fmt, reference_key=REF_KEY)
    assert load_json(dataset.root / RELATIVE_ACTION_PATH) == step.stats_signature(4)
    assert dataset.meta.stats[ACTION]["mean"].shape == (11,)

    recompute_stats(dataset)
    assert not (dataset.root / RELATIVE_ACTION_PATH).exists()
