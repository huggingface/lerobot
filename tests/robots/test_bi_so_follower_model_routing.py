from unittest.mock import patch

from lerobot.robots.so_follower import (
    SO100FollowerConfig,
    SO101FollowerConfig,
)
from lerobot.robots.bi_so_follower import (
    BiSOFollower,
    BiSOFollowerConfig,
)


def _make_config(left_type, right_type):
    return BiSOFollowerConfig(
        left_arm_config=left_type(port="/dev/null"),
        right_arm_config=right_type(port="/dev/null"),
    )


def _verify(left_type, right_type, expected_left, expected_right):
    config = _make_config(left_type, right_type)
    captured = []

    class FakeFollower:
        supports_position_hold = True

        def __init__(self, cfg):
            captured.append(cfg)
            self.config = cfg
            self.cameras = {}

    with patch(
        "lerobot.robots.bi_so_follower.bi_so_follower.SOFollower",
        FakeFollower,
    ):
        BiSOFollower(config)

    assert len(captured) == 2
    assert captured[0].type == expected_left
    assert captured[1].type == expected_right

    return captured


def test_dual_so100_preserves_both_models():
    _verify(
        SO100FollowerConfig,
        SO100FollowerConfig,
        "so100_follower",
        "so100_follower",
    )


def test_dual_so101_preserves_both_models():
    _verify(
        SO101FollowerConfig,
        SO101FollowerConfig,
        "so101_follower",
        "so101_follower",
    )


def test_mixed_so100_so101_preserves_models():
    _verify(
        SO100FollowerConfig,
        SO101FollowerConfig,
        "so100_follower",
        "so101_follower",
    )


def test_mixed_so101_so100_preserves_models():
    _verify(
        SO101FollowerConfig,
        SO100FollowerConfig,
        "so101_follower",
        "so100_follower",
    )
